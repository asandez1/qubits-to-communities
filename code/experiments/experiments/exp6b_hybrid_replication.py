#!/usr/bin/env python3
"""
Experiment 6b: Replication & Validation of the Hybrid Pipeline Finding

Experiment 6 showed: quantum pipeline finds a DIFFERENT matching than classical
with ZERO regret (both equally optimal). This could be:
  (a) Real effect: quantum breaks ties in a systematic way
  (b) Degeneracy artifact: the problem had ties, noise picked one
  (c) Hardware noise: V3 output is random, not structured
  (d) Structural bias: V3 systematically prefers certain pairings
  (e) Nothing: V3 is just classical + noise

This experiment runs three tests that together discriminate between these:

  TEST 1 — MULTIPLE INSTANCES: Run the hybrid pipeline on 6 different
    matching cycles (different categories, different cycle numbers).
    If quantum consistently picks alternatives with zero regret → real effect.
    If it's inconsistent → likely artifact.

  TEST 2 — REPEATED SAME INSTANCE: Submit the SAME V3 circuits to ibm_fez
    3 times. If quantum always picks the same alternative → structured.
    If the choices vary → noise-driven (not a real effect).

  TEST 3 — NOISE CONTROL: Simulate "classical + calibrated noise" and
    check if it produces the same tiebreak distribution as quantum.
    If yes → quantum is just noisy classical.
    If no → quantum has structure beyond noise.

Budget: 3-4 hardware jobs total (batched).
"""

import json, logging, os, sys, time
from collections import Counter, defaultdict
import numpy as np
from scipy import stats
from itertools import product as iter_product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Reuse all exp6 machinery
from experiments.exp6_hybrid_pipeline import (
    build_v3_circuit, get_ancilla_prob, extract_matching_cycle,
    compute_classical_weights, build_qubo, solve_qubo_exhaustive,
    extract_assignments, train_v3_on_skills,
)


# =====================================================================
# Harvest multiple matching instances
# =====================================================================

def harvest_instances(config_path, num_members=30, n_cycles_to_scan=10,
                      min_pairs=4, max_pairs=12, seed=42,
                      prefer_snapshot: bool = True):
    """
    Return 8 matching instances for the hardware benchmark.

    If a frozen v2 snapshot exists at
    `results/exp6d/hardware_instances_v2.json` (created by
    `scripts/snapshot_hardware_instances.py`), load from it via the
    v1-compat bridge — this is the paper's canonical path and requires no
    simulation engine at load time.

    Otherwise (legacy path), run the v1 simulation from `config_path` with
    the given seed and scan matching cycles. Kept for backward-compat and
    for regenerating the snapshot; not used by any current paper experiment.
    """
    # --- Preferred path: frozen v2 snapshot via bridge -------------------
    if prefer_snapshot:
        default_snapshot = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "exp6d", "hardware_instances_v2.json",
        )
        if os.path.exists(default_snapshot):
            from core.benchmark_fixtures import hardware_instances_v1_compat
            return hardware_instances_v1_compat(snapshot_path=default_snapshot)

    # --- Legacy path: run v1 simulation from config ----------------------
    from core.simulation_engine import SimulationEngine
    from core.matching_service import MatchingStrategy, MatchingService

    with open(config_path) as f:
        config = json.load(f)

    engine = SimulationEngine(
        config=config, matching_strategy=MatchingStrategy.GREEDY,
        num_members=num_members, num_cycles=500, seed=seed,
        topology_sample_interval=99999,
    )
    engine.run()

    categories = list(config["categories"].keys())
    matcher = MatchingService(MatchingStrategy.QUBO, config.get("matching", {}))
    instances = []

    # Sample across cycles to get variety
    sample_cycles = [50, 100, 150, 200, 250, 300, 350, 400, 450, 499]

    for target_cycle in sample_cycles:
        # Set a specific RNG state for this cycle
        rng_state = np.random.default_rng(seed + target_cycle)
        for m in engine.members_list:
            m._rng = rng_state

        prices = engine.economy.get_all_prices()
        all_reqs, all_offs = [], []
        for m in engine.members_list:
            all_reqs.extend(m.generate_requests(target_cycle, categories))
            all_offs.extend(m.declare_availability(target_cycle, prices))

        for cat in categories:
            reqs = [(i, r) for i, r in enumerate(all_reqs) if r.category == cat]
            offs = [(i, o) for i, o in enumerate(all_offs) if o.category == cat]
            pairs = []
            for r_idx, req in reqs:
                for o_idx, offer in offs:
                    if offer.provider_id == req.requester_id:
                        continue
                    member = engine.members.get(offer.provider_id)
                    if member is None or member.skill_levels.get(cat, 0) < 0.3:
                        continue
                    pairs.append((r_idx, o_idx, req, offer, member))

            if min_pairs <= len(pairs) <= max_pairs:
                instances.append({
                    "engine": engine,
                    "category": cat,
                    "cycle": target_cycle,
                    "price": prices[cat],
                    "categories": categories,
                    "prices": prices,
                    "requests": reqs,
                    "offers": offs,
                    "eligible_pairs": pairs,
                    "matcher": matcher,
                })

    # Diversify — pick instances from different categories/cycles
    seen = set()
    diverse = []
    for inst in instances:
        key = (inst["category"], len(inst["eligible_pairs"]))
        if key not in seen:
            diverse.append(inst)
            seen.add(key)
    return diverse[:8]  # cap at 8 to respect hardware budget


# =====================================================================
# Analyze a single pipeline result
# =====================================================================

def analyze_degeneracy(Q, n_vars, tolerance=1e-6):
    """
    How many assignments achieve the optimal energy (or near-optimal)?
    Returns: (n_optima, gap_to_next_best)
    """
    energies = []
    for bits in iter_product([0, 1], repeat=n_vars):
        e = 0.0
        for (i, j), val in Q.items():
            if i == j:
                e += val * bits[i]
            else:
                e += val * bits[i] * bits[j]
        energies.append((e, bits))
    energies.sort(key=lambda x: x[0])

    best_e = energies[0][0]
    n_optima = sum(1 for e, _ in energies if abs(e - best_e) < tolerance)
    # Find the first strictly higher-energy solution
    gap = 0.0
    for e, _ in energies:
        if abs(e - best_e) >= tolerance:
            gap = e - best_e
            break
    return n_optima, gap, [bits for e, bits in energies[:n_optima]]


# =====================================================================
# Quantum weight computation (batched across instances)
# =====================================================================

def batched_v3_on_hardware(instances, v3_theta, backend_name, shots=4096, run_label=""):
    """
    Batch all V3 circuit evaluations for multiple instances into one
    hardware job. Returns dict: instance_idx -> {pair_idx -> p1}.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    emb_cats = ['electrical', 'cooking', 'transport']
    def mem_emb(m):
        raw = np.array([m.skill_levels.get(c, 0.0) for c in emb_cats])
        return 0.1 + (np.pi - 0.2) * raw
    def tsk_emb(cat):
        raw = np.array([1.0 if c == cat else 0.0 for c in emb_cats])
        return 0.1 + (np.pi - 0.2) * raw

    # Build all circuits
    all_circuits = []
    index_map = []  # (inst_idx, pair_idx)
    for inst_idx, inst in enumerate(instances):
        for pair_idx, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
            v1 = mem_emb(member)
            v2 = tsk_emb(req.category)
            all_circuits.append(build_v3_circuit(v1, v2, v3_theta))
            index_map.append((inst_idx, pair_idx))

    logger.info(f"  [{run_label}] Building {len(all_circuits)} V3 circuits for {len(instances)} instances")

    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    transpiled = [pm.run(c) for c in all_circuits]

    logger.info(f"  [{run_label}] Backend: {backend.name}, submitting batch of {len(transpiled)}")
    sampler = SamplerV2(mode=backend)
    t0 = time.time()
    job = sampler.run(transpiled, shots=shots)
    logger.info(f"  [{run_label}] Job: {job.job_id()} — waiting...")
    result = job.result()
    hw_time = time.time() - t0
    logger.info(f"  [{run_label}] Hardware: {hw_time:.1f}s")

    # Parse results
    probs = defaultdict(dict)
    for circuit_idx, (inst_idx, pair_idx) in enumerate(index_map):
        counts = result[circuit_idx].data.meas.get_counts()
        p1 = get_ancilla_prob(counts)
        probs[inst_idx][pair_idx] = p1
    return dict(probs), {"job_id": job.job_id(), "hw_time": hw_time}


def compute_weights_from_probs(inst, probs_for_instance):
    """Given {pair_idx -> p1}, compute full composite weights."""
    matcher = inst["matcher"]
    prices = inst["prices"]
    alpha = matcher._weights["alpha"]
    beta = matcher._weights["beta"]
    gamma = matcher._weights["gamma"]
    delta = matcher._weights["delta"]

    weights = {}
    similarities = {}
    for pair_idx, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
        p1 = probs_for_instance[pair_idx]
        v3_sim = 1.0 - p1
        similarities[pair_idx] = v3_sim
        rep = member.reputation
        skill = member.skill_levels.get(req.category, 0.0)
        price = prices.get(req.category, 5.0) / 10.0
        weights[pair_idx] = alpha * v3_sim + beta * rep + gamma * skill + delta * price
    return weights, similarities


# =====================================================================
# Noise control: classical + calibrated Gaussian noise
# =====================================================================

def noise_control_simulation(inst, sigma_estimate, n_trials=20):
    """
    Run n_trials with classical weights + calibrated Gaussian noise on the
    similarity term. Returns the distribution of selected bitstrings.
    """
    from core.member import CommunityMember
    matcher = inst["matcher"]
    prices = inst["prices"]
    cats = inst["categories"]
    alpha = matcher._weights["alpha"]
    beta = matcher._weights["beta"]
    gamma = matcher._weights["gamma"]
    delta = matcher._weights["delta"]

    rng = np.random.default_rng(1234)
    trial_solutions = []

    for trial in range(n_trials):
        weights = {}
        for idx, (r_idx, o_idx, req, offer, member) in enumerate(inst["eligible_pairs"]):
            mem_emb = member.get_capability_embedding(cats)
            task_emb = CommunityMember.get_task_embedding(req.category, cats)
            nm, nt = np.linalg.norm(mem_emb), np.linalg.norm(task_emb)
            cos_sim = float(np.dot(mem_emb, task_emb) / (nm * nt)) if nm > 0 and nt > 0 else 0
            # Add Gaussian noise to the similarity term only
            noisy_sim = cos_sim + rng.normal(0, sigma_estimate)
            rep = member.reputation
            skill = member.skill_levels.get(req.category, 0.0)
            price = prices.get(req.category, 5.0) / 10.0
            weights[idx] = alpha * noisy_sim + beta * rep + gamma * skill + delta * price

        Q, var_map, _ = build_qubo(inst, weights)
        bits, _ = solve_qubo_exhaustive(Q, len(inst["eligible_pairs"]))
        trial_solutions.append(tuple(bits))

    # Count unique solutions
    counter = Counter(trial_solutions)
    return counter


# =====================================================================
# Main experiment
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument("--members", type=int, default=30)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--spsa-iter", type=int, default=150)
    parser.add_argument("--n-repetitions", type=int, default=3,
                        help="Number of hardware runs on same instance (TEST 2)")
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp6b"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "config", "community_economy.json")

    logger.info("=" * 70)
    logger.info("EXPERIMENT 6b: Hybrid Pipeline Replication & Validation")
    logger.info(f"  Backend: {args.backend}")
    logger.info("=" * 70)

    # --- Setup ---
    logger.info("\n[SETUP] Harvesting matching instances across cycles/categories...")
    instances = harvest_instances(config_path, args.members)
    logger.info(f"  Harvested {len(instances)} diverse instances:")
    for i, inst in enumerate(instances):
        logger.info(f"    {i}: {inst['category']} @ cycle {inst['cycle']}: "
                    f"{len(inst['eligible_pairs'])} pairs")

    logger.info("\n[SETUP] Training V3 once on simulator...")
    v3_theta = train_v3_on_skills(config_path, num_members=100,
                                    n_iter=args.spsa_iter, shots=2048)

    # Precompute classical results + degeneracy analysis for all instances
    logger.info("\n[SETUP] Computing classical baseline + degeneracy for all instances...")
    classical_results = []
    for i, inst in enumerate(instances):
        n = len(inst["eligible_pairs"])
        w_c, s_c = compute_classical_weights(inst)
        Q_c, vm_c, _ = build_qubo(inst, w_c)
        c_bits, c_energy = solve_qubo_exhaustive(Q_c, n)
        n_optima, gap, optima_bits = analyze_degeneracy(Q_c, n)
        classical_results.append({
            "n_pairs": n,
            "weights": w_c,
            "similarities": s_c,
            "Q": Q_c,
            "var_map": vm_c,
            "bits": c_bits,
            "energy": c_energy,
            "n_optima": n_optima,
            "energy_gap": gap,
            "optima": optima_bits,
        })
        logger.info(f"  [{i}] {inst['category']}: n_optima={n_optima}, gap={gap:.4f}, "
                    f"assignments={sum(c_bits)}")

    # =================================================================
    # TEST 1: Multiple instances, one hardware job each (batched)
    # =================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: MULTIPLE INSTANCES on hardware (is the effect consistent?)")
    logger.info("=" * 70)

    try:
        probs_dict, hw_meta_1 = batched_v3_on_hardware(
            instances, v3_theta, args.backend, args.shots, "TEST1")
    except Exception as e:
        logger.error(f"  TEST 1 FAILED: {e}")
        import traceback; traceback.print_exc()
        return

    test1_results = []
    for i, inst in enumerate(instances):
        n = len(inst["eligible_pairs"])
        w_q, s_q = compute_weights_from_probs(inst, probs_dict[i])
        Q_q, vm_q, _ = build_qubo(inst, w_q)
        q_bits, q_energy = solve_qubo_exhaustive(Q_q, n)

        # Correlation with classical
        sc_arr = np.array([classical_results[i]["similarities"][k]
                           for k in sorted(classical_results[i]["similarities"].keys())])
        sq_arr = np.array([s_q[k] for k in sorted(s_q.keys())])
        r_sim = float(stats.pearsonr(sc_arr, sq_arr)[0]) if np.std(sq_arr) > 0 else 0.0

        # Objective comparison
        c_bits = classical_results[i]["bits"]
        w_c = classical_results[i]["weights"]
        obj_c_under_c = sum(w_c[k] * c_bits[k] for k in range(n))
        obj_q_under_c = sum(w_c[k] * q_bits[k] for k in range(n))
        regret = (obj_c_under_c - obj_q_under_c) / obj_c_under_c if obj_c_under_c != 0 else 0

        identical = tuple(c_bits) == tuple(q_bits)
        # If quantum solution is one of the classical optima
        q_is_co_optimal = tuple(q_bits) in [tuple(b) for b in classical_results[i]["optima"]]

        test1_results.append({
            "instance_idx": i,
            "category": inst["category"],
            "cycle": inst["cycle"],
            "n_pairs": n,
            "n_classical_optima": classical_results[i]["n_optima"],
            "energy_gap": classical_results[i]["energy_gap"],
            "sim_correlation": r_sim,
            "classical_bits": c_bits,
            "quantum_bits": q_bits,
            "identical": identical,
            "q_is_co_optimal": q_is_co_optimal,
            "classical_regret": float(regret),
        })

        status = "IDENTICAL" if identical else ("CO-OPTIMAL" if q_is_co_optimal else "DIFFERENT")
        logger.info(f"  [{i}] {inst['category']:10s} n={n:2d} n_opt={classical_results[i]['n_optima']:2d} "
                    f"r={r_sim:.3f} regret={regret*100:+.2f}% [{status}]")

    # =================================================================
    # TEST 2: Same instances, repeated runs (is the choice stable?)
    # =================================================================
    logger.info("\n" + "=" * 70)
    logger.info(f"TEST 2: REPEATED SAME INSTANCES (is the tiebreak systematic or noise-driven?)")
    logger.info("=" * 70)

    # Focus on first 3 instances for repetition (smaller batch per repetition)
    instances_for_rep = instances[:3]
    repetition_bits = defaultdict(list)  # inst_idx -> list of bits-tuples

    for rep in range(args.n_repetitions):
        logger.info(f"\n  Repetition {rep+1}/{args.n_repetitions}...")
        try:
            probs_rep, _ = batched_v3_on_hardware(
                instances_for_rep, v3_theta, args.backend, args.shots,
                f"REP{rep+1}")
        except Exception as e:
            logger.error(f"  Rep {rep+1} FAILED: {e}")
            continue

        for i, inst in enumerate(instances_for_rep):
            n = len(inst["eligible_pairs"])
            w_q, _ = compute_weights_from_probs(inst, probs_rep[i])
            Q_q, _, _ = build_qubo(inst, w_q)
            q_bits, _ = solve_qubo_exhaustive(Q_q, n)
            repetition_bits[i].append(tuple(q_bits))

    test2_results = []
    for i, inst in enumerate(instances_for_rep):
        bits_list = repetition_bits[i]
        counter = Counter(bits_list)
        n_unique = len(counter)
        most_common, most_count = counter.most_common(1)[0] if counter else (None, 0)
        consistency = most_count / len(bits_list) if bits_list else 0
        test2_results.append({
            "instance_idx": i,
            "category": inst["category"],
            "n_repetitions": len(bits_list),
            "n_unique_solutions": n_unique,
            "consistency": consistency,
            "most_common": list(most_common) if most_common else None,
        })
        logger.info(f"  [{i}] {inst['category']:10s}: {len(bits_list)} reps → "
                    f"{n_unique} unique solutions, consistency={consistency*100:.0f}%")

    # =================================================================
    # TEST 3: Noise control (is V3 just classical + noise?)
    # =================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: NOISE CONTROL (classical + calibrated noise)")
    logger.info("=" * 70)

    # Estimate the sigma that approximates V3 hardware deviation from classical
    sigma_estimates = []
    for i, inst in enumerate(instances):
        if i >= len(test1_results):
            continue
        sc_arr = np.array([classical_results[i]["similarities"][k]
                           for k in sorted(classical_results[i]["similarities"].keys())])
        # V3 hardware similarities reconstructed from probs
        probs_i = probs_dict[i]
        sq_arr = np.array([1.0 - probs_i[k] for k in sorted(probs_i.keys())])
        residuals = sq_arr - sc_arr
        sigma_estimates.append(np.std(residuals))

    sigma = float(np.mean(sigma_estimates))
    logger.info(f"  Calibrated noise σ = {sigma:.4f} (from V3 hardware residuals)")

    test3_results = []
    for i, inst in enumerate(instances_for_rep):
        counter = noise_control_simulation(inst, sigma, n_trials=30)
        n_unique_noise = len(counter)
        noise_consistency = counter.most_common(1)[0][1] / sum(counter.values()) if counter else 0

        # Compare with TEST 2
        hw_n_unique = test2_results[i]["n_unique_solutions"] if i < len(test2_results) else 0
        hw_consistency = test2_results[i]["consistency"] if i < len(test2_results) else 0

        test3_results.append({
            "instance_idx": i,
            "category": inst["category"],
            "noise_n_unique": n_unique_noise,
            "noise_consistency": noise_consistency,
            "hw_n_unique": hw_n_unique,
            "hw_consistency": hw_consistency,
        })
        logger.info(f"  [{i}] {inst['category']:10s}: "
                    f"noise→{n_unique_noise} uniq ({noise_consistency*100:.0f}% stable) "
                    f"vs HW→{hw_n_unique} uniq ({hw_consistency*100:.0f}% stable)")

    # =================================================================
    # SUMMARY — classify the finding
    # =================================================================
    logger.info("\n" + "=" * 70)
    logger.info("VERDICT: Is the hybrid pipeline finding real?")
    logger.info("=" * 70)

    # TEST 1 summary
    n_instances = len(test1_results)
    n_identical = sum(1 for r in test1_results if r["identical"])
    n_co_optimal = sum(1 for r in test1_results if r["q_is_co_optimal"] and not r["identical"])
    n_different = n_instances - n_identical - n_co_optimal
    avg_regret = np.mean([r["classical_regret"] for r in test1_results])
    avg_sim_corr = np.mean([r["sim_correlation"] for r in test1_results])
    degenerate_instances = sum(1 for r in test1_results if r["n_classical_optima"] > 1)

    logger.info(f"\n  TEST 1 (n={n_instances} instances):")
    logger.info(f"    Instances with ties (n_optima > 1): {degenerate_instances}/{n_instances}")
    logger.info(f"    Quantum = classical: {n_identical}")
    logger.info(f"    Quantum co-optimal (different but equal): {n_co_optimal}")
    logger.info(f"    Quantum suboptimal: {n_different}")
    logger.info(f"    Mean classical regret: {avg_regret*100:.2f}%")
    logger.info(f"    Mean similarity correlation (HW vs classical): {avg_sim_corr:.3f}")

    # TEST 2 summary
    logger.info(f"\n  TEST 2 (consistency across {args.n_repetitions} hardware runs):")
    avg_hw_consistency = np.mean([r["consistency"] for r in test2_results])
    logger.info(f"    Mean consistency: {avg_hw_consistency*100:.0f}%")
    logger.info(f"    (100% = always same solution; 50% = random among 2)")

    # TEST 3 summary
    logger.info(f"\n  TEST 3 (noise control with σ = {sigma:.4f}):")
    avg_noise_consistency = np.mean([r["noise_consistency"] for r in test3_results])
    logger.info(f"    Mean noise-control consistency: {avg_noise_consistency*100:.0f}%")
    logger.info(f"    HW consistency: {avg_hw_consistency*100:.0f}%")
    logger.info(f"    → If HW >> noise control: quantum has structure beyond noise")
    logger.info(f"    → If HW ≈ noise control: V3 is classical + noise")

    # Classify the finding
    logger.info("\n  CLASSIFICATION:")
    if n_different > n_co_optimal + n_identical:
        verdict = "QUANTUM PRODUCES DIFFERENT (possibly suboptimal) MATCHINGS"
    elif avg_regret < 0.01 and n_co_optimal > 0:
        if avg_hw_consistency > avg_noise_consistency + 0.15:
            verdict = "REAL EFFECT: quantum systematically breaks ties with higher consistency than noise"
        elif avg_hw_consistency > 0.7:
            verdict = "STRUCTURED TIEBREAKING (but not distinguishable from correlated noise)"
        else:
            verdict = "HARDWARE NOISE DOMINATES (quantum ≈ noisy classical)"
    else:
        verdict = "MIXED / INCONCLUSIVE"

    logger.info(f"    → {verdict}")

    # Save all results
    summary = {
        "backend": args.backend,
        "n_instances": n_instances,
        "n_degenerate_instances": degenerate_instances,
        "test1_instances": test1_results,
        "test2_repetitions": test2_results,
        "test3_noise_control": test3_results,
        "calibrated_sigma": sigma,
        "mean_classical_regret": float(avg_regret),
        "mean_sim_correlation": float(avg_sim_corr),
        "hw_consistency": float(avg_hw_consistency),
        "noise_consistency": float(avg_noise_consistency),
        "verdict": verdict,
    }
    out_path = os.path.join(args.output, "exp6b_replication.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
