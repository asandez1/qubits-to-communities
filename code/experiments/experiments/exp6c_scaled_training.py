#!/usr/bin/env python3
"""
Experiment 6c: Scaled V3 Training — Testing the Q-Manifold Scaling Law

Experiment 6b showed that V3 with 40 training pairs fails to generalize
across diverse matching instances (r = 0.19, 2.44% regret). Sández (2025)
projects classical parity at ~100-120 training pairs via the empirical
4.25x scaling law from 12→40 pairs.

THIS experiment tests that prediction directly on OrquestIA's domain:
  - Train V3 at FOUR sizes: 40, 80, 100, 120 pairs
  - Each training on simulator (local, free)
  - For each trained V3, evaluate on the SAME 8 diverse instances from Exp 6b
  - Measure the sim correlation and classical regret at each training size
  - Plot the scaling curve

If the scaling law holds:
  r(40) = 0.19 → r(80) = ~0.40 → r(100) = ~0.60 → r(120) = ~0.70+

Only the final (best-trained) V3 is evaluated on ibm_fez — 1 hardware job.

This:
  (a) Tests the scaling law predicted by Sández (2025) in a new domain
  (b) If confirmed: validates the hybrid pipeline claim once data is scaled
  (c) If contradicted: reveals Q-Manifold scaling is ConceptNet-specific
"""

import json, logging, os, sys, time
from collections import Counter, defaultdict
import numpy as np
from scipy import stats
from itertools import product as iter_product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from experiments.exp6_hybrid_pipeline import (
    build_v3_circuit, get_ancilla_prob, compute_classical_weights,
    build_qubo, solve_qubo_exhaustive,
)
from experiments.exp6b_hybrid_replication import (
    harvest_instances, analyze_degeneracy, compute_weights_from_probs,
)


# =====================================================================
# Scaled V3 training
# =====================================================================

def build_training_set(config_path, num_members, n_pairs_per_class, seed=42):
    """Build balanced training set with exactly n_pairs_per_class per class."""
    from core.simulation_engine import SimulationEngine
    from core.matching_service import MatchingStrategy

    with open(config_path) as f:
        config = json.load(f)
    engine = SimulationEngine(
        config=config, matching_strategy=MatchingStrategy.GREEDY,
        num_members=num_members, num_cycles=501, seed=seed,
        topology_sample_interval=99999,
    )
    engine.run()

    cats = ['electrical', 'cooking', 'transport']
    embeddings = {}
    for m in engine.members_list:
        raw = np.array([m.skill_levels.get(c, 0.0) for c in cats])
        embeddings[m.member_id] = 0.1 + (np.pi - 0.2) * raw

    members = engine.members_list
    similar, dissimilar = [], []
    for i, mi in enumerate(members):
        for j, mj in enumerate(members):
            if j <= i: continue
            ei, ej = embeddings[mi.member_id], embeddings[mj.member_id]
            raw_i = np.array([mi.skill_levels.get(c, 0) for c in cats])
            raw_j = np.array([mj.skill_levels.get(c, 0) for c in cats])
            ni, nj = np.linalg.norm(raw_i), np.linalg.norm(raw_j)
            if ni > 0 and nj > 0:
                cos = float(np.dot(raw_i, raw_j) / (ni * nj))
                if cos > 0.8:
                    similar.append((ei, ej, 1.0))
                elif cos < 0.2:
                    dissimilar.append((ei, ej, 0.0))

    rng = np.random.default_rng(seed)
    rng.shuffle(similar); rng.shuffle(dissimilar)
    n_per = min(n_pairs_per_class, len(similar), len(dissimilar))
    train_data = similar[:n_per] + dissimilar[:n_per]
    rng.shuffle(train_data)
    return train_data


def train_v3_spsa(train_data, n_iter=200, shots=2048, lr=0.15, seed=42):
    """SPSA training (from exp5_proven_v3.py logic)."""
    from qiskit.primitives import StatevectorSampler
    sampler = StatevectorSampler()
    np.random.seed(seed)

    n_params = 21
    theta = np.random.uniform(-0.01, 0.01, n_params)
    best_theta, best_loss = theta.copy(), float("inf")

    for it in range(n_iter):
        delta = 2 * np.random.randint(0, 2, size=n_params) - 1
        c_k = 0.1 / (it + 1) ** 0.101
        a_k = lr / (it + 1) ** 0.602

        circuits = []
        for v1, v2, _ in train_data:
            circuits.append(build_v3_circuit(v1, v2, theta + c_k * delta))
            circuits.append(build_v3_circuit(v1, v2, theta - c_k * delta))

        job = sampler.run(circuits, shots=shots)
        result = job.result()

        lp, lm = 0.0, 0.0
        for i, (v1, v2, target) in enumerate(train_data):
            pp = get_ancilla_prob(result[2*i].data.meas.get_counts())
            pm = get_ancilla_prob(result[2*i+1].data.meas.get_counts())
            pp = np.clip(pp, 1e-7, 1-1e-7); pm = np.clip(pm, 1e-7, 1-1e-7)
            lp += -(target*np.log(pp) + (1-target)*np.log(1-pp))
            lm += -(target*np.log(pm) + (1-target)*np.log(1-pm))
        lp /= len(train_data); lm /= len(train_data)

        theta = theta - a_k * (lp - lm) / (2 * c_k) * delta
        if (lp + lm) / 2 < best_loss:
            best_loss = (lp + lm) / 2
            best_theta = theta.copy()

        if it % 50 == 0:
            logger.info(f"      Iter {it}: loss={(lp+lm)/2:.4f} best={best_loss:.4f}")

    return best_theta, best_loss


# =====================================================================
# Evaluate V3 on instances (simulator — local, fast)
# =====================================================================

def evaluate_v3_on_instances(instances, v3_theta, classical_results, on_hardware=False,
                              backend_name=None, shots=4096, run_label=""):
    """
    For each instance, compute quantum weights and measure correlation + regret.
    If on_hardware=True, submit batched V3 circuits to IBM; else use simulator.
    """
    from qiskit.primitives import StatevectorSampler
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    emb_cats = ['electrical', 'cooking', 'transport']
    def mem_emb(m):
        raw = np.array([m.skill_levels.get(c, 0.0) for c in emb_cats])
        return 0.1 + (np.pi - 0.2) * raw
    def tsk_emb(cat):
        raw = np.array([1.0 if c == cat else 0.0 for c in emb_cats])
        return 0.1 + (np.pi - 0.2) * raw

    all_circuits = []
    index_map = []
    for inst_idx, inst in enumerate(instances):
        for pair_idx, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
            v1 = mem_emb(member)
            v2 = tsk_emb(req.category)
            all_circuits.append(build_v3_circuit(v1, v2, v3_theta))
            index_map.append((inst_idx, pair_idx))

    if on_hardware:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        backend = service.backend(backend_name)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        transpiled = [pm.run(c) for c in all_circuits]
        logger.info(f"    [{run_label}] Submitting {len(transpiled)} circuits to {backend_name}")
        sampler = SamplerV2(mode=backend)
        t0 = time.time()
        job = sampler.run(transpiled, shots=shots)
        logger.info(f"    [{run_label}] Job: {job.job_id()}")
        result = job.result()
        logger.info(f"    [{run_label}] Hardware: {time.time()-t0:.1f}s")
        hw_meta = {"job_id": job.job_id(), "hw_time": time.time()-t0}
    else:
        sampler = StatevectorSampler()
        result = sampler.run(all_circuits, shots=shots).result()
        hw_meta = None

    # Parse results
    probs = defaultdict(dict)
    for circuit_idx, (inst_idx, pair_idx) in enumerate(index_map):
        counts = result[circuit_idx].data.meas.get_counts()
        p1 = get_ancilla_prob(counts)
        probs[inst_idx][pair_idx] = p1

    # Per-instance analysis
    results = []
    for i, inst in enumerate(instances):
        n = len(inst["eligible_pairs"])
        w_q, s_q = compute_weights_from_probs(inst, probs[i])
        Q_q, vm_q, _ = build_qubo(inst, w_q)
        q_bits, q_energy = solve_qubo_exhaustive(Q_q, n)

        # Compare with classical
        sc_arr = np.array([classical_results[i]["similarities"][k]
                           for k in sorted(classical_results[i]["similarities"].keys())])
        sq_arr = np.array([s_q[k] for k in sorted(s_q.keys())])
        r_sim = float(stats.pearsonr(sc_arr, sq_arr)[0]) if np.std(sq_arr) > 0 else 0.0

        c_bits = classical_results[i]["bits"]
        w_c = classical_results[i]["weights"]
        obj_c = sum(w_c[k] * c_bits[k] for k in range(n))
        obj_q = sum(w_c[k] * q_bits[k] for k in range(n))
        regret = (obj_c - obj_q) / obj_c if obj_c != 0 else 0

        identical = tuple(c_bits) == tuple(q_bits)
        q_is_co_optimal = tuple(q_bits) in [tuple(b) for b in classical_results[i]["optima"]]

        results.append({
            "instance_idx": i,
            "category": inst["category"],
            "n_pairs": n,
            "sim_correlation": r_sim,
            "classical_regret": float(regret),
            "identical": identical,
            "q_is_co_optimal": q_is_co_optimal,
        })

    mean_r = float(np.mean([r["sim_correlation"] for r in results]))
    mean_regret = float(np.mean([r["classical_regret"] for r in results]))
    n_co_optimal = sum(1 for r in results if r["q_is_co_optimal"])
    n_identical = sum(1 for r in results if r["identical"])

    return {
        "instances": results,
        "mean_sim_correlation": mean_r,
        "mean_classical_regret": mean_regret,
        "n_co_optimal": n_co_optimal,
        "n_identical": n_identical,
        "hw_meta": hw_meta,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 40, 50, 60],
                        help="Training pairs per class (total = 2x)")
    parser.add_argument("--train-members", type=int, default=100)
    parser.add_argument("--eval-members", type=int, default=30)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--spsa-iter", type=int, default=150)
    parser.add_argument("--skip-hardware", action="store_true",
                        help="Run only simulator (no hardware job)")
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp6c"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "config", "community_economy.json")

    logger.info("=" * 70)
    logger.info("EXPERIMENT 6c: Scaled V3 Training — Testing Q-Manifold Scaling Law")
    logger.info(f"  Training sizes per class: {args.sizes} (total = 2x)")
    logger.info(f"  Backend: {args.backend} (hardware run at final size only)")
    logger.info("=" * 70)

    # --- SETUP: harvest instances + classical baseline ---
    logger.info("\n[SETUP] Harvesting 8 evaluation instances...")
    instances = harvest_instances(config_path, args.eval_members)
    logger.info(f"  Got {len(instances)} instances:")

    classical_results = []
    for i, inst in enumerate(instances):
        n = len(inst["eligible_pairs"])
        w_c, s_c = compute_classical_weights(inst)
        Q_c, vm_c, _ = build_qubo(inst, w_c)
        c_bits, c_energy = solve_qubo_exhaustive(Q_c, n)
        n_opt, gap, opt_bits = analyze_degeneracy(Q_c, n)
        classical_results.append({
            "n_pairs": n,
            "weights": w_c,
            "similarities": s_c,
            "bits": c_bits,
            "energy": c_energy,
            "n_optima": n_opt,
            "optima": opt_bits,
        })
        logger.info(f"    [{i}] {inst['category']:10s} n={n:2d} n_optima={n_opt}")

    # --- SCALING: train V3 at each size, evaluate on simulator ---
    sim_results = {}
    trained_thetas = {}

    for size in args.sizes:
        total_pairs = 2 * size
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING SIZE: {total_pairs} pairs ({size} per class)")
        logger.info(f"{'='*70}")

        train_data = build_training_set(config_path, args.train_members, size)
        logger.info(f"  Training set: {len(train_data)} pairs")

        t0 = time.time()
        theta, loss = train_v3_spsa(train_data, n_iter=args.spsa_iter, shots=2048)
        logger.info(f"  Training done: {time.time()-t0:.1f}s, loss={loss:.4f}")
        trained_thetas[total_pairs] = theta

        logger.info(f"  Evaluating on simulator ({len(instances)} instances)...")
        sim_eval = evaluate_v3_on_instances(
            instances, theta, classical_results,
            on_hardware=False, shots=args.shots)
        sim_results[total_pairs] = sim_eval

        logger.info(f"  Simulator result at {total_pairs} pairs:")
        logger.info(f"    Mean sim correlation: r = {sim_eval['mean_sim_correlation']:+.4f}")
        logger.info(f"    Mean classical regret: {sim_eval['mean_classical_regret']*100:+.2f}%")
        logger.info(f"    Co-optimal: {sim_eval['n_co_optimal']}/{len(instances)}")
        logger.info(f"    Identical: {sim_eval['n_identical']}/{len(instances)}")

    # --- Build scaling curve ---
    logger.info("\n" + "=" * 70)
    logger.info("SCALING CURVE (simulator)")
    logger.info("=" * 70)
    logger.info(f"  {'n_pairs':>10} {'mean_r':>10} {'regret %':>12} {'co-opt':>10}")
    sizes_sorted = sorted(sim_results.keys())
    for n in sizes_sorted:
        r = sim_results[n]
        logger.info(f"  {n:>10} {r['mean_sim_correlation']:>+10.4f} "
                    f"{r['mean_classical_regret']*100:>+12.2f}% "
                    f"{r['n_co_optimal']}/{len(instances)}")

    # --- Hardware run at best size ---
    hw_result = None
    if not args.skip_hardware:
        # Pick the size with best sim correlation
        best_size = max(sim_results.keys(),
                        key=lambda s: sim_results[s]["mean_sim_correlation"])
        logger.info(f"\n{'='*70}")
        logger.info(f"HARDWARE RUN at {best_size} pairs (best simulator r = "
                     f"{sim_results[best_size]['mean_sim_correlation']:.4f})")
        logger.info(f"{'='*70}")

        try:
            hw_result = evaluate_v3_on_instances(
                instances, trained_thetas[best_size], classical_results,
                on_hardware=True, backend_name=args.backend, shots=args.shots,
                run_label=f"{best_size}pairs_HW")

            logger.info(f"\n  Hardware result at {best_size} pairs:")
            logger.info(f"    Mean sim correlation: r = {hw_result['mean_sim_correlation']:+.4f}")
            logger.info(f"    Mean classical regret: {hw_result['mean_classical_regret']*100:+.2f}%")
            logger.info(f"    Co-optimal: {hw_result['n_co_optimal']}/{len(instances)}")
            logger.info(f"    Identical: {hw_result['n_identical']}/{len(instances)}")

            # Compare with simulator at same size
            sim_r = sim_results[best_size]["mean_sim_correlation"]
            hw_r = hw_result["mean_sim_correlation"]
            delta = hw_r - sim_r
            logger.info(f"\n    HW vs Sim at {best_size} pairs: Δr = {delta:+.4f}")
            if delta > 0.05:
                logger.info(f"    → HARDWARE ADVANTAGE (like Q-Manifold +18% effect)")
            elif delta > -0.05:
                logger.info(f"    → Hardware matches simulator")
            else:
                logger.info(f"    → Hardware degraded")

        except Exception as e:
            logger.error(f"  Hardware FAILED: {e}")
            import traceback; traceback.print_exc()

    # --- VERDICT on scaling law ---
    logger.info("\n" + "=" * 70)
    logger.info("SCALING LAW VERIFICATION (Sández 2025)")
    logger.info("=" * 70)

    # Q-Manifold law: 12→40 pairs gives 4.25x improvement (r=0.08 → r=0.34)
    # Projected parity (r ≈ 0.86) at ~100-120 pairs
    logger.info(f"\n  Q-Manifold reference:")
    logger.info(f"    12 pairs → r = 0.08 (baseline)")
    logger.info(f"    40 pairs → r = 0.34 (4.25x)")
    logger.info(f"    Projection: ~100-120 pairs → r ≈ 0.80-0.86")

    logger.info(f"\n  OrquestIA observed (simulator):")
    for n in sizes_sorted:
        logger.info(f"    {n:3d} pairs → r = {sim_results[n]['mean_sim_correlation']:+.4f}")

    # Fit linear trend
    if len(sizes_sorted) >= 2:
        ns = np.array(sizes_sorted)
        rs = np.array([sim_results[n]["mean_sim_correlation"] for n in sizes_sorted])
        if np.std(ns) > 0:
            slope, intercept, r_fit, _, _ = stats.linregress(ns, rs)
            logger.info(f"\n  Linear fit: r(n) = {slope:.4f}·n + {intercept:.4f} (R² = {r_fit**2:.3f})")
            # Extrapolate to Q-Manifold parity target
            target_r = 0.70
            if slope > 1e-6:
                n_target = (target_r - intercept) / slope
                logger.info(f"  Extrapolation: r = 0.70 reached at n ≈ {n_target:.0f} pairs")

    # Save
    summary = {
        "sizes_tested": sizes_sorted,
        "simulator_results": {str(n): {
            "mean_sim_correlation": sim_results[n]["mean_sim_correlation"],
            "mean_classical_regret": sim_results[n]["mean_classical_regret"],
            "n_co_optimal": sim_results[n]["n_co_optimal"],
            "n_identical": sim_results[n]["n_identical"],
            "per_instance": sim_results[n]["instances"],
        } for n in sizes_sorted},
        "hardware_result": {
            "size": best_size if not args.skip_hardware else None,
            "mean_sim_correlation": hw_result["mean_sim_correlation"] if hw_result else None,
            "mean_classical_regret": hw_result["mean_classical_regret"] if hw_result else None,
            "n_co_optimal": hw_result["n_co_optimal"] if hw_result else None,
            "hw_meta": hw_result["hw_meta"] if hw_result else None,
        } if hw_result else None,
    }
    out_path = os.path.join(args.output, "exp6c_scaled.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
