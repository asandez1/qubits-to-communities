#!/usr/bin/env python3
"""
Experiment 6d: V3-Extended — 9D Embeddings, 19 Qubits

Hypothesis (from Exp 6/6b/6c analysis):
  The hybrid pipeline failure was caused by forced dimensionality reduction
  (6D → 3D) required to fit V3's 7-qubit architecture. This forced tasks
  into one-hot encoding [1,0,0] vs rich member vectors, creating a
  training/evaluation distribution mismatch.

Prediction:
  With enough qubits to represent the full skill space (9D per vector,
  requiring 2·9 + 1 = 19 qubits on ibm_fez's 156-qubit processor), both
  members AND tasks live in the same continuous distribution → no mismatch
  → hybrid pipeline correlation should recover toward r ≈ 0.7+.

Architecture changes vs V3 (7 qubits):
  - 9 qubits for member vector (full 6 skills + reputation + time + social_capital)
  - 9 qubits for task vector   (full 6-skill requirement vector, not one-hot)
  - 1 ancilla
  - Total: 19 qubits (well within ibm_fez's 156)

Everything else stays identical: SPSA training, BCE loss, ancilla measurement,
[0.1, π-0.1] scaling.

Budget: 1 hardware job. If it works, next step is multi-cycle replication.
"""

import json, logging, os, sys, time
from collections import defaultdict
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from experiments.exp6b_hybrid_replication import (
    harvest_instances, analyze_degeneracy,
)
from experiments.exp6_hybrid_pipeline import (
    compute_classical_weights, build_qubo, solve_qubo_exhaustive,
)


# =====================================================================
# V3-EXTENDED: 19-qubit circuit with 9D per vector
# =====================================================================

N_PER_VEC = 9  # 9 features per vector
N_TOTAL = 2 * N_PER_VEC + 1  # 19 qubits
ANC = N_TOTAL - 1  # index 18
N_PARAMS = 3 * N_TOTAL  # 57 trainable parameters (3 layers × 19 qubits)


def build_v3_extended(v1, v2, theta):
    """
    19-qubit V3 circuit — exact same structure as V3, just wider.
      - 9 RY encode v1, 9 RY encode v2
      - 3 trainable layers with CX routing to ancilla + cross-register CX
      - Measure all, extract ancilla
    """
    from qiskit import QuantumCircuit
    assert len(v1) == N_PER_VEC and len(v2) == N_PER_VEC
    n = N_PER_VEC
    qc = QuantumCircuit(N_TOTAL)

    # Encoding
    for i in range(n):
        qc.ry(float(v1[i]), i)
    for i in range(n):
        qc.ry(float(v2[i]), n + i)
    qc.barrier()

    # Layer 1: local rotations on all 19 qubits
    for i in range(N_TOTAL):
        qc.ry(float(theta[i]), i)
    qc.barrier()

    # Entanglement: all 18 data qubits → ancilla
    for i in range(n):
        qc.cx(i, ANC)
    for i in range(n):
        qc.cx(n + i, ANC)
    qc.barrier()

    # Layer 2
    for i in range(N_TOTAL):
        qc.ry(float(theta[N_TOTAL + i]), i)
    # Cross register CX (v1 ↔ v2)
    for i in range(n):
        qc.cx(i, n + i)
    qc.barrier()

    # Layer 3
    for i in range(N_TOTAL):
        qc.ry(float(theta[2 * N_TOTAL + i]), i)
    # Final ancilla connections
    qc.cx(0, ANC)
    qc.cx(n, ANC)

    qc.measure_all()
    return qc


def get_ancilla_prob(counts, n_total=N_TOTAL):
    total = sum(counts.values())
    p1 = 0
    for bs, count in counts.items():
        bs = bs.zfill(n_total)
        if bs[0] == '1':  # ancilla is qubit N_TOTAL-1 → leftmost bit
            p1 += count
    return p1 / total


# =====================================================================
# 9D Embeddings (no forced reduction)
# =====================================================================

# Axis order for both member and task vectors:
FULL_AXES = [
    'electrical', 'plumbing', 'tutoring', 'transport', 'cooking', 'childcare',
    'reputation', 'time_availability', 'social_capital',
]


def scale_to_v3_range(x):
    """Scale value in [0, 1] to [0.1, π-0.1]."""
    return 0.1 + (np.pi - 0.2) * x


def member_embedding_9d(member):
    """Full 9D member embedding."""
    vec = np.zeros(9)
    for i, cat in enumerate(FULL_AXES[:6]):
        vec[i] = member.skill_levels.get(cat, 0.0)
    vec[6] = member.reputation
    vec[7] = member.time_availability
    vec[8] = member.social_capital
    # Clip to [0, 1] (time can go slightly out via noise)
    vec = np.clip(vec, 0, 1)
    return scale_to_v3_range(vec)


def task_embedding_9d(task_category):
    """Full 9D task embedding — NOT one-hot anymore.

    The task requires a specific skill (main axis = 1.0), moderate reputation,
    and typical time/social requirements. This keeps the task vector in the
    same continuous distribution as member vectors.
    """
    vec = np.zeros(9)
    # Main skill axis
    for i, cat in enumerate(FULL_AXES[:6]):
        if cat == task_category:
            vec[i] = 1.0  # task requires this skill at full strength
        else:
            vec[i] = 0.0
    # Typical task requirements
    vec[6] = 0.5  # reputation requirement (moderate)
    vec[7] = 0.3  # time requirement (modest)
    vec[8] = 0.2  # social capital requirement (low)
    return scale_to_v3_range(vec)


def member_embedding_9d_raw(member):
    """Unscaled 9D vector (for computing ground-truth cosine similarity)."""
    vec = np.zeros(9)
    for i, cat in enumerate(FULL_AXES[:6]):
        vec[i] = member.skill_levels.get(cat, 0.0)
    vec[6] = member.reputation
    vec[7] = member.time_availability
    vec[8] = member.social_capital
    return np.clip(vec, 0, 1)


def task_embedding_9d_raw(task_category):
    vec = np.zeros(9)
    for i, cat in enumerate(FULL_AXES[:6]):
        vec[i] = 1.0 if cat == task_category else 0.0
    vec[6] = 0.5
    vec[7] = 0.3
    vec[8] = 0.2
    return vec


# =====================================================================
# Build training set: member-task pairs (the correct distribution)
# =====================================================================

def build_training_set(config_path, num_members, n_pairs_per_class, seed=42):
    """
    Build balanced training set of (member, task) pairs.
    Label = 1 if member's skill in task.category is high (> 0.6), else 0.
    This directly trains V3 for the evaluation distribution.
    """
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

    categories = list(config["categories"].keys())
    rng = np.random.default_rng(seed)

    similar, dissimilar = [], []
    for member in engine.members_list:
        for cat in categories:
            v1 = member_embedding_9d(member)
            v2 = task_embedding_9d(cat)
            skill = member.skill_levels.get(cat, 0.0)
            if skill > 0.6:
                similar.append((v1, v2, 1.0))
            elif skill < 0.1:
                dissimilar.append((v1, v2, 0.0))

    rng.shuffle(similar); rng.shuffle(dissimilar)
    n_per = min(n_pairs_per_class, len(similar), len(dissimilar))
    train_data = similar[:n_per] + dissimilar[:n_per]
    rng.shuffle(train_data)

    logger.info(f"  Training: {len(similar)} high-skill + {len(dissimilar)} low-skill "
                f"available → using {n_per} per class ({len(train_data)} total)")
    return train_data


# =====================================================================
# SPSA training (identical to proven V3 code, adapted for larger N_PARAMS)
# =====================================================================

def train_v3_extended(train_data, n_iter=200, shots=2048, lr=0.15, seed=42):
    """SPSA training. Uses AerSimulator (compiled C++, ~600× faster than StatevectorSampler at 19q)."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    sim = AerSimulator(method='statevector')

    class AerWrapper:
        def run(self, circuits, shots=shots):
            tqcs = transpile(circuits, sim, optimization_level=0)
            result = sim.run(tqcs, shots=shots).result()
            class R:
                def __init__(self, r): self._r = r
                def __getitem__(self, i):
                    class Data:
                        def __init__(self, counts):
                            class M:
                                def __init__(self, c): self._c = c
                                def get_counts(self): return self._c
                            self.meas = M(counts)
                    return type('X', (), {'data': Data(self._r.get_counts(i))})()
            return type('Job', (), {'result': lambda self: R(result)})()
    sampler = AerWrapper()
    np.random.seed(seed)

    theta = np.random.uniform(-0.01, 0.01, N_PARAMS)
    best_theta, best_loss = theta.copy(), float("inf")

    logger.info(f"  SPSA: {N_PARAMS} params, {n_iter} iters, {len(train_data)} pairs")

    for it in range(n_iter):
        delta = 2 * np.random.randint(0, 2, size=N_PARAMS) - 1
        c_k = 0.1 / (it + 1) ** 0.101
        a_k = lr / (it + 1) ** 0.602

        circuits = []
        for v1, v2, _ in train_data:
            circuits.append(build_v3_extended(v1, v2, theta + c_k * delta))
            circuits.append(build_v3_extended(v1, v2, theta - c_k * delta))

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

        if it % 25 == 0:
            logger.info(f"    Iter {it}: loss={(lp+lm)/2:.4f} best={best_loss:.4f}")

    logger.info(f"  Training done: best_loss={best_loss:.4f}")
    return best_theta, best_loss


# =====================================================================
# Evaluation on instances — same 8 instances as Exp 6b
# =====================================================================

def compute_v3_extended_weights(inst, theta, sampler, shots=4096):
    """Compute V3-extended similarity for each pair in a matching instance."""
    matcher = inst["matcher"]
    prices = inst["prices"]
    alpha = matcher._weights["alpha"]
    beta = matcher._weights["beta"]
    gamma = matcher._weights["gamma"]
    delta = matcher._weights["delta"]

    circuits = []
    for r_idx, o_idx, req, offer, member in inst["eligible_pairs"]:
        v1 = member_embedding_9d(member)
        v2 = task_embedding_9d(req.category)
        circuits.append(build_v3_extended(v1, v2, theta))

    result = sampler.run(circuits, shots=shots).result()

    weights = {}
    similarities = {}
    for idx, (r_idx, o_idx, req, offer, member) in enumerate(inst["eligible_pairs"]):
        counts = result[idx].data.meas.get_counts()
        p1 = get_ancilla_prob(counts)
        v3_sim = 1.0 - p1
        similarities[idx] = v3_sim

        rep = member.reputation
        skill = member.skill_levels.get(req.category, 0.0)
        price = prices.get(req.category, 5.0) / 10.0
        weights[idx] = alpha * v3_sim + beta * rep + gamma * skill + delta * price

    return weights, similarities, circuits


def compute_9d_cosine(inst):
    """Compute ground-truth cosine similarity on 9D raw vectors (same distribution as V3 sees)."""
    similarities = {}
    for idx, (r_idx, o_idx, req, offer, member) in enumerate(inst["eligible_pairs"]):
        v1 = member_embedding_9d_raw(member)
        v2 = task_embedding_9d_raw(req.category)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        cos = float(np.dot(v1, v2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
        similarities[idx] = cos
    return similarities


def evaluate_on_instances(instances, theta, classical_results, on_hardware=False,
                           backend_name=None, shots=4096):
    """Evaluate V3-extended on all instances. On sim or hardware."""
    from qiskit.primitives import StatevectorSampler
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    if on_hardware:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        backend = service.backend(backend_name)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        logger.info(f"  Backend: {backend.name} ({backend.num_qubits} qubits)")

        # Build all circuits, transpile, batch-submit
        all_circuits = []
        pair_map = []
        for inst_idx, inst in enumerate(instances):
            for pair_idx, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
                v1 = member_embedding_9d(member)
                v2 = task_embedding_9d(req.category)
                qc = build_v3_extended(v1, v2, theta)
                all_circuits.append(pm.run(qc))
                pair_map.append((inst_idx, pair_idx))

        logger.info(f"  Transpiled {len(all_circuits)} circuits")
        depths = [c.depth() for c in all_circuits]
        logger.info(f"  Depth range: {min(depths)}-{max(depths)}")

        sampler = SamplerV2(mode=backend)
        t0 = time.time()
        job = sampler.run(all_circuits, shots=shots)
        logger.info(f"  Job: {job.job_id()} — waiting...")
        result = job.result()
        hw_time = time.time() - t0
        logger.info(f"  Hardware: {hw_time:.1f}s")

        probs = defaultdict(dict)
        for c_idx, (inst_idx, pair_idx) in enumerate(pair_map):
            counts = result[c_idx].data.meas.get_counts()
            probs[inst_idx][pair_idx] = get_ancilla_prob(counts)
        hw_meta = {"job_id": job.job_id(), "hw_time": hw_time, "depth_range": [min(depths), max(depths)]}
    else:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        aer_sim = AerSimulator(method='statevector')
        probs = defaultdict(dict)
        for inst_idx, inst in enumerate(instances):
            circuits = []
            for r, o, req, offer, member in inst["eligible_pairs"]:
                v1 = member_embedding_9d(member)
                v2 = task_embedding_9d(req.category)
                circuits.append(build_v3_extended(v1, v2, theta))
            tqcs = transpile(circuits, aer_sim, optimization_level=0)
            result = aer_sim.run(tqcs, shots=shots).result()
            for pair_idx in range(len(inst["eligible_pairs"])):
                counts = result.get_counts(pair_idx)
                probs[inst_idx][pair_idx] = get_ancilla_prob(counts)
        hw_meta = None

    # Per-instance analysis
    results = []
    for i, inst in enumerate(instances):
        n = len(inst["eligible_pairs"])

        # Compute V3-extended weights
        matcher = inst["matcher"]
        prices = inst["prices"]
        alpha = matcher._weights["alpha"]; beta = matcher._weights["beta"]
        gamma = matcher._weights["gamma"]; delta = matcher._weights["delta"]

        v3_sims = {}
        weights_q = {}
        for pair_idx, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
            p1 = probs[i][pair_idx]
            v3_sim = 1.0 - p1
            v3_sims[pair_idx] = v3_sim
            rep = member.reputation
            skill = member.skill_levels.get(req.category, 0.0)
            price = prices.get(req.category, 5.0) / 10.0
            weights_q[pair_idx] = alpha * v3_sim + beta * rep + gamma * skill + delta * price

        # Solve quantum QUBO
        Q_q, vm_q, _ = build_qubo(inst, weights_q)
        q_bits, q_energy = solve_qubo_exhaustive(Q_q, n)

        # Compare similarity to classical 9D cosine (same distribution ground truth)
        cos_9d = compute_9d_cosine(inst)
        cos_arr = np.array([cos_9d[k] for k in sorted(cos_9d.keys())])
        v3_arr = np.array([v3_sims[k] for k in sorted(v3_sims.keys())])
        r_sim = float(stats.pearsonr(cos_arr, v3_arr)[0]) if np.std(v3_arr) > 0 else 0.0

        # Classical regret (using original classical weights from exp6 protocol)
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
            "sim_correlation_9d": r_sim,  # V3 vs 9D cosine (same distribution)
            "classical_regret": float(regret),
            "identical": identical,
            "q_is_co_optimal": q_is_co_optimal,
            "v3_similarities": v3_sims,
        })

    mean_r = float(np.mean([r["sim_correlation_9d"] for r in results]))
    mean_regret = float(np.mean([r["classical_regret"] for r in results]))
    n_co_optimal = sum(1 for r in results if r["q_is_co_optimal"])
    n_identical = sum(1 for r in results if r["identical"])

    return {
        "instances": results,
        "mean_sim_correlation_9d": mean_r,
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
    parser.add_argument("--members", type=int, default=30)
    parser.add_argument("--train-members", type=int, default=100)
    parser.add_argument("--n-train-per-class", type=int, default=40)
    parser.add_argument("--spsa-iter", type=int, default=150)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--skip-hardware", action="store_true")
    parser.add_argument("--theta-path", default=None,
                        help="Path to pretrained theta.npy. If provided, STEP 1 "
                             "(training) is skipped and the loaded theta is used "
                             "for both simulator and hardware evaluation. This is "
                             "the right flag for cross-backend differential tests "
                             "— it isolates backend-vs-backend effects by keeping "
                             "the circuit parameters identical.")
    parser.add_argument("--harvest-seed", type=int, default=42,
                        help="Seed passed to harvest_instances. Default 42 "
                             "reproduces the original Exp 6d instance draw. "
                             "Pass a different int (e.g. 7, 1234) to test on "
                             "a new set of 8 instances with the same theta — "
                             "the right flag for n=1 vs reproducible-effect tests.")
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp6d"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "config", "community_economy.json")

    logger.info("=" * 70)
    logger.info("EXPERIMENT 6d: V3-Extended (19 qubits, 9D embeddings)")
    logger.info(f"  Hypothesis: removing forced dim reduction fixes the hybrid pipeline")
    logger.info(f"  Architecture: {N_TOTAL} qubits = 2·{N_PER_VEC} data + 1 ancilla, "
                f"{N_PARAMS} trainable params")
    logger.info(f"  Backend: {args.backend}")
    logger.info("=" * 70)

    # --- SETUP: harvest 8 instances (same as Exp 6b) ---
    logger.info("\n[SETUP] Harvesting 8 diverse matching instances "
                f"(harvest-seed={args.harvest_seed})...")
    instances = harvest_instances(config_path, args.members, seed=args.harvest_seed)
    logger.info(f"  Got {len(instances)} instances")

    classical_results = []
    for i, inst in enumerate(instances):
        n = len(inst["eligible_pairs"])
        w_c, s_c = compute_classical_weights(inst)
        Q_c, vm_c, _ = build_qubo(inst, w_c)
        c_bits, c_energy = solve_qubo_exhaustive(Q_c, n)
        n_opt, gap, opt_bits = analyze_degeneracy(Q_c, n)
        classical_results.append({
            "n_pairs": n, "weights": w_c, "similarities": s_c,
            "bits": c_bits, "energy": c_energy,
            "n_optima": n_opt, "optima": opt_bits,
        })

    # --- STEP 1: Train V3-extended on simulator (or load pretrained theta) ---
    if args.theta_path:
        logger.info(f"\n[STEP 1] Skipped — loading pretrained theta from {args.theta_path}")
        theta = np.load(args.theta_path)
        assert theta.shape == (N_PARAMS,), (
            f"Loaded theta has shape {theta.shape}, expected ({N_PARAMS},). "
            f"Is this theta from a different architecture?"
        )
        loss = None
    else:
        logger.info(f"\n[STEP 1] Training V3-extended on member-task pairs...")
        train_data = build_training_set(config_path, args.train_members, args.n_train_per_class)
        t0 = time.time()
        theta, loss = train_v3_extended(train_data, n_iter=args.spsa_iter, shots=2048)
        logger.info(f"  Total training time: {time.time()-t0:.1f}s")

        # Persist theta for downstream experiments (Exp 7, cross-backend tests)
        theta_path = os.path.join(args.output, "theta.npy")
        np.save(theta_path, theta)
        logger.info(f"  Saved theta to {theta_path}")

    # --- STEP 2: Evaluate on simulator ---
    logger.info(f"\n[STEP 2] Evaluating on SIMULATOR (all 8 instances)...")
    sim_eval = evaluate_on_instances(instances, theta, classical_results,
                                       on_hardware=False, shots=args.shots)
    logger.info(f"\n  Simulator result:")
    logger.info(f"    Mean similarity correlation (vs 9D cosine): r = {sim_eval['mean_sim_correlation_9d']:+.4f}")
    logger.info(f"    Mean classical regret: {sim_eval['mean_classical_regret']*100:+.2f}%")
    logger.info(f"    Co-optimal: {sim_eval['n_co_optimal']}/{len(instances)}")
    logger.info(f"    Identical: {sim_eval['n_identical']}/{len(instances)}")

    # Per-instance breakdown
    logger.info(f"\n  Per-instance breakdown:")
    for r in sim_eval["instances"]:
        status = "IDENTICAL" if r["identical"] else ("CO-OPTIMAL" if r["q_is_co_optimal"] else "SUBOPTIMAL")
        logger.info(f"    [{r['instance_idx']}] {r['category']:10s} r={r['sim_correlation_9d']:+.3f} "
                    f"regret={r['classical_regret']*100:+.2f}% [{status}]")

    # --- STEP 3: Hardware ---
    hw_eval = None
    if not args.skip_hardware:
        logger.info(f"\n[STEP 3] Evaluating on {args.backend} (1 batched hardware job)...")
        try:
            hw_eval = evaluate_on_instances(instances, theta, classical_results,
                                              on_hardware=True, backend_name=args.backend,
                                              shots=args.shots)
            logger.info(f"\n  Hardware result:")
            logger.info(f"    Mean similarity correlation: r = {hw_eval['mean_sim_correlation_9d']:+.4f}")
            logger.info(f"    Mean classical regret: {hw_eval['mean_classical_regret']*100:+.2f}%")
            logger.info(f"    Co-optimal: {hw_eval['n_co_optimal']}/{len(instances)}")
            logger.info(f"    Identical: {hw_eval['n_identical']}/{len(instances)}")
        except Exception as e:
            logger.error(f"  Hardware FAILED: {e}")
            import traceback; traceback.print_exc()

    # --- VERDICT ---
    logger.info("\n" + "=" * 70)
    logger.info("VERDICT — Does removing dim reduction fix the hybrid pipeline?")
    logger.info("=" * 70)
    logger.info(f"\n  Reference points from earlier experiments:")
    logger.info(f"    Exp 6b (7q, 3D, member-member training): r = 0.19, regret = 2.44%")
    logger.info(f"    Exp 6c (7q, 3D, scaled 40→120): r flat at 0.20, regret ≈ 2.4%")
    logger.info(f"    ISOLATED V3 semantic (Exp 5d): r = 0.70")

    logger.info(f"\n  Exp 6d (19q, 9D, member-task training):")
    logger.info(f"    Simulator: r = {sim_eval['mean_sim_correlation_9d']:+.4f}, "
                f"regret = {sim_eval['mean_classical_regret']*100:+.2f}%")
    if hw_eval:
        logger.info(f"    Hardware:  r = {hw_eval['mean_sim_correlation_9d']:+.4f}, "
                    f"regret = {hw_eval['mean_classical_regret']*100:+.2f}%")

    sim_r = sim_eval["mean_sim_correlation_9d"]
    if sim_r > 0.5:
        verdict = "YES — removing dim reduction DOES fix the hybrid pipeline"
    elif sim_r > 0.3:
        verdict = "PARTIAL — improvement but not full recovery"
    else:
        verdict = "NO — dim reduction was not the sole issue"
    logger.info(f"\n  → {verdict}")

    # Save
    summary = {
        "architecture": "V3-extended: 19 qubits, 9D per vector",
        "n_params": N_PARAMS,
        "n_train_pairs": len(train_data) if not args.theta_path else None,
        "simulator": {
            "mean_sim_correlation_9d": sim_eval["mean_sim_correlation_9d"],
            "mean_classical_regret": sim_eval["mean_classical_regret"],
            "n_co_optimal": sim_eval["n_co_optimal"],
            "n_identical": sim_eval["n_identical"],
            "per_instance": sim_eval["instances"],
        },
        "hardware": {
            "mean_sim_correlation_9d": hw_eval["mean_sim_correlation_9d"] if hw_eval else None,
            "mean_classical_regret": hw_eval["mean_classical_regret"] if hw_eval else None,
            "n_co_optimal": hw_eval["n_co_optimal"] if hw_eval else None,
            "n_identical": hw_eval["n_identical"] if hw_eval else None,
            "hw_meta": hw_eval["hw_meta"] if hw_eval else None,
            "per_instance": hw_eval["instances"] if hw_eval else None,
        } if hw_eval else None,
        "verdict": verdict,
        "backend": args.backend,
        "theta_source": args.theta_path or "trained_this_run",
    }
    # Tag output with backend and harvest seed to avoid overwriting earlier runs.
    # Default (ibm_fez, seed=42) stays at the original filename for back-compat.
    if args.backend == "ibm_fez" and args.harvest_seed == 42:
        fname = "exp6d_v3_extended.json"
    else:
        suffix = f"_{args.backend}"
        if args.harvest_seed != 42:
            suffix += f"_seed{args.harvest_seed}"
        fname = f"exp6d_v3_extended{suffix}.json"
    summary["harvest_seed"] = args.harvest_seed
    out_path = os.path.join(args.output, fname)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
