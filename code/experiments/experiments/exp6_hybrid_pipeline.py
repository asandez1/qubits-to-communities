#!/usr/bin/env python3
"""
Experiment 6: End-to-End V3 → QUBO Hybrid Pipeline

Tests the paper's core architectural claim: Stage 1 (quantum semantic matching)
feeding Stage 3 (classical QUBO optimization) produces measurably different
coordination outcomes than an all-classical pipeline.

Protocol:
  1. Extract a real matching cycle from the simulation (N members, M requests)
  2. Compute matching weights TWO ways:
     (A) Classical: cosine similarity on skill embeddings
     (B) Quantum:   V3 circuit on ibm_fez — ancilla P(|0|) as similarity proxy
  3. Feed each weight set into the SAME classical QUBO solver
  4. Compare:
     - Do the two pipelines produce the SAME assignments?
     - If different, which produces higher total w·x (objective)?
     - Are the resulting social graph topologies different?
     - Is quantum-computed similarity noisier, smoother, or structurally biased?

Budget: 1-2 ibm_fez jobs (one batch of pairs).

This is the first hardware-validated end-to-end quantum-classical social
coordination pipeline. Whatever the result, it calibrates where quantum
hardware changes real sociological outcomes vs where it's a wash.
"""

import json, logging, os, sys, time
import numpy as np
from scipy import stats
from itertools import product as iter_product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# V3 CIRCUIT (exact proven code from Sández 2025)
# =====================================================================

def build_v3_circuit(v1, v2, theta):
    """Exact V3 from quantum_learning_v3.py — 7 qubits, 21 params."""
    from qiskit import QuantumCircuit
    n = len(v1)
    n_total = 2 * n + 1
    anc = n_total - 1
    qc = QuantumCircuit(n_total)

    # Encoding
    for i in range(n): qc.ry(float(v1[i]), i)
    for i in range(n): qc.ry(float(v2[i]), n + i)
    qc.barrier()

    # Layer 1
    for i in range(n_total): qc.ry(float(theta[i]), i)
    qc.barrier()

    # Data → ancilla
    for i in range(n): qc.cx(i, anc)
    for i in range(n): qc.cx(n + i, anc)
    qc.barrier()

    # Layer 2
    for i in range(n_total): qc.ry(float(theta[n_total + i]), i)
    for i in range(n): qc.cx(i, n + i)
    qc.barrier()

    # Layer 3
    for i in range(n_total): qc.ry(float(theta[2 * n_total + i]), i)
    qc.cx(0, anc)
    qc.cx(n, anc)
    qc.measure_all()
    return qc


def get_ancilla_prob(counts, n_total=7):
    total = sum(counts.values())
    p1 = 0
    for bs, count in counts.items():
        bs = bs.zfill(n_total)
        if bs[0] == '1':
            p1 += count
    return p1 / total


# =====================================================================
# Extract matching cycle
# =====================================================================

def extract_matching_cycle(config_path, num_members=30, target_cycle=50, seed=42):
    """Run simulation to target cycle, extract a single category's matching problem."""
    from core.simulation_engine import SimulationEngine
    from core.matching_service import MatchingStrategy, MatchingService

    with open(config_path) as f:
        config = json.load(f)

    engine = SimulationEngine(
        config=config, matching_strategy=MatchingStrategy.GREEDY,
        num_members=num_members, num_cycles=target_cycle + 1, seed=seed,
        topology_sample_interval=99999,
    )
    engine.run()

    categories = list(config["categories"].keys())
    prices = engine.economy.get_all_prices()
    matcher = MatchingService(MatchingStrategy.QUBO, config.get("matching", {}))

    # Generate requests/offers for the cycle
    all_reqs, all_offs = [], []
    for m in engine.members_list:
        all_reqs.extend(m.generate_requests(target_cycle, categories))
        all_offs.extend(m.declare_availability(target_cycle, prices))

    # Pick category with enough pairs for a meaningful test (4-8 variables)
    best_cat = None
    best_pairs = None
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
        if 4 <= len(pairs) <= 20:  # sweet spot for both exhaustive + hardware
            if best_cat is None or abs(len(pairs) - 10) < abs(len(best_pairs) - 10):
                best_cat = cat
                best_pairs = pairs
                best_reqs = reqs
                best_offs = offs

    if best_cat is None:
        logger.error("No suitable category found. Adjust num_members or target_cycle.")
        return None

    logger.info(f"  Selected category: {best_cat} ({len(best_reqs)} requests, "
                f"{len(best_offs)} offers, {len(best_pairs)} eligible pairs)")
    logger.info(f"  Price: {prices[best_cat]:.2f}")

    return {
        "engine": engine,
        "category": best_cat,
        "cycle": target_cycle,
        "price": prices[best_cat],
        "categories": categories,
        "prices": prices,
        "requests": best_reqs,
        "offers": best_offs,
        "eligible_pairs": best_pairs,
        "matcher": matcher,
    }


# =====================================================================
# Weight computation — TWO methods
# =====================================================================

def compute_classical_weights(cycle_data):
    """Classical: use the paper's exact composite weight formula."""
    matcher = cycle_data["matcher"]
    prices = cycle_data["prices"]
    cats = cycle_data["categories"]
    weights = {}
    similarities = {}  # store just the sim term for comparison

    for idx, (r_idx, o_idx, req, offer, member) in enumerate(cycle_data["eligible_pairs"]):
        # Classical cosine similarity (what V3 replaces)
        from core.member import CommunityMember
        mem_emb = member.get_capability_embedding(cats)
        task_emb = CommunityMember.get_task_embedding(req.category, cats)
        norm_m = np.linalg.norm(mem_emb)
        norm_t = np.linalg.norm(task_emb)
        sim = float(np.dot(mem_emb, task_emb) / (norm_m * norm_t)) if norm_m > 0 and norm_t > 0 else 0
        similarities[idx] = sim

        # Full composite weight (paper Section 3.6.1)
        w = matcher._compute_weight(member, req, prices, cats)
        weights[idx] = w

    return weights, similarities


def compute_quantum_weights(cycle_data, v3_theta, backend_name, shots=4096):
    """
    Quantum: use V3 circuit on ibm_fez to compute sim(e_i, e_j),
    then combine with the same beta/gamma/delta terms classically.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    matcher = cycle_data["matcher"]
    prices = cycle_data["prices"]
    cats = cycle_data["categories"]
    alpha = matcher._weights["alpha"]
    beta = matcher._weights["beta"]
    gamma = matcher._weights["gamma"]
    delta = matcher._weights["delta"]

    # Build reduced embeddings (3D, same as V3 paper)
    # Use [electrical, cooking, transport] — differentiating skill axes
    emb_cats = ['electrical', 'cooking', 'transport']

    def member_emb(m):
        raw = np.array([m.skill_levels.get(c, 0.0) for c in emb_cats])
        # V3 scaling: [0.1, π-0.1]
        return 0.1 + (np.pi - 0.2) * raw

    def task_emb(cat_name):
        # One-hot-like encoding in reduced space
        raw = np.array([1.0 if c == cat_name else 0.0 for c in emb_cats])
        return 0.1 + (np.pi - 0.2) * raw

    # Build V3 circuits for all pairs
    logger.info(f"  Building {len(cycle_data['eligible_pairs'])} V3 circuits...")
    circuits = []
    pair_indices = []

    for idx, (r_idx, o_idx, req, offer, member) in enumerate(cycle_data["eligible_pairs"]):
        v1 = member_emb(member)
        v2 = task_emb(req.category)
        qc = build_v3_circuit(v1, v2, v3_theta)
        circuits.append(qc)
        pair_indices.append(idx)

    # Submit to hardware
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.backend(backend_name)
    logger.info(f"  Backend: {backend.name}, pending={backend.status().pending_jobs}")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    transpiled = [pm.run(c) for c in circuits]
    depths = [t.depth() for t in transpiled]
    logger.info(f"  Transpiled {len(transpiled)} circuits, depth: {min(depths)}-{max(depths)}")

    sampler = SamplerV2(mode=backend)
    logger.info(f"  Submitting batch ({shots} shots × {len(transpiled)} circuits)...")
    t0 = time.time()
    job = sampler.run(transpiled, shots=shots)
    logger.info(f"  Job: {job.job_id()} — waiting...")
    result = job.result()
    hw_time = time.time() - t0
    logger.info(f"  Hardware batch: {hw_time:.1f}s")

    # Extract V3 similarities and build full composite weights
    quantum_weights = {}
    quantum_similarities = {}

    for idx, (r_idx, o_idx, req, offer, member) in enumerate(cycle_data["eligible_pairs"]):
        counts = result[idx].data.meas.get_counts()
        p1 = get_ancilla_prob(counts)
        # V3 convention: P(|0⟩) = similarity
        v3_sim = 1.0 - p1
        quantum_similarities[idx] = v3_sim

        # Combine with the same other terms as classical (beta·rep + gamma·prox + delta·price)
        rep = member.reputation
        skill_level = member.skill_levels.get(req.category, 0.0)
        proximity = skill_level
        price = prices.get(req.category, 5.0) / 10.0

        w = alpha * v3_sim + beta * rep + gamma * proximity + delta * price
        quantum_weights[idx] = w

    return quantum_weights, quantum_similarities, {
        "job_id": job.job_id(),
        "hw_time": hw_time,
        "depth_range": (min(depths), max(depths)),
        "shots": shots,
    }


# =====================================================================
# QUBO Solve — same solver both times
# =====================================================================

def build_qubo(cycle_data, weights):
    """Build QUBO with given weight dict. Returns Q and var_map."""
    from collections import defaultdict

    pairs = cycle_data["eligible_pairs"]
    n = len(pairs)
    max_w = max(abs(w) for w in weights.values())
    penalty = 10 * max_w

    Q = {}
    var_map = {}
    for i, (r_idx, o_idx, _, _, _) in enumerate(pairs):
        var_map[(r_idx, o_idx)] = i
        Q[(i, i)] = -weights[i]

    # Constraint: one provider per request
    req_vars, off_vars = defaultdict(list), defaultdict(list)
    for (r, o), v in var_map.items():
        req_vars[r].append(v)
        off_vars[o].append(v)

    for vl in req_vars.values():
        for vi in vl:
            Q[(vi, vi)] = Q.get((vi, vi), 0) - penalty
        for i in range(len(vl)):
            for j in range(i + 1, len(vl)):
                k = (min(vl[i], vl[j]), max(vl[i], vl[j]))
                Q[k] = Q.get(k, 0) + 2 * penalty

    for vl in off_vars.values():
        if len(vl) <= 1:
            continue
        for i in range(len(vl)):
            for j in range(i + 1, len(vl)):
                k = (min(vl[i], vl[j]), max(vl[i], vl[j]))
                Q[k] = Q.get(k, 0) + 2 * penalty

    return Q, var_map, penalty


def solve_qubo_exhaustive(Q, n):
    """Exact solve for n <= 20. Same solver for both pipelines = fair comparison."""
    best_energy = float("inf")
    best_bits = None
    for bits in iter_product([0, 1], repeat=n):
        e = 0.0
        for (i, j), val in Q.items():
            if i == j:
                e += val * bits[i]
            else:
                e += val * bits[i] * bits[j]
        if e < best_energy:
            best_energy = e
            best_bits = bits
    return list(best_bits), best_energy


def extract_assignments(bits, var_map, pairs):
    """Decode bits → list of (requester_id, provider_id) assignments."""
    inv_map = {v: k for k, v in var_map.items()}
    assignments = []
    for v, bit in enumerate(bits):
        if bit == 1:
            r_idx, o_idx = inv_map[v]
            for pr_idx, po_idx, req, offer, member in pairs:
                if pr_idx == r_idx and po_idx == o_idx:
                    assignments.append({
                        "requester": req.requester_id,
                        "provider": member.member_id,
                        "category": req.category,
                    })
                    break
    return assignments


# =====================================================================
# Compare pipelines
# =====================================================================

def compare_pipelines(classical_result, quantum_result, weights_c, weights_q, sims_c, sims_q, pairs):
    """Rich comparison between classical and quantum pipelines."""
    comparison = {}

    # 1. Assignment agreement
    c_assign = {(a["requester"], a["provider"]) for a in classical_result["assignments"]}
    q_assign = {(a["requester"], a["provider"]) for a in quantum_result["assignments"]}

    n_c = len(c_assign)
    n_q = len(q_assign)
    shared = c_assign & q_assign

    comparison["classical_assignments"] = n_c
    comparison["quantum_assignments"] = n_q
    comparison["shared_assignments"] = len(shared)
    comparison["jaccard"] = len(shared) / len(c_assign | q_assign) if (c_assign | q_assign) else 0.0
    comparison["identical_solutions"] = c_assign == q_assign

    # 2. Similarity correlation
    sims_c_arr = np.array([sims_c[i] for i in sorted(sims_c.keys())])
    sims_q_arr = np.array([sims_q[i] for i in sorted(sims_q.keys())])
    if np.std(sims_c_arr) > 0 and np.std(sims_q_arr) > 0:
        r, p_val = stats.pearsonr(sims_c_arr, sims_q_arr)
        comparison["sim_correlation"] = float(r)
        comparison["sim_p_value"] = float(p_val)
    else:
        comparison["sim_correlation"] = 0.0
        comparison["sim_p_value"] = 1.0

    # 3. Weight correlation
    w_c_arr = np.array([weights_c[i] for i in sorted(weights_c.keys())])
    w_q_arr = np.array([weights_q[i] for i in sorted(weights_q.keys())])
    if np.std(w_c_arr) > 0 and np.std(w_q_arr) > 0:
        r_w, _ = stats.pearsonr(w_c_arr, w_q_arr)
        comparison["weight_correlation"] = float(r_w)
    else:
        comparison["weight_correlation"] = 0.0

    # 4. Objective values (apples to apples: evaluate each solution under both weight sets)
    def obj(bits, weights):
        return sum(weights[i] * bits[i] for i in range(len(bits)))

    c_bits = classical_result["bits"]
    q_bits = quantum_result["bits"]

    comparison["classical_obj_under_classical_weights"] = obj(c_bits, weights_c)
    comparison["classical_obj_under_quantum_weights"] = obj(c_bits, weights_q)
    comparison["quantum_obj_under_classical_weights"] = obj(q_bits, weights_c)
    comparison["quantum_obj_under_quantum_weights"] = obj(q_bits, weights_q)

    # 5. Interpretation
    if classical_result["bits"] == quantum_result["bits"]:
        comparison["verdict"] = "IDENTICAL — quantum weights produce same assignments"
    else:
        # Does the quantum solution perform well under classical weights?
        c_under_c = comparison["classical_obj_under_classical_weights"]
        q_under_c = comparison["quantum_obj_under_classical_weights"]
        regret = (c_under_c - q_under_c) / c_under_c if c_under_c != 0 else 0
        comparison["quantum_regret"] = float(regret)
        if abs(regret) < 0.05:
            comparison["verdict"] = f"DIFFERENT but near-optimal (regret {regret*100:.1f}%)"
        elif regret < 0:
            comparison["verdict"] = f"QUANTUM BETTER under classical weights ({-regret*100:.1f}%)"
        else:
            comparison["verdict"] = f"CLASSICAL BETTER under classical weights ({regret*100:.1f}% quantum regret)"

    return comparison


# =====================================================================
# Train V3 (reuse proven code)
# =====================================================================

def train_v3_on_skills(config_path, num_members=100, n_iter=200, shots=2048):
    """Train V3 once on simulator using member skill pairs. Returns theta."""
    from core.simulation_engine import SimulationEngine
    from core.matching_service import MatchingStrategy
    from qiskit.primitives import StatevectorSampler

    with open(config_path) as f:
        config = json.load(f)
    engine = SimulationEngine(
        config=config, matching_strategy=MatchingStrategy.GREEDY,
        num_members=num_members, num_cycles=501, seed=42,
        topology_sample_interval=99999,
    )
    engine.run()

    # 3D skill embeddings
    cats = ['electrical', 'cooking', 'transport']
    embeddings = {}
    for m in engine.members_list:
        raw = np.array([m.skill_levels.get(c, 0.0) for c in cats])
        embeddings[m.member_id] = 0.1 + (np.pi - 0.2) * raw

    # Build pairs with binary labels
    members = engine.members_list
    similar_pairs, dissimilar_pairs = [], []
    for i, mi in enumerate(members):
        for j, mj in enumerate(members):
            if j <= i:
                continue
            ei, ej = embeddings[mi.member_id], embeddings[mj.member_id]
            raw_i = np.array([mi.skill_levels.get(c, 0) for c in cats])
            raw_j = np.array([mj.skill_levels.get(c, 0) for c in cats])
            ni, nj = np.linalg.norm(raw_i), np.linalg.norm(raw_j)
            if ni > 0 and nj > 0:
                cos = float(np.dot(raw_i, raw_j) / (ni * nj))
                if cos > 0.8:
                    similar_pairs.append((ei, ej, 1.0))
                elif cos < 0.2:
                    dissimilar_pairs.append((ei, ej, 0.0))

    rng = np.random.default_rng(42)
    rng.shuffle(similar_pairs); rng.shuffle(dissimilar_pairs)
    n_per_class = min(len(similar_pairs), len(dissimilar_pairs), 20)
    train_data = similar_pairs[:n_per_class] + dissimilar_pairs[:n_per_class]
    rng.shuffle(train_data)

    logger.info(f"  Training V3 on {len(train_data)} member pairs...")

    sampler = StatevectorSampler()
    n_params = 21
    theta = np.random.uniform(-0.01, 0.01, n_params)
    best_theta, best_loss = theta.copy(), float("inf")

    for it in range(n_iter):
        delta = 2 * np.random.randint(0, 2, size=n_params) - 1
        c_k = 0.1 / (it + 1) ** 0.101
        a_k = 0.15 / (it + 1) ** 0.602

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
            logger.info(f"    Iter {it}: loss={(lp+lm)/2:.4f} best={best_loss:.4f}")

    logger.info(f"  V3 trained: best_loss={best_loss:.4f}")
    return best_theta


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument("--members", type=int, default=30)
    parser.add_argument("--cycle", type=int, default=50)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--spsa-iter", type=int, default=150)
    parser.add_argument("--train-members", type=int, default=100)
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp6"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "config", "community_economy.json")

    logger.info("=" * 64)
    logger.info("EXPERIMENT 6: V3 → QUBO End-to-End Hybrid Pipeline")
    logger.info(f"  Backend: {args.backend}")
    logger.info(f"  Protocol: Train V3 locally, then apply to matching cycle")
    logger.info("=" * 64)

    # STEP 1: Extract matching cycle
    logger.info("\n[STEP 1] Extracting real matching cycle...")
    cycle_data = extract_matching_cycle(config_path, args.members, args.cycle)
    if cycle_data is None:
        return
    n_pairs = len(cycle_data["eligible_pairs"])

    # STEP 2: Train V3 on simulator
    logger.info("\n[STEP 2] Training V3 circuit on simulator...")
    v3_theta = train_v3_on_skills(config_path, args.train_members, args.spsa_iter, shots=2048)

    # STEP 3: Compute CLASSICAL weights + solve QUBO
    logger.info("\n[STEP 3] CLASSICAL pipeline: cosine sim + composite weights → QUBO...")
    weights_c, sims_c = compute_classical_weights(cycle_data)
    Q_c, var_map_c, pen_c = build_qubo(cycle_data, weights_c)
    c_bits, c_energy = solve_qubo_exhaustive(Q_c, n_pairs)
    c_assignments = extract_assignments(c_bits, var_map_c, cycle_data["eligible_pairs"])
    logger.info(f"  Classical: {sum(c_bits)} assignments, energy={c_energy:.4f}")
    for a in c_assignments[:5]:
        logger.info(f"    {a['requester']} → {a['provider']}")
    classical_result = {"bits": c_bits, "energy": c_energy, "assignments": c_assignments}

    # STEP 4: Compute QUANTUM weights on hardware + solve QUBO
    logger.info(f"\n[STEP 4] QUANTUM pipeline: V3 on {args.backend} + composite → QUBO...")
    try:
        weights_q, sims_q, hw_meta = compute_quantum_weights(
            cycle_data, v3_theta, args.backend, args.shots)
        Q_q, var_map_q, pen_q = build_qubo(cycle_data, weights_q)
        q_bits, q_energy = solve_qubo_exhaustive(Q_q, n_pairs)
        q_assignments = extract_assignments(q_bits, var_map_q, cycle_data["eligible_pairs"])
        logger.info(f"  Quantum: {sum(q_bits)} assignments, energy={q_energy:.4f}")
        for a in q_assignments[:5]:
            logger.info(f"    {a['requester']} → {a['provider']}")
        quantum_result = {"bits": q_bits, "energy": q_energy, "assignments": q_assignments}
    except Exception as e:
        logger.error(f"  Hardware FAILED: {e}")
        import traceback; traceback.print_exc()
        return

    # STEP 5: Compare
    logger.info("\n[STEP 5] Comparing pipelines...")
    comparison = compare_pipelines(
        classical_result, quantum_result,
        weights_c, weights_q, sims_c, sims_q,
        cycle_data["eligible_pairs"]
    )

    # SUMMARY
    logger.info("\n" + "=" * 64)
    logger.info("END-TO-END HYBRID PIPELINE RESULT")
    logger.info("=" * 64)
    logger.info(f"  Category: {cycle_data['category']} | {n_pairs} eligible pairs")
    logger.info(f"  Similarity correlation (classical vs V3): r = {comparison['sim_correlation']:.4f}")
    logger.info(f"  Weight correlation: r = {comparison['weight_correlation']:.4f}")
    logger.info(f"  Classical assignments: {comparison['classical_assignments']}")
    logger.info(f"  Quantum assignments:   {comparison['quantum_assignments']}")
    logger.info(f"  Shared assignments:    {comparison['shared_assignments']} "
                f"(Jaccard = {comparison['jaccard']:.3f})")
    logger.info(f"\n  Objective values (higher = better):")
    logger.info(f"    Classical solution under classical weights: "
                f"{comparison['classical_obj_under_classical_weights']:.4f}")
    logger.info(f"    Quantum solution under classical weights:   "
                f"{comparison['quantum_obj_under_classical_weights']:.4f}")
    logger.info(f"    Classical solution under quantum weights:   "
                f"{comparison['classical_obj_under_quantum_weights']:.4f}")
    logger.info(f"    Quantum solution under quantum weights:     "
                f"{comparison['quantum_obj_under_quantum_weights']:.4f}")
    logger.info(f"\n  VERDICT: {comparison['verdict']}")

    # Save
    results = {
        "n_pairs": n_pairs,
        "category": cycle_data["category"],
        "cycle": args.cycle,
        "backend": args.backend,
        "hardware_meta": hw_meta,
        "similarities_classical": {str(k): v for k, v in sims_c.items()},
        "similarities_quantum": {str(k): v for k, v in sims_q.items()},
        "weights_classical": {str(k): v for k, v in weights_c.items()},
        "weights_quantum": {str(k): v for k, v in weights_q.items()},
        "classical_assignments": c_assignments,
        "quantum_assignments": q_assignments,
        "classical_energy": c_energy,
        "quantum_energy": q_energy,
        "comparison": comparison,
    }
    out_path = os.path.join(args.output, "exp6_hybrid_pipeline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
