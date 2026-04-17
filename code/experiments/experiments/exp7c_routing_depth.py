#!/usr/bin/env python3
"""
Experiment 7c: Routing-Depth Noise Simulation

Tests whether routing-amplified physical depth (149-261 gates on heavy-hex)
with calibrated NISQ noise reproduces an ibm_fez-like hardware advantage.

Background:
  Arm B (logical depth ~30, ibm_fez NoiseModel): 1.48% ± 0.00% (30 MC traj)
  ibm_fez hardware (physical depth 167-270):      0.61%
  K_c classical kernel:                            0.86%

The gap between Arm B and hardware is hypothesized to arise from routing-
amplified physical depth: transpilation against the heavy-hex coupling map
inserts SWAP chains that inflate depth from ~30 to 150-270. This experiment
tests whether that depth regime, combined with calibrated noise, pushes
regret below the K_c ceiling (0.86%).

Approach:
  1. Transpile V3 circuits against FakeTorino (133q Heron-class, heavy-hex,
     CZ native) at optimization_level=3 → depth 149-261
  2. Compact transpiled circuits to active-qubit subspace (19-23 qubits)
  3. Build per-qubit noise model for compact indices from backend calibration
  4. Simulate N MC trajectories via AerSimulator(statevector + noise)
  5. Compare regret distribution to Arm A/B, K_c, and hardware references

Uses FakeTorino as a Heron-class proxy for ibm_fez. Both are heavy-hex
processors with CZ native gates. When ibm_fez credentials are available,
re-run with --use-ibm-fez for exact noise calibration.
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
for noisy in ("qiskit", "qiskit.transpiler", "qiskit.compiler", "stevedore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from experiments.exp6b_hybrid_replication import harvest_instances, analyze_degeneracy
from experiments.exp6_hybrid_pipeline import (
    compute_classical_weights, build_qubo, solve_qubo_exhaustive,
)
from experiments.exp6d_v3_extended import (
    N_PARAMS, build_v3_extended, get_ancilla_prob,
    member_embedding_9d, task_embedding_9d, compute_9d_cosine,
)


# =====================================================================
# Circuit compaction: 133-qubit → active-qubit subspace
# =====================================================================

def compact_circuit(tqc):
    """Remap a transpiled circuit to use only its active qubits.

    Returns (compact_qc, active_physical, phys_to_compact).
    """
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(tqc)

    # Identify physical qubits that participate in any gate or measurement
    active_physical = set()
    for node in dag.op_nodes():
        for q in node.qargs:
            active_physical.add(tqc.find_bit(q).index)
    active_physical = sorted(active_physical)
    n_active = len(active_physical)

    phys_to_compact = {old: new for new, old in enumerate(active_physical)}

    # Rebuild on compact register, preserving classical bits
    compact = QuantumCircuit(n_active, tqc.num_clbits)

    for inst in tqc.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        new_qubits = [phys_to_compact[tqc.find_bit(q).index] for q in inst.qubits]
        new_clbits = [tqc.find_bit(c).index for c in inst.clbits] if inst.clbits else []
        compact.append(op, new_qubits, new_clbits)

    return compact, active_physical, phys_to_compact


def build_compact_noise_model(backend, active_physical, phys_to_compact):
    """Build a NoiseModel for compact qubit indices using backend calibration.

    Uses depolarizing_error approximation (standard for NoiseModel.from_backend).
    Maps per-qubit error rates from physical indices to compact indices.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

    target = backend.target
    nm = NoiseModel()

    # Single-qubit gate errors
    for gate_name in ["sx", "x", "rz", "id"]:
        if gate_name not in target.operation_names:
            continue
        gate_props = target[gate_name]
        for phys_q in active_physical:
            props = gate_props.get((phys_q,))
            if props and props.error is not None and props.error > 0:
                err = depolarizing_error(props.error, 1)
                nm.add_quantum_error(err, gate_name, [phys_to_compact[phys_q]])

    # Two-qubit gate errors (CZ for Heron-class)
    for gate_name in ["cz", "cx", "ecr"]:
        if gate_name not in target.operation_names:
            continue
        gate_props = target[gate_name]
        for qargs, props in gate_props.items():
            if qargs is None or len(qargs) != 2:
                continue
            q0, q1 = qargs
            if q0 in phys_to_compact and q1 in phys_to_compact:
                if props and props.error is not None and props.error > 0:
                    err = depolarizing_error(props.error, 2)
                    nm.add_quantum_error(
                        err, gate_name,
                        [phys_to_compact[q0], phys_to_compact[q1]],
                    )

    # Readout errors
    if "measure" in target.operation_names:
        meas_props = target["measure"]
        for phys_q in active_physical:
            props = meas_props.get((phys_q,))
            if props and props.error is not None and props.error > 0:
                p_err = min(props.error, 0.5)  # clamp to valid range
                nm.add_readout_error(
                    [[1 - p_err, p_err], [p_err, 1 - p_err]],
                    [phys_to_compact[phys_q]],
                )

    return nm


# =====================================================================
# Main experiment
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp 7c: Routing-depth noise simulation"
    )
    parser.add_argument("--n-trajectories", type=int, default=10,
                        help="Number of MC trajectories (default 10)")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--members", type=int, default=30)
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--opt-level", type=int, default=3,
                        help="Transpilation optimization level (default 3)")
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exp7"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "community_economy.json",
    )

    logger.info("=" * 72)
    logger.info("EXPERIMENT 7c: Routing-Depth Noise Simulation")
    logger.info(f"  {args.n_trajectories} MC trajectories, shots={args.shots}, "
                f"opt_level={args.opt_level}, device={args.device}")
    logger.info("=" * 72)

    # --- SETUP ---
    logger.info("\n[SETUP] Harvest instances + load theta...")
    instances = harvest_instances(config_path, args.members)
    theta = np.load(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exp6d", "theta.npy",
    ))
    logger.info(f"  {len(instances)} instances, theta shape {theta.shape}")

    # Classical ground truth
    classical_results = []
    for inst in instances:
        n = len(inst["eligible_pairs"])
        w_c, s_c = compute_classical_weights(inst)
        Q_c, _, _ = build_qubo(inst, w_c)
        c_bits, c_energy = solve_qubo_exhaustive(Q_c, n)
        n_opt, gap, opt_bits = analyze_degeneracy(Q_c, n)
        classical_results.append({
            "n_pairs": n, "weights": w_c, "similarities": s_c,
            "bits": c_bits, "energy": c_energy,
            "n_optima": n_opt, "optima": opt_bits,
        })

    # --- BACKEND ---
    logger.info("\n[BACKEND] Loading FakeTorino (133q Heron-class, heavy-hex, CZ native)...")
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    fake_backend = FakeTorino()
    logger.info(f"  {fake_backend.name}: {fake_backend.num_qubits}q, "
                f"native gates: {[g for g in fake_backend.target.operation_names if g in ('sx','cz','rz','x')]}")

    # --- BUILD + TRANSPILE ALL CIRCUITS ---
    logger.info("\n[TRANSPILE] Building V3 circuits and transpiling at "
                f"opt_level={args.opt_level} against heavy-hex coupling map...")
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    all_logical = []      # original circuits
    pair_map = []          # (instance_idx, pair_idx)
    for i, inst in enumerate(instances):
        for j, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
            v1 = member_embedding_9d(member)
            v2 = task_embedding_9d(req.category)
            all_logical.append(build_v3_extended(v1, v2, theta))
            pair_map.append((i, j))

    t0 = time.time()
    all_transpiled = transpile(
        all_logical, fake_backend, optimization_level=args.opt_level
    )
    transpile_time = time.time() - t0
    logger.info(f"  {len(all_transpiled)} circuits transpiled in {transpile_time:.1f}s")

    # Analyze transpiled depths
    depths = [tqc.depth() for tqc in all_transpiled]
    logger.info(f"  Depth range: {min(depths)}-{max(depths)} "
                f"(mean {np.mean(depths):.0f})")

    # --- COMPACT ALL CIRCUITS ---
    logger.info("\n[COMPACT] Compacting to active-qubit subspace...")
    all_compact = []
    all_noise_models = []
    n_active_list = []

    for idx, tqc in enumerate(all_transpiled):
        compact_qc, active_phys, p2c = compact_circuit(tqc)
        compact_nm = build_compact_noise_model(fake_backend, active_phys, p2c)
        all_compact.append(compact_qc)
        all_noise_models.append((compact_nm, active_phys, p2c))
        n_active_list.append(len(active_phys))

    logger.info(f"  Active qubits range: {min(n_active_list)}-{max(n_active_list)}")

    # Group circuits by noise model signature (same active qubits = same noise model)
    # For simplicity, we'll run each circuit individually since noise models may differ
    # across circuits (different routing → different physical qubits → different noise)

    # --- VALIDATE: single noiseless run first ---
    logger.info("\n[VALIDATE] Noiseless control on compact circuits...")
    sim_noiseless = AerSimulator(method="statevector")
    noiseless_regrets = []
    for i, inst in enumerate(instances):
        n = len(inst["eligible_pairs"])
        weights_q = {}
        for j, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
            c_idx = next(ci for ci, (ii, jj) in enumerate(pair_map) if ii == i and jj == j)
            qc = all_compact[c_idx]
            result = sim_noiseless.run(qc, shots=args.shots).result()
            p1 = get_ancilla_prob(result.get_counts(0))
            sim_val = 1.0 - p1
            rep = member.reputation
            skill = member.skill_levels.get(req.category, 0.0)
            price = inst["prices"].get(req.category, 5.0) / 10.0
            matcher = inst["matcher"]
            weights_q[j] = (matcher._weights["alpha"] * sim_val
                          + matcher._weights["beta"] * rep
                          + matcher._weights["gamma"] * skill
                          + matcher._weights["delta"] * price)

        Q_q, _, _ = build_qubo(inst, weights_q)
        q_bits, _ = solve_qubo_exhaustive(Q_q, n)
        c_bits = classical_results[i]["bits"]
        w_c = classical_results[i]["weights"]
        obj_c = sum(w_c[k] * c_bits[k] for k in range(n))
        obj_q = sum(w_c[k] * q_bits[k] for k in range(n))
        regret = (obj_c - obj_q) / obj_c if obj_c != 0 else 0.0
        noiseless_regrets.append(regret)

    logger.info(f"  Noiseless compact: {np.mean(noiseless_regrets)*100:.3f}% mean regret "
                f"(should match Arm A ~1.48%)")

    # --- MC TRAJECTORIES WITH ROUTING-DEPTH NOISE ---
    logger.info(f"\n[MC TRAJECTORIES] Running {args.n_trajectories} trajectories "
                f"with routing-depth noise on {args.device}...")
    n_inst = len(instances)
    all_traj_regrets = []      # (n_traj, n_inst)
    all_traj_co_optimal = []   # (n_traj, n_inst) bool
    all_mean_regret = []

    for traj in range(args.n_trajectories):
        seed = args.seed_base + traj
        t0 = time.time()

        traj_regrets = []
        traj_co_opt = []

        for i, inst in enumerate(instances):
            n = len(inst["eligible_pairs"])
            matcher = inst["matcher"]
            prices = inst["prices"]
            alpha = matcher._weights["alpha"]
            beta = matcher._weights["beta"]
            gamma = matcher._weights["gamma"]
            delta = matcher._weights["delta"]

            weights_q = {}
            for j, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
                c_idx = next(ci for ci, (ii, jj) in enumerate(pair_map) if ii == i and jj == j)
                compact_qc = all_compact[c_idx]
                compact_nm = all_noise_models[c_idx][0]

                sim = AerSimulator(
                    noise_model=compact_nm,
                    method="statevector",
                    device=args.device,
                )
                result = sim.run(compact_qc, shots=args.shots,
                                seed_simulator=seed).result()
                p1 = get_ancilla_prob(result.get_counts(0))
                v3_sim = 1.0 - p1

                rep = member.reputation
                skill = member.skill_levels.get(req.category, 0.0)
                price = prices.get(req.category, 5.0) / 10.0
                weights_q[j] = alpha * v3_sim + beta * rep + gamma * skill + delta * price

            Q_q, _, _ = build_qubo(inst, weights_q)
            q_bits, _ = solve_qubo_exhaustive(Q_q, n)

            c_bits = classical_results[i]["bits"]
            w_c = classical_results[i]["weights"]
            obj_c = sum(w_c[k] * c_bits[k] for k in range(n))
            obj_q = sum(w_c[k] * q_bits[k] for k in range(n))
            regret = (obj_c - obj_q) / obj_c if obj_c != 0 else 0.0
            q_is_co_opt = tuple(q_bits) in [tuple(b) for b in classical_results[i]["optima"]]

            traj_regrets.append(float(regret))
            traj_co_opt.append(q_is_co_opt)

        mean_r = float(np.mean(traj_regrets))
        n_co = sum(traj_co_opt)
        all_traj_regrets.append(traj_regrets)
        all_traj_co_optimal.append(traj_co_opt)
        all_mean_regret.append(mean_r)
        elapsed = time.time() - t0

        logger.info(
            f"  [traj {traj+1:3d}/{args.n_trajectories}] seed={seed:3d} "
            f"regret={mean_r*100:.2f}% co-opt={n_co}/{n_inst} ({elapsed:.1f}s)"
        )

    # --- AGGREGATE ---
    regret_arr = np.array(all_traj_regrets)       # (N_traj, N_inst)
    co_opt_arr = np.array(all_traj_co_optimal)    # (N_traj, N_inst) bool
    mean_regret_arr = np.array(all_mean_regret)

    logger.info("\n" + "=" * 72)
    logger.info("ROUTING-DEPTH NOISE DISTRIBUTION")
    logger.info("=" * 72)
    logger.info(f"  Backend proxy: FakeTorino (133q Heron-class, heavy-hex, CZ native)")
    logger.info(f"  Transpiled depth range: {min(depths)}-{max(depths)} "
                f"(mean {np.mean(depths):.0f})")
    logger.info(f"  Active qubits range: {min(n_active_list)}-{max(n_active_list)}")
    logger.info(f"  Noiseless compact control: {np.mean(noiseless_regrets)*100:.3f}% mean regret")
    logger.info(f"")
    logger.info(f"  Routing-depth noisy mean regret: "
                f"{mean_regret_arr.mean()*100:.3f}% ± {mean_regret_arr.std()*100:.3f}%")
    logger.info(f"  Median: {np.median(mean_regret_arr)*100:.3f}%")
    logger.info(f"  Min: {mean_regret_arr.min()*100:.3f}%, "
                f"Max: {mean_regret_arr.max()*100:.3f}%")

    logger.info(f"\n  Per-instance co-optimal probability "
                f"(out of {args.n_trajectories} trajectories):")
    for i in range(n_inst):
        p = co_opt_arr[:, i].mean()
        cat = instances[i]["category"]
        np_ = len(instances[i]["eligible_pairs"])
        logger.info(f"    [{i}] {cat:10s} n={np_:2d} pairs  P(co-opt)={p:.2f}")

    # Reference comparisons
    logger.info(f"\n  Reference comparisons:")
    logger.info(f"    Arm A (noiseless, logical depth):   1.48%")
    logger.info(f"    Arm B (noisy, logical depth ~30):   1.48% ± 0.00%  (30 traj)")
    logger.info(f"    Noiseless compact (routing depth):  "
                f"{np.mean(noiseless_regrets)*100:.3f}%")
    logger.info(f"    >>> Exp 7c (noisy, routing depth):  "
                f"{mean_regret_arr.mean()*100:.3f}% ± {mean_regret_arr.std()*100:.3f}% <<<")
    logger.info(f"    K_c classical kernel:               0.86%")
    logger.info(f"    ibm_fez hardware:                   0.61%")
    logger.info(f"    ibm_marrakesh hardware:              1.48%")

    # Percentile analysis
    if mean_regret_arr.std() > 0:
        logger.info(f"\n  Percentile of references in Exp 7c distribution:")
        logger.info(f"    ibm_fez 0.61%:      "
                    f"{(mean_regret_arr <= 0.0061).mean()*100:.1f}%")
        logger.info(f"    K_c 0.86%:          "
                    f"{(mean_regret_arr <= 0.0086).mean()*100:.1f}%")
        logger.info(f"    ibm_marrakesh 1.48%: "
                    f"{(mean_regret_arr <= 0.0148).mean()*100:.1f}%")

    # Interpretation
    mean_r = mean_regret_arr.mean()
    logger.info(f"\n  INTERPRETATION:")
    if mean_r < 0.0086:
        logger.info(f"    Routing-depth noise pushes regret BELOW K_c ceiling (0.86%).")
        logger.info(f"    → Noise-as-regularization mechanism CONFIRMED in simulation.")
        logger.info(f"    → Physical depth is the operative variable, not chip-specific.")
    elif mean_r < 0.0148 - 0.001:
        logger.info(f"    Routing-depth noise reduces regret but stays ABOVE K_c (0.86%).")
        logger.info(f"    → Physical depth has an effect but doesn't fully explain fez.")
        logger.info(f"    → ibm_fez's specific noise calibration may contribute.")
    else:
        logger.info(f"    Routing-depth noise has NO effect (matches logical-depth Arm B).")
        logger.info(f"    → The fez advantage requires something beyond routing depth.")
        logger.info(f"    → Possible: chip-specific calibration, coherent noise structure,")
        logger.info(f"      or an artifact of the FakeTorino proxy.")

    # --- SAVE ---
    summary = {
        "experiment": "exp7c_routing_depth",
        "backend_proxy": "FakeTorino",
        "backend_qubits": fake_backend.num_qubits,
        "n_trajectories": args.n_trajectories,
        "seed_base": args.seed_base,
        "shots": args.shots,
        "opt_level": args.opt_level,
        "device": args.device,
        "transpiled_depth_range": [int(min(depths)), int(max(depths))],
        "transpiled_depth_mean": float(np.mean(depths)),
        "active_qubits_range": [int(min(n_active_list)), int(max(n_active_list))],
        "noiseless_compact_regret": float(np.mean(noiseless_regrets)),
        "mean_regret_per_trajectory": all_mean_regret,
        "per_instance_regrets": regret_arr.tolist(),
        "per_instance_co_optimal": co_opt_arr.tolist(),
        "aggregate": {
            "mean": float(mean_regret_arr.mean()),
            "std": float(mean_regret_arr.std()),
            "median": float(np.median(mean_regret_arr)),
            "min": float(mean_regret_arr.min()),
            "max": float(mean_regret_arr.max()),
            "per_instance_co_opt_prob": [
                float(co_opt_arr[:, i].mean()) for i in range(n_inst)
            ],
        },
        "references": {
            "arm_a_noiseless": 0.0148,
            "arm_b_logical_depth_noisy": 0.0148,
            "kc_classical_kernel": 0.0086,
            "ibm_fez_hardware": 0.0061,
            "ibm_marrakesh_hardware": 0.0148,
        },
    }
    out_path = os.path.join(args.output, "exp7c_routing_depth.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
