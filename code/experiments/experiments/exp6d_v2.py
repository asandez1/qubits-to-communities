#!/usr/bin/env python3
"""
Experiment 6d (v2) — V3-Extended hardware benchmark on Heron R2, unified v2 codepath.

Drop-in replacement for `exp6d_v3_extended.py` that:
  1. Loads the 8 frozen benchmark instances via `core.benchmark_fixtures.hardware_tier()`
     — no v1 simulation harvest, no `community_economy.json` dependency.
  2. Computes classical weights via `core.benchmark_model.hardware_classical_weights`
     — ported from the paper's Section 3.6.1 formula, byte-identical to v1.
  3. Builds the QUBO via `core.benchmark_model.hardware_qubo` — byte-identical to v1.
  4. Runs the V3-extended 19-qubit circuit (reused from original exp6d module)
     on a backend passed via --backend, with the same pretrained θ.

Verified equivalence: `experiments/scripts/verify_v2_equivalence.py` confirms
all 8 instances match the v1 pipeline byte-for-byte.

Usage:
    python exp6d_v2.py --backend ibm_kingston --theta-path ../results/exp6d/theta.npy
"""
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from core.benchmark_fixtures import hardware_tier
from core.benchmark_model import (
    hardware_classical_weights, hardware_qubo, hardware_qubo_solve_exhaustive,
    hardware_qubo_analyze_degeneracy, hardware_evaluate_quantum,
)
# Reuse the quantum circuit + training + embeddings from the original exp6d —
# these operate on member attributes (skill_levels, reputation, etc.) that
# both v1 and v2 Member classes expose identically.
from experiments.exp6d_v3_extended import (
    build_v3_extended, get_ancilla_prob,
    member_embedding_9d, task_embedding_9d,
    N_PARAMS, N_TOTAL, train_v3_extended, build_training_set,
)


# =====================================================================
# Core evaluation on a v2 MatchingInstance
# =====================================================================

def _eval_instance(instance, meta, similarities, cref):
    """Thin wrapper around `hardware_evaluate_quantum` for result recording."""
    rec = hardware_evaluate_quantum(
        instance=instance,
        classical_weights=cref["weights"],
        classical_optima=cref["optima"],
        quantum_similarities=similarities,
        matcher_weights=meta["matcher_weights"],
    )
    rec["instance_id"] = instance.instance_id
    rec["category"] = meta["category_label"]
    rec["cycle"] = meta["cycle"]
    return rec


# =====================================================================
# Run V3 on simulator OR hardware for all 8 instances
# =====================================================================

def run_v3_on_instances(instances, theta, on_hardware=False,
                       backend_name="ibm_fez", shots=4096):
    """Submit V3-extended circuits for all pairs across all 8 instances in
    one batched job. Returns {instance_idx: {var_idx: similarity}}."""
    from qiskit import transpile

    # Collect all circuits in a single flat list with instance_idx, var_idx mapping
    circuits = []
    circuit_idx_map = []  # (instance_idx, var_idx)
    for inst_i, inst in enumerate(instances):
        for pair in inst.pairs:
            m = inst.member(pair)
            t = inst.task(pair)
            v1 = member_embedding_9d(m)
            v2 = task_embedding_9d(t.required_skill)
            circuits.append(build_v3_extended(v1, v2, theta))
            circuit_idx_map.append((inst_i, pair.var_idx))
    logger.info(f"  Prepared {len(circuits)} V3 circuits across {len(instances)} instances")

    if on_hardware:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        backend = service.backend(backend_name)
        logger.info(f"  Backend: {backend.name}, pending_jobs={backend.status().pending_jobs}")
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        transpiled = [pm.run(c) for c in circuits]
        depths = [t.depth() for t in transpiled]
        logger.info(f"  Transpiled: depth {min(depths)}-{max(depths)}")
        sampler = SamplerV2(mode=backend)
        t0 = time.time()
        job = sampler.run(transpiled, shots=shots)
        logger.info(f"  Submitted: job_id={job.job_id()}")
        result = job.result()
        hw_time = time.time() - t0
        logger.info(f"  Hardware batch complete: {hw_time:.1f}s")
        hardware_meta = {"job_id": job.job_id(), "hw_time": hw_time,
                         "depth_range": [min(depths), max(depths)], "shots": shots}
    else:
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method="statevector")
        tqcs = transpile(circuits, sim, optimization_level=0)
        result = sim.run(tqcs, shots=shots).result()
        hardware_meta = {"backend": "AerSimulator(noiseless)", "shots": shots}

    # Extract similarities per (instance, var)
    sims_by_instance: dict[int, dict[int, float]] = defaultdict(dict)
    for flat_idx, (inst_i, var_idx) in enumerate(circuit_idx_map):
        counts = result[flat_idx].data.meas.get_counts() if on_hardware else \
                 result.get_counts(flat_idx)
        p1 = get_ancilla_prob(counts)
        sims_by_instance[inst_i][var_idx] = 1.0 - p1
    return sims_by_instance, hardware_meta


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exp 6d v2 — unified v2 codepath")
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--skip-hardware", action="store_true")
    parser.add_argument("--theta-path", default=None,
                        help="Path to pretrained theta.npy. If omitted, train "
                             "V3-extended from scratch (~3 min CPU).")
    parser.add_argument("--spsa-iter", type=int, default=150)
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "exp6d_v2"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    logger.info("=" * 70)
    logger.info("EXPERIMENT 6d v2 — unified v2 codepath")
    logger.info(f"  Backend: {args.backend}, skip_hardware={args.skip_hardware}")
    logger.info("=" * 70)

    # --- Load v2 hardware instances ---
    logger.info("\n[1] Loading frozen hardware benchmark instances (v2)...")
    instances, metadata = hardware_tier()
    logger.info(f"  {len(instances)} instances: "
                f"{[m['category_label'] for m in metadata]}")

    # --- Classical reference solutions ---
    logger.info("\n[2] Computing classical baseline (paper §3.6.1 formula)...")
    classical_refs = []
    for inst, meta in zip(instances, metadata):
        c_weights, c_sims = hardware_classical_weights(inst, meta["matcher_weights"])
        c_Q, _, _ = hardware_qubo(inst, c_weights)
        n_opt, gap, optima = hardware_qubo_analyze_degeneracy(c_Q, inst.n_vars)
        c_bits, c_energy = hardware_qubo_solve_exhaustive(c_Q, inst.n_vars)
        classical_refs.append({
            "weights": c_weights, "similarities": c_sims,
            "bits": c_bits, "energy": c_energy,
            "optima": optima, "n_optima": n_opt, "gap": gap,
        })
    logger.info(f"  All {len(instances)} classical optima computed")

    # --- Load or train θ ---
    if args.theta_path and os.path.exists(args.theta_path):
        theta = np.load(args.theta_path)
        assert theta.shape == (N_PARAMS,), (
            f"theta shape mismatch: {theta.shape} vs ({N_PARAMS},)")
        logger.info(f"\n[3] Loaded θ from {args.theta_path}")
    else:
        logger.info(f"\n[3] Training V3-extended from scratch ({args.spsa_iter} iters)...")
        config_path = os.path.join(ROOT, "config", "community_economy.json")
        train_data = build_training_set(config_path, 100, 40)
        t0 = time.time()
        theta, _ = train_v3_extended(train_data, n_iter=args.spsa_iter, shots=2048)
        logger.info(f"  Training took {time.time()-t0:.1f}s")
        np.save(os.path.join(args.output, "theta.npy"), theta)

    # --- Simulator evaluation ---
    logger.info("\n[4] Noiseless simulator evaluation...")
    sim_sims, _ = run_v3_on_instances(instances, theta, on_hardware=False,
                                       shots=args.shots)
    sim_records = []
    for i, (inst, meta, cref) in enumerate(zip(instances, metadata, classical_refs)):
        rec = _eval_instance(inst, meta, sim_sims[i], cref)
        rec["instance_idx"] = i
        sim_records.append(rec)
    sim_regret = float(np.mean([r["classical_regret"] for r in sim_records]))
    sim_co_opt = sum(1 for r in sim_records if r["q_is_co_optimal"])
    logger.info(f"  Simulator: regret={sim_regret*100:+.3f}% "
                f"co-opt={sim_co_opt}/{len(instances)}")

    output: dict = {
        "experiment": "Exp 6d v2 (unified v2 codepath)",
        "backend": args.backend,
        "n_params": N_PARAMS,
        "n_total_qubits": N_TOTAL,
        "simulator": {
            "mean_classical_regret": sim_regret,
            "n_co_optimal": sim_co_opt,
            "n_identical": sum(1 for r in sim_records if r["identical"]),
            "per_instance": sim_records,
        },
    }

    # --- Hardware evaluation ---
    if not args.skip_hardware:
        logger.info(f"\n[5] Hardware evaluation on {args.backend}...")
        hw_sims, hw_meta = run_v3_on_instances(instances, theta,
                                               on_hardware=True,
                                               backend_name=args.backend,
                                               shots=args.shots)
        hw_records = []
        for i, (inst, meta, cref) in enumerate(zip(instances, metadata, classical_refs)):
            rec = _eval_instance(inst, meta, hw_sims[i], cref)
            rec["instance_idx"] = i
            hw_records.append(rec)
        hw_regret = float(np.mean([r["classical_regret"] for r in hw_records]))
        hw_co_opt = sum(1 for r in hw_records if r["q_is_co_optimal"])
        logger.info(f"  Hardware: regret={hw_regret*100:+.3f}% "
                    f"co-opt={hw_co_opt}/{len(instances)}")
        output["hardware"] = {
            "mean_classical_regret": hw_regret,
            "n_co_optimal": hw_co_opt,
            "n_identical": sum(1 for r in hw_records if r["identical"]),
            "per_instance": hw_records,
            "hardware_metadata": hw_meta,
        }

    # --- Save ---
    fname = f"exp6d_v2_{args.backend}.json"
    out_path = os.path.join(args.output, fname)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
