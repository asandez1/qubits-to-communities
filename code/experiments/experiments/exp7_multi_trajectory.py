#!/usr/bin/env python3
"""
Experiment 7b: Many-trajectory Arm B distribution.

Runs N independent Monte Carlo trajectories of the ibm_fez noisy simulator
(AerSimulator with NoiseModel + GPU) on the same 8 instances and same theta,
varying only seed_simulator. Produces:
  - Per-trajectory mean regret and co-optimal count
  - Per-instance flip probability (fraction of trajectories where each instance
    is co-optimal)
  - Aggregate mean ± std of regret distribution

This directly tests whether the ibm_fez 0.61% hardware result is within the
expected distribution of a noisy simulator, and whether marrakesh's 1.48% is
similarly typical.

Requires: venv_gpu with qiskit-aer-gpu + LD_LIBRARY_PATH set for nvidia libs.
NoiseModel loaded from /tmp/fez_noise_model_v2.json (exported from qubo-experiment venv).
"""

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


def load_noise_model(path="/tmp/fez_noise_model_v2.json"):
    """Load ibm_fez NoiseModel exported from qubo-experiment venv."""
    from qiskit_aer.noise import NoiseModel

    def decode_hook(obj):
        if "__ndarray__" in obj:
            return np.array(obj["data"], dtype=obj["dtype"])
        if "__complex__" in obj:
            return complex(obj["re"], obj["im"])
        return obj

    with open(path) as f:
        d = json.load(f, object_hook=decode_hook)
    return NoiseModel.from_dict(d)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trajectories", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--members", type=int, default=30)
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--noise-model-path", default="/tmp/fez_noise_model_v2.json")
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
    logger.info(f"EXP 7b: Many-trajectory Arm B distribution ({args.n_trajectories} trajectories)")
    logger.info(f"  device={args.device}, shots={args.shots}, seed_base={args.seed_base}")
    logger.info("=" * 72)

    # --- SETUP ---
    logger.info("\n[SETUP] Harvest instances + load theta…")
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

    # --- BUILD CIRCUITS ONCE ---
    logger.info("\n[CIRCUITS] Building + transpiling all circuits…")
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    nm = load_noise_model(args.noise_model_path)
    basis = list(nm.basis_gates)

    all_circuits = []
    pair_map = []  # (instance_idx, pair_idx)
    for i, inst in enumerate(instances):
        for j, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
            v1 = member_embedding_9d(member)
            v2 = task_embedding_9d(req.category)
            all_circuits.append(build_v3_extended(v1, v2, theta))
            pair_map.append((i, j))
    tqcs = transpile(all_circuits, basis_gates=basis, optimization_level=1)
    logger.info(f"  {len(tqcs)} circuits transpiled")

    sim = AerSimulator(noise_model=nm, basis_gates=basis,
                       method="statevector", device=args.device)
    logger.info(f"  Simulator: {args.device}")

    # --- RUN TRAJECTORIES ---
    n_inst = len(instances)
    all_regrets = []        # (n_traj, n_inst)
    all_co_optimal = []     # (n_traj, n_inst) bool
    all_mean_regret = []

    for traj in range(args.n_trajectories):
        seed = args.seed_base + traj
        t0 = time.time()
        result = sim.run(tqcs, shots=args.shots, seed_simulator=seed).result()
        elapsed = time.time() - t0

        # Extract per-instance results
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
                counts = result.get_counts(c_idx)
                p1 = get_ancilla_prob(counts)
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
        all_regrets.append(traj_regrets)
        all_co_optimal.append(traj_co_opt)
        all_mean_regret.append(mean_r)

        logger.info(
            f"  [traj {traj+1:3d}/{args.n_trajectories}] seed={seed:3d} "
            f"regret={mean_r*100:+.2f}% co-opt={n_co}/{n_inst} "
            f"({elapsed:.1f}s)"
        )

    # --- AGGREGATE ---
    regret_arr = np.array(all_regrets)       # (N_traj, N_inst)
    co_opt_arr = np.array(all_co_optimal)    # (N_traj, N_inst) bool
    mean_regret_arr = np.array(all_mean_regret)

    logger.info("\n" + "=" * 72)
    logger.info("DISTRIBUTION SUMMARY")
    logger.info("=" * 72)
    logger.info(f"  Mean regret: {mean_regret_arr.mean()*100:.3f}% ± {mean_regret_arr.std()*100:.3f}%")
    logger.info(f"  Median: {np.median(mean_regret_arr)*100:.3f}%")
    logger.info(f"  Min: {mean_regret_arr.min()*100:.3f}%, Max: {mean_regret_arr.max()*100:.3f}%")
    logger.info(f"  25th–75th: {np.percentile(mean_regret_arr,25)*100:.3f}%–{np.percentile(mean_regret_arr,75)*100:.3f}%")

    logger.info(f"\n  Per-instance co-optimal probability (out of {args.n_trajectories} trajectories):")
    for i in range(n_inst):
        p = co_opt_arr[:, i].mean()
        logger.info(f"    [{i}] {instances[i]['category']:10s} "
                    f"n={len(instances[i]['eligible_pairs']):2d} pairs  "
                    f"P(co-opt)={p:.2f}")

    # Hardware references
    logger.info(f"\n  ibm_fez reference:      0.61% — percentile in Arm B dist: "
                f"{(mean_regret_arr <= 0.0061).mean()*100:.1f}%")
    logger.info(f"  ibm_marrakesh reference: 1.48% — percentile: "
                f"{(mean_regret_arr <= 0.0148).mean()*100:.1f}%")
    logger.info(f"  K_c kernel reference:    0.86% — percentile: "
                f"{(mean_regret_arr <= 0.0086).mean()*100:.1f}%")

    # Save
    summary = {
        "n_trajectories": args.n_trajectories,
        "seed_base": args.seed_base,
        "shots": args.shots,
        "device": args.device,
        "mean_regret_per_trajectory": all_mean_regret,
        "per_instance_regrets": regret_arr.tolist(),
        "per_instance_co_optimal": co_opt_arr.tolist(),
        "aggregate": {
            "mean": float(mean_regret_arr.mean()),
            "std": float(mean_regret_arr.std()),
            "median": float(np.median(mean_regret_arr)),
            "min": float(mean_regret_arr.min()),
            "max": float(mean_regret_arr.max()),
            "q25": float(np.percentile(mean_regret_arr, 25)),
            "q75": float(np.percentile(mean_regret_arr, 75)),
            "per_instance_co_opt_prob": [float(co_opt_arr[:, i].mean()) for i in range(n_inst)],
        },
        "percentiles": {
            "fez_0.61pct": float((mean_regret_arr <= 0.0061).mean()),
            "marrakesh_1.48pct": float((mean_regret_arr <= 0.0148).mean()),
            "kc_0.86pct": float((mean_regret_arr <= 0.0086).mean()),
        },
    }
    out_path = os.path.join(args.output, "exp7b_multi_trajectory.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
