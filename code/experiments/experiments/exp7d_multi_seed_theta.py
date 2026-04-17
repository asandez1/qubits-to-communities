#!/usr/bin/env python3
"""
Experiment 7d: Multi-seed θ training (Phase 2A).

Retires the "single-seed θ" reviewer objection by training V3-extended from
scratch with 10 different numpy seeds and evaluating each θ on the same 8
benchmark instances used in Exp 6/7. Reports mean ± std of classical regret
and co-optimal count.

If the noiseless ceiling (1.48%) is stable across seeds (σ small), the
mechanism-isolation conclusions generalize. If σ is large, the paper must
report parameter-sensitivity bands.

Usage:
    python exp7d_multi_seed_theta.py --seeds 0 1 2 3 4 5 6 7 8 9 --workers 4
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from experiments.exp6d_v3_extended import (
    N_PARAMS, build_training_set, train_v3_extended,
    build_v3_extended, get_ancilla_prob,
    member_embedding_9d, task_embedding_9d,
)
from experiments.exp7_mechanism_isolation import (
    harvest_instances, compute_classical_weights,
    build_qubo, solve_qubo_exhaustive, analyze_degeneracy,
    _evaluate_instance, _summarize,
)


CONFIG_PATH = os.path.join(ROOT, "config", "community_economy.json")


def train_and_eval_one_seed(
    seed: int,
    n_members: int = 30,
    train_members: int = 100,
    n_pairs_per_class: int = 40,
    spsa_iter: int = 150,
    shots: int = 2048,
    eval_shots: int = 4096,
) -> dict:
    """Train V3-extended with a specific seed, evaluate on 8 instances."""
    # Constrain openmp threads per subprocess to avoid 4×N thread oversubscription
    # when running under ProcessPoolExecutor. Must be set BEFORE qiskit_aer import.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    t_start = time.time()
    logger.info(f"[seed={seed}] START")

    # Deterministic instance harvesting (seed for the QUBO instance draw is fixed
    # by `harvest_instances`; only the θ seed varies per run).
    instances = harvest_instances(CONFIG_PATH, n_members)

    # Classical baselines for regret computation (same for all seeds).
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

    # Train θ with the given seed.
    train_data = build_training_set(CONFIG_PATH, train_members, n_pairs_per_class)
    t0 = time.time()
    theta, best_loss = train_v3_extended(
        train_data, n_iter=spsa_iter, shots=shots, seed=seed,
    )
    train_time = time.time() - t0

    # Evaluate on the 8 instances (noiseless).
    sim = AerSimulator(method="statevector")
    per_instance = []
    for i, inst in enumerate(instances):
        circuits = []
        for r, o, req, offer, member in inst["eligible_pairs"]:
            v1 = member_embedding_9d(member)
            v2 = task_embedding_9d(req.category)
            circuits.append(build_v3_extended(v1, v2, theta))
        tqcs = transpile(circuits, sim, optimization_level=0)
        result = sim.run(tqcs, shots=eval_shots).result()
        sims = {p: 1.0 - get_ancilla_prob(result.get_counts(p))
                for p in range(len(inst["eligible_pairs"]))}
        rec = _evaluate_instance(inst, sims, classical_results[i])
        rec["instance_idx"] = i
        per_instance.append(rec)
    summary = _summarize(per_instance)

    total_time = time.time() - t_start
    logger.info(
        f"[seed={seed}] DONE: regret={summary['mean_classical_regret']*100:+.2f}% "
        f"co-opt={summary['n_co_optimal']}/{len(instances)} "
        f"identical={summary['n_identical']}/{len(instances)} "
        f"r={summary['mean_sim_correlation_9d']:+.3f} "
        f"(train={train_time:.0f}s total={total_time:.0f}s loss={best_loss:.4f})"
    )

    return {
        "seed": seed,
        "best_loss": float(best_loss),
        "train_time_sec": float(train_time),
        "total_time_sec": float(total_time),
        "theta_norm": float(np.linalg.norm(theta)),
        **summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Exp 7d: Multi-seed θ training")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(10)),
                        help="SPSA training seeds (default: 0..9)")
    parser.add_argument("--members", type=int, default=30,
                        help="Members in harvested instances (matches Exp 6d)")
    parser.add_argument("--train-members", type=int, default=100)
    parser.add_argument("--n-train-per-class", type=int, default=40)
    parser.add_argument("--spsa-iter", type=int, default=150)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--eval-shots", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel training workers (CPU AerSimulator per worker)")
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "exp7d"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    logger.info("=" * 72)
    logger.info("EXPERIMENT 7d: Multi-seed θ training (Phase 2A)")
    logger.info(f"  Seeds: {args.seeds}, SPSA iter: {args.spsa_iter}, "
                f"Workers: {args.workers}")
    logger.info("=" * 72)

    t0 = time.time()
    seed_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                train_and_eval_one_seed,
                s, args.members, args.train_members, args.n_train_per_class,
                args.spsa_iter, args.shots, args.eval_shots,
            ): s for s in args.seeds
        }
        for fut in as_completed(futs):
            try:
                seed_results.append(fut.result())
            except Exception as e:
                logger.error(f"  FAILED seed={futs[fut]}: {e}")

    elapsed = time.time() - t0
    logger.info(f"\nAll seeds done: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    seed_results.sort(key=lambda r: r["seed"])
    regrets = np.array([r["mean_classical_regret"] for r in seed_results])
    co_opts = np.array([r["n_co_optimal"] for r in seed_results])
    corrs = np.array([r["mean_sim_correlation_9d"] for r in seed_results])

    logger.info("\n" + "=" * 72)
    logger.info("MULTI-SEED θ RESULTS (Phase 2A)")
    logger.info("=" * 72)
    logger.info(f"  Seeds tested: {len(seed_results)}")
    logger.info(f"  Regret:    mean={regrets.mean()*100:+.2f}% ± {regrets.std()*100:.2f}pp  "
                f"[min={regrets.min()*100:+.2f}%, max={regrets.max()*100:+.2f}%]")
    logger.info(f"  Co-opt:    mean={co_opts.mean():.1f}/8 ± {co_opts.std():.2f}  "
                f"[min={co_opts.min()}/8, max={co_opts.max()}/8]")
    logger.info(f"  Sim corr:  mean={corrs.mean():+.3f} ± {corrs.std():.3f}")
    logger.info(f"  Ref (Exp 6d, seed=42, ibm_marrakesh/noiseless): +1.48%, 5/8")
    logger.info("\n  Per-seed detail:")
    logger.info(f"  {'seed':>4s} {'regret':>7s} {'co-opt':>6s} {'identical':>9s} "
                f"{'r':>7s} {'loss':>6s} {'time':>5s}")
    logger.info(f"  {'-'*4} {'-'*7} {'-'*6} {'-'*9} {'-'*7} {'-'*6} {'-'*5}")
    for r in seed_results:
        logger.info(
            f"  {r['seed']:>4d} {r['mean_classical_regret']*100:+6.2f}% "
            f"{r['n_co_optimal']:>3d}/8  {r['n_identical']:>3d}/8     "
            f"{r['mean_sim_correlation_9d']:+6.3f} "
            f"{r['best_loss']:6.4f} {r['total_time_sec']:>3.0f}s"
        )

    # Save
    output_file = os.path.join(args.output, "multi_seed_theta.json")
    with open(output_file, "w") as f:
        json.dump({
            "experiment": "Exp 7d: Multi-seed θ training",
            "seeds": args.seeds,
            "spsa_iter": args.spsa_iter,
            "members": args.members,
            "train_members": args.train_members,
            "n_train_per_class": args.n_train_per_class,
            "elapsed_sec": elapsed,
            "aggregate": {
                "n_seeds": len(seed_results),
                "regret_mean": float(regrets.mean()),
                "regret_std": float(regrets.std()),
                "regret_min": float(regrets.min()),
                "regret_max": float(regrets.max()),
                "coopt_mean": float(co_opts.mean()),
                "coopt_std": float(co_opts.std()),
                "coopt_min": int(co_opts.min()),
                "coopt_max": int(co_opts.max()),
                "simcorr_mean": float(corrs.mean()),
                "simcorr_std": float(corrs.std()),
            },
            "per_seed": seed_results,
        }, f, indent=2)
    logger.info(f"\n  Saved: {output_file}")


if __name__ == "__main__":
    main()
