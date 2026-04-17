#!/usr/bin/env python3
"""
Experiment 9c: Quantum-Classical Bridge Experiment (Phase 2D).

Adds a 6th architecture to Exp 9: V3-extended 19-qubit circuit computing
the matching similarity w_ij in each cycle of a 30-day society simulation.
This is the first integrated pipeline run — quantum+society meet.

We use the pretrained θ from Exp 6d (results/exp6d/theta.npy) and the
noiseless AerSimulator (v2 statevector). When IBM Quantum credentials are
restored, add `--noise-model ibm_marrakesh` to re-run with calibrated noise.

Usage:
    python exp9c_quantum_bridge.py --days 30 --tier m
"""
import json
import logging
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from core.domain import (
    MatchingInstance, acceptance_weights, linear_weights, skill_only_weights,
)
from core.history import CycleSnapshot, SimulationHistory
from core.cycle_engine import CycleEngine, create_initial_snapshot
from core.benchmark_model import (
    QUBOBuilder, solve_simulated_annealing, evaluate_solution,
)
from core.benchmark_fixtures import tier_m, tier_l
from experiments.exp9_pipeline_comparison import run_pipeline
from experiments.exp9b_factorial_ablation import kc_kernel_weights  # numpy-2-safe
from experiments.exp6d_v3_extended import (
    N_PARAMS, build_v3_extended, get_ancilla_prob,
    member_embedding_9d, task_embedding_9d,
)


def build_v3_weight_fn(theta: np.ndarray, shots: int = 4096,
                       use_gpu: bool = False, noise_model=None):
    """Returns a function that computes w_ij via V3-extended noiseless sim.

    The function has the signature expected by run_pipeline: takes a
    MatchingInstance, returns {var_idx: weight}.
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    device = "GPU" if use_gpu else "CPU"
    if noise_model is not None:
        sim = AerSimulator(noise_model=noise_model, method="statevector",
                           device=device)
    else:
        sim = AerSimulator(method="statevector", device=device)

    def wfn(instance: MatchingInstance) -> dict:
        if instance.n_vars == 0:
            return {}
        circuits = []
        pair_list = list(instance.pairs)
        for pair in pair_list:
            m = instance.member(pair)
            t = instance.task(pair)
            v1 = member_embedding_9d(m)
            v2 = task_embedding_9d(t.required_skill)
            circuits.append(build_v3_extended(v1, v2, theta))
        tqcs = transpile(circuits, sim, optimization_level=0)
        result = sim.run(tqcs, shots=shots).result()
        weights = {}
        for idx, pair in enumerate(pair_list):
            p1 = get_ancilla_prob(result.get_counts(idx))
            # Similarity = 1 - P(ancilla = |1>).
            sim_score = float(1.0 - p1)
            weights[pair.var_idx] = max(sim_score, 0.001)
        return weights

    return wfn


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exp 9c: Quantum-Classical Bridge")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--cycles-per-day", type=int, default=1)
    parser.add_argument("--tier", choices=["s", "m", "l"], default="m")
    parser.add_argument("--sa-reads", type=int, default=500)
    parser.add_argument("--shots", type=int, default=4096,
                        help="Shots per V3 circuit evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--theta-path", default=os.path.join(
        ROOT, "results", "exp6d", "theta.npy"))
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU AerSimulator (qiskit-aer-gpu)")
    parser.add_argument("--archs", nargs="+",
                        default=["linear", "acceptance", "kc_9d", "v3_noiseless"],
                        help="Which architectures to run "
                             "(options: linear/skill_only/acceptance/kc_9d/kc_12d/"
                             "v3_noiseless/v3_noisy)")
    parser.add_argument("--noise-backend", default="ibm_marrakesh",
                        help="IBM backend whose NoiseModel drives v3_noisy")
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "exp9c"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load θ
    if not os.path.exists(args.theta_path):
        raise SystemExit(f"θ not found at {args.theta_path}. Run Exp 6d first.")
    theta = np.load(args.theta_path)
    assert theta.shape == (N_PARAMS,), f"bad θ shape: {theta.shape}"
    logger.info(f"Loaded θ from {args.theta_path}, norm={np.linalg.norm(theta):.3f}")

    # Fixture
    if args.tier == "s":
        inst = tier_m()  # s-tier too small for multi-arch comparison
    elif args.tier == "m":
        inst = tier_m()
    else:
        inst = tier_l()

    total_cycles = args.days * args.cycles_per_day
    logger.info("=" * 72)
    logger.info("EXPERIMENT 9c: Quantum-Classical Bridge (Phase 2D)")
    logger.info(f"  Tier: {args.tier.upper()}, Days: {args.days}, "
                f"Cycles/day: {args.cycles_per_day} → {total_cycles} cycles")
    logger.info(f"  V3-extended noiseless, shots={args.shots}, "
                f"device={'GPU' if args.use_gpu else 'CPU'}")
    logger.info(f"  Archs: {args.archs}")
    logger.info("=" * 72)

    # Build architectures
    all_archs = {
        "linear": lambda i: linear_weights(i),
        "skill_only": lambda i: skill_only_weights(i),
        "acceptance": lambda i: acceptance_weights(i),
        "kc_9d": lambda i: kc_kernel_weights(i, n_dims=9),
        "kc_12d": lambda i: kc_kernel_weights(i, n_dims=12),
        "v3_noiseless": build_v3_weight_fn(theta, shots=args.shots,
                                           use_gpu=args.use_gpu,
                                           noise_model=None),
    }
    if "v3_noisy" in args.archs:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_aer.noise import NoiseModel
        logger.info(f"Fetching NoiseModel from {args.noise_backend}...")
        svc = QiskitRuntimeService(channel="ibm_quantum_platform")
        backend = svc.backend(args.noise_backend)
        nm = NoiseModel.from_backend(backend)
        logger.info(f"  NoiseModel loaded: {len(nm.basis_gates)} basis gates")
        all_archs["v3_noisy"] = build_v3_weight_fn(
            theta, shots=args.shots, use_gpu=args.use_gpu, noise_model=nm,
        )
    architectures = {a: all_archs[a] for a in args.archs}

    # Run each
    initial = create_initial_snapshot(inst.members, inst.tasks, inst.categories, inst.prices)
    engine = CycleEngine(seed=args.seed, demurrage_grace_cycles=10 * args.cycles_per_day)

    results = {}
    for name, wfn in architectures.items():
        logger.info(f"\n  Running {name}...")
        t0 = time.time()
        history = run_pipeline(name, initial, wfn, args.days, engine,
                               cycles_per_day=args.cycles_per_day,
                               sa_reads=args.sa_reads)
        elapsed = time.time() - t0
        results[name] = (history, elapsed)
        logger.info(f"    {name}: {elapsed:.1f}s, {history.n_cycles} cycles")

    # Metrics
    logger.info("\n" + "=" * 72)
    logger.info(f"RESULTS — After {args.days} days ({total_cycles} cycles)")
    logger.info("=" * 72)
    logger.info(f"\n  {'Architecture':<14s} {'Gini':>6s} {'HH_Gini':>7s} "
                f"{'Fill%':>6s} {'Skill':>6s} {'EDef':>6s} {'Time':>6s}")
    logger.info(f"  {'-'*14} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    summary_rows = []
    for name, (history, elapsed) in results.items():
        final = history.state_at(total_cycles)
        bals = np.array([m.credit_balance for m in final.members])
        gini = history.gini_at(total_cycles)
        hh = {}
        for m in final.members:
            hid = (getattr(m, 'household_id', '') or '') or m.member_id
            hh.setdefault(hid, []).append(m.credit_balance)
        hh_means = np.sort(np.array([np.mean(v) for v in hh.values()]))
        n_hh = len(hh_means)
        if n_hh and hh_means.sum() > 0:
            idx = np.arange(1, n_hh + 1)
            hh_gini = float(
                (2 * np.sum(idx * hh_means) - (n_hh + 1) * hh_means.sum())
                / (n_hh * hh_means.sum())
            )
        else:
            hh_gini = 0.0

        skills, fulfillments, deficits = [], [], []
        for c in range(max(1, total_cycles - 4), total_cycles + 1):
            s = history.state_at(c)
            n_tasks_c = len(s.tasks)
            n_assign_c = len(s.assignments)
            if n_tasks_c > 0:
                fulfillments.append(n_assign_c / n_tasks_c)
            for m in s.members:
                cap = getattr(m, 'energy_capacity', 0.0)
                en = getattr(m, 'energy', 0.0)
                if cap > 0:
                    deficits.append(max(0.0, cap - en))
            for a in s.assignments:
                member = s.member_by_id(a.get("member_id", ""))
                task = next((t for t in s.tasks if t.task_id == a.get("task_id", "")), None)
                if member and task:
                    skills.append(member.skill_in(task.required_skill))

        fill_rate = float(np.mean(fulfillments)) if fulfillments else 0.0
        e_def = float(np.mean(deficits)) if deficits else 0.0
        mean_skill = float(np.mean(skills)) if skills else 0.0

        logger.info(
            f"  {name:<14s} {gini:6.3f} {hh_gini:7.3f} "
            f"{fill_rate*100:5.1f}% {mean_skill:6.3f} "
            f"{e_def:5.2f}h {elapsed:5.1f}s"
        )
        summary_rows.append({
            "arch": name, "gini": float(gini), "hh_gini": float(hh_gini),
            "fulfillment_rate": fill_rate, "mean_provider_skill": mean_skill,
            "mean_energy_deficit_h": e_def,
            "elapsed_sec": float(elapsed),
            "avg_balance": float(bals.mean()),
            "min_balance": float(bals.min()),
        })

    # Compare v3 vs linear
    if "v3_noiseless" in results and "linear" in results:
        v3 = next(r for r in summary_rows if r["arch"] == "v3_noiseless")
        lin = next(r for r in summary_rows if r["arch"] == "linear")
        logger.info("\n  v3_noiseless vs linear:")
        logger.info(f"    Gini:     {v3['gini']:.3f} vs {lin['gini']:.3f}  Δ={v3['gini']-lin['gini']:+.3f}")
        logger.info(f"    HH_Gini:  {v3['hh_gini']:.3f} vs {lin['hh_gini']:.3f}  Δ={v3['hh_gini']-lin['hh_gini']:+.3f}")
        logger.info(f"    Fill%:    {v3['fulfillment_rate']*100:.1f}% vs {lin['fulfillment_rate']*100:.1f}%  "
                    f"Δ={(v3['fulfillment_rate']-lin['fulfillment_rate'])*100:+.1f}pp")
        logger.info(f"    Skill:    {v3['mean_provider_skill']:.3f} vs {lin['mean_provider_skill']:.3f}  "
                    f"Δ={v3['mean_provider_skill']-lin['mean_provider_skill']:+.3f}")

    # Gini trajectories
    logger.info("\n  Gini over time:")
    sample_points = list(range(0, total_cycles + 1, max(1, total_cycles // 5)))
    if total_cycles not in sample_points:
        sample_points.append(total_cycles)
    header = f"  {'Cycle':>6s}"
    for name in results:
        header += f" {name:>14s}"
    logger.info(header)
    for c in sample_points:
        line = f"  {c:6d}"
        for name, (history, _) in results.items():
            line += f" {history.gini_at(c):14.4f}"
        logger.info(line)

    # Save
    with open(os.path.join(args.output, "bridge.json"), "w") as f:
        json.dump({
            "experiment": "Exp 9c: Quantum-Classical Bridge",
            "tier": args.tier,
            "n_days": args.days,
            "cycles_per_day": args.cycles_per_day,
            "total_cycles": total_cycles,
            "seed": args.seed,
            "theta_path": args.theta_path,
            "shots_per_sim": args.shots,
            "archs": list(results.keys()),
            "summary": summary_rows,
            "gini_trajectory": {
                name: [(c, float(history.gini_at(c))) for c in sample_points]
                for name, (history, _) in results.items()
            },
        }, f, indent=2)
    logger.info(f"\n  Saved: {os.path.join(args.output, 'bridge.json')}")


if __name__ == "__main__":
    main()
