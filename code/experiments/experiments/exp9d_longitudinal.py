#!/usr/bin/env python3
"""
Experiment 9d: Longitudinal Society Simulation (Phase 2G).

Extends Exp 9 to 180 cycles (6× the 30-cycle default) with:
  - Age advancement: members age by 1 cycle = 6 days (365/60 cycles per year).
  - Overnight recovery: existing logic (age-based).
  - Simplified archetype transitions at age thresholds
    (gig → parent at 32, parent → professional at 40, retiree at 65).

Goal: observe whether architecture differences persist or compress as the
population ages, and whether the community collapses (Gini → 1) or stabilizes.

Usage:
    python exp9d_longitudinal.py --cycles 180 --tier m
"""
import argparse
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
    MatchingInstance, Member,
    acceptance_weights, linear_weights, skill_only_weights,
)
from core.history import CycleSnapshot, SimulationHistory
from core.cycle_engine import CycleEngine, create_initial_snapshot
from core.benchmark_model import (
    QUBOBuilder, solve_simulated_annealing, evaluate_solution,
)
from core.benchmark_fixtures import tier_m, tier_l
from experiments.exp9b_factorial_ablation import kc_kernel_weights


# ============================================================
# Ageing wrapper
# ============================================================

def advance_age(snapshot, cycle_count, cycles_per_year=60):
    """Return a snapshot with every member aged by 1/cycles_per_year years.

    Also adjusts energy_capacity by re-computing from age (handled in v2 domain
    because `energy_capacity` is a computed field of Member).
    """
    delta_age = 1.0 / cycles_per_year
    new_members = []
    for m in snapshot.members:
        # Integer age increment: accumulate fractional years via cycles counter
        # on a custom attribute. Simplest approximation: increment age every
        # `cycles_per_year` cycles by 1. Check divisibility on cycle_count.
        if cycle_count > 0 and cycle_count % cycles_per_year == 0:
            new_age = min(m.age + 1, 100)
            new_members.append(m.model_copy(update={"age": new_age}))
        else:
            new_members.append(m)
    return CycleSnapshot(
        cycle=snapshot.cycle, members=tuple(new_members),
        tasks=snapshot.tasks, categories=snapshot.categories,
        prices=snapshot.prices, economy_state=snapshot.economy_state,
        assignments=snapshot.assignments,
    )


def run_longitudinal(name, initial, weight_fn, n_cycles, engine, sa_reads=500,
                     cycles_per_year=60):
    history = SimulationHistory(f"/tmp/exp9d_{name}", seed=engine._base_seed)
    snap = initial
    history.append(snap)

    for c in range(n_cycles):
        instance = snap.to_matching_instance(skill_threshold=0.3)
        if instance.n_vars == 0:
            snap = CycleSnapshot(
                cycle=snap.cycle + 1, members=snap.members, tasks=snap.tasks,
                categories=snap.categories, prices=snap.prices,
                economy_state=snap.economy_state,
            )
            history.append(snap)
            continue
        weights = weight_fn(instance)
        builder = QUBOBuilder(instance, weights)
        Q = builder.build()
        bits, _ = solve_simulated_annealing(Q, builder.n_vars, num_reads=sa_reads)
        result = evaluate_solution(instance, builder, bits)
        snap = engine.advance(snap, result.assignments, reset_energy=True)
        # Age-advance once per cycles_per_year cycles.
        snap = advance_age(snap, c + 1, cycles_per_year=cycles_per_year)
        history.append(snap)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=180)
    parser.add_argument("--tier", choices=["m", "l"], default="m")
    parser.add_argument("--cycles-per-year", type=int, default=60,
                        help="How many cycles advance age by 1 year (default 60 = ~6 days/cycle)")
    parser.add_argument("--sa-reads", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--archs", nargs="+",
                        default=["linear", "acceptance", "kc_9d"])
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "exp9d"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    inst = tier_m() if args.tier == "m" else tier_l()
    logger.info("=" * 72)
    logger.info("EXPERIMENT 9d: Longitudinal Society Simulation (Phase 2G)")
    logger.info(f"  Tier: {args.tier.upper()}, Cycles: {args.cycles}, "
                f"Cycles/year: {args.cycles_per_year}, "
                f"Total simulated: {args.cycles / args.cycles_per_year:.1f} years")
    logger.info(f"  Archs: {args.archs}")
    logger.info("=" * 72)

    initial = create_initial_snapshot(inst.members, inst.tasks, inst.categories, inst.prices)
    engine = CycleEngine(seed=args.seed, demurrage_grace_cycles=10)

    archs = {
        "linear": lambda i: linear_weights(i),
        "skill_only": lambda i: skill_only_weights(i),
        "acceptance": lambda i: acceptance_weights(i),
        "kc_9d": lambda i: kc_kernel_weights(i, n_dims=9),
        "kc_12d": lambda i: kc_kernel_weights(i, n_dims=12),
    }
    architectures = {a: archs[a] for a in args.archs}

    results = {}
    for name, wfn in architectures.items():
        logger.info(f"\n  Running {name}...")
        t0 = time.time()
        history = run_longitudinal(name, initial, wfn, args.cycles, engine,
                                   sa_reads=args.sa_reads,
                                   cycles_per_year=args.cycles_per_year)
        elapsed = time.time() - t0
        results[name] = (history, elapsed)
        logger.info(f"    {name}: {elapsed:.1f}s, {history.n_cycles} cycles")

    # Gini trajectory over time
    logger.info("\n" + "=" * 72)
    logger.info(f"GINI TRAJECTORY — {args.cycles} cycles "
                f"({args.cycles / args.cycles_per_year:.1f} years)")
    logger.info("=" * 72)
    checkpoints = list(range(0, args.cycles + 1, args.cycles // 10))
    if args.cycles not in checkpoints:
        checkpoints.append(args.cycles)
    header = f"  {'Cycle':>6s} {'Years':>6s}"
    for name in results:
        header += f" {name:>12s}"
    logger.info(header)
    for c in checkpoints:
        line = f"  {c:>6d} {c/args.cycles_per_year:>6.1f}"
        for name, (h, _) in results.items():
            line += f" {h.gini_at(c):12.4f}"
        logger.info(line)

    # Per-arch final-state metrics
    logger.info("\n  Final state (cycle={}):".format(args.cycles))
    logger.info(f"  {'arch':<14s} {'Gini':>6s} {'HH_Gini':>7s} "
                f"{'AvgBal':>7s} {'MinBal':>7s} {'MaxAge':>7s} {'AvgAge':>7s}")
    summary_rows = []
    for name, (history, elapsed) in results.items():
        final = history.state_at(args.cycles)
        bals = [m.credit_balance for m in final.members]
        ages = [m.age for m in final.members]
        gini = history.gini_at(args.cycles)
        hh = {}
        for m in final.members:
            hid = (m.household_id or "") or m.member_id
            hh.setdefault(hid, []).append(m.credit_balance)
        hh_means = np.sort(np.array([np.mean(v) for v in hh.values()]))
        if len(hh_means) and hh_means.sum() > 0:
            idx = np.arange(1, len(hh_means) + 1)
            hh_gini = float((2 * np.sum(idx * hh_means) - (len(hh_means) + 1) * hh_means.sum())
                            / (len(hh_means) * hh_means.sum()))
        else:
            hh_gini = 0.0
        logger.info(
            f"  {name:<14s} {gini:6.3f} {hh_gini:7.3f} "
            f"{np.mean(bals):7.1f} {min(bals):7.1f} "
            f"{max(ages):6.0f} {np.mean(ages):6.1f}"
        )
        summary_rows.append({
            "arch": name,
            "gini": float(gini), "hh_gini": float(hh_gini),
            "avg_balance": float(np.mean(bals)), "min_balance": float(min(bals)),
            "max_age": int(max(ages)), "avg_age": float(np.mean(ages)),
            "elapsed_sec": float(elapsed),
        })

    with open(os.path.join(args.output, "longitudinal.json"), "w") as f:
        json.dump({
            "experiment": "Exp 9d: Longitudinal",
            "tier": args.tier, "n_cycles": args.cycles,
            "cycles_per_year": args.cycles_per_year,
            "seed": args.seed, "archs": list(results.keys()),
            "summary": summary_rows,
            "gini_trajectory": {
                name: [(c, float(h.gini_at(c))) for c in checkpoints]
                for name, (h, _) in results.items()
            },
        }, f, indent=2)
    logger.info(f"\n  Saved: {os.path.join(args.output, 'longitudinal.json')}")


if __name__ == "__main__":
    main()
