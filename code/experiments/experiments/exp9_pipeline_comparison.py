#!/usr/bin/env python3
"""
Experiment 9: Pipeline Architecture Comparison on the Society Model

Runs the same community through multiple coordination cycles under
different weight functions (pipeline architectures), tracking how each
one shapes the community over time.

Architectures compared:
  1. Linear   — paper's α·sim + β·rep + γ·skill + δ·price (Exp 6/7 baseline)
  2. Skill-only — raw skill match (simplest possible)
  3. Acceptance — sigmoid P(accept) with distance, schedule, economics (v2 model)
  4. K_c 9D    — Shin-Teo-Jeong kernel on 9D embeddings (Exp 7 Arm D)
  5. K_c 12D   — K_c on full feature vector (Exp 8)

Each architecture runs an independent simulation from the same initial
state. After N cycles, compare: Gini, mean provider skill, mean net value,
money-losing assignments, total credits in circulation.

This answers: "Which pipeline architecture produces the best community
outcomes over time — not just the best single-cycle assignment?"

Usage:
    python -m experiments.exp9_pipeline_comparison [--cycles 50] [--tier m]
"""

import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from core.domain import (
    MatchingInstance, Member, Task, TaskType, SkillCategory,
    Location, TimeWindow,
    acceptance_weights, linear_weights, skill_only_weights, uniform_weights,
)
from core.history import CycleSnapshot, SimulationHistory
from core.cycle_engine import CycleEngine, create_initial_snapshot
from core.benchmark_model import (
    QUBOBuilder, solve_simulated_annealing, evaluate_solution,
)
from core.benchmark_fixtures import tier_s, tier_m, tier_l


# =====================================================================
# K_c kernel weight functions (from Exp 7/8, adapted for v2 models)
# =====================================================================

def _phi(x, lo=0.1, hi=np.pi - 0.1):
    return lo + (hi - lo) * np.asarray(x, dtype=np.float64)


def _kc_kernel(X1, X2):
    phi1 = _phi(X1)
    phi2 = _phi(X2)
    diff = phi1[:, None, :] - phi2[None, :, :]
    factors = 1.0 + np.cos(diff)
    factors = np.clip(factors, 1e-12, 2.0)
    log_k = np.sum(np.log(factors), axis=-1)
    return np.exp(log_k)


def _build_kc_training_data(instance: MatchingInstance, n_dims: int):
    """Build (X, y) training data from instance members/tasks."""
    X_list, y_list = [], []
    cats = list(instance.categories)
    for pair in instance.pairs:
        m = instance.member(pair)
        t = instance.task(pair)
        skill = m.skill_in(t.required_skill)
        label = 1.0 if skill > 0.6 else (0.0 if skill < 0.2 else None)
        if label is None:
            continue

        if n_dims == 9:
            # 9D: 6 skill categories + reputation + time_availability + social_capital
            vec = [m.skill_levels.get(c, 0.0) for c in cats[:6]]
            vec += [m.reputation, m.time_availability, m.social_capital]
            vec = vec[:9]  # pad/trim to 9
            while len(vec) < 9:
                vec.append(0.0)
        else:
            # 12D: 6 skills + reputation + proximity + price + time + social + trust
            vec = [m.skill_levels.get(c, 0.0) for c in cats[:6]]
            vec.append(m.reputation)
            vec.append(m.skill_in(t.required_skill))  # proximity
            vec.append(min(t.price / 10.0, 1.0))
            vec.append(m.time_availability)
            vec.append(m.social_capital)
            trust = m.skill_in(t.required_skill) / 0.3 if 0.3 > 0 else 1.0
            vec.append(min(trust, 1.0))

        X_list.append(np.clip(vec, 0, 1))
        y_list.append(label)

    return np.array(X_list), np.array(y_list)


def kc_kernel_weights(instance: MatchingInstance, n_dims: int = 9,
                      lam: float = 0.01) -> dict[int, float]:
    """K_c kernel ridge regression weights (Exp 7 Arm D / Exp 8 style)."""
    X_train, y_train = _build_kc_training_data(instance, n_dims)
    if len(X_train) < 5:
        # Not enough training data — fall back to skill_only
        return skill_only_weights(instance)

    K = _kc_kernel(X_train, X_train)
    k_scale = float(np.mean(np.diag(K)))
    K_norm = K / max(k_scale, 1e-12)
    M = K_norm.shape[0]
    alpha = np.linalg.solve(K_norm + lam * np.eye(M), y_train)

    cats = list(instance.categories)
    weights = {}
    for pair in instance.pairs:
        m = instance.member(pair)
        t = instance.task(pair)

        if n_dims == 9:
            vec = [m.skill_levels.get(c, 0.0) for c in cats[:6]]
            vec += [m.reputation, m.time_availability, m.social_capital]
            vec = vec[:9]
            while len(vec) < 9:
                vec.append(0.0)
        else:
            vec = [m.skill_levels.get(c, 0.0) for c in cats[:6]]
            vec.append(m.reputation)
            vec.append(m.skill_in(t.required_skill))
            vec.append(min(t.price / 10.0, 1.0))
            vec.append(m.time_availability)
            vec.append(m.social_capital)
            trust = m.skill_in(t.required_skill) / 0.3
            vec.append(min(trust, 1.0))

        x_test = np.clip(np.array([vec]), 0, 1)
        K_test = _kc_kernel(x_test, X_train) / max(k_scale, 1e-12)
        raw = np.clip(K_test @ alpha, 0.0, 1.0)
        pred = float(np.asarray(raw).ravel()[0])
        weights[pair.var_idx] = max(pred, 0.001)

    return weights


# =====================================================================
# Run one architecture for N cycles
# =====================================================================

def run_pipeline(
    name: str,
    initial_snap: CycleSnapshot,
    weight_fn,
    n_days: int,
    engine: CycleEngine,
    cycles_per_day: int = 1,
    sa_reads: int = 500,
) -> SimulationHistory:
    """Run a full simulation with a specific weight function.

    Args:
        n_days: number of simulated days
        cycles_per_day: 1 = one matching round per day (energy resets each cycle)
                        3 = three rounds per day (energy depletes across rounds,
                            resets only at day start)
    """
    history = SimulationHistory(f"/tmp/exp9_{name}", seed=engine._base_seed)
    snap = initial_snap
    history.append(snap)

    for day in range(n_days):
        for subcycle in range(cycles_per_day):
            is_day_start = (subcycle == 0)

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
            bits, energy = solve_simulated_annealing(Q, builder.n_vars, num_reads=sa_reads)
            result = evaluate_solution(instance, builder, bits)

            snap = engine.advance(snap, result.assignments, reset_energy=is_day_start)
            history.append(snap)

    return history


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exp 9: Pipeline Architecture Comparison")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--cycles-per-day", type=int, default=1,
                        help="1 = one round/day (energy resets each cycle), "
                             "3 = three rounds/day (energy depletes across rounds)")
    parser.add_argument("--tier", choices=["s", "m", "l"], default="m")
    parser.add_argument("--sa-reads", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exp9"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load fixture
    if args.tier == "s":
        inst = tier_s()
        threshold = 0.0
    elif args.tier == "m":
        inst = tier_m()
        threshold = 0.4
    else:
        inst = tier_l()
        threshold = 0.3

    total_cycles = args.days * args.cycles_per_day
    mode_label = f"{args.cycles_per_day}x/day" if args.cycles_per_day > 1 else "1x/day"

    logger.info("=" * 72)
    logger.info("EXPERIMENT 9: Pipeline Architecture Comparison")
    logger.info(f"  Tier: {args.tier.upper()}, Days: {args.days}, "
                f"Cycles/day: {args.cycles_per_day} ({total_cycles} total cycles)")
    logger.info(f"  Mode: {mode_label} — energy {'depletes across rounds' if args.cycles_per_day > 1 else 'resets each cycle'}")
    logger.info(f"  {inst.summary()}")
    logger.info("=" * 72)

    # Initial snapshot
    initial = create_initial_snapshot(inst.members, inst.tasks, inst.categories, inst.prices)
    engine = CycleEngine(seed=args.seed, demurrage_grace_cycles=10 * args.cycles_per_day)

    # Define architectures
    architectures = {
        "linear": lambda i: linear_weights(i),
        "skill_only": lambda i: skill_only_weights(i),
        "acceptance": lambda i: acceptance_weights(i),
        "kc_9d": lambda i: kc_kernel_weights(i, n_dims=9),
        "kc_12d": lambda i: kc_kernel_weights(i, n_dims=12),
    }

    # Run each architecture
    results = {}
    for name, wfn in architectures.items():
        logger.info(f"\n  Running {name}...")
        t0 = time.time()
        history = run_pipeline(name, initial, wfn, args.days, engine,
                               cycles_per_day=args.cycles_per_day,
                               sa_reads=args.sa_reads)
        elapsed = time.time() - t0
        results[name] = history
        logger.info(f"    {name}: {elapsed:.1f}s, {history.n_cycles} cycles")

    # === COMPARISON TABLE ===
    logger.info("\n" + "=" * 72)
    logger.info("RESULTS — After %d days (%d cycles, %s)", args.days, total_cycles, mode_label)
    logger.info("=" * 72)

    # Final state metrics
    logger.info(f"\n  {'Architecture':<14s} {'Gini':>6s} {'HH_Gini':>7s} "
                f"{'AvgBal':>7s} {'MinBal':>7s} {'AvgRep':>7s}")
    logger.info(f"  {'-'*14} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for name, history in results.items():
        final = history.state_at(total_cycles)
        bals = [m.credit_balance for m in final.members]
        reps = [m.reputation for m in final.members]
        gini = history.gini_at(total_cycles)
        # Household Gini computed from final state
        hh = {}
        for m in final.members:
            hid = getattr(m, 'household_id', '') or m.member_id
            hh.setdefault(hid, []).append(m.credit_balance)
        hh_means = [np.mean(v) for v in hh.values()]
        hh_arr = np.sort(np.array(hh_means))
        n_hh = len(hh_arr)
        if n_hh and hh_arr.sum() > 0:
            idx = np.arange(1, n_hh + 1)
            hh_gini = float((2 * np.sum(idx * hh_arr) - (n_hh + 1) * hh_arr.sum()) / (n_hh * hh_arr.sum()))
        else:
            hh_gini = 0.0
        logger.info(
            f"  {name:<14s} {gini:6.3f} {hh_gini:7.3f} "
            f"{np.mean(bals):7.1f} {min(bals):7.1f} "
            f"{np.mean(reps):7.3f}"
        )

    # Per-cycle assignment quality (last 5 cycles average)
    logger.info(f"\n  Last {min(5, total_cycles)} cycles — average assignment quality:")
    logger.info(f"  {'Architecture':<14s} {'Skill':>6s} {'NetVal':>7s} "
                f"{'Dist':>6s} {'Conv':>5s} {'Loss':>5s} {'Fill%':>6s} {'EDef':>6s}")
    logger.info(f"  {'-'*14} {'-'*6} {'-'*7} {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*6}")

    for name, history in results.items():
        skills, nets, dists, convs, losses = [], [], [], [], []
        fulfillments, deficits = [], []
        start = max(1, total_cycles - 4)
        for c in range(start, total_cycles + 1):
            snap = history.state_at(c)
            n_tasks_c = len(snap.tasks)
            n_assign_c = len(snap.assignments)
            if n_tasks_c > 0:
                fulfillments.append(n_assign_c / n_tasks_c)
            # End-of-cycle energy deficit
            for m in snap.members:
                cap = getattr(m, 'energy_capacity', 0.0)
                en = getattr(m, 'energy', 0.0)
                if cap > 0:
                    deficits.append(max(0.0, cap - en))
            for a in snap.assignments:
                member = snap.member_by_id(a.get("member_id", ""))
                task_id = a.get("task_id", "")
                task = None
                for t in snap.tasks:
                    if t.task_id == task_id:
                        task = t
                        break
                if member and task:
                    sk = member.skill_in(task.required_skill)
                    dist = member.location.distance_to(task.location)
                    net = task.price - dist * 1.0
                    conv = member.schedule.convenience(0, task.time_window)
                    skills.append(sk)
                    nets.append(net)
                    dists.append(dist)
                    convs.append(conv)
                    if net < 0:
                        losses.append(1)

        fill_pct = 100.0 * np.mean(fulfillments) if fulfillments else 0.0
        e_def = np.mean(deficits) if deficits else 0.0
        logger.info(
            f"  {name:<14s} {np.mean(skills) if skills else 0:6.3f} "
            f"{np.mean(nets) if nets else 0:7.2f} "
            f"{np.mean(dists) if dists else 0:6.2f} "
            f"{np.mean(convs) if convs else 0:5.2f} "
            f"{len(losses):5d} "
            f"{fill_pct:5.1f}% "
            f"{e_def:5.2f}h"
        )

    # Gini evolution
    logger.info(f"\n  Gini evolution:")
    sample_points = list(range(0, total_cycles + 1, max(1, total_cycles // 5)))
    if total_cycles not in sample_points:
        sample_points.append(total_cycles)
    header = f"  {'Cycle':>6s}"
    for name in results:
        header += f" {name:>10s}"
    logger.info(header)
    for c in sample_points:
        line = f"  {c:6d}"
        for name, history in results.items():
            line += f" {history.gini_at(c):10.4f}"
        logger.info(line)

    # Save
    summary = {
        "experiment": "Exp 9: Pipeline Architecture Comparison",
        "tier": args.tier,
        "n_days": args.days,
        "cycles_per_day": args.cycles_per_day,
        "total_cycles": total_cycles,
        "seed": args.seed,
        "architectures": list(results.keys()),
    }
    for name, history in results.items():
        final = history.state_at(total_cycles)
        bals = [m.credit_balance for m in final.members]
        summary[name] = {
            "gini_final": history.gini_at(total_cycles),
            "total_credits": sum(bals),
            "avg_balance": float(np.mean(bals)),
            "avg_reputation": float(np.mean([m.reputation for m in final.members])),
            "gini_series": [(c, history.gini_at(c))
                           for c in range(0, total_cycles + 1, max(1, total_cycles // 10))],
        }

    cpd_tag = f"_{args.cycles_per_day}x" if args.cycles_per_day > 1 else ""
    out_path = os.path.join(args.output, f"exp9_{args.tier}{cpd_tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
