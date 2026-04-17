#!/usr/bin/env python3
"""
Experiment 9b: Factorial Ablation of Tier A Society Features (Phase 2B).

Turns Exp 9's descriptive "Tier A compresses architecture differences" into
a causal claim by isolating the marginal contribution of each realism feature.

Design: 2^4 factorial × 5 pipeline architectures on M-tier, 30 cycles 1x/day.

Factors (each ON/OFF):
  - AGE     : age-based energy capacity + task eligibility (ON = real ages,
              OFF = all members aged 35 with peak energy)
  - TOOLS   : tool hard-filter (ON = real tools, OFF = no required_tools)
  - HH      : household credit pooling (ON = real household_id, OFF = all solo)
  - URGENCY : task urgency multiplier (ON = real urgencies, OFF = all 0.5)

Architectures: linear, skill_only, acceptance, kc_9d, kc_12d.

Output: results/exp9b/factorial.json + markdown summary.
"""
import itertools
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

from core.domain import (
    MatchingInstance, Member, Task, TaskType, EffortProfile,
    acceptance_weights, linear_weights, skill_only_weights,
)
from core.history import SimulationHistory
from core.cycle_engine import CycleEngine, create_initial_snapshot
from core.benchmark_model import (
    QUBOBuilder, solve_simulated_annealing, evaluate_solution,
)
from core.benchmark_fixtures import tier_m
from experiments.exp9_pipeline_comparison import (
    _phi as _ek_phi,
    _kc_kernel as _ek_kernel,
    _build_kc_training_data as _ek_build_train,
)


def kc_kernel_weights(instance: MatchingInstance, n_dims: int = 9,
                      lam: float = 0.01) -> dict:
    """Patched K_c kernel ridge (numpy 2.x compatible): flattens prediction
    before float() conversion. Otherwise identical to exp9's kc_kernel_weights."""
    X_train, y_train = _ek_build_train(instance, n_dims)
    if len(X_train) < 5:
        return skill_only_weights(instance)

    K = _ek_kernel(X_train, X_train)
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
        K_test = _ek_kernel(x_test, X_train) / max(k_scale, 1e-12)
        raw = np.clip(K_test @ alpha, 0.0, 1.0)
        pred = float(np.asarray(raw).ravel()[0])
        weights[pair.var_idx] = max(pred, 0.001)

    return weights


# ============================================================
# Feature stripping — build the 16 ablated fixtures
# ============================================================

def strip_features(
    inst: MatchingInstance,
    age_on: bool,
    tools_on: bool,
    hh_on: bool,
    urg_on: bool,
) -> MatchingInstance:
    """Return a copy of `inst` with selected Tier A features stripped."""
    new_members = []
    for m in inst.members:
        updates = {}
        if not age_on:
            # Peak age 35 → max energy capacity; removes age eligibility filter.
            updates["age"] = 35
            updates["energy"] = -1.0  # auto → energy_capacity
        if not tools_on:
            updates["tools"] = frozenset()
        if not hh_on:
            updates["household_id"] = ""
        if updates:
            new_members.append(m.model_copy(update=updates))
        else:
            new_members.append(m)

    new_tasks = []
    for t in inst.tasks:
        updates = {}
        if not tools_on:
            # Strip required_tools from task_type.effort
            new_effort = t.task_type.effort.model_copy(update={"required_tools": frozenset()})
            new_task_type = t.task_type.model_copy(update={"effort": new_effort})
            updates["task_type"] = new_task_type
        if not urg_on:
            updates["urgency"] = 0.5
        if updates:
            new_tasks.append(t.model_copy(update=updates))
        else:
            new_tasks.append(t)

    return MatchingInstance(
        instance_id=inst.instance_id + "_ablated",
        members=tuple(new_members),
        tasks=tuple(new_tasks),
        categories=inst.categories,
        prices=inst.prices,
        skill_threshold=inst.skill_threshold,
    )


# ============================================================
# Run one (cell, architecture) simulation
# ============================================================

def _arch_weight_fn(name):
    """Return (name, callable). Avoids pickling lambdas."""
    return {
        "linear": lambda i: linear_weights(i),
        "skill_only": lambda i: skill_only_weights(i),
        "acceptance": lambda i: acceptance_weights(i),
        "kc_9d": lambda i: kc_kernel_weights(i, n_dims=9),
        "kc_12d": lambda i: kc_kernel_weights(i, n_dims=12),
    }[name]


def run_one_cell(
    cell_id: str,
    arch: str,
    age_on: bool, tools_on: bool, hh_on: bool, urg_on: bool,
    n_days: int = 30, sa_reads: int = 500, seed: int = 42,
) -> dict:
    """Run one architecture on one ablated fixture. Returns a metrics dict."""
    inst_full = tier_m()
    inst = strip_features(inst_full, age_on, tools_on, hh_on, urg_on)

    initial = create_initial_snapshot(inst.members, inst.tasks, inst.categories, inst.prices)
    engine = CycleEngine(seed=seed, demurrage_grace_cycles=10)

    wfn = _arch_weight_fn(arch)
    history = SimulationHistory(f"/tmp/exp9b_{cell_id}_{arch}", seed=seed)
    snap = initial
    history.append(snap)

    total_cycles = n_days
    for _ in range(total_cycles):
        matching = snap.to_matching_instance(skill_threshold=0.3)
        if matching.n_vars == 0:
            # No eligible pairs this cycle — advance without assignments.
            from core.history import CycleSnapshot
            snap = CycleSnapshot(
                cycle=snap.cycle + 1, members=snap.members, tasks=snap.tasks,
                categories=snap.categories, prices=snap.prices,
                economy_state=snap.economy_state,
            )
            history.append(snap)
            continue
        weights = wfn(matching)
        builder = QUBOBuilder(matching, weights)
        Q = builder.build()
        bits, _ = solve_simulated_annealing(Q, builder.n_vars, num_reads=sa_reads)
        result = evaluate_solution(matching, builder, bits)
        snap = engine.advance(snap, result.assignments, reset_energy=True)
        history.append(snap)

    # === Metrics on final state + last 5 cycles average ===
    final = history.state_at(total_cycles)
    bals = np.array([m.credit_balance for m in final.members])
    gini = history.gini_at(total_cycles)

    # Household Gini
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

    # Fulfillment + skill + energy deficit (last 5 cycles)
    skills, fulfillments, deficits = [], [], []
    start = max(1, total_cycles - 4)
    for c in range(start, total_cycles + 1):
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

    return {
        "cell_id": cell_id,
        "arch": arch,
        "age_on": age_on, "tools_on": tools_on, "hh_on": hh_on, "urg_on": urg_on,
        "n_cycles": total_cycles,
        "gini": float(gini),
        "hh_gini": float(hh_gini),
        "avg_balance": float(bals.mean()),
        "min_balance": float(bals.min()),
        "mean_provider_skill": float(np.mean(skills)) if skills else 0.0,
        "fulfillment_rate": float(np.mean(fulfillments)) if fulfillments else 0.0,
        "mean_energy_deficit_h": float(np.mean(deficits)) if deficits else 0.0,
    }


# ============================================================
# Main: full factorial
# ============================================================

ARCHS = ["linear", "skill_only", "acceptance", "kc_9d", "kc_12d"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exp 9b: Factorial ablation of Tier A features")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--sa-reads", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "exp9b"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 2^4 cells × 5 architectures
    cells = list(itertools.product([False, True], repeat=4))
    jobs = []
    for cell in cells:
        age_on, tools_on, hh_on, urg_on = cell
        cell_id = (
            ("A" if age_on else "a") +
            ("T" if tools_on else "t") +
            ("H" if hh_on else "h") +
            ("U" if urg_on else "u")
        )
        for arch in ARCHS:
            jobs.append((cell_id, arch, age_on, tools_on, hh_on, urg_on,
                         args.days, args.sa_reads, args.seed))

    logger.info("=" * 72)
    logger.info("EXPERIMENT 9b: Factorial Ablation of Tier A Features")
    logger.info(f"  Cells: {len(cells)} (2^4), Architectures: {len(ARCHS)}, "
                f"Total runs: {len(jobs)}")
    logger.info(f"  Days/run: {args.days}, Workers: {args.workers}")
    logger.info("=" * 72)

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one_cell, *j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
                results.append(r)
                logger.info(f"  [{i:>3d}/{len(jobs)}] {r['cell_id']} {r['arch']:<10s} "
                            f"gini={r['gini']:.3f} hh_gini={r['hh_gini']:.3f} "
                            f"fill={r['fulfillment_rate']*100:.1f}% "
                            f"skill={r['mean_provider_skill']:.3f}")
            except Exception as e:
                j = futs[fut]
                logger.error(f"  FAILED {j[0]} {j[1]}: {e}")

    elapsed = time.time() - t0
    logger.info(f"\nTotal wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # === Summary table ===
    logger.info("\n" + "=" * 72)
    logger.info("MARGINAL EFFECTS — averaged across architectures")
    logger.info("=" * 72)
    logger.info("Cell encoding: A/a=age, T/t=tools, H/h=households, U/u=urgency "
                "(uppercase=ON, lowercase=OFF)")

    # Aggregate per cell (mean across 5 archs) + per arch (mean across 16 cells)
    by_cell = {}
    for r in results:
        by_cell.setdefault(r["cell_id"], []).append(r)

    logger.info(f"\n  {'Cell':<6s} {'Gini':>6s} {'HH_Gini':>7s} {'Fill%':>6s} "
                f"{'Skill':>6s} {'EDef':>6s}")
    logger.info(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")
    for cell_id in sorted(by_cell):
        rs = by_cell[cell_id]
        logger.info(
            f"  {cell_id:<6s} "
            f"{np.mean([r['gini'] for r in rs]):6.3f} "
            f"{np.mean([r['hh_gini'] for r in rs]):7.3f} "
            f"{np.mean([r['fulfillment_rate'] for r in rs])*100:5.1f}% "
            f"{np.mean([r['mean_provider_skill'] for r in rs]):6.3f} "
            f"{np.mean([r['mean_energy_deficit_h'] for r in rs]):5.2f}h"
        )

    # Marginal effect of each factor: mean over ON minus mean over OFF
    logger.info("\n  FACTOR MARGINAL EFFECT on Gini (ON − OFF, mean across archs+other factors):")
    for factor in ["age_on", "tools_on", "hh_on", "urg_on"]:
        on_vals = [r["gini"] for r in results if r[factor]]
        off_vals = [r["gini"] for r in results if not r[factor]]
        logger.info(f"    {factor:<10s}: ON={np.mean(on_vals):.3f}  "
                    f"OFF={np.mean(off_vals):.3f}  "
                    f"Δ={np.mean(on_vals) - np.mean(off_vals):+.3f}")

    logger.info("\n  FACTOR MARGINAL EFFECT on Fulfillment rate:")
    for factor in ["age_on", "tools_on", "hh_on", "urg_on"]:
        on_vals = [r["fulfillment_rate"] for r in results if r[factor]]
        off_vals = [r["fulfillment_rate"] for r in results if not r[factor]]
        logger.info(f"    {factor:<10s}: ON={np.mean(on_vals)*100:5.1f}%  "
                    f"OFF={np.mean(off_vals)*100:5.1f}%  "
                    f"Δ={(np.mean(on_vals)-np.mean(off_vals))*100:+5.1f}pp")

    logger.info("\n  FACTOR MARGINAL EFFECT on HH Gini:")
    for factor in ["age_on", "tools_on", "hh_on", "urg_on"]:
        on_vals = [r["hh_gini"] for r in results if r[factor]]
        off_vals = [r["hh_gini"] for r in results if not r[factor]]
        logger.info(f"    {factor:<10s}: ON={np.mean(on_vals):.3f}  "
                    f"OFF={np.mean(off_vals):.3f}  "
                    f"Δ={np.mean(on_vals) - np.mean(off_vals):+.3f}")

    # Save
    with open(os.path.join(args.output, "factorial.json"), "w") as f:
        json.dump({
            "experiment": "Exp 9b: Factorial Ablation of Tier A Features",
            "n_days": args.days,
            "n_cells": len(cells),
            "archs": ARCHS,
            "seed": args.seed,
            "elapsed_sec": elapsed,
            "results": results,
        }, f, indent=2)
    logger.info(f"\n  Saved: {os.path.join(args.output, 'factorial.json')}")


if __name__ == "__main__":
    main()
