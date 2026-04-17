"""
Experiment 4: Bridge Resistance Under Adversarial Extraction

Tests whether the asymmetric fiat bridge prevents profitable adversarial extraction.

Three adversary strategies at four population levels (5%, 10%, 15%, 20%):
  A. Simple arbitrage: provide low-effort services, redeem aggressively
  B. Credit hoarding + dump: accumulate for 500 cycles, then mass-redeem
  C. Sybil splitting: split redemptions across sub-accounts to dodge progressive discount

For each (strategy x population), 500 members, 1000 cycles, QUBO matching.

Confirms if: no adversarial strategy achieves positive net extraction
(fiat withdrawn > opportunity cost of participation) with progressive discount tiers.

Output: extraction profitability chart, treasury reserve over time, economy health metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.matching_service import MatchingStrategy
from core.simulation_engine import SimulationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Adversarial population fractions to test
ADV_FRACTIONS = [0.05, 0.10, 0.15, 0.20]

# Adversary strategy names
ADV_STRATEGIES = ["simple_arbitrage", "hoard_and_dump", "sybil_splitting"]


def _make_config_with_adversary_fraction(
    base_config: dict,
    adv_fraction: float,
    adv_strategy: str,
) -> dict:
    """Create a config with a specific adversarial population fraction.

    Scales down non-adversarial archetypes proportionally to make room for
    the adversarial fraction.
    """
    config = json.loads(json.dumps(base_config))  # deep copy

    archetypes = config["archetypes"]
    non_adv_names = [k for k in archetypes if k != "adversarial_extractor"]

    # Current non-adversarial total weight
    non_adv_total = sum(archetypes[k]["weight"] for k in non_adv_names)

    # Scale down non-adversarial to make room
    remaining = 1.0 - adv_fraction
    for k in non_adv_names:
        archetypes[k]["weight"] = archetypes[k]["weight"] / non_adv_total * remaining

    # Set adversarial weight
    archetypes["adversarial_extractor"]["weight"] = adv_fraction

    # Set adversarial strategy behavior
    archetypes["adversarial_extractor"]["strategy"] = "bridge_arbitrage"

    # Store strategy name for engine-level behavior override
    config["_adv_strategy"] = adv_strategy

    return config


def _run_single(args: Tuple) -> dict:
    """Run one (strategy, fraction) scenario."""
    config, adv_strategy, adv_fraction, num_members, num_cycles, seed, output_dir = args

    import logging as _log
    _log.basicConfig(
        level=_log.INFO,
        format=f"%(asctime)s [{adv_strategy}:{adv_fraction:.0%}] %(message)s"
    )
    _logger = _log.getLogger(__name__)

    from core.matching_service import MatchingStrategy as MS
    from core.simulation_engine import SimulationEngine as SE
    from core.member import MemberArchetype

    scenario_config = _make_config_with_adversary_fraction(config, adv_fraction, adv_strategy)

    def progress(cycle, total):
        if cycle % 200 == 0:
            _logger.info(f"Cycle {cycle}/{total}")

    engine = SE(
        config=scenario_config,
        matching_strategy=MS.QUBO,
        num_members=num_members,
        num_cycles=num_cycles,
        seed=seed,
        topology_sample_interval=500,  # minimal topology overhead
        progress_callback=progress,
    )

    # Tag adversarial members for tracking
    adv_ids = set()
    non_adv_ids = set()
    for m in engine.members_list:
        if m.archetype == MemberArchetype.ADVERSARIAL_EXTRACTOR:
            adv_ids.add(m.member_id)
        else:
            non_adv_ids.add(m.member_id)

    # Override bridge behavior based on adversary strategy
    for m in engine.members_list:
        if m.member_id not in adv_ids:
            continue
        if adv_strategy == "hoard_and_dump":
            # Don't redeem until cycle 500, then dump everything
            m.bridge_propensity = 0.0  # will be overridden dynamically
        elif adv_strategy == "sybil_splitting":
            # Similar to simple_arbitrage but will split at redemption
            m.bridge_propensity = 0.95

    # Track initial adversary credits (their "investment")
    adv_initial_credits = sum(engine.economy.get_balance(mid) for mid in adv_ids)

    # Run simulation with per-cycle tracking
    gini_trajectory = []
    match_rate_trajectory = []
    reserve_ratio_trajectory = []
    adv_balance_trajectory = []
    adv_bridge_total = 0.0
    adv_bridge_fiat_total = 0.0
    cycle_bridge_events = []

    # We need to hook into the simulation to customize bridge behavior.
    # Override the bridge decisions for hoard_and_dump and sybil_splitting.
    original_run_cycle = engine._run_cycle

    def _custom_run_cycle(cycle):
        nonlocal adv_bridge_total, adv_bridge_fiat_total

        # For hoard_and_dump: switch on aggressive redemption at cycle 500
        if adv_strategy == "hoard_and_dump":
            for m in engine.members_list:
                if m.member_id in adv_ids:
                    if cycle >= 500:
                        m.bridge_propensity = 0.95
                    else:
                        m.bridge_propensity = 0.0

        record = original_run_cycle(cycle)

        # Track adversary-specific bridge activity
        # After the cycle, check what changed
        for m in engine.members_list:
            if m.member_id in adv_ids:
                # Track balance
                pass

        # Record metrics
        em = record.economy
        gini_trajectory.append(em.gini)
        match_rate_trajectory.append(em.match_rate)
        reserve_ratio_trajectory.append(em.reserve_ratio)
        adv_balance_trajectory.append(
            sum(engine.economy.get_balance(mid) for mid in adv_ids)
        )

        return record

    engine._run_cycle = _custom_run_cycle

    # For sybil_splitting, we need to intercept bridge_redeem calls.
    # The sybil strategy splits one large redemption into multiple small ones
    # to stay below progressive discount thresholds.
    if adv_strategy == "sybil_splitting":
        original_bridge_redeem = engine.economy.bridge_redeem

        def _sybil_bridge_redeem(member_id, amount, current_cycle):
            if member_id not in adv_ids:
                return original_bridge_redeem(member_id, amount, current_cycle)

            # Split into 3-5 sub-redemptions below the first discount threshold
            # First threshold is at 50 credits with 0% discount
            max_per_split = 45.0  # stay below 50 threshold
            if amount <= max_per_split:
                return original_bridge_redeem(member_id, amount, current_cycle)

            # Split into chunks — but RESET the member's redemption history
            # (simulating different accounts)
            total_credits = 0.0
            total_fiat = 0.0
            remaining = amount

            # Save and reset redemption history (simulating Sybil accounts)
            saved_redeemed = engine.economy._bridge_redemptions.get(member_id, 0)

            while remaining > 0:
                chunk = min(max_per_split, remaining)
                # Reset history to simulate a fresh Sybil account
                engine.economy._bridge_redemptions[member_id] = 0
                result = original_bridge_redeem(member_id, chunk, current_cycle)
                if result.success:
                    total_credits += result.credits_spent
                    total_fiat += result.fiat_received
                    remaining -= chunk
                else:
                    break

            # Restore (accumulated) history
            engine.economy._bridge_redemptions[member_id] = saved_redeemed + total_credits

            from core.credit_economy import BridgeResult
            if total_credits > 0:
                return BridgeResult(True, total_credits, total_fiat,
                                    0.0, "Sybil-split redemption")
            return BridgeResult(False, 0, 0, 0, "Sybil split failed")

        engine.economy.bridge_redeem = _sybil_bridge_redeem

    # Run the simulation
    results = engine.run()

    # Compute adversary extraction metrics
    adv_final_credits = sum(engine.economy.get_balance(mid) for mid in adv_ids)
    adv_total_redeemed = sum(
        engine.economy._bridge_redemptions.get(mid, 0) for mid in adv_ids
    )

    # Compute fiat received by adversaries
    # Since we can't directly track fiat per member from the engine,
    # we estimate from the bridge discount tiers
    adv_total_earned = sum(
        engine.members.get(mid).total_earned if engine.members.get(mid) else 0
        for mid in adv_ids
    )
    adv_total_spent = sum(
        engine.members.get(mid).total_spent if engine.members.get(mid) else 0
        for mid in adv_ids
    )
    adv_total_provided = sum(
        engine.members.get(mid).total_provided if engine.members.get(mid) else 0
        for mid in adv_ids
    )

    # Net credit flow: earned - spent - initial_grant = net from economy
    net_credit_earned = adv_total_earned - adv_total_spent

    # Opportunity cost: adversaries could have earned wages in fiat market
    # Approximate as: services_provided * base_hourly_rate * hours_equivalent
    # Since adversaries provide transport (base_price 6), their opportunity cost
    # per service in the real economy is approximately the base price in fiat terms
    opportunity_cost_per_service = 6.0  # base fiat value of transport service
    total_opportunity_cost = adv_total_provided * opportunity_cost_per_service

    # Estimate fiat received (accounting for progressive discounts)
    # This is approximate — actual fiat depends on which discount tiers were hit
    # We use the total redeemed and average discount
    avg_discount = 0.0
    for mid in adv_ids:
        total = engine.economy._bridge_redemptions.get(mid, 0)
        for tier in sorted(engine.economy.bridge_discounts,
                           key=lambda t: t["threshold"], reverse=True):
            if total >= tier["threshold"]:
                avg_discount = max(avg_discount, tier["discount"])
                break

    estimated_fiat = adv_total_redeemed * (1.0 - avg_discount)

    # Net extraction = fiat received - opportunity cost
    net_extraction = estimated_fiat - total_opportunity_cost

    # Economy health
    avg_gini = float(np.mean(gini_trajectory[-100:])) if len(gini_trajectory) >= 100 else 0
    avg_match = float(np.mean(match_rate_trajectory[-100:])) if len(match_rate_trajectory) >= 100 else 0
    final_reserve = reserve_ratio_trajectory[-1] if reserve_ratio_trajectory else 0

    scenario_result = {
        "adv_strategy": adv_strategy,
        "adv_fraction": adv_fraction,
        "num_adversaries": len(adv_ids),
        "num_honest": len(non_adv_ids),
        "adv_initial_credits": float(adv_initial_credits),
        "adv_final_credits": float(adv_final_credits),
        "adv_total_redeemed": float(adv_total_redeemed),
        "adv_total_earned": float(adv_total_earned),
        "adv_total_spent": float(adv_total_spent),
        "adv_total_provided": int(adv_total_provided),
        "estimated_fiat_received": float(estimated_fiat),
        "opportunity_cost": float(total_opportunity_cost),
        "net_extraction": float(net_extraction),
        "extraction_profitable": net_extraction > 0,
        "avg_discount_tier": float(avg_discount),
        "avg_gini_last100": float(avg_gini),
        "avg_match_rate_last100": float(avg_match),
        "final_reserve_ratio": float(final_reserve),
        "final_treasury_fiat": float(results.cycle_records[-1].economy.treasury_fiat),
        "wall_time": float(results.wall_time),
        "gini_trajectory": [float(g) for g in gini_trajectory],
        "match_rate_trajectory": [float(m) for m in match_rate_trajectory],
        "reserve_ratio_trajectory": [float(r) for r in reserve_ratio_trajectory],
        "adv_balance_trajectory": [float(b) for b in adv_balance_trajectory],
    }

    _logger.info(
        f"Done: extraction={'PROFITABLE' if net_extraction > 0 else 'UNPROFITABLE'} "
        f"(net={net_extraction:.1f}), redeemed={adv_total_redeemed:.0f}, "
        f"opp_cost={total_opportunity_cost:.0f}, reserve={final_reserve:.3f}"
    )

    return scenario_result


def run_experiment(
    config_path: str,
    output_dir: str,
    num_members: int = 500,
    num_cycles: int = 1000,
    seed: int = 42,
    max_workers: int = 0,
):
    """Run Experiment 4: Bridge Resistance Under Adversarial Extraction."""
    import multiprocessing
    if max_workers <= 0:
        max_workers = min(len(ADV_STRATEGIES) * len(ADV_FRACTIONS), multiprocessing.cpu_count())

    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: Bridge Resistance Under Adversarial Extraction")
    logger.info(f"  Members: {num_members}, Cycles: {num_cycles}, Seed: {seed}")
    logger.info(f"  Strategies: {ADV_STRATEGIES}")
    logger.info(f"  Adversarial fractions: {ADV_FRACTIONS}")
    logger.info(f"  Workers: {max_workers}")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Build job list: one per (strategy x fraction)
    jobs = []
    for adv_strategy in ADV_STRATEGIES:
        for adv_fraction in ADV_FRACTIONS:
            jobs.append((config, adv_strategy, adv_fraction, num_members, num_cycles, seed, output_dir))

    logger.info(f"  Launching {len(jobs)} scenarios...")

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single, job): (job[1], job[2]) for job in jobs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                logger.info(
                    f"  {key[0]} @ {key[1]:.0%}: "
                    f"net_extraction={result['net_extraction']:.1f} "
                    f"({'PROFITABLE' if result['extraction_profitable'] else 'UNPROFITABLE'})"
                )
            except Exception as e:
                logger.error(f"  FAILED {key}: {e}")
                import traceback
                traceback.print_exc()

    # Organize results
    results_by_strategy = {}
    for r in all_results:
        key = r["adv_strategy"]
        results_by_strategy.setdefault(key, []).append(r)
    for key in results_by_strategy:
        results_by_strategy[key].sort(key=lambda x: x["adv_fraction"])

    # Generate figures
    _plot_extraction_profitability(results_by_strategy, output_dir)
    _plot_treasury_reserve(results_by_strategy, output_dir)
    _plot_economy_health(results_by_strategy, output_dir)
    _plot_adversary_balances(results_by_strategy, output_dir)

    # Validation
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)

    any_profitable = False
    for strat in ADV_STRATEGIES:
        logger.info(f"\n  Strategy: {strat}")
        for r in results_by_strategy.get(strat, []):
            status = "PROFITABLE" if r["extraction_profitable"] else "UNPROFITABLE"
            if r["extraction_profitable"]:
                any_profitable = True
            logger.info(
                f"    {r['adv_fraction']:.0%} adversaries: "
                f"net={r['net_extraction']:.1f} [{status}], "
                f"redeemed={r['adv_total_redeemed']:.0f}, "
                f"match={r['avg_match_rate_last100']:.3f}, "
                f"reserve={r['final_reserve_ratio']:.3f}"
            )

    overall_pass = not any_profitable
    logger.info(f"\n  OVERALL: {'PASS — no profitable extraction' if overall_pass else 'FAIL — some strategies profitable'}")

    # Save summary
    summary = {
        "scenarios": [{k: v for k, v in r.items()
                       if k not in ("gini_trajectory", "match_rate_trajectory",
                                    "reserve_ratio_trajectory", "adv_balance_trajectory")}
                      for r in all_results],
        "any_profitable_extraction": any_profitable,
        "overall_pass": overall_pass,
    }
    with open(os.path.join(output_dir, "exp4_validation.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save full trajectories
    with open(os.path.join(output_dir, "exp4_trajectories.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results, summary


def _plot_extraction_profitability(results_by_strategy: dict, output_dir: str):
    """Bar chart: net extraction by strategy and adversarial fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    STRAT_COLORS = {
        "simple_arbitrage": "#1f77b4",
        "hoard_and_dump": "#ff7f0e",
        "sybil_splitting": "#d62728",
    }
    STRAT_LABELS = {
        "simple_arbitrage": "Simple Arbitrage",
        "hoard_and_dump": "Hoard & Dump",
        "sybil_splitting": "Sybil Splitting",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ADV_FRACTIONS))
    width = 0.25

    for i, strat in enumerate(ADV_STRATEGIES):
        results = results_by_strategy.get(strat, [])
        if not results:
            continue
        vals = [r["net_extraction"] for r in results]
        bars = ax.bar(x + i * width, vals, width,
                      label=STRAT_LABELS.get(strat, strat),
                      color=STRAT_COLORS.get(strat, "gray"),
                      edgecolor="black", linewidth=0.5)

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="Break-even")
    ax.set_xlabel("Adversarial Population Fraction")
    ax.set_ylabel("Net Extraction (fiat)")
    ax.set_title("Adversarial Extraction Profitability by Strategy")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{f:.0%}" for f in ADV_FRACTIONS])
    ax.legend()
    fig.savefig(os.path.join(output_dir, "exp4_extraction_profitability.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_treasury_reserve(results_by_strategy: dict, output_dir: str):
    """Line chart: treasury reserve ratio over time for each scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, axes = plt.subplots(1, len(ADV_STRATEGIES), figsize=(5 * len(ADV_STRATEGIES), 4),
                             sharey=True)
    if len(ADV_STRATEGIES) == 1:
        axes = [axes]

    frac_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, strat in enumerate(ADV_STRATEGIES):
        ax = axes[idx]
        results = results_by_strategy.get(strat, [])
        for j, r in enumerate(results):
            traj = r.get("reserve_ratio_trajectory", [])
            if not traj:
                continue
            # Smooth
            window = 20
            smoothed = []
            for i in range(len(traj)):
                start = max(0, i - window + 1)
                smoothed.append(float(np.mean(traj[start:i + 1])))
            ax.plot(smoothed, label=f"{r['adv_fraction']:.0%}",
                    color=frac_colors[j % len(frac_colors)], linewidth=1.2)

        ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.7, linewidth=0.8)
        ax.axhline(y=0.05, color="darkred", linestyle=":", alpha=0.7, linewidth=0.8)
        ax.set_xlabel("Cycle")
        ax.set_title(strat.replace("_", " ").title())
        if idx == 0:
            ax.set_ylabel("Reserve Ratio")
        ax.legend(title="Adv %", fontsize=8)

    fig.suptitle("Treasury Reserve Ratio Under Adversarial Extraction", fontsize=13, y=1.02)
    fig.savefig(os.path.join(output_dir, "exp4_treasury_reserve.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_economy_health(results_by_strategy: dict, output_dir: str):
    """2x1 panel: Gini and Match Rate vs adversarial fraction for each strategy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    STRAT_COLORS = {
        "simple_arbitrage": "#1f77b4",
        "hoard_and_dump": "#ff7f0e",
        "sybil_splitting": "#d62728",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for strat in ADV_STRATEGIES:
        results = results_by_strategy.get(strat, [])
        if not results:
            continue
        fracs = [r["adv_fraction"] for r in results]
        ginis = [r["avg_gini_last100"] for r in results]
        matches = [r["avg_match_rate_last100"] for r in results]
        color = STRAT_COLORS.get(strat, "gray")
        label = strat.replace("_", " ").title()

        ax1.plot(fracs, ginis, "o-", color=color, label=label, linewidth=1.5)
        ax2.plot(fracs, matches, "o-", color=color, label=label, linewidth=1.5)

    ax1.set_xlabel("Adversarial Fraction")
    ax1.set_ylabel("Gini Coefficient")
    ax1.set_title("Credit Inequality vs Adversarial Population")
    ax1.legend(fontsize=9)

    ax2.axhline(y=0.7, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Adversarial Fraction")
    ax2.set_ylabel("Match Rate")
    ax2.set_title("Match Rate vs Adversarial Population")
    ax2.legend(fontsize=9)

    fig.savefig(os.path.join(output_dir, "exp4_economy_health.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_adversary_balances(results_by_strategy: dict, output_dir: str):
    """Line chart: aggregate adversary balance over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, axes = plt.subplots(1, len(ADV_STRATEGIES), figsize=(5 * len(ADV_STRATEGIES), 4),
                             sharey=True)
    if len(ADV_STRATEGIES) == 1:
        axes = [axes]

    frac_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, strat in enumerate(ADV_STRATEGIES):
        ax = axes[idx]
        results = results_by_strategy.get(strat, [])
        for j, r in enumerate(results):
            traj = r.get("adv_balance_trajectory", [])
            if not traj:
                continue
            window = 20
            smoothed = []
            for i in range(len(traj)):
                start = max(0, i - window + 1)
                smoothed.append(float(np.mean(traj[start:i + 1])))
            ax.plot(smoothed, label=f"{r['adv_fraction']:.0%}",
                    color=frac_colors[j % len(frac_colors)], linewidth=1.2)

        ax.set_xlabel("Cycle")
        ax.set_title(strat.replace("_", " ").title())
        if idx == 0:
            ax.set_ylabel("Aggregate Adversary Credits")
        ax.legend(title="Adv %", fontsize=8)

    fig.suptitle("Adversary Credit Balances Over Time", fontsize=13, y=1.02)
    fig.savefig(os.path.join(output_dir, "exp4_adversary_balances.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Bridge Resistance")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "community_economy.json"))
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exp4"))
    parser.add_argument("--members", type=int, default=500)
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    run_experiment(args.config, args.output, args.members, args.cycles, args.seed, args.workers)


if __name__ == "__main__":
    main()
