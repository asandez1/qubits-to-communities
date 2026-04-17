"""
Experiment 1: Credit Circulation Dynamics

500 members, 6 categories, 1000 cycles, QUBO matching.
Measures: credit velocity, Gini coefficient, match rate, per-category price evolution.

Confirms if:
- Bonding curve prices converge within 100 cycles
- Gini < 0.4
- Match rate > 70%

Output: 4 figures (price evolution, Gini, velocity, match rate)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.credit_economy import CreditEconomyService
from core.matching_service import MatchingStrategy
from core.simulation_engine import SimulationEngine
from analysis.metrics import price_convergence_cycle
from analysis.visualization import (
    plot_price_evolution,
    plot_gini_over_time,
    plot_velocity_over_time,
    plot_match_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_experiment(
    config_path: str,
    output_dir: str,
    num_members: int = 500,
    num_cycles: int = 1000,
    seed: int = 42,
):
    """Run Experiment 1: Credit Circulation Dynamics."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Credit Circulation Dynamics")
    logger.info(f"  Members: {num_members}, Cycles: {num_cycles}, Seed: {seed}")
    logger.info("=" * 60)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Run simulation with QUBO matching
    def progress(cycle, total):
        logger.info(f"  Cycle {cycle}/{total}")

    engine = SimulationEngine(
        config=config,
        matching_strategy=MatchingStrategy.QUBO,
        num_members=num_members,
        num_cycles=num_cycles,
        seed=seed,
        topology_sample_interval=100,
        progress_callback=progress,
    )

    results = engine.run()

    # Extract time series
    price_histories = {cat: [] for cat in config["categories"]}
    gini_values = []
    velocity_values = []
    match_rates = []

    for record in results.cycle_records:
        em = record.economy
        for cat in config["categories"]:
            price_histories[cat].append(em.prices.get(cat, 0))
        gini_values.append(em.gini)
        velocity_values.append(em.velocity)
        match_rates.append(em.match_rate)

    # Save results JSON
    os.makedirs(output_dir, exist_ok=True)
    results.save(os.path.join(output_dir, "exp1_results.json"))

    # Generate figures
    plot_price_evolution(
        price_histories,
        os.path.join(output_dir, "exp1_price_evolution.png"),
    )
    plot_gini_over_time(
        gini_values,
        os.path.join(output_dir, "exp1_gini.png"),
    )
    plot_velocity_over_time(
        velocity_values,
        os.path.join(output_dir, "exp1_velocity.png"),
    )
    plot_match_rate(
        match_rates,
        os.path.join(output_dir, "exp1_match_rate.png"),
    )

    # Validation criteria
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)

    # 1. Price convergence within 100 cycles
    convergence_results = {}
    all_converged = True
    for cat, prices in price_histories.items():
        conv_cycle = price_convergence_cycle(prices, tolerance=0.15)
        convergence_results[cat] = conv_cycle
        status = "PASS" if conv_cycle <= 100 else "FAIL"
        if conv_cycle > 100:
            all_converged = False
        logger.info(f"  Price convergence [{cat}]: cycle {conv_cycle} [{status}]")

    # 2. Final Gini < 0.4
    final_gini = gini_values[-1] if gini_values else 1.0
    avg_gini = sum(gini_values[-100:]) / max(1, len(gini_values[-100:]))
    gini_pass = avg_gini < 0.4
    logger.info(f"  Gini (last 100 avg): {avg_gini:.3f} [{'PASS' if gini_pass else 'FAIL'}]")

    # 3. Match rate > 70%
    avg_match = sum(match_rates[-100:]) / max(1, len(match_rates[-100:]))
    match_pass = avg_match > 0.7
    logger.info(f"  Match rate (last 100 avg): {avg_match:.3f} [{'PASS' if match_pass else 'FAIL'}]")

    # Summary
    all_pass = all_converged and gini_pass and match_pass
    logger.info(f"\n  OVERALL: {'ALL CRITERIA MET' if all_pass else 'SOME CRITERIA NOT MET'}")
    logger.info(f"  Wall time: {results.wall_time:.1f}s")

    # Save validation summary
    summary = {
        "convergence_cycles": convergence_results,
        "all_converged_by_100": bool(all_converged),
        "final_gini": float(final_gini),
        "avg_gini_last100": float(avg_gini),
        "gini_pass": bool(gini_pass),
        "avg_match_rate_last100": float(avg_match),
        "match_rate_pass": bool(match_pass),
        "overall_pass": bool(all_pass),
        "wall_time": float(results.wall_time),
    }
    with open(os.path.join(output_dir, "exp1_validation.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Credit Circulation Dynamics")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "community_economy.json",
    ))
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exp1",
    ))
    parser.add_argument("--members", type=int, default=500)
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(args.config, args.output, args.members, args.cycles, args.seed)


if __name__ == "__main__":
    main()
