"""
Experiment 1b: Credit Redistribution — Fixing the Gini Problem

Baseline Gini = 0.70 due to bonding curve structural inequality.
Tests three redistribution mechanisms against the baseline:
  - Variant A: Progressive demurrage (rate scales with balance size)
  - Variant B: Progressive protocol fee (higher fee on high-price categories,
               surplus redistributed to below-median members)
  - Variant C: Combined (progressive demurrage + progressive fee)

500 members, 1000 cycles, QUBO matching, same seed for comparability.

Confirms if any variant achieves Gini < 0.4 while maintaining match rate > 70%.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.matching_service import MatchingStrategy
from core.simulation_engine import SimulationEngine
from analysis.metrics import price_convergence_cycle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VARIANTS = [
    ("baseline", "flat", "flat"),
    ("progressive_demurrage", "progressive", "flat"),
    ("progressive_fee", "flat", "progressive"),
    ("combined", "progressive", "progressive"),
]


def _run_variant(args: Tuple) -> Tuple[str, dict]:
    """Run a single variant. Designed for ProcessPoolExecutor."""
    config, variant_name, demurrage_mode, fee_mode, num_members, num_cycles, seed = args

    import logging as _log
    _log.basicConfig(level=_log.INFO, format=f"%(asctime)s [{variant_name}] %(message)s")
    _logger = _log.getLogger(__name__)

    from core.matching_service import MatchingStrategy as MS
    from core.simulation_engine import SimulationEngine as SE

    # Override economy config with redistribution modes
    config = dict(config)  # shallow copy
    config["_demurrage_mode"] = demurrage_mode
    config["_fee_mode"] = fee_mode

    def progress(cycle, total):
        if cycle % 200 == 0:
            _logger.info(f"Cycle {cycle}/{total}")

    engine = SE(
        config=config,
        matching_strategy=MS.QUBO,
        num_members=num_members,
        num_cycles=num_cycles,
        seed=seed,
        topology_sample_interval=200,
        progress_callback=progress,
    )

    # Apply redistribution modes to the economy service
    engine.economy.demurrage_mode = demurrage_mode
    engine.economy.fee_mode = fee_mode

    results = engine.run()

    # Extract time series
    gini_values = []
    velocity_values = []
    match_rates = []
    price_histories = {cat: [] for cat in config["categories"]}

    for record in results.cycle_records:
        em = record.economy
        gini_values.append(em.gini)
        velocity_values.append(em.velocity)
        match_rates.append(em.match_rate)
        for cat in config["categories"]:
            price_histories[cat].append(em.prices.get(cat, 0))

    # Compute summary stats
    final_gini = gini_values[-1] if gini_values else 1.0
    avg_gini_last100 = float(np.mean(gini_values[-100:])) if len(gini_values) >= 100 else final_gini
    avg_match_last100 = float(np.mean(match_rates[-100:])) if len(match_rates) >= 100 else 0.0
    avg_velocity_last100 = float(np.mean(velocity_values[-100:])) if len(velocity_values) >= 100 else 0.0

    # Price convergence
    convergence_results = {}
    for cat, prices in price_histories.items():
        convergence_results[cat] = price_convergence_cycle(prices, tolerance=0.15)

    summary = {
        "variant": variant_name,
        "demurrage_mode": demurrage_mode,
        "fee_mode": fee_mode,
        "final_gini": float(final_gini),
        "avg_gini_last100": float(avg_gini_last100),
        "avg_match_rate_last100": float(avg_match_last100),
        "avg_velocity_last100": float(avg_velocity_last100),
        "gini_pass": avg_gini_last100 < 0.4,
        "match_rate_pass": avg_match_last100 > 0.7,
        "convergence_cycles": convergence_results,
        "wall_time": float(results.wall_time),
        "gini_trajectory": [float(g) for g in gini_values],
        "match_rate_trajectory": [float(m) for m in match_rates],
        "velocity_trajectory": [float(v) for v in velocity_values],
        "price_histories": {cat: [float(p) for p in ph] for cat, ph in price_histories.items()},
    }

    _logger.info(
        f"Done: Gini={avg_gini_last100:.3f}, Match={avg_match_last100:.3f}, "
        f"Velocity={avg_velocity_last100:.4f}, Time={results.wall_time:.0f}s"
    )
    return variant_name, summary


def run_experiment(
    config_path: str,
    output_dir: str,
    num_members: int = 500,
    num_cycles: int = 1000,
    seed: int = 42,
    max_workers: int = 0,
):
    """Run Experiment 1b: Credit Redistribution."""
    import multiprocessing
    if max_workers <= 0:
        max_workers = min(len(VARIANTS), multiprocessing.cpu_count())

    logger.info("=" * 60)
    logger.info("EXPERIMENT 1b: Credit Redistribution (Gini Fix)")
    logger.info(f"  Members: {num_members}, Cycles: {num_cycles}, Seed: {seed}")
    logger.info(f"  Variants: {[v[0] for v in VARIANTS]}")
    logger.info(f"  Workers: {max_workers}")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Build job list
    jobs = []
    for variant_name, demurrage_mode, fee_mode in VARIANTS:
        jobs.append((config, variant_name, demurrage_mode, fee_mode, num_members, num_cycles, seed))

    # Execute in parallel
    all_results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_variant, job): job[1] for job in jobs}
        for future in as_completed(futures):
            try:
                variant_name, summary = future.result()
                all_results[variant_name] = summary
                logger.info(
                    f"  {variant_name}: Gini={summary['avg_gini_last100']:.3f}, "
                    f"Match={summary['avg_match_rate_last100']:.3f}"
                )
            except Exception as e:
                variant_name = futures[future]
                logger.error(f"  FAILED {variant_name}: {e}")
                import traceback
                traceback.print_exc()

    # Generate comparison figures
    _plot_gini_comparison(all_results, output_dir)
    _plot_match_rate_comparison(all_results, output_dir)
    _plot_velocity_comparison(all_results, output_dir)
    _plot_price_comparison(all_results, config, output_dir)

    # Validation summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)

    for vname in ["baseline", "progressive_demurrage", "progressive_fee", "combined"]:
        if vname not in all_results:
            continue
        r = all_results[vname]
        gini_status = "PASS" if r["gini_pass"] else "FAIL"
        match_status = "PASS" if r["match_rate_pass"] else "FAIL"
        logger.info(
            f"  {vname:25s}: Gini={r['avg_gini_last100']:.3f} [{gini_status}], "
            f"Match={r['avg_match_rate_last100']:.3f} [{match_status}], "
            f"Velocity={r['avg_velocity_last100']:.4f}"
        )

    # Find the best variant that passes both criteria
    best = None
    for vname in ["combined", "progressive_demurrage", "progressive_fee"]:
        if vname in all_results and all_results[vname]["gini_pass"] and all_results[vname]["match_rate_pass"]:
            best = vname
            break

    if best:
        logger.info(f"\n  WINNER: {best} (Gini={all_results[best]['avg_gini_last100']:.3f})")
    else:
        # Find lowest Gini that maintains match rate
        candidates = [
            (vname, r) for vname, r in all_results.items()
            if r["match_rate_pass"] and vname != "baseline"
        ]
        if candidates:
            best = min(candidates, key=lambda x: x[1]["avg_gini_last100"])[0]
            logger.info(f"\n  BEST (match rate preserved): {best} (Gini={all_results[best]['avg_gini_last100']:.3f})")
        else:
            logger.info("\n  No variant achieves both targets simultaneously.")

    # Save full results (excluding large trajectories for the summary)
    summary_data = {}
    for vname, r in all_results.items():
        summary_data[vname] = {k: v for k, v in r.items()
                               if k not in ("gini_trajectory", "match_rate_trajectory",
                                            "velocity_trajectory", "price_histories")}
    summary_data["best_variant"] = best

    with open(os.path.join(output_dir, "exp1b_validation.json"), "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    # Save full trajectories separately
    with open(os.path.join(output_dir, "exp1b_trajectories.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results, summary_data


def _plot_gini_comparison(results: dict, output_dir: str):
    """Plot Gini trajectories for all variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    VARIANT_COLORS = {
        "baseline": "#1f77b4",
        "progressive_demurrage": "#ff7f0e",
        "progressive_fee": "#2ca02c",
        "combined": "#d62728",
    }
    VARIANT_LABELS = {
        "baseline": "Baseline (flat)",
        "progressive_demurrage": "Progressive Demurrage",
        "progressive_fee": "Progressive Fee",
        "combined": "Combined",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for vname in ["baseline", "progressive_demurrage", "progressive_fee", "combined"]:
        if vname not in results:
            continue
        gini = results[vname]["gini_trajectory"]
        # Smooth with moving average
        window = 20
        smoothed = []
        for i in range(len(gini)):
            start = max(0, i - window + 1)
            smoothed.append(float(np.mean(gini[start:i + 1])))
        ax.plot(smoothed, label=VARIANT_LABELS.get(vname, vname),
                color=VARIANT_COLORS.get(vname, "gray"), linewidth=1.5)

    ax.axhline(y=0.4, color="red", linestyle="--", alpha=0.7, label="Target (0.4)")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Credit Inequality Under Redistribution Mechanisms")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(output_dir, "exp1b_gini_comparison.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_match_rate_comparison(results: dict, output_dir: str):
    """Plot match rate trajectories for all variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    VARIANT_COLORS = {
        "baseline": "#1f77b4",
        "progressive_demurrage": "#ff7f0e",
        "progressive_fee": "#2ca02c",
        "combined": "#d62728",
    }
    VARIANT_LABELS = {
        "baseline": "Baseline (flat)",
        "progressive_demurrage": "Progressive Demurrage",
        "progressive_fee": "Progressive Fee",
        "combined": "Combined",
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    for vname in ["baseline", "progressive_demurrage", "progressive_fee", "combined"]:
        if vname not in results:
            continue
        mr = results[vname]["match_rate_trajectory"]
        window = 20
        smoothed = []
        for i in range(len(mr)):
            start = max(0, i - window + 1)
            smoothed.append(float(np.mean(mr[start:i + 1])))
        ax.plot(smoothed, label=VARIANT_LABELS.get(vname, vname),
                color=VARIANT_COLORS.get(vname, "gray"), linewidth=1.5)

    ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.7, label="Target (70%)")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Match Rate")
    ax.set_title("Match Rate Under Redistribution Mechanisms")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(output_dir, "exp1b_match_rate_comparison.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_velocity_comparison(results: dict, output_dir: str):
    """Plot velocity trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    VARIANT_COLORS = {
        "baseline": "#1f77b4",
        "progressive_demurrage": "#ff7f0e",
        "progressive_fee": "#2ca02c",
        "combined": "#d62728",
    }
    VARIANT_LABELS = {
        "baseline": "Baseline (flat)",
        "progressive_demurrage": "Progressive Demurrage",
        "progressive_fee": "Progressive Fee",
        "combined": "Combined",
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    for vname in ["baseline", "progressive_demurrage", "progressive_fee", "combined"]:
        if vname not in results:
            continue
        vel = results[vname]["velocity_trajectory"]
        window = 20
        smoothed = []
        for i in range(len(vel)):
            start = max(0, i - window + 1)
            smoothed.append(float(np.mean(vel[start:i + 1])))
        ax.plot(smoothed, label=VARIANT_LABELS.get(vname, vname),
                color=VARIANT_COLORS.get(vname, "gray"), linewidth=1.5)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Velocity")
    ax.set_title("Credit Velocity Under Redistribution Mechanisms")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(output_dir, "exp1b_velocity_comparison.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_price_comparison(results: dict, config: dict, output_dir: str):
    """Plot price evolution for baseline vs best variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)
    colors = sns.color_palette("deep", 8)

    CATEGORY_COLORS = {
        "electrical": colors[0], "plumbing": colors[1], "tutoring": colors[2],
        "transport": colors[3], "cooking": colors[4], "childcare": colors[5],
    }

    # Plot baseline and combined side by side
    for vname in ["baseline", "combined"]:
        if vname not in results or "price_histories" not in results[vname]:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for cat in config["categories"]:
            prices = results[vname]["price_histories"].get(cat, [])
            if not prices:
                continue
            window = 20
            smoothed = []
            for i in range(len(prices)):
                start = max(0, i - window + 1)
                smoothed.append(float(np.mean(prices[start:i + 1])))
            ax.plot(smoothed, label=cat, color=CATEGORY_COLORS.get(cat, "gray"), linewidth=1.5)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Price (credits)")
        label = "Baseline" if vname == "baseline" else "Combined Redistribution"
        ax.set_title(f"Price Evolution — {label}")
        ax.legend(loc="upper right", framealpha=0.9)
        fig.savefig(os.path.join(output_dir, f"exp1b_prices_{vname}.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Experiment 1b: Credit Redistribution")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "community_economy.json"))
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exp1b"))
    parser.add_argument("--members", type=int, default=500)
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    run_experiment(args.config, args.output, args.members, args.cycles, args.seed, args.workers)


if __name__ == "__main__":
    main()
