"""
Experiment 3: Topological Performativity (MOST IMPORTANT)

Run the SAME economy three times with different matching strategies:
  - Run A: Random matching
  - Run B: Greedy matching
  - Run C: QUBO matching

Same random seed for member generation, different matching only.
After 1000 cycles, compare social graphs.

Confirms if:
- QUBO produces significantly more inter-cluster bridges (p < 0.01, Mann-Whitney U)
- PI > 2x random baseline

Output: 3 network visualizations, bar chart of bridge counts, PI comparison

Parallelized across runs using multiprocessing (one process per strategy+seed).
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
from core.simulation_engine import SimulationEngine, SimulationResults
from core.topology_analyzer import TopologyAnalyzer
from analysis.metrics import mann_whitney_test
from analysis.visualization import (
    plot_network_comparison,
    plot_bridge_comparison,
    plot_pi_comparison,
    plot_bridge_evolution,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


STRATEGIES = [
    ("random", MatchingStrategy.RANDOM),
    ("greedy", MatchingStrategy.GREEDY),
    ("qubo", MatchingStrategy.QUBO),
]


def _run_single_process(args: Tuple) -> Tuple[str, int, str]:
    """Worker function for parallel execution. Returns (strategy, seed, result_path)."""
    config, strategy_name, strategy_value, num_members, num_cycles, seed, output_dir = args

    # Re-import in subprocess
    import logging as _log
    _log.basicConfig(level=_log.INFO, format=f"%(asctime)s [{strategy_name}:s{seed}] %(message)s")
    _logger = _log.getLogger(__name__)

    from core.matching_service import MatchingStrategy as MS
    from core.simulation_engine import SimulationEngine as SE
    from core.topology_analyzer import TopologyAnalyzer as TA

    strategy = MS(strategy_value)

    def progress(cycle, total):
        if cycle % 200 == 0:
            _logger.info(f"Cycle {cycle}/{total}")

    engine = SE(
        config=config,
        matching_strategy=strategy,
        num_members=num_members,
        num_cycles=num_cycles,
        seed=seed,
        topology_sample_interval=50,
        progress_callback=progress,
    )
    results = engine.run()

    # Compute bridge count
    bridges = 0
    if results.social_graph and results.social_graph.number_of_edges() > 0:
        analyzer = TA(results.social_graph)
        bridges = analyzer.inter_cluster_bridges()

    _logger.info(f"Done: {bridges} bridges, {results.social_graph.number_of_edges()} edges, {results.wall_time:.0f}s")

    # Save result to disk (avoid passing large objects between processes)
    run_idx = seed - 42  # convention: seed = base_seed + run_idx
    result_path = os.path.join(output_dir, f"exp3_{strategy_name}_run{run_idx}.json")
    results.save(result_path)

    # Save graph separately for first run's visualization
    graph_path = os.path.join(output_dir, f"_graph_{strategy_name}_run{run_idx}.json")
    import networkx as nx
    if results.social_graph:
        graph_data = nx.node_link_data(results.social_graph)
        with open(graph_path, "w") as f:
            json.dump(graph_data, f)

    # Return summary data
    return (strategy_name, seed, bridges, results.wall_time,
            results.final_topology.__dict__ if results.final_topology else {},
            len(results.novel_connections), len(results.organic_transactions),
            result_path, graph_path)


def run_experiment(
    config_path: str,
    output_dir: str,
    num_members: int = 500,
    num_cycles: int = 1000,
    seed: int = 42,
    num_runs: int = 7,
    max_workers: int = 0,
):
    """Run Experiment 3: Topological Performativity (parallelized)."""
    if max_workers <= 0:
        # Use up to num_runs * 3 workers, capped by CPU count
        import multiprocessing
        max_workers = min(num_runs * 3, multiprocessing.cpu_count())

    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Topological Performativity (parallel)")
    logger.info(f"  Members: {num_members}, Cycles: {num_cycles}, Seed: {seed}")
    logger.info(f"  Strategies: {[s[0] for s in STRATEGIES]}")
    logger.info(f"  Runs per strategy: {num_runs}")
    logger.info(f"  Workers: {max_workers}")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Build job list: one job per (strategy, seed)
    jobs = []
    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        for strategy_name, strategy in STRATEGIES:
            jobs.append((config, strategy_name, strategy.value, num_members, num_cycles, run_seed, output_dir))

    logger.info(f"  Launching {len(jobs)} simulation jobs...")

    # Execute in parallel
    completed_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_process, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
                completed_results.append(result)
                strategy_name, run_seed, bridges, wall_time = result[:4]
                logger.info(f"  Completed {strategy_name} seed={run_seed}: {bridges} bridges ({wall_time:.0f}s)")
            except Exception as e:
                job = futures[future]
                logger.error(f"  FAILED {job[1]} seed={job[4]}: {e}")

    # Collect results by strategy
    bridge_samples: Dict[str, List[int]] = {"random": [], "greedy": [], "qubo": []}
    topo_data: Dict[str, dict] = {}
    wall_times: Dict[str, List[float]] = {"random": [], "greedy": [], "qubo": []}

    for res in sorted(completed_results, key=lambda r: (r[0], r[1])):
        strategy_name, run_seed, bridges, wall_time, topo, n_novel, n_organic, result_path, graph_path = res
        bridge_samples[strategy_name].append(bridges)
        wall_times[strategy_name].append(wall_time)
        if strategy_name not in topo_data and topo:
            topo_data[strategy_name] = topo

    # Log bridge summary
    logger.info("\n" + "=" * 60)
    logger.info("BRIDGE COUNTS")
    for s in ["random", "greedy", "qubo"]:
        vals = bridge_samples[s]
        logger.info(f"  {s:8s}: {vals}  mean={np.mean(vals):.0f}  std={np.std(vals):.0f}")

    # Load first run graphs for visualization
    graphs_for_viz = {}
    for strategy_name, _ in STRATEGIES:
        graph_path = os.path.join(output_dir, f"_graph_{strategy_name}_run0.json")
        if os.path.exists(graph_path):
            import networkx as nx
            with open(graph_path) as f:
                graphs_for_viz[strategy_name] = nx.node_link_graph(json.load(f))

    if graphs_for_viz:
        plot_network_comparison(
            graphs_for_viz,
            os.path.join(output_dir, "exp3_networks.png"),
        )

    # Bridge bar chart
    plot_bridge_comparison(
        bridge_samples,
        os.path.join(output_dir, "exp3_bridges.png"),
    )

    # Performativity Index (from first run's data)
    pi_data: Dict[str, Dict[str, float]] = {}
    for strategy_name, _ in STRATEGIES:
        graph_path = os.path.join(output_dir, f"_graph_{strategy_name}_run0.json")
        result_path = os.path.join(output_dir, f"exp3_{strategy_name}_run0.json")
        if os.path.exists(graph_path) and os.path.exists(result_path):
            import networkx as nx
            with open(graph_path) as f:
                G = nx.node_link_graph(json.load(f))
            with open(result_path) as f:
                rdata = json.load(f)
            analyzer = TopologyAnalyzer(G)
            # Reconstruct novel/organic from cycle records (approximate via graph)
            n_edges = G.number_of_edges()
            # Use topology-based PI approximation
            bridges = analyzer.inter_cluster_bridges()
            total_possible = max(1, G.number_of_nodes() * (G.number_of_nodes() - 1) // 2)
            pi_approx = bridges / max(1, n_edges)
            random_baseline = n_edges / max(1, total_possible)
            pi_data[strategy_name] = {
                "pi": pi_approx,
                "bridges": bridges,
                "edges": n_edges,
                "random_baseline": random_baseline,
            }
            logger.info(f"  [{strategy_name}] Bridge ratio (bridges/edges): {pi_approx:.4f}")

    if pi_data:
        plot_pi_comparison(
            pi_data,
            os.path.join(output_dir, "exp3_pi.png"),
        )

    # Statistical tests
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)

    qubo_bridges = bridge_samples.get("qubo", [])
    random_bridges = bridge_samples.get("random", [])
    greedy_bridges = bridge_samples.get("greedy", [])

    if qubo_bridges and random_bridges:
        mw_qr = mann_whitney_test(qubo_bridges, random_bridges)
        logger.info(f"  Mann-Whitney (QUBO > Random): U={mw_qr['statistic']:.1f}, p={mw_qr['p_value']:.6f}")
        logger.info(f"    Significant (p < 0.01): {mw_qr['significant']}")
    else:
        mw_qr = {"statistic": 0, "p_value": 1.0, "significant": False}

    if qubo_bridges and greedy_bridges:
        mw_qg = mann_whitney_test(qubo_bridges, greedy_bridges)
        logger.info(f"  Mann-Whitney (QUBO > Greedy): U={mw_qg['statistic']:.1f}, p={mw_qg['p_value']:.6f}")
        logger.info(f"    Significant (p < 0.01): {mw_qg['significant']}")
    else:
        mw_qg = {"statistic": 0, "p_value": 1.0, "significant": False}

    # PI ratio
    qubo_pi = pi_data.get("qubo", {}).get("pi", 0)
    random_pi = pi_data.get("random", {}).get("pi", 0.001)
    pi_ratio = qubo_pi / max(0.001, random_pi)
    pi_pass = pi_ratio > 2.0
    logger.info(f"  PI bridge-ratio (QUBO / Random): {pi_ratio:.2f} [{'PASS' if pi_pass else 'FAIL'}]")

    # Non-overlap test
    if qubo_bridges and greedy_bridges:
        non_overlap = min(qubo_bridges) > max(greedy_bridges)
        logger.info(f"  Non-overlapping (QUBO min > Greedy max): {min(qubo_bridges)} > {max(greedy_bridges)} = {non_overlap}")

    bridge_pass = bool(mw_qg.get("significant", False))
    overall = bridge_pass and pi_pass
    logger.info(f"\n  OVERALL: {'ALL CRITERIA MET' if overall else 'SOME CRITERIA NOT MET'}")

    # Summary
    summary = {
        "num_runs": num_runs,
        "num_members": num_members,
        "num_cycles": num_cycles,
        "bridge_counts": {k: [int(v) for v in vals] for k, vals in bridge_samples.items()},
        "bridge_means": {k: float(np.mean(v)) if v else 0 for k, v in bridge_samples.items()},
        "bridge_stds": {k: float(np.std(v)) if v else 0 for k, v in bridge_samples.items()},
        "mann_whitney_qubo_vs_random": mw_qr,
        "mann_whitney_qubo_vs_greedy": mw_qg,
        "performativity_index": pi_data,
        "pi_ratio_qubo_random": float(pi_ratio),
        "pi_pass": bool(pi_pass),
        "bridge_significant": bridge_pass,
        "overall_pass": bool(overall),
        "wall_times": {k: [round(t, 1) for t in v] for k, v in wall_times.items()},
    }

    if topo_data:
        summary["topology_comparison"] = {}
        for s, td in topo_data.items():
            summary["topology_comparison"][s] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in td.items()
            }

    with open(os.path.join(output_dir, "exp3_validation.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Cleanup temp graph files
    for strategy_name, _ in STRATEGIES:
        for run_idx in range(num_runs):
            gp = os.path.join(output_dir, f"_graph_{strategy_name}_run{run_idx}.json")
            if os.path.exists(gp):
                os.remove(gp)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Topological Performativity")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "community_economy.json",
    ))
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exp3",
    ))
    parser.add_argument("--members", type=int, default=500)
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=7, help="Runs per strategy for statistical power")
    parser.add_argument("--workers", type=int, default=0, help="Max parallel workers (0=auto)")
    args = parser.parse_args()

    run_experiment(args.config, args.output, args.members, args.cycles, args.seed, args.runs, args.workers)


if __name__ == "__main__":
    main()
