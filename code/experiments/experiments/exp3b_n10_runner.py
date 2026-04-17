"""Runner for Exp 3 at n=10 seeds (Phase 2C).

Dedicated output directory so existing n=7 results are preserved.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from experiments.exp3_performativity import run_experiment

if __name__ == "__main__":
    run_experiment(
        config_path=os.path.join(ROOT, "config", "community_economy.json"),
        output_dir=os.path.join(ROOT, "results", "exp3_n10"),
        num_members=500,
        num_cycles=1000,
        seed=42,
        num_runs=10,
        max_workers=30,
    )
