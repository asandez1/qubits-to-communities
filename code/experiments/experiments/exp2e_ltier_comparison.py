#!/usr/bin/env python3
"""
Experiment 2E: L-tier Scaling of Pipeline Comparison (Phase 2E).

Re-runs exp9_pipeline_comparison on the L-tier fixture (50 members × 25 tasks,
428 QUBO vars) to check whether the kernel advantage persists at non-trivial
scale or collapses as the optimizer becomes less decisive.

This is a thin wrapper around exp9 — same code, larger fixture.

Usage:
    python exp2e_ltier_comparison.py --days 30
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from experiments.exp9_pipeline_comparison import main as exp9_main

if __name__ == "__main__":
    # Inject L-tier default; let exp9_main parse everything else.
    if "--tier" not in sys.argv:
        sys.argv += ["--tier", "l"]
    if "--output" not in sys.argv:
        sys.argv += ["--output", os.path.join(ROOT, "results", "exp2e_ltier")]
    if "--days" not in sys.argv:
        sys.argv += ["--days", "30"]
    exp9_main()
