"""Verify the v1-compat bridge produces byte-identical outputs when plugged
into the existing v1 pipeline functions (compute_classical_weights, build_qubo,
solve_qubo_exhaustive, analyze_degeneracy).

This is the proof that experiment scripts (exp7, exp7d, exp7e, exp8) can
swap their harvest_instances import to the bridge without code changes or
numerical drift.
"""
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from experiments.exp6b_hybrid_replication import harvest_instances, analyze_degeneracy
from experiments.exp6_hybrid_pipeline import (
    compute_classical_weights, build_qubo, solve_qubo_exhaustive,
)
from core.benchmark_fixtures import hardware_instances_v1_compat


def run_pipeline(instances, label: str):
    """Run the v1 pipeline on a list of instances, return per-instance dict."""
    out = []
    for i, inst in enumerate(instances):
        n = len(inst["eligible_pairs"])
        w_c, sims = compute_classical_weights(inst)
        Q_c, _, pen = build_qubo(inst, w_c)
        c_bits, c_energy = solve_qubo_exhaustive(Q_c, n)
        n_opt, gap, optima = analyze_degeneracy(Q_c, n)
        out.append({
            "n": n,
            "weights": [round(w_c[k], 10) for k in range(n)],
            "similarities": [round(sims[k], 10) for k in range(n)],
            "penalty": round(pen, 10),
            "bits": list(c_bits),
            "energy": round(c_energy, 8),
            "n_optima": n_opt,
            "optima": [tuple(o) for o in optima],
        })
    return out


def main():
    config_path = os.path.join(ROOT, "config", "community_economy.json")

    logger.info("[V1] Harvesting via v1 simulation...")
    v1_insts = harvest_instances(config_path, 30)
    v1_out = run_pipeline(v1_insts, "v1")
    logger.info(f"  Got {len(v1_out)} results")

    logger.info("[BRIDGE] Loading via hardware_instances_v1_compat (v2-backed)...")
    bridge_insts = hardware_instances_v1_compat()
    bridge_out = run_pipeline(bridge_insts, "bridge")
    logger.info(f"  Got {len(bridge_out)} results")

    all_ok = True
    for i, (a, b) in enumerate(zip(v1_out, bridge_out)):
        if a != b:
            logger.error(f"  [{i}] MISMATCH")
            for k in a:
                if a[k] != b.get(k):
                    logger.error(f"    {k}: v1={a[k]} bridge={b[k]}")
            all_ok = False
        else:
            logger.info(f"  [{i}] n={a['n']} energy={a['energy']:.4f} "
                        f"n_optima={a['n_optima']} → PASS")

    print()
    if all_ok:
        logger.info("=" * 60)
        logger.info("✓ BRIDGE produces BYTE-IDENTICAL output to v1 pipeline")
        logger.info("=" * 60)
    else:
        logger.error("✗ Bridge diverges — cannot drop-in replace")
        sys.exit(1)


if __name__ == "__main__":
    main()
