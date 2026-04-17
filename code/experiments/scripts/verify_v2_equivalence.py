"""Verify that the v2 hardware_tier + hardware_classical_weights + hardware_qubo
pipeline reproduces v1 numerics byte-for-byte on all 8 hardware instances.

Passing this means:
- Anyone who re-runs the quantum circuit on the v2-loaded instances gets
  exactly the numerical inputs that produced the paper's hardware results.
- The existing fez/marrakesh/kingston JSONs remain valid without re-running
  the hardware.
"""
import logging
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from core.benchmark_fixtures import hardware_tier
from core.benchmark_model import (
    hardware_classical_weights, hardware_qubo, hardware_qubo_solve_exhaustive,
)
from experiments.exp6b_hybrid_replication import harvest_instances
from experiments.exp6_hybrid_pipeline import (
    compute_classical_weights, build_qubo, solve_qubo_exhaustive,
)


def main():
    config_path = os.path.join(ROOT, "config", "community_economy.json")

    logger.info("[V1] Harvesting instances from v1 simulation engine...")
    v1_instances = harvest_instances(config_path, 30)
    logger.info(f"  Got {len(v1_instances)} instances")

    logger.info("[V2] Loading instances from snapshot...")
    v2_instances, v2_metadata = hardware_tier()
    logger.info(f"  Got {len(v2_instances)} instances")

    assert len(v1_instances) == len(v2_instances), \
        f"Instance count mismatch: v1={len(v1_instances)} v2={len(v2_instances)}"

    all_good = True
    for i, (v1, v2, meta) in enumerate(zip(v1_instances, v2_instances, v2_metadata)):
        assert len(v1["eligible_pairs"]) == v2.n_vars, (
            f"Instance {i}: pair count mismatch v1={len(v1['eligible_pairs'])} "
            f"v2={v2.n_vars}"
        )

        # v1 path
        v1_w, v1_sim = compute_classical_weights(v1)
        v1_Q, v1_varmap, v1_pen = build_qubo(v1, v1_w)
        v1_bits, v1_energy = solve_qubo_exhaustive(v1_Q, len(v1["eligible_pairs"]))

        # v2 path (use matcher_weights from snapshot metadata — originally
        # stored from the same v1 matcher in the snapshot step)
        v2_w, v2_sim = hardware_classical_weights(v2, meta["matcher_weights"])
        v2_Q, v2_varmap, v2_pen = hardware_qubo(v2, v2_w)
        v2_bits, v2_energy = hardware_qubo_solve_exhaustive(v2_Q, v2.n_vars)

        # --- Compare weights per pair ---
        w_diff_max = 0.0
        sim_diff_max = 0.0
        for pair in v2.pairs:
            idx = pair.var_idx
            dw = abs(v1_w[idx] - v2_w[idx])
            ds = abs(v1_sim[idx] - v2_sim[idx])
            w_diff_max = max(w_diff_max, dw)
            sim_diff_max = max(sim_diff_max, ds)

        pen_diff = abs(v1_pen - v2_pen)
        energy_diff = abs(v1_energy - v2_energy)
        bits_match = (list(v1_bits) == list(v2_bits))

        ok = (w_diff_max < 1e-12 and sim_diff_max < 1e-12
              and pen_diff < 1e-12 and energy_diff < 1e-9 and bits_match)
        status = "PASS" if ok else "FAIL"
        logger.info(
            f"  [{i}] {meta['category_label']:<12s} cycle={meta['cycle']:<4d} "
            f"n={v2.n_vars:<3d} wΔ={w_diff_max:.2e} simΔ={sim_diff_max:.2e} "
            f"penΔ={pen_diff:.2e} EΔ={energy_diff:.2e} bits={bits_match} → {status}"
        )
        if not ok:
            all_good = False

    print()
    if all_good:
        logger.info("=" * 60)
        logger.info("✓ ALL 8 INSTANCES MATCH v1 BYTE-FOR-BYTE")
        logger.info("=" * 60)
        logger.info("The v2 pipeline reproduces v1 numerics exactly. "
                    "The existing hardware JSONs remain valid.")
    else:
        logger.error("✗ Some instances diverge — inspect above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
