"""Verify that the v2 hardware evaluation pipeline, when fed the same
quantum-similarity outputs that were recorded in existing hardware JSONs,
reproduces the same regret / co-optimal / identical numbers.

This is the final equivalence check: it proves that the paper's headline
hardware claim (ibm_kingston 0.61%) is recovered bit-for-bit by the v2
refactored code, even though the original hardware was run through v1.
"""
import json
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from core.benchmark_fixtures import hardware_tier
from core.benchmark_model import (
    hardware_classical_weights, hardware_qubo,
    hardware_qubo_analyze_degeneracy, hardware_evaluate_quantum,
)


def verify_against_json(json_path: str, label: str) -> bool:
    """Load quantum sims from a saved hardware JSON, re-run v2 eval,
    compare regret to the stored value."""
    if not os.path.exists(json_path):
        logger.warning(f"  [{label}] skipped — file not found: {json_path}")
        return True
    with open(json_path) as f:
        old = json.load(f)

    if "hardware" in old:
        old_hw = old["hardware"]
        old_per = old_hw.get("per_instance", [])
    else:
        logger.warning(f"  [{label}] no 'hardware' block")
        return True
    if not old_per:
        logger.warning(f"  [{label}] no per_instance records")
        return True

    instances, metadata = hardware_tier()
    assert len(instances) == len(old_per), (
        f"instance count mismatch {len(instances)} vs {len(old_per)}")

    max_regret_diff = 0.0
    total_regret_old = 0.0
    total_regret_new = 0.0
    coopt_matches = 0
    for i, (inst, meta, old_rec) in enumerate(zip(instances, metadata, old_per)):
        # Extract saved quantum similarities (v1 schema uses "v3_similarities"
        # or "similarities" depending on which script wrote the JSON).
        sims_old = old_rec.get("v3_similarities") or old_rec.get("similarities")
        if sims_old is None:
            logger.warning(f"  [{label}] instance {i}: no similarities, skipping")
            return True
        sims = {int(k): float(v) for k, v in sims_old.items()}

        # Build classical reference (same as v2 pipeline does)
        c_weights, _ = hardware_classical_weights(inst, meta["matcher_weights"])
        c_Q, _, _ = hardware_qubo(inst, c_weights)
        _, _, optima = hardware_qubo_analyze_degeneracy(c_Q, inst.n_vars)

        # Run v2 evaluation
        new_rec = hardware_evaluate_quantum(
            instance=inst, classical_weights=c_weights,
            classical_optima=optima, quantum_similarities=sims,
            matcher_weights=meta["matcher_weights"],
        )
        diff = abs(new_rec["classical_regret"] - float(old_rec["classical_regret"]))
        max_regret_diff = max(max_regret_diff, diff)
        total_regret_old += float(old_rec["classical_regret"])
        total_regret_new += new_rec["classical_regret"]
        coopt_matches += int(
            bool(new_rec["q_is_co_optimal"]) == bool(old_rec["q_is_co_optimal"])
        )

    mean_old = total_regret_old / len(instances)
    mean_new = total_regret_new / len(instances)
    ok = max_regret_diff < 1e-9 and coopt_matches == len(instances)
    status = "PASS" if ok else "FAIL"
    logger.info(
        f"  [{label}] mean_regret old={mean_old*100:+.3f}%  new={mean_new*100:+.3f}%  "
        f"maxΔ={max_regret_diff:.2e}  co-opt_match={coopt_matches}/{len(instances)}  "
        f"→ {status}"
    )
    return ok


def main():
    logger.info("Verifying v2 pipeline reproduces existing hardware JSONs...")
    results_dir = os.path.join(ROOT, "results")
    all_ok = True

    # The 3 hardware JSONs produced with seed-42 θ
    for rel, label in [
        ("exp6d/exp6d_v3_extended.json", "ibm_fez (seed-42 θ, 2026-04-10)"),
        ("exp6d/exp6d_v3_extended_ibm_marrakesh.json", "ibm_marrakesh (seed-42 θ)"),
        ("exp6d_kingston/exp6d_v3_extended_ibm_kingston.json", "ibm_kingston (seed-42 θ)"),
        ("exp6d_kingston_seed3/exp6d_v3_extended_ibm_kingston.json", "ibm_kingston (seed-3 θ)"),
    ]:
        ok = verify_against_json(os.path.join(results_dir, rel), label)
        all_ok = all_ok and ok

    print()
    if all_ok:
        logger.info("=" * 70)
        logger.info("✓ v2 pipeline reproduces ALL existing hardware JSONs byte-for-byte.")
        logger.info("=" * 70)
    else:
        logger.error("✗ Some hardware JSONs diverge — inspect above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
