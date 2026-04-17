#!/usr/bin/env python3
"""
Experiment 8: Kernel-Weighted QUBO Architecture

Replaces the hand-tuned linear weight formula
    w_ij = α·sim + β·rep + γ·skill + δ·price
with a Shin-Teo-Jeong basis-equivalent kernel K_c trained via kernel ridge
regression on the full constraint feature vector z_ij.

Exp 7 Arm D already showed that K_c on V3's 9D similarity space achieves
0.86% regret (7/8 co-optimal) — 42% better than the 1.48% noiseless quantum
ceiling. This experiment tests whether encoding the FULL constraint vector
(skill match per category, reputation, proximity, price, time, social capital,
trust) into the kernel captures cross-feature interactions the linear formula
misses.

Methods compared:
  1. Linear:    w_ij = α·cosine_sim + β·rep + γ·skill + δ·price  (current pipeline)
  2. K_c 9D:    w_ij = α·K_c_9d_pred + β·rep + γ·skill + δ·price  (Arm D replication)
  3. K_c 12D:   w_ij = α·K_c_12d_pred + β·rep + γ·skill + δ·price  (this experiment)
  4. K_c pure:  w_ij = K_c_12d_pred  (kernel encodes all features directly)

Metrics:
  - Classical regret: how different is the assignment from the linear-optimal?
    (Note: this is biased toward linear — it's 0% by definition for linear.)
  - Mean provider skill: average skill_level of assigned providers in their
    task category. This is an EXTERNAL quality metric independent of the
    weight formula.
  - N assignments: how many pairs each method activates.

Protocol:
  1. Harvest the same 8 instances used in Exp 6/7 (via harvest_instances)
  2. Build the full feature vector z_ij for each (member, task) pair
  3. Train K_c KRR on labeled data from the simulation
  4. Compute kernel-weighted QUBO weights using KRR predictions
  5. Solve QUBO with these kernel weights (same build_qubo + solve_qubo_exhaustive)
  6. Evaluate each assignment via external quality metrics

Budget: CPU only, no hardware time.
"""

import json
import logging
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from experiments.exp6b_hybrid_replication import harvest_instances, analyze_degeneracy
from experiments.exp6_hybrid_pipeline import (
    compute_classical_weights,
    build_qubo,
    solve_qubo_exhaustive,
)
from experiments.exp6d_v3_extended import (
    N_PARAMS,
    FULL_AXES,
    member_embedding_9d,
    task_embedding_9d,
    compute_9d_cosine,
    build_training_set,
    scale_to_v3_range,
)


# =====================================================================
# Feature vector z_ij — full constraint encoding
# =====================================================================

# z_ij axes:
#   [0:6]  skill match per category (member skill_level for each of 6 categories)
#   [6]    reputation
#   [7]    geographic proximity proxy (skill level in task category, normalized)
#   [8]    price sensitivity (bonding curve price / 10, normalized to ~[0,1])
#   [9]    time availability
#   [10]   social capital
#   [11]   trust score (member skill in task category / threshold, clamped to [0,1])

N_FEATURES = 12
SKILL_CATEGORIES = FULL_AXES[:6]  # electrical, plumbing, tutoring, transport, cooking, childcare
TRUST_THRESHOLD = 0.3  # from config matching.skill_gap_threshold


def build_feature_vector(member, req_category, price):
    """Build the full ~12D constraint feature vector z_ij for a (member, task) pair.

    All features are in [0, 1] before angle scaling.
    """
    z = np.zeros(N_FEATURES)

    # Skill match per category (6 dims)
    for i, cat in enumerate(SKILL_CATEGORIES):
        z[i] = member.skill_levels.get(cat, 0.0)

    # Reputation
    z[6] = member.reputation

    # Geographic proximity proxy (skill level in task category)
    z[7] = member.skill_levels.get(req_category, 0.0)

    # Price sensitivity (bonding curve price / 10, rough normalization)
    z[8] = min(price / 10.0, 1.0)

    # Time availability
    z[9] = member.time_availability

    # Social capital
    z[10] = member.social_capital

    # Trust score (skill / threshold, clamped)
    skill_in_cat = member.skill_levels.get(req_category, 0.0)
    z[11] = min(skill_in_cat / TRUST_THRESHOLD, 1.0) if TRUST_THRESHOLD > 0 else 1.0

    z = np.clip(z, 0.0, 1.0)
    return z


# =====================================================================
# K_c kernel on full feature vectors
# =====================================================================

def _phi(x, lo=0.1, hi=np.pi - 0.1):
    """V3 angle encoding: linear map [0, 1] -> [0.1, pi-0.1]."""
    return lo + (hi - lo) * np.asarray(x, dtype=np.float64)


def _kc_kernel(X1, X2):
    """Basis-equivalent classical kernel K_c(x_i, x_j) on batched inputs.

    X1: (M1, D), X2: (M2, D) — raw features in [0,1], angle-scaled internally.
    Returns: (M1, M2) kernel matrix prod_alpha [1 + cos(phi(x_i) - phi(x_j))].
    Computed in log-space to avoid overflow when D is large.
    """
    phi1 = _phi(X1)
    phi2 = _phi(X2)
    diff = phi1[:, None, :] - phi2[None, :, :]
    factors = 1.0 + np.cos(diff)  # shape (M1, M2, D), each in [0, 2]
    factors = np.clip(factors, 1e-12, 2.0)
    log_k = np.sum(np.log(factors), axis=-1)
    return np.exp(log_k)


def train_kc_kernel_ridge(X_train, y_train, lam=1e-2):
    """Kernel ridge regression in the K_c RKHS.

    X_train: (M, D) raw features in [0, 1]
    y_train: (M,) labels in {0, 1}
    Returns: (alpha, X_train, k_scale)
    """
    K = _kc_kernel(X_train, X_train)
    k_scale = float(np.mean(np.diag(K)))
    K_norm = K / k_scale

    M = K_norm.shape[0]
    alpha = np.linalg.solve(K_norm + lam * np.eye(M), y_train)
    logger.info(f"  K_c KRR: {M} train points, D={X_train.shape[1]}, "
                f"lambda={lam}, k_scale={k_scale:.3e}")
    return alpha, X_train, k_scale


def predict_kc(X_test, alpha, X_train, k_scale):
    """Predict using trained KRR model. Returns scores clamped to [0, 1]."""
    K_test = _kc_kernel(X_test, X_train) / k_scale
    preds = K_test @ alpha
    return np.clip(preds, 0.0, 1.0)


# =====================================================================
# Build training data with full z_ij feature vectors
# =====================================================================

def build_full_training_set(config_path, num_members=100, n_pairs_per_class=40, seed=42):
    """Build training pairs with full z_ij feature vectors.

    Label = 1 if member's skill in task category > 0.6, else 0 (same protocol
    as build_training_set in exp6d but over the full 12D z_ij instead of 9D).
    """
    from core.simulation_engine import SimulationEngine
    from core.matching_service import MatchingStrategy

    with open(config_path) as f:
        config = json.load(f)
    engine = SimulationEngine(
        config=config, matching_strategy=MatchingStrategy.GREEDY,
        num_members=num_members, num_cycles=501, seed=seed,
        topology_sample_interval=99999,
    )
    engine.run()

    categories = list(config["categories"].keys())
    prices = engine.economy.get_all_prices()
    rng = np.random.default_rng(seed)

    similar, dissimilar = [], []
    for member in engine.members_list:
        for cat in categories:
            z = build_feature_vector(member, cat, prices.get(cat, 5.0))
            skill = member.skill_levels.get(cat, 0.0)
            if skill > 0.6:
                similar.append((z, 1.0))
            elif skill < 0.1:
                dissimilar.append((z, 0.0))

    rng.shuffle(similar)
    rng.shuffle(dissimilar)
    n_per = min(n_pairs_per_class, len(similar), len(dissimilar))
    train_data = similar[:n_per] + dissimilar[:n_per]
    rng.shuffle(train_data)

    logger.info(f"  Full z_ij training: {len(similar)} high-skill + "
                f"{len(dissimilar)} low-skill available -> "
                f"using {n_per} per class ({len(train_data)} total), "
                f"D={N_FEATURES}")
    return train_data


def build_9d_training_set(config_path, num_members=100, n_pairs_per_class=40, seed=42):
    """Wrapper around exp6d's build_training_set for the 9D Arm D baseline."""
    return build_training_set(config_path, num_members, n_pairs_per_class, seed=seed)


# =====================================================================
# External quality metrics — independent of weight formula
# =====================================================================

def _compute_external_quality(inst, bits):
    """Compute quality metrics that do NOT depend on which weight formula was used.

    - mean_provider_skill: average skill_level of assigned providers in their
      task category. Higher = better matching.
    - n_assignments: how many pairs were activated.
    - mean_provider_reputation: average reputation of assigned providers.
    - total_skill: sum of provider skills across assignments (raw coordination value).
    """
    pairs = inst["eligible_pairs"]
    skills = []
    reps = []
    for idx, (r, o, req, offer, member) in enumerate(pairs):
        if bits[idx] == 1:
            skills.append(member.skill_levels.get(req.category, 0.0))
            reps.append(member.reputation)

    n_assigned = len(skills)
    return {
        "n_assignments": n_assigned,
        "mean_provider_skill": float(np.mean(skills)) if skills else 0.0,
        "total_skill": float(np.sum(skills)),
        "mean_provider_reputation": float(np.mean(reps)) if reps else 0.0,
    }


# =====================================================================
# Evaluate one instance with BOTH regret and external quality
# =====================================================================

def _evaluate_instance(inst, weights, classical_result, label=""):
    """Score one instance given arbitrary QUBO weights.

    Returns both:
    - classical_regret: distance from linear-optimal (biased toward linear)
    - external quality: mean_provider_skill, n_assignments (unbiased)
    """
    n = len(inst["eligible_pairs"])
    Q, _, _ = build_qubo(inst, weights)
    bits, energy = solve_qubo_exhaustive(Q, n)

    # Classical regret (relative to linear-optimal solution)
    c_bits = classical_result["bits"]
    w_c = classical_result["weights"]
    obj_c = sum(w_c[k] * c_bits[k] for k in range(n))
    obj_this = sum(w_c[k] * bits[k] for k in range(n))
    regret = (obj_c - obj_this) / obj_c if obj_c != 0 else 0.0

    identical = tuple(c_bits) == tuple(bits)
    is_co_optimal = tuple(bits) in [tuple(b) for b in classical_result["optima"]]

    # External quality (independent of weight formula)
    quality = _compute_external_quality(inst, bits)

    # Also compute quality of the linear-optimal for comparison
    quality_linear = _compute_external_quality(inst, c_bits)

    return {
        "instance_idx": None,
        "category": inst["category"],
        "n_pairs": n,
        "classical_regret": float(regret),
        "identical_to_linear": identical,
        "is_co_optimal": is_co_optimal,
        "n_assignments": quality["n_assignments"],
        "mean_provider_skill": quality["mean_provider_skill"],
        "total_skill": quality["total_skill"],
        "mean_provider_reputation": quality["mean_provider_reputation"],
        "linear_mean_provider_skill": quality_linear["mean_provider_skill"],
        "linear_n_assignments": quality_linear["n_assignments"],
        "label": label,
    }


def _summarize(per_instance, n_instances):
    return {
        "mean_classical_regret": float(
            np.mean([r["classical_regret"] for r in per_instance])
        ),
        "n_co_optimal": int(sum(1 for r in per_instance if r["is_co_optimal"])),
        "n_identical": int(sum(1 for r in per_instance if r["identical_to_linear"])),
        "mean_provider_skill": float(
            np.mean([r["mean_provider_skill"] for r in per_instance])
        ),
        "mean_total_skill": float(
            np.mean([r["total_skill"] for r in per_instance])
        ),
        "mean_n_assignments": float(
            np.mean([r["n_assignments"] for r in per_instance])
        ),
        "mean_provider_reputation": float(
            np.mean([r["mean_provider_reputation"] for r in per_instance])
        ),
        "per_instance": per_instance,
    }


# =====================================================================
# Method 1: Linear baseline (current pipeline)
# =====================================================================

def run_linear_baseline(instances, classical_results):
    """Linear: w_ij = alpha*cosine_sim + beta*rep + gamma*skill + delta*price."""
    logger.info("[LINEAR] Current pipeline (cosine sim + hand-tuned coefficients)")

    per_instance = []
    for i, inst in enumerate(instances):
        w_c, _ = compute_classical_weights(inst)
        rec = _evaluate_instance(inst, w_c, classical_results[i], label="linear")
        rec["instance_idx"] = i
        per_instance.append(rec)

    summary = _summarize(per_instance, len(instances))
    summary["method"] = "linear"
    logger.info(
        f"  LINEAR -> skill={summary['mean_provider_skill']:.3f} "
        f"assigns={summary['mean_n_assignments']:.1f} "
        f"regret=0.00% (by definition)"
    )
    return summary


# =====================================================================
# Method 2: K_c on 9D (Arm D replication)
# =====================================================================

def run_kc_9d(instances, classical_results, train_data_9d, lam=1e-2):
    """K_c on 9D angle-encoded vectors: replaces cosine_sim with K_c prediction."""
    logger.info("[K_c 9D] Kernel on 9D V3 embeddings (Arm D replication)")

    X_train_angles = np.array(
        [np.concatenate([v1, v2]) for v1, v2, _ in train_data_9d], dtype=np.float64
    )
    y_train = np.array([label for _, _, label in train_data_9d], dtype=np.float64)

    def _kc_kernel_prescaled(X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        factors = 1.0 + np.cos(diff)
        factors = np.clip(factors, 1e-12, 2.0)
        log_k = np.sum(np.log(factors), axis=-1)
        return np.exp(log_k)

    K = _kc_kernel_prescaled(X_train_angles, X_train_angles)
    k_scale = float(np.mean(np.diag(K)))
    K_norm = K / k_scale
    M = K_norm.shape[0]
    alpha_krr = np.linalg.solve(K_norm + lam * np.eye(M), y_train)
    logger.info(f"  KRR (9D): {M} train, D=18, lambda={lam}, k_scale={k_scale:.3e}")

    per_instance = []
    for i, inst in enumerate(instances):
        X_test = np.array([
            np.concatenate([member_embedding_9d(member), task_embedding_9d(req.category)])
            for (r, o, req, offer, member) in inst["eligible_pairs"]
        ], dtype=np.float64)

        K_test = _kc_kernel_prescaled(X_test, X_train_angles) / k_scale
        preds = np.clip(K_test @ alpha_krr, 0.0, 1.0)

        # Plug K_c prediction into the standard weight formula (replacing cosine_sim)
        matcher = inst["matcher"]
        prices = inst["prices"]
        a, b, g, d = (matcher._weights["alpha"], matcher._weights["beta"],
                       matcher._weights["gamma"], matcher._weights["delta"])

        weights = {}
        for idx, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
            rep = member.reputation
            skill = member.skill_levels.get(req.category, 0.0)
            price = prices.get(req.category, 5.0) / 10.0
            weights[idx] = a * preds[idx] + b * rep + g * skill + d * price

        rec = _evaluate_instance(inst, weights, classical_results[i], label="kc_9d")
        rec["instance_idx"] = i
        per_instance.append(rec)

    summary = _summarize(per_instance, len(instances))
    summary["method"] = "kc_9d"
    summary["ridge_lambda"] = lam
    summary["n_train"] = len(train_data_9d)
    logger.info(
        f"  K_c 9D -> skill={summary['mean_provider_skill']:.3f} "
        f"assigns={summary['mean_n_assignments']:.1f} "
        f"regret={summary['mean_classical_regret']*100:.2f}%"
    )
    return summary


# =====================================================================
# Method 3: K_c on 12D (hybrid — kernel replaces similarity only)
# =====================================================================

def run_kc_12d_hybrid(instances, classical_results, alpha_krr, X_train, k_scale):
    """K_c on 12D full feature vector: replaces cosine_sim with K_c prediction."""
    logger.info("[K_c 12D] Kernel on full 12D feature vector (hybrid)")

    per_instance = []
    for i, inst in enumerate(instances):
        matcher = inst["matcher"]
        prices = inst["prices"]
        a, b, g, d = (matcher._weights["alpha"], matcher._weights["beta"],
                       matcher._weights["gamma"], matcher._weights["delta"])

        X_test = np.array([
            build_feature_vector(member, req.category, prices.get(req.category, 5.0))
            for (r, o, req, offer, member) in inst["eligible_pairs"]
        ])
        preds = predict_kc(X_test, alpha_krr, X_train, k_scale)

        weights = {}
        for idx, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
            rep = member.reputation
            skill = member.skill_levels.get(req.category, 0.0)
            price = prices.get(req.category, 5.0) / 10.0
            weights[idx] = a * preds[idx] + b * rep + g * skill + d * price

        rec = _evaluate_instance(inst, weights, classical_results[i], label="kc_12d")
        rec["instance_idx"] = i
        per_instance.append(rec)

    summary = _summarize(per_instance, len(instances))
    summary["method"] = "kc_12d_hybrid"
    logger.info(
        f"  K_c 12D -> skill={summary['mean_provider_skill']:.3f} "
        f"assigns={summary['mean_n_assignments']:.1f} "
        f"regret={summary['mean_classical_regret']*100:.2f}%"
    )
    return summary


# =====================================================================
# Method 4: K_c pure — w_ij = K_c_pred(z_ij)
# =====================================================================

def run_kc_12d_pure(instances, classical_results, alpha_krr, X_train, k_scale):
    """Pure kernel: w_ij = K_c_pred(z_ij). The kernel encodes all features."""
    logger.info("[K_c PURE] w_ij = K_c(z_ij) — kernel replaces entire weight formula")

    per_instance = []
    for i, inst in enumerate(instances):
        prices = inst["prices"]

        X_test = np.array([
            build_feature_vector(member, req.category, prices.get(req.category, 5.0))
            for (r, o, req, offer, member) in inst["eligible_pairs"]
        ])
        preds = predict_kc(X_test, alpha_krr, X_train, k_scale)

        weights = {idx: float(preds[idx]) for idx in range(len(preds))}

        rec = _evaluate_instance(inst, weights, classical_results[i], label="kc_pure")
        rec["instance_idx"] = i
        per_instance.append(rec)

    summary = _summarize(per_instance, len(instances))
    summary["method"] = "kc_12d_pure"
    logger.info(
        f"  K_c PURE -> skill={summary['mean_provider_skill']:.3f} "
        f"assigns={summary['mean_n_assignments']:.1f} "
        f"regret={summary['mean_classical_regret']*100:.2f}%"
    )
    return summary


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 8: Kernel-Weighted QUBO Architecture")
    parser.add_argument("--members", type=int, default=30)
    parser.add_argument("--train-members", type=int, default=100)
    parser.add_argument("--n-train-per-class", type=int, default=40)
    parser.add_argument("--kc-lambda", type=float, default=1e-2,
                        help="KRR regularization (default 0.01)")
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exp8"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "community_economy.json",
    )

    logger.info("=" * 72)
    logger.info("EXPERIMENT 8: Kernel-Weighted QUBO Architecture")
    logger.info("  Question: Does replacing cosine similarity with K_c kernel")
    logger.info("  improve coordination quality (not just objective score)?")
    logger.info(f"  Feature dimensions: {N_FEATURES} (full z_ij)")
    logger.info(f"  Kernel: Shin-Teo-Jeong K_c (product kernel)")
    logger.info(f"  KRR lambda: {args.kc_lambda}")
    logger.info("=" * 72)

    # --- SETUP: harvest same 8 instances ---
    logger.info("\n[SETUP] Harvesting instances (same protocol as Exp 6b/6d/7)...")
    instances = harvest_instances(config_path, args.members)
    logger.info(f"  Got {len(instances)} instances")

    classical_results = []
    for inst in instances:
        n = len(inst["eligible_pairs"])
        w_c, s_c = compute_classical_weights(inst)
        Q_c, _, _ = build_qubo(inst, w_c)
        c_bits, c_energy = solve_qubo_exhaustive(Q_c, n)
        n_opt, gap, opt_bits = analyze_degeneracy(Q_c, n)
        classical_results.append({
            "n_pairs": n, "weights": w_c, "similarities": s_c,
            "bits": c_bits, "energy": c_energy,
            "n_optima": n_opt, "optima": opt_bits,
        })

    # --- TRAINING DATA ---
    logger.info("\n[TRAIN] Building training data...")

    logger.info("  (a) Full feature vector z_ij (12D):")
    train_data_full = build_full_training_set(
        config_path, args.train_members, args.n_train_per_class)
    X_train_full = np.array([z for z, _ in train_data_full])
    y_train_full = np.array([label for _, label in train_data_full])

    logger.info("  (b) 9D angle-encoded vectors (for K_c 9D baseline):")
    train_data_9d = build_9d_training_set(
        config_path, args.train_members, args.n_train_per_class)

    # --- TRAIN K_c KRR on full features ---
    logger.info("\n[KRR] Training K_c kernel ridge regression on full z_ij...")
    t0 = time.time()
    alpha_krr, X_train_krr, k_scale = train_kc_kernel_ridge(
        X_train_full, y_train_full, lam=args.kc_lambda)
    logger.info(f"  KRR training: {time.time()-t0:.2f}s")

    # --- RUN ALL METHODS ---
    logger.info("")
    linear = run_linear_baseline(instances, classical_results)

    logger.info("")
    kc_9d = run_kc_9d(instances, classical_results, train_data_9d,
                       lam=args.kc_lambda)

    logger.info("")
    kc_12d = run_kc_12d_hybrid(instances, classical_results,
                                alpha_krr, X_train_krr, k_scale)

    logger.info("")
    kc_pure = run_kc_12d_pure(instances, classical_results,
                               alpha_krr, X_train_krr, k_scale)

    # --- RESULTS ---
    methods = [
        ("Linear (current)", linear),
        ("K_c 9D (Arm D)", kc_9d),
        ("K_c 12D (hybrid)", kc_12d),
        ("K_c 12D (pure)", kc_pure),
    ]

    logger.info("\n" + "=" * 72)
    logger.info("EXPERIMENT 8 — RESULTS")
    logger.info("=" * 72)

    # --- TABLE 1: External quality (the unbiased comparison) ---
    logger.info("\n  TABLE 1: External Quality (independent of weight formula)")
    logger.info(f"  {'Method':<22s} {'Skill':>7s} {'TotSkill':>9s} {'Assigns':>8s} {'Rep':>7s}")
    logger.info(f"  {'-'*22} {'-'*7} {'-'*9} {'-'*8} {'-'*7}")
    for name, r in methods:
        logger.info(
            f"  {name:<22s} "
            f"{r['mean_provider_skill']:7.3f} "
            f"{r['mean_total_skill']:9.3f} "
            f"{r['mean_n_assignments']:8.1f} "
            f"{r['mean_provider_reputation']:7.3f}"
        )

    # --- TABLE 2: Regret-based comparison (biased toward linear) ---
    logger.info(f"\n  TABLE 2: Classical Regret (distance from linear-optimal)")
    logger.info(f"  NOTE: Linear is 0%% by definition — this measures deviation, not quality")
    logger.info(f"  {'Method':<22s} {'Regret':>8s} {'Co-opt':>8s} {'Ident':>8s}")
    logger.info(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    for name, r in methods:
        logger.info(
            f"  {name:<22s} "
            f"{r['mean_classical_regret']*100:7.2f}% "
            f"{r['n_co_optimal']:5d}/{len(instances)} "
            f"{r['n_identical']:5d}/{len(instances)}"
        )

    # --- Per-instance breakdown ---
    logger.info(f"\n  Per-instance: Provider Skill (higher = better match)")
    logger.info(f"  {'Inst':>4s} {'Category':>10s} | "
                f"{'Linear':>7s} {'Kc-9D':>7s} {'Kc-12D':>7s} {'Kc-Pur':>7s}")
    logger.info(f"  {'-'*4} {'-'*10} | {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for i in range(len(instances)):
        cat = instances[i]["category"]
        s_lin = linear["per_instance"][i]["mean_provider_skill"]
        s_9d = kc_9d["per_instance"][i]["mean_provider_skill"]
        s_12d = kc_12d["per_instance"][i]["mean_provider_skill"]
        s_pur = kc_pure["per_instance"][i]["mean_provider_skill"]
        # Mark instances where kernel beats linear
        marker = ""
        if s_12d > s_lin + 1e-6:
            marker = " <-- K_c 12D wins"
        elif s_12d < s_lin - 1e-6:
            marker = " <-- Linear wins"
        logger.info(
            f"  [{i:2d}] {cat:>10s} | "
            f"{s_lin:7.3f} {s_9d:7.3f} {s_12d:7.3f} {s_pur:7.3f}{marker}"
        )

    # --- INTERPRETATION ---
    logger.info("\n  " + "-" * 68)
    logger.info("  INTERPRETATION")
    logger.info("  " + "-" * 68)

    # Compare external quality
    skill_lin = linear["mean_provider_skill"]
    skill_9d = kc_9d["mean_provider_skill"]
    skill_12d = kc_12d["mean_provider_skill"]
    skill_pure = kc_pure["mean_provider_skill"]

    logger.info(f"\n  Mean provider skill (external quality):")
    logger.info(f"    Linear:    {skill_lin:.4f}")
    logger.info(f"    K_c 9D:    {skill_9d:.4f}  ({(skill_9d - skill_lin)/skill_lin*100:+.1f}% vs linear)")
    logger.info(f"    K_c 12D:   {skill_12d:.4f}  ({(skill_12d - skill_lin)/skill_lin*100:+.1f}% vs linear)")
    logger.info(f"    K_c pure:  {skill_pure:.4f}  ({(skill_pure - skill_lin)/skill_lin*100:+.1f}% vs linear)")

    # Count per-instance wins
    wins_12d = sum(1 for i in range(len(instances))
                   if kc_12d["per_instance"][i]["mean_provider_skill"] >
                      linear["per_instance"][i]["mean_provider_skill"] + 1e-6)
    wins_lin = sum(1 for i in range(len(instances))
                   if linear["per_instance"][i]["mean_provider_skill"] >
                      kc_12d["per_instance"][i]["mean_provider_skill"] + 1e-6)
    ties = len(instances) - wins_12d - wins_lin

    logger.info(f"\n  Per-instance head-to-head (K_c 12D vs Linear):")
    logger.info(f"    K_c 12D wins: {wins_12d}    Linear wins: {wins_lin}    Ties: {ties}")

    if skill_12d > skill_lin + 1e-4:
        logger.info(f"\n  VERDICT: K_c 12D produces HIGHER-QUALITY assignments")
        logger.info(f"  (better provider-task skill matching) even though it deviates")
        logger.info(f"  from the linear-optimal solution. The linear formula's weights")
        logger.info(f"  are not the ground truth for coordination quality.")
    elif abs(skill_12d - skill_lin) < 1e-4:
        logger.info(f"\n  VERDICT: K_c 12D produces EQUIVALENT-QUALITY assignments.")
        logger.info(f"  Different weight formulas converge to similar real-world outcomes.")
        logger.info(f"  The linear formula is sufficient for these instances.")
    else:
        logger.info(f"\n  VERDICT: Linear produces HIGHER-QUALITY assignments.")
        logger.info(f"  The hand-tuned coefficients capture coordination structure that")
        logger.info(f"  the product kernel misses. The kernel's angle-encoded product")
        logger.info(f"  structure may not suit the correlation patterns in these features.")

    # --- SAVE ---
    summary = {
        "experiment": "Exp 8: Kernel-Weighted QUBO Architecture",
        "n_features": N_FEATURES,
        "n_instances": len(instances),
        "kc_lambda": args.kc_lambda,
        "n_train_full": len(train_data_full),
        "n_train_9d": len(train_data_9d),
        "k_scale_full": k_scale,
        "linear": linear,
        "kc_9d": kc_9d,
        "kc_12d_hybrid": kc_12d,
        "kc_12d_pure": kc_pure,
        "head_to_head_kc12d_vs_linear": {
            "kc12d_wins": wins_12d,
            "linear_wins": wins_lin,
            "ties": ties,
        },
    }
    out_path = os.path.join(args.output, "exp8_kernel_qubo.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
