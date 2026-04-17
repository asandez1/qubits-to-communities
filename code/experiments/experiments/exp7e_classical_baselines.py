#!/usr/bin/env python3
"""
Experiment 7e: Stronger Classical Baselines (Phase 2F).

Head-to-head test against K_c: does LightGBM or a bipartite GNN on the same
z_ij feature vectors beat the Shin-Teo-Jeong dequantization kernel?

If these modern classical models match or exceed K_c (0.86% regret), the
paper must honestly report it — the quantum path has even stronger
non-quantum competition. If K_c remains best, the kernel formulation is
validated as the principled classical baseline.

Uses the same 8 instances, same training data, and same evaluation harness
as Exp 7 Arms A-D.

Usage:
    python exp7e_classical_baselines.py --models lgbm gnn kc
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from experiments.exp7_mechanism_isolation import (
    harvest_instances, compute_classical_weights,
    build_qubo, solve_qubo_exhaustive, analyze_degeneracy,
    _evaluate_instance, _summarize,
)
from experiments.exp6d_v3_extended import (
    build_training_set, member_embedding_9d_raw, task_embedding_9d_raw,
)


CONFIG_PATH = os.path.join(ROOT, "config", "community_economy.json")


def pair_to_z(member, task_category) -> np.ndarray:
    """9D feature vector for a (member, task_category) pair — same as K_c input."""
    m_vec = member_embedding_9d_raw(member)   # 9D [0, 1]
    t_vec = task_embedding_9d_raw(task_category)
    # Simple concatenation would double to 18D. Instead use z = m * t (Hadamard)
    # or cos(phi_m - phi_t)... we use m concatenated with (m - t) to give the
    # model access to both absolute member features AND member-task residual.
    return np.concatenate([m_vec, m_vec - t_vec])  # 18D


def build_train_xy(train_data):
    """Convert (v1, v2, label) triples from build_training_set into X, y.

    train_data contains scaled V3-range vectors. We need raw [0,1] vectors for
    classical models, so we re-harvest the raw features.
    """
    X = []
    y = []
    for v1, v2, label in train_data:
        # Unscale back to [0, 1]
        m_raw = (np.asarray(v1) - 0.1) / (np.pi - 0.2)
        t_raw = (np.asarray(v2) - 0.1) / (np.pi - 0.2)
        m_raw = np.clip(m_raw, 0, 1)
        t_raw = np.clip(t_raw, 0, 1)
        z = np.concatenate([m_raw, m_raw - t_raw])
        X.append(z)
        y.append(float(label))
    return np.array(X), np.array(y)


def instance_X(inst):
    """Per-pair feature vectors for all eligible pairs in an instance."""
    X = []
    for r, o, req, offer, member in inst["eligible_pairs"]:
        X.append(pair_to_z(member, req.category))
    return np.array(X)


# ============================================================
# LightGBM baseline
# ============================================================

def run_lgbm(instances, X_train, y_train, classical_results):
    import lightgbm as lgb
    logger.info("[LGBM] LightGBM gradient boosting on 18D z_ij")
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        min_child_samples=3, verbose=-1, random_state=42,
    )
    clf.fit(X_train, y_train.astype(int))

    per_instance = []
    for i, inst in enumerate(instances):
        X_inst = instance_X(inst)
        preds = clf.predict_proba(X_inst)[:, 1]  # P(class=1)
        sims = {idx: float(np.clip(p, 0, 1)) for idx, p in enumerate(preds)}
        rec = _evaluate_instance(inst, sims, classical_results[i])
        rec["instance_idx"] = i
        per_instance.append(rec)
    summary = _summarize(per_instance)
    logger.info(f"  LGBM → regret={summary['mean_classical_regret']*100:+.2f}% "
                f"co-opt={summary['n_co_optimal']}/{len(instances)} "
                f"identical={summary['n_identical']}/{len(instances)}")
    return summary


# ============================================================
# Bipartite GNN baseline (simple MLP on [member || task] features)
# ============================================================

def run_gnn_mlp(instances, X_train, y_train, classical_results,
                epochs=500, hidden=32, lr=0.01, seed=42):
    """GNN-lite: MLP edge predictor on member-task pairs.

    Mirrors the structural role of a bipartite GNN (learn an edge-score
    function from node-pair features) at the scale of our training data (100
    pairs). A full message-passing GNN would overfit on this sample size.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    d = X_t.shape[1]
    model = nn.Sequential(
        nn.Linear(d, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    logger.info(f"[GNN-MLP] {d}D → {hidden} → {hidden} → 1, {epochs} epochs")
    for ep in range(epochs):
        logits = model(X_t).squeeze(-1)
        loss = bce(logits, y_t)
        opt.zero_grad(); loss.backward(); opt.step()
    final_loss = float(loss.item())

    model.eval()
    per_instance = []
    for i, inst in enumerate(instances):
        X_inst = torch.tensor(instance_X(inst), dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = torch.sigmoid(model(X_inst).squeeze(-1)).cpu().numpy()
        sims = {idx: float(np.clip(p, 0, 1)) for idx, p in enumerate(probs)}
        rec = _evaluate_instance(inst, sims, classical_results[i])
        rec["instance_idx"] = i
        per_instance.append(rec)
    summary = _summarize(per_instance)
    summary["training_loss"] = final_loss
    logger.info(f"  GNN-MLP → regret={summary['mean_classical_regret']*100:+.2f}% "
                f"co-opt={summary['n_co_optimal']}/{len(instances)} "
                f"identical={summary['n_identical']}/{len(instances)} "
                f"loss={final_loss:.4f}")
    return summary


# ============================================================
# K_c replication (for sanity check vs Exp 7 Arm D)
# ============================================================

def run_kc_18d(instances, X_train, y_train, classical_results, lam=1e-2):
    """Shin-Teo-Jeong K_c kernel on the same 18D features, for comparison
    with Arm D (which used 9D member features only)."""
    def phi(x):
        return 0.1 + (np.pi - 0.2) * np.asarray(x)

    def kc(A, B):
        pA, pB = phi(A), phi(B)
        diff = pA[:, None, :] - pB[None, :, :]
        factors = np.clip(1.0 + np.cos(diff), 1e-12, 2.0)
        return np.exp(np.sum(np.log(factors), axis=-1))

    K = kc(X_train, X_train)
    k_scale = float(np.mean(np.diag(K)))
    K_norm = K / max(k_scale, 1e-12)
    n = K_norm.shape[0]
    alpha = np.linalg.solve(K_norm + lam * np.eye(n), y_train)

    per_instance = []
    for i, inst in enumerate(instances):
        X_inst = instance_X(inst)
        K_t = kc(X_inst, X_train) / max(k_scale, 1e-12)
        preds = np.clip(K_t @ alpha, 0, 1)
        sims = {idx: float(p) for idx, p in enumerate(preds)}
        rec = _evaluate_instance(inst, sims, classical_results[i])
        rec["instance_idx"] = i
        per_instance.append(rec)
    summary = _summarize(per_instance)
    logger.info(f"  K_c 18D → regret={summary['mean_classical_regret']*100:+.2f}% "
                f"co-opt={summary['n_co_optimal']}/{len(instances)} "
                f"identical={summary['n_identical']}/{len(instances)}")
    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=int, default=30)
    parser.add_argument("--train-members", type=int, default=100)
    parser.add_argument("--n-train-per-class", type=int, default=40)
    parser.add_argument("--models", nargs="+", default=["lgbm", "gnn", "kc18"],
                        help="Which baselines to run (lgbm, gnn, kc18)")
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "exp7e"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    logger.info("=" * 72)
    logger.info("EXPERIMENT 7e: Stronger Classical Baselines (Phase 2F)")
    logger.info("=" * 72)

    # Harvest the same 8 instances as Exp 7
    instances = harvest_instances(CONFIG_PATH, args.members)
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

    # Build training data (same distribution as V3 training)
    train_data = build_training_set(CONFIG_PATH, args.train_members,
                                    args.n_train_per_class)
    X_train, y_train = build_train_xy(train_data)
    logger.info(f"  Training data: {X_train.shape}, labels: {y_train.mean():.2f} pos rate")

    results = {}
    if "lgbm" in args.models:
        t0 = time.time()
        results["lgbm"] = run_lgbm(instances, X_train, y_train, classical_results)
        logger.info(f"    ({time.time()-t0:.1f}s)")

    if "gnn" in args.models:
        t0 = time.time()
        results["gnn"] = run_gnn_mlp(instances, X_train, y_train, classical_results)
        logger.info(f"    ({time.time()-t0:.1f}s)")

    if "kc18" in args.models:
        t0 = time.time()
        results["kc_18d"] = run_kc_18d(instances, X_train, y_train, classical_results)
        logger.info(f"    ({time.time()-t0:.1f}s)")

    logger.info("\n" + "=" * 72)
    logger.info("SUMMARY — Phase 2F Classical Baselines vs Exp 7 Arm D (K_c 9D)")
    logger.info("=" * 72)
    logger.info(f"  {'Model':<12s} {'Regret':>8s} {'Co-opt':>7s} {'Identical':>10s}")
    logger.info(f"  {'-'*12} {'-'*8} {'-'*7} {'-'*10}")
    logger.info(f"  {'Arm D 9D':<12s} {'0.86%':>8s} {'7/8':>7s} {'6/8':>10s}  (Exp 7 ref)")
    for name, s in results.items():
        logger.info(
            f"  {name:<12s} {s['mean_classical_regret']*100:+6.2f}%  "
            f"{s['n_co_optimal']:>3d}/8   {s['n_identical']:>3d}/8"
        )

    out_path = os.path.join(args.output, "classical_baselines.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "Exp 7e: Classical Baselines",
            "models": list(results.keys()),
            "reference_arm_d_9d": {
                "regret": 0.0086, "co_optimal": 7, "identical": 6,
            },
            "results": {k: {
                "mean_classical_regret": v["mean_classical_regret"],
                "n_co_optimal": v["n_co_optimal"],
                "n_identical": v["n_identical"],
                "mean_sim_correlation_9d": v["mean_sim_correlation_9d"],
            } for k, v in results.items()},
        }, f, indent=2)
    logger.info(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
