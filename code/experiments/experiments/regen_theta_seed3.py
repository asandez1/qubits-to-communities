"""Quick: regenerate θ for seed=3 (which produced 0.61% on noiseless sim in Exp 7d).

Saves to results/exp6d/theta_seed3.npy so exp6d can load it via --theta-path.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# Constrain threads (learned from Phase 2A)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from experiments.exp6d_v3_extended import (
    build_training_set, train_v3_extended, N_PARAMS,
)

CONFIG_PATH = os.path.join(ROOT, "config", "community_economy.json")
OUT = os.path.join(ROOT, "results", "exp6d", "theta_seed3.npy")

if __name__ == "__main__":
    print(f"[seed=3] Building training set...")
    train_data = build_training_set(CONFIG_PATH, 100, 40)
    print(f"[seed=3] Training V3-extended with SPSA seed=3 (Exp 7d confirmed 0.61% outcome)...")
    # Match Phase 2A config exactly: 30 iter, seed=3 → 0.61% (verified in Exp 7d).
    theta, best_loss = train_v3_extended(
        train_data, n_iter=30, shots=2048, lr=0.15, seed=3,
    )
    assert theta.shape == (N_PARAMS,), f"bad shape: {theta.shape}"
    np.save(OUT, theta)
    print(f"[seed=3] Saved θ to {OUT}, norm={np.linalg.norm(theta):.3f}, best_loss={best_loss:.4f}")
