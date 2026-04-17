#!/usr/bin/env python3
"""
Experiment 7: Mechanism Isolation for the Exp 6d Hardware-Over-Simulator Advantage

Exp 6d on ibm_fez produced 0.61% mean regret (6/8 co-optimal) vs 1.48% regret
(5/8 co-optimal) on the noiseless AerSimulator — i.e., hardware OUTPERFORMED
simulator by 59%. This mirrors the +18% hardware-over-simulation effect reported
in Sandez (2025) for semantic representation learning, attributed to NISQ noise
acting as beneficial regularization.

The result is suggestive but, at n=1, anecdotal. Two alternative explanations
must be ruled out before extending the Q-Manifold regularization claim from
representation learning to coordination/optimization:

  (1) Does any tiny function approximator with 57 parameters do just as well?
      A 57-parameter classical MLP trained on the same (member, task, label)
      pairs is the missing parameter-matched baseline.

  (2) Is the advantage driven by noise as such, or by something specific
      (coupling graph, native gate set) to ibm_fez?
      AerSimulator with a NoiseModel derived from ibm_fez's backend properties
      isolates the noise contribution from the hardware's structural features.

This experiment runs three arms on the same 8 instances from Exp 6d, using the
same θ persisted by the patched exp6d_v3_extended.py:

  ARM A  — Noiseless simulator (control; reproduces Exp 6d STEP 2)
  ARM B  — AerSimulator with NoiseModel.from_backend("ibm_fez")
  ARM C  — 57-parameter classical MLP (18→3→1, no biases), plugged into the
           same QUBO pipeline as a drop-in replacement for V3 similarity

Outcome matrix (regret on 8 instances, reference: noiseless=1.48%, hardware=0.61%):

  B ≈ 0.61% and C » 0.61%
    → Noise IS the mechanism; quantum circuit contributes structure classical
      MLP cannot match. Strongest extension of Q-Manifold claim.

  B ≈ 0.61% and C ≈ 0.61%
    → Noise helps but the gain is not quantum-specific at parameter parity.
      Null result on quantum uniqueness but clean on noise regularization.

  B ≈ 1.48% (no recovery) and C » 0.61%
    → Hardware advantage is structural (coupling/gates), not noise. Justifies
      follow-up on a second backend.

  B ≈ 1.48% and C ≈ 0.61%
    → Original 0.61% was a favorable draw; retreat to simulator-only claim.

Budget: no hardware time required. Arms A and B run on CPU; Arm C trains in
seconds on CPU.
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
# Silence qiskit's verbose per-pass transpiler logging
for noisy in ("qiskit", "qiskit.transpiler", "qiskit.compiler", "stevedore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from experiments.exp6b_hybrid_replication import harvest_instances, analyze_degeneracy
from experiments.exp6_hybrid_pipeline import (
    compute_classical_weights,
    build_qubo,
    solve_qubo_exhaustive,
)
from experiments.exp6d_v3_extended import (
    N_PER_VEC,
    N_TOTAL,
    N_PARAMS,
    build_v3_extended,
    get_ancilla_prob,
    member_embedding_9d,
    task_embedding_9d,
    member_embedding_9d_raw,
    task_embedding_9d_raw,
    compute_9d_cosine,
    build_training_set,
    train_v3_extended,
)


# =====================================================================
# Helpers
# =====================================================================

def _assemble_weights(inst, similarities):
    """Combine similarity scores with reputation/skill/price into matcher weights."""
    matcher = inst["matcher"]
    prices = inst["prices"]
    alpha = matcher._weights["alpha"]
    beta = matcher._weights["beta"]
    gamma = matcher._weights["gamma"]
    delta = matcher._weights["delta"]

    weights = {}
    for idx, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
        sim = similarities[idx]
        rep = member.reputation
        skill = member.skill_levels.get(req.category, 0.0)
        price = prices.get(req.category, 5.0) / 10.0
        weights[idx] = alpha * sim + beta * rep + gamma * skill + delta * price
    return weights


def _evaluate_instance(inst, similarities, classical_result):
    """Score one instance given arbitrary similarity scores."""
    n = len(inst["eligible_pairs"])
    weights_q = _assemble_weights(inst, similarities)
    Q_q, _, _ = build_qubo(inst, weights_q)
    q_bits, q_energy = solve_qubo_exhaustive(Q_q, n)

    cos_9d = compute_9d_cosine(inst)
    cos_arr = np.array([cos_9d[k] for k in sorted(cos_9d.keys())])
    sim_arr = np.array([similarities[k] for k in sorted(similarities.keys())])
    r_sim = (
        float(stats.pearsonr(cos_arr, sim_arr)[0]) if np.std(sim_arr) > 0 else 0.0
    )

    c_bits = classical_result["bits"]
    w_c = classical_result["weights"]
    obj_c = sum(w_c[k] * c_bits[k] for k in range(n))
    obj_q = sum(w_c[k] * q_bits[k] for k in range(n))
    regret = (obj_c - obj_q) / obj_c if obj_c != 0 else 0.0

    identical = tuple(c_bits) == tuple(q_bits)
    q_is_co_optimal = tuple(q_bits) in [tuple(b) for b in classical_result["optima"]]

    return {
        "instance_idx": None,  # caller fills
        "category": inst["category"],
        "n_pairs": n,
        "sim_correlation_9d": r_sim,
        "classical_regret": float(regret),
        "identical": identical,
        "q_is_co_optimal": q_is_co_optimal,
        "similarities": {int(k): float(v) for k, v in similarities.items()},
    }


def _summarize(per_instance):
    return {
        "mean_sim_correlation_9d": float(
            np.mean([r["sim_correlation_9d"] for r in per_instance])
        ),
        "mean_classical_regret": float(
            np.mean([r["classical_regret"] for r in per_instance])
        ),
        "n_co_optimal": int(sum(1 for r in per_instance if r["q_is_co_optimal"])),
        "n_identical": int(sum(1 for r in per_instance if r["identical"])),
        "per_instance": per_instance,
    }


# =====================================================================
# ARM A — Noiseless AerSimulator (control, reproduces Exp 6d sim)
# =====================================================================

def run_arm_a_noiseless(instances, theta, classical_results, shots=4096):
    """Reproduces Exp 6d STEP 2 — serves as control for Arm B."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    logger.info("[ARM A] Noiseless AerSimulator (control)")
    sim = AerSimulator(method="statevector")

    per_instance = []
    for i, inst in enumerate(instances):
        circuits = []
        for r, o, req, offer, member in inst["eligible_pairs"]:
            v1 = member_embedding_9d(member)
            v2 = task_embedding_9d(req.category)
            circuits.append(build_v3_extended(v1, v2, theta))

        tqcs = transpile(circuits, sim, optimization_level=0)
        result = sim.run(tqcs, shots=shots).result()

        similarities = {}
        for pair_idx in range(len(inst["eligible_pairs"])):
            p1 = get_ancilla_prob(result.get_counts(pair_idx))
            similarities[pair_idx] = 1.0 - p1

        rec = _evaluate_instance(inst, similarities, classical_results[i])
        rec["instance_idx"] = i
        per_instance.append(rec)

    summary = _summarize(per_instance)
    logger.info(
        f"  ARM A → regret={summary['mean_classical_regret']*100:+.2f}% "
        f"co-opt={summary['n_co_optimal']}/{len(instances)} "
        f"identical={summary['n_identical']}/{len(instances)} "
        f"r={summary['mean_sim_correlation_9d']:+.3f}"
    )
    return summary


# =====================================================================
# ARM B — AerSimulator with ibm_fez NoiseModel
# =====================================================================

def run_arm_b_noisy(instances, theta, classical_results, backend_name="ibm_fez",
                    shots=4096):
    """Injects ibm_fez's calibrated gate-error noise into the simulator.

    Design note: we intentionally DROP the coupling_map constraint. The
    mechanism claim is "NISQ-like gate noise acts as regularization", not
    "ibm_fez's specific qubit topology does". Keeping the coupling map forced
    the transpiler to insert SWAP chains for the 19-qubit circuit on ibm_fez's
    heavy-hex lattice (depth 167-270 in the hardware run), which blew up
    density-matrix simulation beyond tractable runtime. With coupling_map=None
    we keep the calibrated gate-error channels but simulate on the logical
    19-qubit graph the V3 circuit was designed for.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import QiskitRuntimeService

    logger.info(f"[ARM B] AerSimulator + NoiseModel.from_backend('{backend_name}')")
    logger.info("  Fetching backend properties (noise model only, no coupling)...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.backend(backend_name)

    noise_model = NoiseModel.from_backend(backend)
    basis_gates = list(noise_model.basis_gates)
    logger.info(f"  Noise model: {len(basis_gates)} basis gates "
                f"(coupling_map dropped for tractability)")

    # Trajectory-based noise: each shot Monte-Carlo samples the noise channels
    # on a 19-qubit statevector (~4 MB) instead of the full 2^38 density matrix.
    sim = AerSimulator(
        noise_model=noise_model,
        basis_gates=basis_gates,
        method="statevector",
    )

    per_instance = []
    for i, inst in enumerate(instances):
        circuits = []
        for r, o, req, offer, member in inst["eligible_pairs"]:
            v1 = member_embedding_9d(member)
            v2 = task_embedding_9d(req.category)
            circuits.append(build_v3_extended(v1, v2, theta))

        # Light transpile just to decompose to basis_gates — no routing
        tqcs = transpile(circuits, basis_gates=basis_gates, optimization_level=1)
        result = sim.run(tqcs, shots=shots).result()

        similarities = {}
        for pair_idx in range(len(inst["eligible_pairs"])):
            p1 = get_ancilla_prob(result.get_counts(pair_idx))
            similarities[pair_idx] = 1.0 - p1

        rec = _evaluate_instance(inst, similarities, classical_results[i])
        rec["instance_idx"] = i
        per_instance.append(rec)
        logger.info(
            f"  [{i+1}/{len(instances)}] {inst['category']:10s} "
            f"regret={rec['classical_regret']*100:+.2f}% "
            f"{'CO-OPT' if rec['q_is_co_optimal'] else 'SUBOPT'}"
        )

    summary = _summarize(per_instance)
    summary["backend"] = backend_name
    summary["basis_gates"] = basis_gates
    logger.info(
        f"  ARM B → regret={summary['mean_classical_regret']*100:+.2f}% "
        f"co-opt={summary['n_co_optimal']}/{len(instances)} "
        f"identical={summary['n_identical']}/{len(instances)} "
        f"r={summary['mean_sim_correlation_9d']:+.3f}"
    )
    return summary


# =====================================================================
# ARM C — 57-parameter classical MLP baseline
# =====================================================================

def _build_mlp():
    """Parameter-matched MLP: 18 → 3 → 1, no biases → 54 + 3 = 57 params."""
    import torch
    import torch.nn as nn

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(18, 3, bias=False)   # 54 params
            self.fc2 = nn.Linear(3, 1, bias=False)    # 3 params

        def forward(self, x):
            # Tanh keeps the hidden layer comparable in expressive power to RY
            # rotations + entanglement; final sigmoid matches V3's [0,1] output.
            return torch.sigmoid(self.fc2(torch.tanh(self.fc1(x))))

    return TinyMLP()


def train_mlp(train_data, n_epochs=400, lr=5e-2, seed=42):
    """Train the 57-param MLP on the SAME (v1, v2, label) triples used for V3."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    net = _build_mlp()
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params == 57, f"Expected 57 params, got {n_params}"
    logger.info(f"  MLP has {n_params} params (matched to V3-extended)")

    X = torch.tensor(
        np.array([np.concatenate([v1, v2]) for v1, v2, _ in train_data]),
        dtype=torch.float32,
    )
    y = torch.tensor(
        np.array([label for _, _, label in train_data]), dtype=torch.float32
    ).unsqueeze(1)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    for ep in range(n_epochs):
        opt.zero_grad()
        out = net(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 50 == 0:
            logger.info(f"    MLP iter {ep}: loss={loss.item():.4f} best={best_loss:.4f}")

    net.load_state_dict(best_state)
    logger.info(f"  MLP training done: best_loss={best_loss:.4f}")
    return net


def run_arm_c_mlp(instances, train_data, classical_results, seed=42,
                  n_epochs=400, lr=5e-2):
    """Train MLP, evaluate on the same 8 instances, feed through same QUBO pipeline."""
    import torch

    logger.info("[ARM C] 57-parameter classical MLP (parameter-matched baseline)")
    net = train_mlp(train_data, n_epochs=n_epochs, lr=lr, seed=seed)
    net.eval()

    per_instance = []
    with torch.no_grad():
        for i, inst in enumerate(instances):
            similarities = {}
            for pair_idx, (r, o, req, offer, member) in enumerate(inst["eligible_pairs"]):
                v1 = member_embedding_9d(member)
                v2 = task_embedding_9d(req.category)
                x = torch.tensor(
                    np.concatenate([v1, v2]), dtype=torch.float32
                ).unsqueeze(0)
                similarities[pair_idx] = float(net(x).item())

            rec = _evaluate_instance(inst, similarities, classical_results[i])
            rec["instance_idx"] = i
            per_instance.append(rec)

    summary = _summarize(per_instance)
    summary["n_params"] = 57
    summary["training"] = {"n_epochs": n_epochs, "lr": lr, "seed": seed}
    logger.info(
        f"  ARM C → regret={summary['mean_classical_regret']*100:+.2f}% "
        f"co-opt={summary['n_co_optimal']}/{len(instances)} "
        f"identical={summary['n_identical']}/{len(instances)} "
        f"r={summary['mean_sim_correlation_9d']:+.3f}"
    )
    return summary


# =====================================================================
# ARM D — Basis-equivalent classical kernel (Shin, Teo, Jeong, PRR 2024)
# =====================================================================
#
# Ref: Shin, S., Teo, Y. S., & Jeong, H. (2024). Dequantizing quantum
#      machine learning models using tensor networks. Phys. Rev. Research,
#      6, 023218. https://doi.org/10.1103/PhysRevResearch.6.023218
#
# The paper shows that any VQML model f_Q(x; θ) = C^q(θ) · T(x) has a
# classical kernel K_c(x_i, x_j) = <T(x_i) | T(x_j)> whose RKHS strictly
# CONTAINS the RKHS of any quantum kernel built from the same encoding
# (Proposition 3, Eq. 37). The feature map is
#
#   T(x) = ⊗_{α=1..N} (1, cos φ_α(x), sin φ_α(x))^T
#
# so the inner product factorizes as
#
#   K_c(x_i, x_j) = ∏_{α=1..N} [1 + cos(φ_α(x_i) - φ_α(x_j))]
#
# For V3-extended the encoding is φ_α(x) = 0.1 + (π - 0.2) · x_α on a
# concatenated 18-dim input x = [member_9d, task_9d]. We train kernel ridge
# regression on the same 80 (v_mem, v_task, label) pairs used for V3, then
# evaluate per-pair similarity on the same 8 instances. This is a
# theoretically principled classical baseline: by Prop. 3, any function
# V3 can express is also in this kernel's RKHS.

def _phi(x, lo=0.1, hi=np.pi - 0.1):
    """V3 angle encoding: linear map [0, 1] → [0.1, π-0.1]."""
    return lo + (hi - lo) * np.asarray(x, dtype=np.float64)


def _kc_kernel(X1, X2):
    """Basis-equivalent classical kernel K_c(x_i, x_j) on batched inputs.

    X1: (M1, N), X2: (M2, N) — both already mapped via _phi into angle space.
    Returns: (M1, M2) kernel matrix ∏_α [1 + cos(φ_α(x_i) - φ_α(x_j))].
    We compute it in log-space (log1p of each factor, then exp at the end)
    to avoid overflow when factors close to 2 compound across N=18 sites.
    """
    # diff[i, j, α] = φ_α(x_i) - φ_α(x_j)
    diff = X1[:, None, :] - X2[None, :, :]
    factors = 1.0 + np.cos(diff)  # shape (M1, M2, N), each in [0, 2]
    # Product over the feature axis; clip to avoid log(0) if exactly antipodal
    factors = np.clip(factors, 1e-12, 2.0)
    log_k = np.sum(np.log(factors), axis=-1)
    return np.exp(log_k)


def train_kc_kernel_ridge(train_data, lam=1e-2, seed=42):
    """Kernel ridge regression in the basis-equivalent RKHS.

    Follows the setup in Shin-Teo-Jeong Sec. VI (MSE on F-MNIST): KRR with
    regularization λ = 0.01. Labels are {0, 1} (same as V3's BCE target).
    Returns (alpha, X_train_angles, scale) where the prediction for a new
    input x_test is  f(x_test) = K_c(φ(x_test), X_train_angles) · alpha,
    clamped to [0, 1] for downstream compatibility with V3's similarity range.

    Note: train_data already contains angle-encoded vectors in [0.1, π-0.1]
    (output of build_training_set → member_embedding_9d / task_embedding_9d),
    so we consume them directly as X_train_angles — no re-application of φ.
    """
    # v1, v2 in train_data are already scaled angles — use as-is
    X_train_angles = np.array(
        [np.concatenate([v1, v2]) for v1, v2, _ in train_data], dtype=np.float64
    )
    y_train = np.array([label for _, _, label in train_data], dtype=np.float64)

    K = _kc_kernel(X_train_angles, X_train_angles)
    # Normalize by the diagonal to stabilize magnitudes (k_max ~ 2^N = 2^18).
    # This is cosmetic — KRR is invariant to uniform kernel scaling up to λ.
    k_scale = float(np.mean(np.diag(K)))
    K_norm = K / k_scale

    M = K_norm.shape[0]
    alpha = np.linalg.solve(K_norm + lam * np.eye(M), y_train)
    logger.info(f"  K_c kernel: {M} train points, λ={lam}, k_scale={k_scale:.3e}")
    return alpha, X_train_angles, k_scale


def run_arm_d_kernel(instances, train_data, classical_results, lam=1e-2, seed=42):
    """Shin-Teo-Jeong basis-equivalent classical kernel, trained via KRR,
    evaluated on the same 8 instances, feeding the identical QUBO solver."""
    logger.info("[ARM D] Basis-equivalent classical kernel "
                "(Shin-Teo-Jeong, PRR 2024)")
    alpha, X_train_angles, k_scale = train_kc_kernel_ridge(train_data, lam=lam, seed=seed)

    per_instance = []
    for i, inst in enumerate(instances):
        # Use V3's angle-encoded embeddings directly — same feature map V3 sees
        X_test_angles = np.array(
            [np.concatenate([member_embedding_9d(member),
                             task_embedding_9d(req.category)])
             for (r, o, req, offer, member) in inst["eligible_pairs"]],
            dtype=np.float64,
        )

        K_test = _kc_kernel(X_test_angles, X_train_angles) / k_scale
        preds = K_test @ alpha
        preds = np.clip(preds, 0.0, 1.0)  # match V3's [0, 1] similarity range

        similarities = {idx: float(preds[idx]) for idx in range(len(preds))}
        rec = _evaluate_instance(inst, similarities, classical_results[i])
        rec["instance_idx"] = i
        per_instance.append(rec)

    summary = _summarize(per_instance)
    summary["kernel"] = "basis_equivalent_K_c (Shin-Teo-Jeong 2024)"
    summary["ridge_lambda"] = lam
    summary["n_train"] = len(train_data)
    summary["k_scale"] = k_scale
    logger.info(
        f"  ARM D → regret={summary['mean_classical_regret']*100:+.2f}% "
        f"co-opt={summary['n_co_optimal']}/{len(instances)} "
        f"identical={summary['n_identical']}/{len(instances)} "
        f"r={summary['mean_sim_correlation_9d']:+.3f}"
    )
    return summary


# =====================================================================
# Verdict synthesis
# =====================================================================

def _classify(regret, target=0.0061, tol=0.004):
    """Classify regret relative to hardware reference (0.61%, ±0.4%)."""
    if regret < target + tol:
        return "matches_hardware"
    if regret > 0.012:
        return "matches_noiseless_or_worse"
    return "partial"


def synthesize_verdict(hw_ref, arm_a, arm_b, arm_c, arm_d=None):
    r_a = arm_a["mean_classical_regret"]
    r_b = arm_b["mean_classical_regret"]
    r_c = arm_c["mean_classical_regret"]
    r_d = arm_d["mean_classical_regret"] if arm_d else None

    b_status = _classify(r_b)
    c_status = _classify(r_c)
    d_status = _classify(r_d) if arm_d else None

    if b_status == "matches_hardware" and c_status != "matches_hardware":
        verdict = (
            "NOISE-AS-REGULARIZATION confirmed. Noise model alone reproduces the "
            "hardware advantage AND the effect is quantum-specific at parameter "
            "parity (classical MLP cannot match). Strongest extension of the "
            "Q-Manifold hypothesis from representation learning to coordination."
        )
    elif b_status == "matches_hardware" and c_status == "matches_hardware":
        verdict = (
            "NOISE-AS-REGULARIZATION confirmed but NOT quantum-specific. Both the "
            "noisy quantum simulator and a 57-param classical MLP reach hardware "
            "performance. Honest claim: NISQ circuits are competitive with tiny "
            "classical models at parameter parity; regularization mechanism is "
            "real but general."
        )
    elif b_status != "matches_hardware" and c_status != "matches_hardware":
        verdict = (
            "NOISE INSUFFICIENT. The hardware advantage is NOT reproduced by "
            "AerSimulator + ibm_fez NoiseModel. The effect is either structural "
            "(coupling graph, native gate set) or a favorable n=1 draw. "
            "Follow-up: repeat on a second backend with a different coupling "
            "topology, or run more hardware cycles to bound variance."
        )
    else:  # B fails but C matches
        verdict = (
            "NEGATIVE on quantum uniqueness. Classical MLP matches hardware while "
            "noisy simulator does not. The 0.61% on hardware is reachable by a "
            "tiny classical model trained the same way; retreat to simulator-only "
            "claim for the quantum pipeline."
        )

    return {
        "hardware_reference_regret": hw_ref,
        "arm_a_noiseless": r_a,
        "arm_b_noisy_sim": r_b,
        "arm_c_classical_mlp": r_c,
        "arm_d_basis_equivalent_kernel": r_d,
        "arm_b_status": b_status,
        "arm_c_status": c_status,
        "arm_d_status": d_status,
        "verdict": verdict,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=int, default=30)
    parser.add_argument("--train-members", type=int, default=100)
    parser.add_argument("--n-train-per-class", type=int, default=40)
    parser.add_argument("--spsa-iter", type=int, default=150)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--backend", default="ibm_fez",
                        help="Backend whose NoiseModel drives Arm B")
    parser.add_argument("--theta-path", default=None,
                        help="Path to pretrained theta.npy from exp6d. "
                             "If absent, Exp 7 retrains V3-extended in-process.")
    parser.add_argument("--skip-b", action="store_true",
                        help="Skip Arm B (needs IBM Quantum credentials)")
    parser.add_argument("--skip-c", action="store_true")
    parser.add_argument("--skip-d", action="store_true",
                        help="Skip Arm D (Shin-Teo-Jeong basis-equivalent kernel)")
    parser.add_argument("--kc-lambda", type=float, default=1e-2,
                        help="Ridge regularization for Arm D (default 1e-2, "
                             "matching Shin-Teo-Jeong Sec. VI)")
    parser.add_argument("--hw-reference-regret", type=float, default=0.0061,
                        help="Exp 6d hardware reference regret (default 0.61%%)")
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exp7"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "community_economy.json",
    )

    logger.info("=" * 72)
    logger.info("EXPERIMENT 7: Mechanism Isolation")
    logger.info("  Hypothesis: Exp 6d hardware advantage (0.61%% regret vs 1.48%%")
    logger.info("  noiseless) is driven by noise-as-regularization AND is quantum-")
    logger.info("  specific at parameter parity.")
    logger.info("=" * 72)

    # --- SETUP: harvest same 8 instances as Exp 6d ---
    logger.info("\n[SETUP] Harvesting instances (same protocol as Exp 6b/6d)...")
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

    # --- THETA: load or retrain ---
    default_theta_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exp6d", "theta.npy",
    )
    theta_path = args.theta_path or default_theta_path

    if os.path.exists(theta_path):
        theta = np.load(theta_path)
        assert theta.shape == (N_PARAMS,), (
            f"theta shape mismatch: {theta.shape} vs expected ({N_PARAMS},). "
            f"Did Exp 6d change its architecture?"
        )
        logger.info(f"\n[THETA] Loaded from {theta_path}")
    else:
        logger.info(f"\n[THETA] {theta_path} not found → retraining V3-extended")
        train_data_pre = build_training_set(
            config_path, args.train_members, args.n_train_per_class
        )
        t0 = time.time()
        theta, _ = train_v3_extended(
            train_data_pre, n_iter=args.spsa_iter, shots=2048
        )
        logger.info(f"  Retraining took {time.time()-t0:.1f}s")
        np.save(theta_path if args.theta_path else
                os.path.join(args.output, "theta.npy"), theta)

    # Training data for Arm C — must use same pair distribution as V3 training.
    logger.info("\n[TRAIN-DATA] Building (member, task, label) pairs for Arm C...")
    train_data = build_training_set(
        config_path, args.train_members, args.n_train_per_class
    )
    logger.info(f"  {len(train_data)} pairs")

    # --- ARMS ---
    logger.info("")
    arm_a = run_arm_a_noiseless(instances, theta, classical_results, shots=args.shots)

    arm_b = None
    if not args.skip_b:
        try:
            arm_b = run_arm_b_noisy(
                instances, theta, classical_results,
                backend_name=args.backend, shots=args.shots,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  ARM B failed: {exc}")
            import traceback
            traceback.print_exc()

    arm_c = None
    if not args.skip_c:
        try:
            arm_c = run_arm_c_mlp(instances, train_data, classical_results)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  ARM C failed: {exc}")
            import traceback
            traceback.print_exc()

    arm_d = None
    if not args.skip_d:
        try:
            arm_d = run_arm_d_kernel(
                instances, train_data, classical_results, lam=args.kc_lambda,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  ARM D failed: {exc}")
            import traceback
            traceback.print_exc()

    # --- VERDICT ---
    logger.info("\n" + "=" * 72)
    logger.info("VERDICT — mechanism isolation")
    logger.info("=" * 72)
    logger.info(f"  Reference (Exp 6d hardware): regret = {args.hw_reference_regret*100:.2f}%")
    logger.info(f"  ARM A noiseless sim:         regret = {arm_a['mean_classical_regret']*100:+.2f}%")
    if arm_b:
        logger.info(f"  ARM B ibm_fez NoiseModel:    regret = {arm_b['mean_classical_regret']*100:+.2f}%")
    if arm_c:
        logger.info(f"  ARM C 57-param MLP:          regret = {arm_c['mean_classical_regret']*100:+.2f}%")
    if arm_d:
        logger.info(f"  ARM D K_c basis kernel:      regret = {arm_d['mean_classical_regret']*100:+.2f}%")

    verdict = None
    if arm_b and arm_c:
        verdict = synthesize_verdict(args.hw_reference_regret, arm_a, arm_b, arm_c, arm_d)
        logger.info("")
        for line in verdict["verdict"].split(". "):
            line = line.strip()
            if line:
                logger.info(f"  → {line}")

    # --- SAVE ---
    summary = {
        "architecture": "Exp 6d V3-extended (19 qubits, 9D) — same theta, 4 arms",
        "n_instances": len(instances),
        "n_params_v3": N_PARAMS,
        "hardware_reference_regret": args.hw_reference_regret,
        "arm_a_noiseless": arm_a,
        "arm_b_noisy_sim": arm_b,
        "arm_c_classical_mlp": arm_c,
        "arm_d_basis_equivalent_kernel": arm_d,
        "verdict": verdict,
    }
    out_path = os.path.join(args.output, "exp7_mechanism_isolation.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
