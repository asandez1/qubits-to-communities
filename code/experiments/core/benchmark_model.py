"""
QUBO Solver and Evaluation for OrquestIA Coordination Pipeline.

Provides the QUBO Hamiltonian builder, solvers, and solution evaluation.
The domain model (Member, Task, MatchingInstance) lives in core/domain.py.
This module is solver-only — it takes any object with the right attributes
(duck typing) and builds/solves/evaluates the QUBO.

Hamiltonian (7 constraints):

    H = H_obj + λ₁·H_one_member + λ₂·H_task_load + λ₃·H_temporal
               + λ₄·H_skill + λ₅·H_trust + λ₆·H_credit + λ₇·H_effort
               + λ₈·H_energy

Usage:
    from core.domain import MatchingInstance, acceptance_weights
    from core.benchmark_model import QUBOBuilder, solve_exhaustive, evaluate_solution

    instance = MatchingInstance(...)
    weights  = acceptance_weights(instance)
    builder  = QUBOBuilder(instance, weights)
    Q        = builder.build()
    bits, energy = solve_exhaustive(Q, builder.n_vars)
    result   = evaluate_solution(instance, builder, bits)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from itertools import product as iter_product
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Geometry and convenience utilities
# =====================================================================

def _compute_gini(values: List[float]) -> float:
    """Gini coefficient from a list of non-negative values. 0 = equality."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def _loc_xy(loc) -> Tuple[float, float]:
    """Extract (x, y) from a v2 Location object or a bare tuple."""
    if hasattr(loc, 'x'):
        return (loc.x, loc.y)
    return (loc[0], loc[1])


def _distance(loc1, loc2) -> float:
    """Manhattan distance between two locations."""
    x1, y1 = _loc_xy(loc1)
    x2, y2 = _loc_xy(loc2)
    return abs(x1 - x2) + abs(y1 - y2)


def _travel_cost(dist_km: float, cost_per_km: float = 0.5) -> float:
    """Round-trip travel cost in credits.  Default: 0.5 credits/km."""
    return dist_km * cost_per_km * 2


def _convenience(member, task) -> float:
    """How convenient is this task for the member? [0, 1].

    Uses the member's WeeklySchedule if available.
    """
    if hasattr(member, 'schedule') and hasattr(member.schedule, 'convenience'):
        weekday = getattr(task, 'weekday', 0)
        return member.schedule.convenience(weekday, task.time_window)
    return 0.5


def _net_value(member, task, cost_per_km: float = 0.5) -> float:
    """Net credits the provider earns after travel expenses."""
    dist = _distance(member.location, task.location)
    return task.price - _travel_cost(dist, cost_per_km)


# =====================================================================
# QUBO Builder — 7-constraint Hamiltonian
# =====================================================================

class QUBOBuilder:
    """Constructs the QUBO matrix Q for a MatchingInstance.

    H = H_obj + λ₁·H_one_member + λ₂·H_task_load + λ₃·H_temporal
               + λ₄·H_skill + λ₅·H_trust + λ₆·H_credit + λ₇·H_effort
               + λ₈·H_energy

    The Q matrix is a dict {(i, j): coeff} where i <= j.
    Diagonal terms Q[(i,i)] encode linear coefficients.
    Off-diagonal Q[(i,j)] with i<j encode quadratic interactions.
    """

    def __init__(
        self,
        instance,
        weights: Dict[int, float],
        lambda_hard: Optional[float] = None,
        lambda_soft: Optional[float] = None,
        enable_task_load: bool = True,
        enable_temporal: bool = True,
        enable_skill_penalty: bool = True,
        enable_trust_penalty: bool = True,
        enable_credit: bool = True,
        enable_effort: bool = True,
        enable_energy: bool = True,
    ):
        self.instance = instance
        self.weights = weights
        self.n_vars = instance.n_vars

        max_w = max(abs(w) for w in weights.values()) if weights else 1.0
        self.lambda_hard = lambda_hard or 10.0 * max_w
        self.lambda_soft = lambda_soft or 5.0 * max_w

        self.enable_task_load = enable_task_load
        self.enable_temporal = enable_temporal
        self.enable_skill_penalty = enable_skill_penalty
        self.enable_trust_penalty = enable_trust_penalty
        self.enable_credit = enable_credit
        self.enable_effort = enable_effort
        self.enable_energy = enable_energy

        # Pre-compute index lookups
        self._task_to_vars: Dict[int, List[int]] = {}
        self._member_to_vars: Dict[int, List[int]] = {}
        for pair in instance.pairs:
            self._task_to_vars.setdefault(pair.task_idx, []).append(pair.var_idx)
            self._member_to_vars.setdefault(pair.member_idx, []).append(pair.var_idx)

        self._var_to_pair = {p.var_idx: p for p in instance.pairs}

    def build(self) -> Dict[Tuple[int, int], float]:
        Q: Dict[Tuple[int, int], float] = {}
        self._add_objective(Q)
        self._add_one_member_per_task(Q)
        if self.enable_task_load:
            self._add_task_load_limit(Q)
        if self.enable_temporal:
            self._add_temporal_non_overlap(Q)
        if self.enable_skill_penalty:
            self._add_skill_penalty(Q)
        if self.enable_trust_penalty:
            self._add_trust_penalty(Q)
        if self.enable_credit:
            self._add_credit_constraint(Q)
        if self.enable_effort:
            self._add_effort_penalty(Q)
        if self.enable_energy:
            self._add_energy_constraint(Q)
        return Q

    def _add_to_Q(self, Q, i, j, val):
        key = (min(i, j), max(i, j))
        Q[key] = Q.get(key, 0.0) + val

    # --- H_objective: -Σ w_ij · x_ij ---
    def _add_objective(self, Q):
        for var_idx, w in self.weights.items():
            self._add_to_Q(Q, var_idx, var_idx, -w)

    # --- H_one_member: Σ_j (Σ_i x_ij - 1)² ---
    def _add_one_member_per_task(self, Q):
        lam = self.lambda_hard
        for task_idx, var_list in self._task_to_vars.items():
            for v in var_list:
                self._add_to_Q(Q, v, v, -lam)
            for i in range(len(var_list)):
                for j in range(i + 1, len(var_list)):
                    self._add_to_Q(Q, var_list[i], var_list[j], 2 * lam)

    # --- H_task_load: Σ_i max(0, Σ_j x_ij - L_i)² ---
    def _add_task_load_limit(self, Q):
        lam = self.lambda_hard
        for member_idx, var_list in self._member_to_vars.items():
            load = self.instance.members[member_idx].max_task_load
            if len(var_list) <= load:
                continue
            for i in range(len(var_list)):
                for j in range(i + 1, len(var_list)):
                    self._add_to_Q(Q, var_list[i], var_list[j], 2 * lam)

    # --- H_temporal: Σ_i Σ_{(j,k) ∈ T_conflicts} x_ij · x_ik ---
    def _add_temporal_non_overlap(self, Q):
        lam = self.lambda_hard
        inst = self.instance
        for member_idx, var_list in self._member_to_vars.items():
            for i in range(len(var_list)):
                for j in range(i + 1, len(var_list)):
                    pi = self._var_to_pair[var_list[i]]
                    pj = self._var_to_pair[var_list[j]]
                    ti = inst.tasks[pi.task_idx]
                    tj = inst.tasks[pj.task_idx]
                    if ti.time_window.overlaps(tj.time_window):
                        self._add_to_Q(Q, var_list[i], var_list[j], 2 * lam)

    # --- H_skill: Σ_{(i,j): skill < τ_s} x_ij ---
    def _add_skill_penalty(self, Q):
        lam = self.lambda_soft
        for pair in self.instance.pairs:
            m = self.instance.member(pair)
            t = self.instance.task(pair)
            skill = m.skill_levels.get(t.required_skill, 0.0)
            if skill < t.min_skill:
                self._add_to_Q(Q, pair.var_idx, pair.var_idx, lam)

    # --- H_trust: Σ_{(i,j): rep_i < τ_r(j)} x_ij ---
    def _add_trust_penalty(self, Q):
        lam = self.lambda_soft
        for pair in self.instance.pairs:
            m = self.instance.member(pair)
            t = self.instance.task(pair)
            if m.reputation < t.min_trust:
                self._add_to_Q(Q, pair.var_idx, pair.var_idx, lam)

    # --- H_credit: requester can't afford task ---
    def _add_credit_constraint(self, Q):
        lam = self.lambda_soft
        inst = self.instance
        member_by_id = {m.member_id: m for m in inst.members}
        for pair in inst.pairs:
            t = inst.task(pair)
            if t.requester_id and t.requester_id in member_by_id:
                requester = member_by_id[t.requester_id]
                cost = t.price * 1.05
                if requester.credit_balance < cost:
                    self._add_to_Q(Q, pair.var_idx, pair.var_idx, lam)

    # --- H_effort: graduated penalty for inconvenient + low-pay assignments ---
    def _add_effort_penalty(self, Q):
        lam = self.lambda_soft * 0.3
        inst = self.instance
        max_price = max(t.price for t in inst.tasks) or 1.0
        for pair in inst.pairs:
            m = inst.member(pair)
            t = inst.task(pair)
            conv = _convenience(m, t)
            inconvenience = 1.0 - conv
            if inconvenience < 0.1:
                continue
            hourly = t.price / max(t.duration, 0.1)
            pay_factor = 1.0 - min(hourly / max_price, 1.0)
            penalty = inconvenience * (0.3 + 0.7 * pay_factor)
            self._add_to_Q(Q, pair.var_idx, pair.var_idx, lam * penalty)

    # --- H_energy: penalise when total energy cost exceeds member capacity ---
    # For each member, if the sum of energy costs of assigned tasks
    # exceeds their energy, penalise every pair of tasks that together
    # overflow. This is a quadratic penalty similar to task_load but
    # weighted by energy cost instead of count.
    def _add_energy_constraint(self, Q):
        lam = self.lambda_hard
        inst = self.instance
        for member_idx, var_list in self._member_to_vars.items():
            member = inst.members[member_idx]
            energy = getattr(member, 'energy', 999)
            if energy >= 999:
                continue  # no energy tracking
            # Get energy cost per var
            costs = {}
            for v in var_list:
                pair = self._var_to_pair[v]
                task = inst.tasks[pair.task_idx]
                tt = getattr(task, 'task_type', None)
                costs[v] = tt.energy_cost if tt else task.duration
            # Penalise pairs whose combined energy exceeds capacity
            for i in range(len(var_list)):
                for j in range(i + 1, len(var_list)):
                    vi, vj = var_list[i], var_list[j]
                    if costs[vi] + costs[vj] > energy:
                        self._add_to_Q(Q, vi, vj, 2 * lam)

    def active_constraints(self) -> List[str]:
        names = ["one_member_per_task"]
        if self.enable_task_load:
            names.append("task_load_limit")
        if self.enable_temporal:
            names.append("temporal_non_overlap")
        if self.enable_skill_penalty:
            names.append("skill_eligibility")
        if self.enable_trust_penalty:
            names.append("trust_threshold")
        if self.enable_credit:
            names.append("credit_affordability")
        if self.enable_effort:
            names.append("effort_convenience")
        if self.enable_energy:
            names.append("energy_capacity")
        return names


# =====================================================================
# Solvers
# =====================================================================

def solve_exhaustive(Q: Dict[Tuple[int, int], float], n_vars: int,
                     ) -> Tuple[List[int], float]:
    """Exact solver: enumerate all 2^n states. Use for n <= 25."""
    if n_vars > 25:
        raise ValueError(f"Exhaustive search with {n_vars} vars would take "
                         f"2^{n_vars} = {2**n_vars} evaluations. Use SA.")
    best_energy = float("inf")
    best_bits = None
    for bits in iter_product([0, 1], repeat=n_vars):
        e = 0.0
        for (i, j), val in Q.items():
            if i == j:
                e += val * bits[i]
            else:
                e += val * bits[i] * bits[j]
        if e < best_energy:
            best_energy = e
            best_bits = bits
    return list(best_bits), best_energy


def solve_simulated_annealing(
    Q: Dict[Tuple[int, int], float],
    n_vars: int,
    num_reads: int = 1000,
    seed: int = 42,
) -> Tuple[List[int], float]:
    """Simulated annealing solver via dwave-neal. For larger instances."""
    try:
        import neal
    except ImportError:
        raise ImportError("Install dwave-neal: pip install dwave-neal")

    import dimod
    linear = {}
    quadratic = {}
    for (i, j), val in Q.items():
        if i == j:
            linear[i] = linear.get(i, 0.0) + val
        else:
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + val

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads, seed=seed)
    best = response.first
    bits = [int(best.sample.get(i, 0)) for i in range(n_vars)]
    return bits, float(best.energy)


def _evaluate_energy(Q: Dict[Tuple[int, int], float], bits: List[int]) -> float:
    e = 0.0
    for (i, j), val in Q.items():
        if i == j:
            e += val * bits[i]
        else:
            e += val * bits[i] * bits[j]
    return e


def find_all_optima(Q: Dict[Tuple[int, int], float], n_vars: int,
                    tol: float = 1e-6) -> Tuple[List[List[int]], float]:
    """Find ALL optimal solutions (for degeneracy analysis). n <= 25."""
    if n_vars > 25:
        raise ValueError(f"Too many vars ({n_vars}) for exhaustive enumeration")
    best_energy = float("inf")
    all_bits = []
    for bits in iter_product([0, 1], repeat=n_vars):
        e = _evaluate_energy(Q, list(bits))
        if e < best_energy - tol:
            best_energy = e
            all_bits = [list(bits)]
        elif abs(e - best_energy) < tol:
            all_bits.append(list(bits))
    return all_bits, best_energy


# =====================================================================
# Solution evaluation — external quality metrics
# =====================================================================

@dataclass
class SolutionResult:
    """Complete evaluation of a QUBO solution."""
    bits: List[int]
    energy: float
    n_assignments: int

    # External quality
    mean_provider_skill: float
    total_skill: float
    mean_provider_reputation: float
    mean_provider_availability: float

    # Provider economics
    mean_net_value: float
    total_net_value: float
    mean_distance_km: float
    mean_convenience: float
    money_losing_assignments: int

    # Constraint violations
    overloaded_members: int
    temporal_conflicts: int
    skill_violations: int
    trust_violations: int
    total_violations: int

    # Tier A realism metrics
    fulfillment_rate: float = 0.0          # n_assigned / n_tasks
    household_gini: float = 0.0            # Gini of household-mean balances
    mean_energy_deficit: float = 0.0       # avg (capacity - energy) across members

    # Assignment details
    assignments: List[dict] = None

    def to_dict(self):
        return asdict(self)


def evaluate_solution(instance, builder: QUBOBuilder, bits: List[int]) -> SolutionResult:
    """Evaluate a solution with quality metrics and constraint checking."""
    energy = _evaluate_energy(builder.build(), bits)

    assignments = []
    member_task_count: Dict[int, int] = {}
    member_assigned_tasks: Dict[int, List[int]] = {}

    for pair in instance.pairs:
        if bits[pair.var_idx] == 1:
            m = instance.member(pair)
            t = instance.task(pair)
            dist = _distance(m.location, t.location)
            net = _net_value(m, t)
            conv = _convenience(m, t)
            assignments.append({
                "member_id": m.member_id,
                "member_idx": pair.member_idx,
                "task_id": t.task_id,
                "task_idx": pair.task_idx,
                "skill": m.skill_levels.get(t.required_skill, 0.0),
                "reputation": m.reputation,
                "category": t.required_skill,
                "distance_km": dist,
                "net_value": net,
                "convenience": conv,
            })
            member_task_count[pair.member_idx] = (
                member_task_count.get(pair.member_idx, 0) + 1
            )
            member_assigned_tasks.setdefault(pair.member_idx, []).append(
                pair.task_idx
            )

    n_assign = len(assignments)
    skills = [a["skill"] for a in assignments]
    reps = [a["reputation"] for a in assignments]
    avails = [instance.members[a["member_idx"]].time_availability for a in assignments]
    dists = [a["distance_km"] for a in assignments]
    nets = [a["net_value"] for a in assignments]
    convs = [a["convenience"] for a in assignments]

    overloaded = sum(
        1 for m_idx, count in member_task_count.items()
        if count > instance.members[m_idx].max_task_load
    )

    temporal_conflicts = 0
    for m_idx, t_indices in member_assigned_tasks.items():
        for i in range(len(t_indices)):
            for j in range(i + 1, len(t_indices)):
                ti = instance.tasks[t_indices[i]]
                tj = instance.tasks[t_indices[j]]
                if ti.time_window.overlaps(tj.time_window):
                    temporal_conflicts += 1

    skill_violations = sum(
        1 for a in assignments
        if a["skill"] < instance.tasks[a["task_idx"]].min_skill
    )
    trust_violations = sum(
        1 for a in assignments
        if a["reputation"] < instance.tasks[a["task_idx"]].min_trust
    )

    total_v = overloaded + temporal_conflicts + skill_violations + trust_violations

    # --- Tier A metrics ---
    n_tasks = len(instance.tasks)
    fulfillment_rate = n_assign / n_tasks if n_tasks > 0 else 0.0

    # Household Gini: group balances by household_id, take per-household mean,
    # compute Gini over those means. Solo members (empty household_id) count
    # as their own household.
    hh: Dict[str, List[float]] = {}
    for m in instance.members:
        hid = getattr(m, 'household_id', '') or m.member_id
        hh.setdefault(hid, []).append(m.credit_balance)
    hh_means = [float(np.mean(v)) for v in hh.values()]
    hh_gini = _compute_gini(hh_means)

    # Mean energy deficit (how tired members are at end of day)
    deficits = []
    for m in instance.members:
        cap = getattr(m, 'energy_capacity', 0.0)
        en = getattr(m, 'energy', 0.0)
        if cap > 0:
            deficits.append(max(0.0, cap - en))
    mean_energy_deficit = float(np.mean(deficits)) if deficits else 0.0

    return SolutionResult(
        bits=bits, energy=energy, n_assignments=n_assign,
        mean_provider_skill=float(np.mean(skills)) if skills else 0.0,
        total_skill=float(np.sum(skills)),
        mean_provider_reputation=float(np.mean(reps)) if reps else 0.0,
        mean_provider_availability=float(np.mean(avails)) if avails else 0.0,
        mean_net_value=float(np.mean(nets)) if nets else 0.0,
        total_net_value=float(np.sum(nets)),
        mean_distance_km=float(np.mean(dists)) if dists else 0.0,
        mean_convenience=float(np.mean(convs)) if convs else 0.0,
        money_losing_assignments=sum(1 for n in nets if n < 0),
        overloaded_members=overloaded,
        temporal_conflicts=temporal_conflicts,
        skill_violations=skill_violations,
        trust_violations=trust_violations,
        total_violations=total_v,
        fulfillment_rate=fulfillment_rate,
        household_gini=hh_gini,
        mean_energy_deficit=mean_energy_deficit,
        assignments=assignments,
    )


# =====================================================================
# Comparison and validation utilities
# =====================================================================

def compare_methods(
    instance,
    methods: Dict[str, Dict[int, float]],
    solver: str = "exhaustive",
    sa_reads: int = 1000,
) -> Dict[str, SolutionResult]:
    """Run multiple weight functions on the same instance, return results."""
    results = {}
    for name, weights in methods.items():
        builder = QUBOBuilder(instance, weights)
        Q = builder.build()
        if solver == "exhaustive":
            bits, energy = solve_exhaustive(Q, builder.n_vars)
        else:
            bits, energy = solve_simulated_annealing(Q, builder.n_vars,
                                                      num_reads=sa_reads)
        results[name] = evaluate_solution(instance, builder, bits)
    return results


def print_comparison(instance, results: Dict[str, SolutionResult]):
    """Pretty-print a comparison table."""
    print(f"\n  {instance.summary()}")
    print(f"\n  {'Method':<16s} {'Skill':>6s} {'NetVal':>7s} {'Dist':>6s} "
          f"{'Conv':>5s} {'Loss':>5s} {'Viol':>5s}")
    print(f"  {'-'*16} {'-'*6} {'-'*7} {'-'*6} {'-'*5} {'-'*5} {'-'*5}")
    for name, r in results.items():
        print(f"  {name:<16s} {r.mean_provider_skill:6.3f} "
              f"{r.mean_net_value:7.2f} {r.mean_distance_km:6.2f} "
              f"{r.mean_convenience:5.2f} {r.money_losing_assignments:5d} "
              f"{r.total_violations:5d}")

    best_name = max(results, key=lambda n: results[n].total_net_value)
    best = results[best_name]
    if best.assignments:
        print(f"\n  Best net-value ({best_name}) assignments:")
        for a in best.assignments:
            print(f"    {a['member_id']:>10s} -> {a['task_id']:<10s} "
                  f"[{a['category']}] skill={a['skill']:.2f} "
                  f"net={a['net_value']:+.1f} dist={a['distance_km']:.1f}km")


def validate_instance(instance) -> List[str]:
    """Check an instance for common problems. Returns list of warnings."""
    warnings = []

    if instance.n_vars == 0:
        warnings.append("No eligible pairs — all members filtered out by skill threshold")

    tasks_with_providers = set()
    for p in instance.pairs:
        tasks_with_providers.add(p.task_idx)
    for t_idx, task in enumerate(instance.tasks):
        if t_idx not in tasks_with_providers:
            warnings.append(f"Task '{task.task_id}' has no eligible providers")

    members_with_tasks = set()
    for p in instance.pairs:
        members_with_tasks.add(p.member_idx)
    for m_idx, member in enumerate(instance.members):
        if m_idx not in members_with_tasks:
            warnings.append(f"Member '{member.member_id}' is eligible for no tasks")

    n_conflicts = 0
    for m_idx in range(len(instance.members)):
        task_indices = [p.task_idx for p in instance.pairs if p.member_idx == m_idx]
        for i in range(len(task_indices)):
            for j in range(i + 1, len(task_indices)):
                ti = instance.tasks[task_indices[i]]
                tj = instance.tasks[task_indices[j]]
                if ti.time_window.overlaps(tj.time_window):
                    n_conflicts += 1
    if n_conflicts > 0:
        warnings.append(f"{n_conflicts} temporal conflict pairs exist "
                        f"(this is expected — it makes the problem harder)")

    if instance.n_vars > 25:
        warnings.append(f"{instance.n_vars} vars: too large for exhaustive solver, use SA")

    return warnings


# =====================================================================
# Hardware-benchmark formula (paper Section 3.6.1) — v2-native ports
# =====================================================================
# These functions reproduce the exact numerical behavior of the original
# v1 hardware pipeline (`compute_classical_weights` + `build_qubo` from
# `experiments.exp6_hybrid_pipeline`). They operate on v2 MatchingInstance
# objects loaded via `benchmark_fixtures.hardware_tier()`.
#
# Used by: exp6d, exp7, exp7d, exp7e, exp8 (all hardware-path experiments).
# NOT used by: Exp 9 society-model experiments (which use the full
# 7-constraint `QUBOBuilder` + Tier-A-aware weight functions).


def hardware_classical_weights(
    instance,
    matcher_weights: dict,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute the paper Section 3.6.1 composite weight for each eligible pair.

    w_ij = alpha * sim(e_i, e_j) + beta * rep_i + gamma * skill_i + delta * price_j/10

    Where sim(e_i, e_j) is the cosine similarity between the member's skill
    vector (over categories) and a one-hot task vector, with everything else
    taken directly from the v2 Member / Task / MatchingInstance attributes.

    Args
    ----
    instance : MatchingInstance
        v2 instance (typically from `hardware_tier()`).
    matcher_weights : dict
        `{"alpha": ..., "beta": ..., "gamma": ..., "delta": ...}`.
        In the paper's defaults: alpha=0.4, beta=0.2, gamma=0.1, delta=0.3.

    Returns
    -------
    weights : dict[var_idx -> float]  — full composite weight w_ij
    similarities : dict[var_idx -> float]  — just the cosine-sim term
    """
    import numpy as np
    alpha = float(matcher_weights["alpha"])
    beta = float(matcher_weights["beta"])
    gamma = float(matcher_weights["gamma"])
    delta = float(matcher_weights["delta"])
    cats = list(instance.categories)

    weights: dict[int, float] = {}
    similarities: dict[int, float] = {}

    for pair in instance.pairs:
        member = instance.member(pair)
        task = instance.task(pair)
        cat_name = task.required_skill  # SkillCategory.value (str)

        # Cosine similarity: 9-D embedding (6 skills + reputation + time_availability
        # + social_capital). Must match v1 `CommunityMember.get_capability_embedding()`
        # and `get_task_embedding()` exactly — these are what produced the paper's
        # hardware-result numerics.
        mem_vec = np.array(
            [float(member.skill_levels.get(c, 0.0)) for c in cats]
            + [float(member.reputation),
               float(member.time_availability),
               float(member.social_capital)],
            dtype=np.float64,
        )
        # v1 task embedding: one-hot skill + (0.5, 0.5, 0.0) tail for
        # (reputation, time, social) neutral requirements.
        task_vec = np.array(
            [1.0 if c == cat_name else 0.0 for c in cats] + [0.5, 0.5, 0.0],
            dtype=np.float64,
        )
        nm = float(np.linalg.norm(mem_vec))
        nt = float(np.linalg.norm(task_vec))
        sim = float(np.dot(mem_vec, task_vec) / (nm * nt)) if nm > 0 and nt > 0 else 0.0
        similarities[pair.var_idx] = sim

        rep = float(member.reputation)
        skill_level = float(member.skill_levels.get(cat_name, 0.0))
        price_norm = float(instance.prices.get(cat_name, 5.0)) / 10.0

        w = alpha * sim + beta * rep + gamma * skill_level + delta * price_norm
        weights[pair.var_idx] = w

    return weights, similarities


def hardware_qubo(
    instance,
    weights: dict[int, float],
) -> tuple[dict, dict, float]:
    """Build the paper Section 3.6.1 QUBO from v2 instance + weight dict.

    Two hard constraints are enforced via quadratic penalties:
      (C1) One provider per task   (sum_i x_{ij} <= 1 for each task j)
      (C2) One task per provider   (sum_j x_{ij} <= 1 for each member i)

    The other paper-stated constraints (skill eligibility, trust threshold,
    temporal non-overlap) are pre-filtered at instance construction — only
    eligible pairs enter the QUBO. This matches the v1 implementation
    exactly (see `experiments.exp6_hybrid_pipeline.build_qubo`).

    penalty = 10 * max(|w_ij|), i.e. an order of magnitude larger than any
    individual reward so violation is always more expensive than any
    combination of benefits.

    Returns
    -------
    Q : dict[(i, j) -> float]  — QUBO coefficients
    var_map : dict[(member_id, task_id) -> var_idx]  — for assignment decoding
    penalty : float  — the constraint penalty weight actually used
    """
    from collections import defaultdict

    if not weights:
        return {}, {}, 0.0
    max_w = max(abs(w) for w in weights.values())
    penalty = 10.0 * max_w

    Q: dict = {}
    var_map: dict = {}
    member_vars: dict = defaultdict(list)
    task_vars: dict = defaultdict(list)

    for pair in instance.pairs:
        i = pair.var_idx
        member_id = instance.member(pair).member_id
        task_id = instance.task(pair).task_id
        var_map[(member_id, task_id)] = i
        # Reward for assigning (negate to fit energy-minimization):
        Q[(i, i)] = -float(weights[i])
        member_vars[member_id].append(i)
        task_vars[task_id].append(i)

    # C1: one provider per task (penalize pairs sharing a task)
    for vl in task_vars.values():
        for vi in vl:
            Q[(vi, vi)] = Q.get((vi, vi), 0.0) - penalty
        for i in range(len(vl)):
            for j in range(i + 1, len(vl)):
                k = (min(vl[i], vl[j]), max(vl[i], vl[j]))
                Q[k] = Q.get(k, 0.0) + 2.0 * penalty

    # C2: one task per provider (penalize pairs sharing a member)
    for vl in member_vars.values():
        if len(vl) <= 1:
            continue
        for i in range(len(vl)):
            for j in range(i + 1, len(vl)):
                k = (min(vl[i], vl[j]), max(vl[i], vl[j]))
                Q[k] = Q.get(k, 0.0) + 2.0 * penalty

    return Q, var_map, penalty


def hardware_qubo_solve_exhaustive(Q: dict, n_vars: int) -> tuple[list, float]:
    """Exact brute-force solver. Same implementation as v1 `solve_qubo_exhaustive`.

    Enumerates all 2^n bit strings; returns the one with minimum energy.
    Practical up to n ≈ 22 (each benchmark instance has 4–12 vars).
    """
    from itertools import product as iter_product

    best_energy = float("inf")
    best_bits = None
    for bits in iter_product([0, 1], repeat=n_vars):
        e = 0.0
        for (i, j), val in Q.items():
            if i == j:
                e += val * bits[i]
            else:
                e += val * bits[i] * bits[j]
        if e < best_energy:
            best_energy = e
            best_bits = bits
    return list(best_bits), float(best_energy)


def hardware_qubo_analyze_degeneracy(
    Q: dict, n_vars: int, tolerance: float = 1e-6,
) -> tuple[int, float, list[tuple[int, ...]]]:
    """Find all bit-strings within `tolerance` of the minimum energy.

    Used by the hardware-path experiments to detect when multiple assignments
    are co-optimal (degenerate ground states). Matches v1
    `experiments.exp6b_hybrid_replication.analyze_degeneracy` exactly.

    Returns
    -------
    n_optima : int
        Count of bit-strings within `tolerance` of the minimum.
    gap : float
        Energy gap to the next non-optimal bit-string.
    optima : list[tuple[int, ...]]
        All bit-strings achieving the minimum (up to tolerance).
    """
    from itertools import product as iter_product

    energies = []
    for bits in iter_product([0, 1], repeat=n_vars):
        e = 0.0
        for (i, j), val in Q.items():
            if i == j:
                e += val * bits[i]
            else:
                e += val * bits[i] * bits[j]
        energies.append((e, bits))
    energies.sort(key=lambda x: x[0])
    best_e = energies[0][0]
    optima = [bits for e, bits in energies if e - best_e < tolerance]
    # Gap to first non-optimum
    gap = 0.0
    for e, _ in energies:
        if e - best_e >= tolerance:
            gap = e - best_e
            break
    return len(optima), float(gap), optima


def hardware_evaluate_quantum(
    instance,
    classical_weights: dict,
    classical_optima: list,
    quantum_similarities: dict,
    matcher_weights: dict,
) -> dict:
    """Score quantum similarities against a classical reference.

    Reproduces the v1 `_evaluate_instance` protocol exactly:
      1. Build composite v3 weights from quantum similarities.
      2. Solve QUBO with those weights.
      3. Compute pure-reward regret = (obj_c - obj_q) / obj_c where
         obj = sum(classical_weights[k] * bits[k]).
      4. Mark co-optimal iff quantum bits appear in `classical_optima`.

    Args
    ----
    instance : MatchingInstance
    classical_weights : dict[var_idx -> float]
        Paper §3.6.1 formula weights (from `hardware_classical_weights`).
    classical_optima : list[tuple[int]]
        All bit-strings in the degenerate classical ground-state set
        (from `hardware_qubo_analyze_degeneracy`).
    quantum_similarities : dict[var_idx -> float]
        Output of the V3-extended circuit (or any similarity oracle).
    matcher_weights : dict
        {alpha, beta, gamma, delta} — same as used by hardware_classical_weights.

    Returns
    -------
    dict with keys: quantum_bits, classical_regret, q_is_co_optimal, identical,
    n_pairs, similarities.
    """
    n = instance.n_vars
    alpha = float(matcher_weights["alpha"])
    beta = float(matcher_weights["beta"])
    gamma = float(matcher_weights["gamma"])
    delta = float(matcher_weights["delta"])

    # Build v3 composite weights (replace cosine-sim term with quantum sim)
    v3_weights: dict[int, float] = {}
    for pair in instance.pairs:
        member = instance.member(pair)
        task = instance.task(pair)
        cat = task.required_skill
        sim = float(quantum_similarities[pair.var_idx])
        rep = float(member.reputation)
        skill = float(member.skill_levels.get(cat, 0.0))
        price = float(instance.prices.get(cat, 5.0)) / 10.0
        v3_weights[pair.var_idx] = alpha * sim + beta * rep + gamma * skill + delta * price

    Q_q, _, _ = hardware_qubo(instance, v3_weights)
    q_bits, q_energy = hardware_qubo_solve_exhaustive(Q_q, n)

    # Pure-reward regret (NOT energy-based — penalties cancel when bits
    # satisfy the hard constraints, which is always true here by
    # construction because the QUBO's ground state is a valid assignment)
    c_bits = list(classical_optima[0]) if classical_optima else [0] * n
    obj_c = sum(float(classical_weights[k]) * c_bits[k] for k in range(n))
    obj_q = sum(float(classical_weights[k]) * q_bits[k] for k in range(n))
    regret = (obj_c - obj_q) / obj_c if abs(obj_c) > 1e-12 else 0.0

    q_is_co_optimal = tuple(q_bits) in [tuple(b) for b in classical_optima]
    identical = tuple(c_bits) == tuple(q_bits)

    return {
        "n_pairs": n,
        "quantum_bits": q_bits,
        "classical_bits": c_bits,
        "n_classical_optima": len(classical_optima),
        "classical_regret": float(regret),
        "q_is_co_optimal": bool(q_is_co_optimal),
        "identical": bool(identical),
        "similarities": {int(k): float(v) for k, v in quantum_similarities.items()},
    }
