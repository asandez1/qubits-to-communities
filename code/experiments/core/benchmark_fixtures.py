"""
Benchmark Fixtures v2 — Fixed test instances using the Pydantic domain model.

Tiers:
  S  — Paper's 4x3 worked example (12 vars, 3 constraints).
  M  — "Neighborhood": 12 members, 6 tasks, 3 categories on a 10x10km grid.
       Explicit construction — each member/task encodes a specific trade-off.
  L  — "District": 50 members, 25 tasks, 6 categories on a 20x20km grid.
       Generated from archetype profiles via CommunityConfig.

Usage:
    from core.benchmark_fixtures import tier_s, tier_m, tier_l, all_tiers
    instance = tier_m()
"""

import json
import os

from core.domain import (
    CommunityConfig,
    EligiblePair,
    Location,
    MatchingInstance,
    Member,
    SkillCategory as SC,
    Task,
    TaskType,
    TimeBlock,
    TimeWindow,
    DaySchedule,
    WeeklySchedule,
)


# =====================================================================
# S-TIER: Paper worked example (Section 3.6.1)
# =====================================================================

def tier_s() -> MatchingInstance:
    """Paper's 4x3 worked example.

    T1 and T3 overlap — member C cannot do both.
    Expected optimal: A->T1, B->T2, C->T3.
    """
    elec = TaskType.simple("repair", "electrical", duration=2, price=10)
    tutor = TaskType.simple("lesson", "tutoring", duration=1.5, price=5)
    trans = TaskType.simple("delivery", "transport", duration=1, price=6)

    members = (
        Member(member_id="A",
               skill_levels={SC.ELECTRICAL: 0.92, SC.TUTORING: 0.15, SC.TRANSPORT: 0.78},
               reputation=0.8, time_availability=0.6, social_capital=0.3, max_task_load=2),
        Member(member_id="B",
               skill_levels={SC.ELECTRICAL: 0.30, SC.TUTORING: 0.88, SC.TRANSPORT: 0.22},
               reputation=0.7, time_availability=0.5, social_capital=0.4, max_task_load=2),
        Member(member_id="C",
               skill_levels={SC.ELECTRICAL: 0.85, SC.TUTORING: 0.40, SC.TRANSPORT: 0.90},
               reputation=0.9, time_availability=0.7, social_capital=0.5, max_task_load=2),
        Member(member_id="D",
               skill_levels={SC.ELECTRICAL: 0.10, SC.TUTORING: 0.75, SC.TRANSPORT: 0.60},
               reputation=0.6, time_availability=0.8, social_capital=0.2, max_task_load=2),
    )

    tasks = (
        Task(task_id="T1", task_type=elec, time_window=TimeWindow.from_hours(0, 5)),
        Task(task_id="T2", task_type=tutor, time_window=TimeWindow.from_hours(5, 10)),
        Task(task_id="T3", task_type=trans, time_window=TimeWindow.from_hours(0, 5)),
    )

    return MatchingInstance(
        instance_id="S_paper_example",
        members=members, tasks=tasks,
        categories=("electrical", "tutoring", "transport"),
        prices={"electrical": 10.0, "tutoring": 5.0, "transport": 6.0},
        skill_threshold=0.0,
        description="Paper Section 3.6.1 worked example. T1/T3 overlap. Expected: A->T1, B->T2, C->T3.",
    )


# =====================================================================
# M-TIER: "Neighborhood" — explicit construction on a 10x10km grid
# =====================================================================

def tier_m() -> MatchingInstance:
    """Medium instance: distance, schedules, and economics all interact.

    12 members, 6 tasks, 3 categories. Each member has a specific
    schedule (work 9-5, early bird, night owl, etc.), location, and
    credit balance that creates genuine trade-offs the weight function
    must resolve.
    """
    # Task types — electricians need toolkit + electrical_tools; cooks need kitchen;
    # catering requires a car/van to haul food. Tutoring requires no tools.
    elec_repair = TaskType.simple("repair", "electrical", duration=2, price=12, min_skill=0.4,
                                   required_tools=frozenset({"toolkit", "electrical_tools"}))
    elec_install = TaskType.simple("install", "electrical", duration=3, price=8, min_skill=0.5,
                                    min_trust=0.5,
                                    required_tools=frozenset({"toolkit", "electrical_tools"}))
    tutor_cheap = TaskType.simple("basic_lesson", "tutoring", duration=2, price=2, min_skill=0.4)
    tutor_premium = TaskType.simple("exam_prep", "tutoring", duration=2, price=9, min_skill=0.5,
                                     min_trust=0.5)
    cook_catering = TaskType.simple("catering", "cooking", duration=2.5, price=10, min_skill=0.4,
                                     required_tools=frozenset({"kitchen"}),
                                     required_vehicle="car")
    cook_cheap = TaskType.simple("meal_prep", "cooking", duration=1.5, price=3, min_skill=0.4,
                                  required_tools=frozenset({"kitchen"}))

    # Schedules
    morning_worker = WeeklySchedule.from_preferred_hours(6, 12)
    afternoon_worker = WeeklySchedule.from_preferred_hours(12, 18)
    evening_worker = WeeklySchedule.from_preferred_hours(16, 22)
    nine_to_five = WeeklySchedule(default_day=DaySchedule(commitments=(
        TimeBlock.from_hours(9, 17, "work"),)))
    flexible = WeeklySchedule()  # all day available

    # Household groupings (Tier A realism):
    #   hh_e1    : E1 (senior) + E2 (apprentice)            — master/apprentice
    #   hh_e3    : E3 + Co3                                  — father/son multi-gen
    #   hh_tu1   : Tu1 (retiree, solo)
    #   hh_fam1  : Tu2 + Co1                                 — working couple
    #   hh_fam2  : Tu4 + G2                                  — couple with kids
    #   hh_teen  : Tu3 (teen, lives "alone" for simplicity)
    #   hh_co2   : Co2 (solo)
    #   hh_g1    : G1 (solo)
    ELEC = frozenset({"toolkit", "electrical_tools"})
    KITCHEN = frozenset({"kitchen", "baking_supplies"})
    TOOLKIT = frozenset({"toolkit"})

    members = (
        # --- Electricians ---
        Member(member_id="E1", household_id="hh_e1",      # senior master, van
               tools=ELEC, vehicle="van",
               skill_levels={SC.ELECTRICAL: 0.92, SC.TUTORING: 0.15, SC.COOKING: 0.10},
               reputation=0.88, time_availability=0.5, social_capital=0.6,
               max_task_load=2, credit_balance=130.0, age=62,
               location=Location(x=1, y=9), schedule=morning_worker),
        Member(member_id="E2", household_id="hh_e1",      # young apprentice, bike
               tools=ELEC, vehicle="bike",
               skill_levels={SC.ELECTRICAL: 0.80, SC.TUTORING: 0.25, SC.COOKING: 0.20},
               reputation=0.50, time_availability=0.6, social_capital=0.3,
               max_task_load=2, credit_balance=8.0, age=26,
               location=Location(x=8, y=8), schedule=evening_worker),
        Member(member_id="E3", household_id="hh_e3",      # mid-career father, van
               tools=ELEC, vehicle="van",
               skill_levels={SC.ELECTRICAL: 0.70, SC.TUTORING: 0.40, SC.COOKING: 0.35},
               reputation=0.65, time_availability=0.7, social_capital=0.4,
               max_task_load=2, credit_balance=45.0, age=42,
               location=Location(x=5, y=5), schedule=nine_to_five),

        # --- Tutors ---
        Member(member_id="Tu1", household_id="hh_tu1",    # retiree solo, car
               tools=frozenset(), vehicle="car",
               skill_levels={SC.ELECTRICAL: 0.10, SC.TUTORING: 0.90, SC.COOKING: 0.15},
               reputation=0.85, time_availability=0.6, social_capital=0.7,
               max_task_load=2, credit_balance=70.0, age=68,
               location=Location(x=2, y=2), schedule=afternoon_worker),
        Member(member_id="Tu2", household_id="hh_fam1",   # young mom (partner Co1), bike
               tools=frozenset(), vehicle="bike",
               skill_levels={SC.ELECTRICAL: 0.20, SC.TUTORING: 0.85, SC.COOKING: 0.30},
               reputation=0.75, time_availability=0.7, social_capital=0.5,
               max_task_load=2, credit_balance=35.0, age=28,
               location=Location(x=9, y=3), schedule=morning_worker),
        Member(member_id="Tu3", household_id="hh_teen",   # teenager, no vehicle
               tools=frozenset(), vehicle="none",
               skill_levels={SC.ELECTRICAL: 0.15, SC.TUTORING: 0.75, SC.COOKING: 0.40},
               reputation=0.40, time_availability=0.8, social_capital=0.2,
               max_task_load=2, credit_balance=3.0, age=17,
               location=Location(x=5, y=1), schedule=evening_worker),
        Member(member_id="Tu4", household_id="hh_fam2",   # parent (partner G2), car
               tools=frozenset(), vehicle="car",
               skill_levels={SC.ELECTRICAL: 0.08, SC.TUTORING: 0.68, SC.COOKING: 0.25},
               reputation=0.30, time_availability=0.9, social_capital=0.1,
               max_task_load=2, credit_balance=12.0, age=38,
               location=Location(x=7, y=7), schedule=flexible),

        # --- Cooks ---
        Member(member_id="Co1", household_id="hh_fam1",   # senior chef (partner Tu2), car
               tools=KITCHEN, vehicle="car",
               skill_levels={SC.ELECTRICAL: 0.05, SC.TUTORING: 0.20, SC.COOKING: 0.88},
               reputation=0.82, time_availability=0.5, social_capital=0.6,
               max_task_load=2, credit_balance=90.0, age=58,
               location=Location(x=3, y=1), schedule=morning_worker),
        Member(member_id="Co2", household_id="hh_co2",    # solo cook, bike
               tools=KITCHEN, vehicle="bike",
               skill_levels={SC.ELECTRICAL: 0.10, SC.TUTORING: 0.15, SC.COOKING: 0.80},
               reputation=0.60, time_availability=0.6, social_capital=0.4,
               max_task_load=2, credit_balance=25.0, age=44,
               location=Location(x=8, y=2), schedule=afternoon_worker),
        Member(member_id="Co3", household_id="hh_e3",     # young son (father E3), bike
               tools=KITCHEN, vehicle="bike",
               skill_levels={SC.ELECTRICAL: 0.08, SC.TUTORING: 0.12, SC.COOKING: 0.72},
               reputation=0.35, time_availability=0.8, social_capital=0.2,
               max_task_load=2, credit_balance=5.0, age=23,
               location=Location(x=5, y=9), schedule=evening_worker),

        # --- Generalists ---
        Member(member_id="G1", household_id="hh_g1",      # solo generalist, car
               tools=TOOLKIT, vehicle="car",
               skill_levels={SC.ELECTRICAL: 0.55, SC.TUTORING: 0.50, SC.COOKING: 0.52},
               reputation=0.70, time_availability=0.6, social_capital=0.5,
               max_task_load=2, credit_balance=50.0, age=35,
               location=Location(x=4, y=5), schedule=flexible),
        Member(member_id="G2", household_id="hh_fam2",    # generalist (partner Tu4), car
               tools=TOOLKIT, vehicle="car",
               skill_levels={SC.ELECTRICAL: 0.45, SC.TUTORING: 0.55, SC.COOKING: 0.48},
               reputation=0.55, time_availability=0.7, social_capital=0.3,
               max_task_load=2, credit_balance=18.0, age=48,
               location=Location(x=6, y=6), schedule=nine_to_five),
    )

    # --- 6 Tasks with specific trade-offs ---
    tasks = (
        # elec_1: NW corner, URGENT (leak risk), high price
        Task(task_id="elec_1", task_type=elec_repair,
             time_window=TimeWindow.from_hours(7, 9),
             location=Location(x=2, y=9), requester_reputation=0.80,
             urgency=0.9),
        # elec_2: SE corner, afternoon, moderate — E2 close, E1 far, needs trust
        Task(task_id="elec_2", task_type=elec_install,
             time_window=TimeWindow.from_hours(14, 17),
             location=Location(x=7, y=3), requester_reputation=0.70),
        # tutor_1: SE, morning, TERRIBLE deal (2cr/2h, bad requester)
        Task(task_id="tutor_1", task_type=tutor_cheap,
             time_window=TimeWindow.from_hours(8, 10),
             location=Location(x=9, y=2), requester_reputation=0.25),
        # tutor_2: SW, afternoon, GREAT deal (9cr/2h, good requester, needs trust)
        Task(task_id="tutor_2", task_type=tutor_premium,
             time_window=TimeWindow.from_hours(13, 15),
             location=Location(x=3, y=3), requester_reputation=0.85),
        # cook_1: N center, midday, URGENT catering for event, requires car
        Task(task_id="cook_1", task_type=cook_catering,
             time_window=TimeWindow.from_hours(11, 14),
             location=Location(x=4, y=9), requester_reputation=0.78,
             urgency=0.85),
        # cook_2: SE, evening, cheap bad deal
        Task(task_id="cook_2", task_type=cook_cheap,
             time_window=TimeWindow.from_hours(18, 20),
             location=Location(x=7, y=1), requester_reputation=0.30),
    )

    return MatchingInstance(
        instance_id="M_neighborhood",
        members=members, tasks=tasks,
        categories=("electrical", "tutoring", "cooking"),
        prices={"electrical": 10.0, "tutoring": 5.0, "cooking": 4.0},
        skill_threshold=0.4,
        description=(
            "Neighborhood: 12 members, 6 tasks, 3 categories on 10x10km grid. "
            "Schedules (9-5 workers, morning/afternoon/evening preferences), "
            "distance costs, credit balances 3-130cr, prices 2-12cr. "
            "Weight function determines assignments."
        ),
    )


# =====================================================================
# L-TIER: "District" — generated from CommunityConfig
# =====================================================================

def tier_l(seed: int = 42) -> MatchingInstance:
    """Large instance: 50 members, 25 tasks generated from archetype profiles.

    Uses CommunityConfig to generate members and tasks deterministically
    from the v2 JSON config. Same seed = same instance.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "community_economy_v2.json",
    )
    with open(config_path) as f:
        raw = json.load(f)
    config = CommunityConfig.model_validate(raw)

    members = config.generate_members(50, seed=seed)
    tasks = config.generate_tasks(25, seed=seed)

    all_cats = tuple(c.value for c in SC)
    def _cat_str(c):
        name = c.name
        return name.value if hasattr(name, "value") else str(name)
    prices = {_cat_str(c): c.base_price for c in config.categories.values()}

    return MatchingInstance(
        instance_id="L_district",
        members=members, tasks=tasks,
        categories=all_cats,
        prices=prices,
        skill_threshold=0.3,
        description=(
            f"District: 50 members (5 archetypes), 25 tasks, 6 categories "
            f"on 20x20km grid. Generated from community_economy_v2.json "
            f"seed={seed}. Schedules include 9-5 workers, school pickups, "
            f"free retirees, flexible gig workers."
        ),
    )


# =====================================================================
# Hardware-benchmark tier (frozen 8 instances from Exp 6d)
# =====================================================================

def hardware_tier(
    snapshot_path: str = None,
) -> tuple[list[MatchingInstance], list[dict]]:
    """Load the 8 frozen matching instances used by all hardware experiments.

    These are the exact 8 instances that produced the paper's §4/§5 hardware
    results (ibm_fez 0.61%, ibm_marrakesh 1.48%, ibm_kingston 0.61%). The
    snapshot was originally extracted from the v1 simulation engine at
    `community_economy.json`, seed=42, cycles [50,100,150,...,499] — that
    extraction is now frozen as a JSON file, so these instances are
    deterministic and require no simulation-engine dependency at load time.

    Returns
    -------
    instances : list[MatchingInstance]
        8 v2 MatchingInstance objects with `pairs` pre-populated to match
        the original v1 eligible-pair layout byte-for-byte.
    metadata : list[dict]
        Per-instance metadata: {"category_label", "cycle", "matcher_weights",
        "pair_refs"}. `pair_refs` gives each pair's v1 (req_idx, offer_idx)
        assignment used by the paper's Section 3.6.1 QUBO construction; see
        `paper_classical_weights` in `core.benchmark_model`.

    Notes
    -----
    - The 9 quantum-relevant Member attributes (6 skills + reputation +
      time_availability + social_capital) are preserved with full fidelity.
    - location/schedule/age/tools/vehicle are filled with defaults; they
      are not accessed by the hardware-path quantum circuit or by the
      paper's Section 3.6.1 weight formula.
    - The snapshot is version-controlled at
      `results/exp6d/hardware_instances_v2.json`.
    """
    if snapshot_path is None:
        snapshot_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "exp6d", "hardware_instances_v2.json",
        )
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(
            f"Hardware snapshot not found at {snapshot_path}. "
            "Run `python experiments/scripts/snapshot_hardware_instances.py` "
            "to regenerate it from the v1 simulation."
        )
    with open(snapshot_path) as f:
        snap = json.load(f)

    instances = []
    metadata = []
    for inst_blob in snap["instances"]:
        # Reconstruct v2 Member and Task tuples, preserving order
        members = tuple(
            Member.model_validate(m) for m in inst_blob["members"]
        )
        tasks = tuple(
            Task.model_validate(t) for t in inst_blob["tasks"]
        )

        # Build member_id → idx and task_id → idx maps for pair resolution
        mid_to_idx = {m.member_id: i for i, m in enumerate(members)}
        tid_to_idx = {t.task_id: i for i, t in enumerate(tasks)}

        # Build v2 EligiblePair list from the v1 pair_refs (guaranteed
        # to match the v1 variable indexing exactly).
        pairs = []
        for var_idx, ref in enumerate(inst_blob["eligible_pair_refs"]):
            pairs.append(EligiblePair(
                member_idx=mid_to_idx[ref["member_id"]],
                task_idx=tid_to_idx[ref["task_id"]],
                var_idx=var_idx,
            ))

        # Build MatchingInstance with pairs pre-populated — the
        # `@model_validator` respects the explicit pairs kwarg and skips the
        # auto-filter (see domain.py line 604: `if self.pairs: return self`).
        inst = MatchingInstance(
            instance_id=inst_blob["instance_id"],
            members=members,
            tasks=tasks,
            categories=tuple(inst_blob["categories"]),
            prices=dict(inst_blob["prices"]),
            skill_threshold=0.3,
            description=f"{inst_blob['category_label']} @ cycle {inst_blob['cycle']}",
            pairs=tuple(pairs),
        )
        instances.append(inst)
        metadata.append({
            "category_label": inst_blob["category_label"],
            "cycle": inst_blob["cycle"],
            "matcher_weights": inst_blob["matcher_weights"],
            "pair_refs": inst_blob["eligible_pair_refs"],
        })

    return instances, metadata


def hardware_instances_v1_compat(snapshot_path: str = None) -> list[dict]:
    """Return the 8 hardware benchmark instances as v1-compatible dicts.

    This is a backward-compatibility bridge: existing v1 experiment scripts
    (exp6d_v3_extended, exp7_mechanism_isolation, exp7d_multi_seed_theta,
    exp7e_classical_baselines, exp8_kernel_qubo) call `harvest_instances()`
    and expect a list of dicts with keys `eligible_pairs`, `categories`,
    `prices`, `matcher`, etc.

    This function produces the same shape, but the underlying member/task
    objects are **lightweight proxies** that expose just the v1 attributes
    (skill_levels, reputation, time_availability, social_capital, category,
    requester_id) needed by the v1 weight formula — no v1 simulation engine,
    no community_economy.json dependency at load time.

    Replacing `from experiments.exp6b_hybrid_replication import harvest_instances`
    with `from core.benchmark_fixtures import hardware_instances_v1_compat as harvest_instances`
    in any v1 script makes it consume the frozen snapshot instead.
    """
    from types import SimpleNamespace

    instances, metadata = hardware_tier(snapshot_path=snapshot_path)
    result = []

    # Build a lightweight "matcher" proxy with the same _weights interface
    # and _compute_weight method that v1 scripts call.
    class _MatcherProxy:
        def __init__(self, matcher_weights: dict):
            self._weights = dict(matcher_weights)

        def _compute_weight(self, member, request, prices, all_categories):
            import numpy as np
            alpha = self._weights["alpha"]
            beta = self._weights["beta"]
            gamma = self._weights["gamma"]
            delta = self._weights["delta"]
            mem_emb = member.get_capability_embedding(all_categories)
            task_emb = _compat_task_embedding(request.category, all_categories)
            nm = float(np.linalg.norm(mem_emb))
            nt = float(np.linalg.norm(task_emb))
            sim = float(np.dot(mem_emb, task_emb) / (nm * nt)) if nm > 0 and nt > 0 else 0.0
            rep = member.reputation
            skill = member.skill_levels.get(request.category, 0.0)
            price = prices.get(request.category, 5.0) / 10.0
            return alpha * sim + beta * rep + gamma * skill + delta * price

    for inst, meta in zip(instances, metadata):
        # Wrap each v2 Member into a v1-compat proxy with get_capability_embedding
        member_proxies = {}
        for m in inst.members:
            member_proxies[m.member_id] = _MemberProxy(m)

        # Build v1-shape eligible_pairs: list of (r_idx, o_idx, req, offer, member)
        eligible_pairs = []
        for pair in inst.pairs:
            m = inst.member(pair)
            t = inst.task(pair)
            req = SimpleNamespace(
                request_id=t.task_id,
                requester_id=t.requester_id,
                category=t.required_skill,
                cycle=meta["cycle"],
                assigned_to=None,
                fulfilled=False,
            )
            offer = SimpleNamespace(
                offer_id=f"offer_{m.member_id}_{t.required_skill}",
                provider_id=m.member_id,
                category=t.required_skill,
                skill_level=m.skill_levels.get(t.required_skill, 0.0),
                cycle=meta["cycle"],
            )
            # Find original v1 (req_idx, offer_idx) from the snapshot
            ref = meta["pair_refs"][pair.var_idx]
            eligible_pairs.append((
                ref["req_idx"], ref["offer_idx"],
                req, offer, member_proxies[m.member_id],
            ))

        result.append({
            "engine": None,  # no simulation engine dependency
            "category": meta["category_label"],
            "cycle": meta["cycle"],
            "price": inst.prices.get(meta["category_label"], 5.0),
            "categories": list(inst.categories),
            "prices": dict(inst.prices),
            "requests": [],   # populated by downstream code only if needed
            "offers": [],
            "eligible_pairs": eligible_pairs,
            "matcher": _MatcherProxy(meta["matcher_weights"]),
        })

    return result


def _compat_task_embedding(category: str, all_categories: list) -> list:
    """v1-compat task embedding used by the matcher proxy."""
    import numpy as np
    emb = [1.0 if c == category else 0.0 for c in all_categories]
    emb += [0.5, 0.5, 0.0]  # reputation / time / social neutrals
    return np.array(emb, dtype=np.float64)


class _MemberProxy:
    """Wraps a v2 Member with v1-style attribute access.

    Exposes: member_id, skill_levels (dict[str, float]), reputation,
    time_availability, social_capital, get_capability_embedding(cats).
    """
    def __init__(self, v2_member):
        self._m = v2_member
        self.member_id = v2_member.member_id
        # Convert SkillCategory keys to strings (v1 uses str keys)
        self.skill_levels = {
            (k.value if hasattr(k, "value") else str(k)): float(v)
            for k, v in v2_member.skill_levels.items()
        }
        self.reputation = float(v2_member.reputation)
        self.time_availability = float(v2_member.time_availability)
        self.social_capital = float(v2_member.social_capital)
        self.archetype = v2_member.archetype
        self.credit_balance = float(v2_member.credit_balance)
        self.max_task_load = int(v2_member.max_task_load)

    def get_capability_embedding(self, all_categories: list):
        import numpy as np
        emb = [self.skill_levels.get(c, 0.0) for c in all_categories]
        emb += [self.reputation, self.time_availability, self.social_capital]
        return np.array(emb, dtype=np.float64)


# =====================================================================
# Convenience
# =====================================================================

def all_tiers():
    return [("S", tier_s()), ("M", tier_m()), ("L", tier_l())]


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from core.benchmark_model import (
        QUBOBuilder, solve_exhaustive, solve_simulated_annealing,
        evaluate_solution, compare_methods, print_comparison,
        validate_instance, find_all_optima,
    )
    from core.domain import (
        acceptance_weights as v2_acceptance,
        linear_weights as v2_linear,
        skill_only_weights as v2_skill,
        uniform_weights as v2_uniform,
    )

    print("=" * 72)
    print("BENCHMARK FIXTURES v2 — Validation")
    print("=" * 72)

    for tier_name, inst in all_tiers():
        print(f"\n{'─' * 72}")
        print(f"  TIER {tier_name}: {inst.summary()}")
        print(f"  {inst.description}")

        warnings = validate_instance(inst)
        if warnings:
            for w in warnings:
                print(f"  WARNING: {w}")

        # Build weight functions (v2 domain-native)
        methods = {
            "linear": v2_linear(inst),
            "skill_only": v2_skill(inst),
            "acceptance": v2_acceptance(inst),
            "uniform": v2_uniform(inst),
        }

        if inst.n_vars <= 25:
            results = compare_methods(inst, methods, solver="exhaustive")
            builder = QUBOBuilder(inst, methods["linear"])
            optima, opt_e = find_all_optima(builder.build(), builder.n_vars)
            print(f"\n  Linear optimum: {len(optima)} solution(s) at energy {opt_e:.3f}")
        else:
            results = compare_methods(inst, methods, solver="sa", sa_reads=2000)
            print(f"\n  (solved via SA, 2000 reads)")

        print_comparison(inst, results)

    # Save fixtures
    fixture_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
    os.makedirs(fixture_dir, exist_ok=True)
    for tier_name, inst in all_tiers():
        path = os.path.join(fixture_dir, f"tier_{tier_name.lower()}_v2.json")
        with open(path, "w") as f:
            f.write(inst.model_dump_json(indent=2))
        print(f"\nSaved: {path}")

    print("\nAll fixtures validated.")
