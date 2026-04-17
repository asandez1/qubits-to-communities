"""
OrquestIA Domain Model v2 — Pydantic-based society model.

Design principles (from external evaluation):
  1. Frozen models — QUBO needs immutable snapshots. Mutations create new
     states via model_copy(update={...}).
  2. Integer time — 15-minute blocks (0–96 per day) eliminate floating-point
     comparison bugs.  Block 0 = 00:00, 36 = 09:00, 68 = 17:00, 96 = 24:00.
  3. Sigmoid acceptance — P_accept = 1 / (1 + e^{-k(V_net - V_threshold)})
     grounded in behavioral economics.
  4. Model validators — reject impossible states at construction time.
  5. Enums over strings — StrEnum for categories prevents typos.

All models serialize to JSON via .model_dump() / .model_validate().
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


# =====================================================================
# Enums — finite universe of categories
# =====================================================================

class SkillCategory(StrEnum):
    """The 6 service categories from the OrquestIA paper."""
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    TUTORING = "tutoring"
    TRANSPORT = "transport"
    COOKING = "cooking"
    CHILDCARE = "childcare"


# =====================================================================
# Primitives (frozen value objects)
# =====================================================================

class Location(BaseModel):
    """Geographic position on a grid (km units)."""
    model_config = ConfigDict(frozen=True)

    x: float = 0.0
    y: float = 0.0

    def manhattan_distance(self, other: Location) -> float:
        """Manhattan distance — realistic for grid-based communities."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def euclidean_distance(self, other: Location) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to(self, other: Location) -> float:
        """Default: Manhattan distance."""
        return self.manhattan_distance(other)


# Time is measured in 15-minute blocks.  One day = 96 blocks.
# Block 0 = 00:00, block 36 = 09:00, block 68 = 17:00, block 96 = 24:00.
BLOCKS_PER_HOUR = 4
BLOCKS_PER_DAY = 96


_VEHICLE_RANK = {"none": 0, "bike": 1, "car": 2, "van": 3}


def vehicle_satisfies(have: str, need: str) -> bool:
    """Return True if vehicle `have` meets requirement `need`.

    Hierarchy: van > car > bike > none.  A van satisfies any requirement;
    a bike cannot satisfy a 'car' or 'van' requirement.
    """
    return _VEHICLE_RANK.get(have, 0) >= _VEHICLE_RANK.get(need, 0)


def hours_to_block(hours: float) -> int:
    """Convert fractional hours to the nearest 15-min block."""
    return max(0, min(BLOCKS_PER_DAY, round(hours * BLOCKS_PER_HOUR)))


def block_to_hours(block: int) -> float:
    """Convert a 15-min block back to hours."""
    return block / BLOCKS_PER_HOUR


class TimeWindow(BaseModel):
    """A time interval [start, end) in 15-minute blocks (0–96)."""
    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0, le=BLOCKS_PER_DAY)
    end: int = Field(ge=0, le=BLOCKS_PER_DAY)

    @model_validator(mode="after")
    def _end_after_start(self):
        if self.end <= self.start:
            raise ValueError(
                f"TimeWindow end ({self.end}) must be strictly after "
                f"start ({self.start})"
            )
        return self

    @classmethod
    def from_hours(cls, start_h: float, end_h: float) -> TimeWindow:
        """Convenience: create from fractional hours (e.g. 9.0, 17.0)."""
        return cls(start=hours_to_block(start_h), end=hours_to_block(end_h))

    @property
    def duration_blocks(self) -> int:
        return self.end - self.start

    @property
    def duration_hours(self) -> float:
        return self.duration_blocks / BLOCKS_PER_HOUR

    def overlaps(self, other: TimeWindow) -> bool:
        return self.start < other.end and other.start < self.end

    def overlap_blocks(self, other: TimeWindow) -> int:
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def overlap_fraction(self, other: TimeWindow) -> float:
        """Fraction of self that overlaps with other."""
        if self.duration_blocks == 0:
            return 0.0
        return self.overlap_blocks(other) / self.duration_blocks

    def __repr__(self):
        sh, sm = divmod(self.start * 15, 60)
        eh, em = divmod(self.end * 15, 60)
        return f"TimeWindow({sh:02d}:{sm:02d}–{eh:02d}:{em:02d})"


# =====================================================================
# Schedule system
# =====================================================================

class TimeBlock(BaseModel):
    """A blocked period in a member's day (commitment)."""
    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0, le=BLOCKS_PER_DAY)
    end: int = Field(ge=0, le=BLOCKS_PER_DAY)
    label: str = ""  # "work", "school_pickup", "gym"

    @model_validator(mode="after")
    def _end_after_start(self):
        if self.end <= self.start:
            raise ValueError(f"TimeBlock end ({self.end}) must be after start ({self.start})")
        return self

    @classmethod
    def from_hours(cls, start_h: float, end_h: float, label: str = "") -> TimeBlock:
        return cls(start=hours_to_block(start_h), end=hours_to_block(end_h), label=label)


class DaySchedule(BaseModel):
    """A member's schedule for one day: list of blocked commitments."""
    model_config = ConfigDict(frozen=True)

    commitments: tuple[TimeBlock, ...] = ()

    def is_available(self, start: int, end: int) -> bool:
        """Can the member work from block `start` to block `end`?"""
        for c in self.commitments:
            if start < c.end and c.start < end:
                return False
        return True

    def available_windows(self) -> list[TimeWindow]:
        """Compute free windows by inverting blocked periods."""
        # Sort commitments by start
        sorted_c = sorted(self.commitments, key=lambda c: c.start)
        windows = []
        cursor = 0
        for c in sorted_c:
            if c.start > cursor:
                windows.append(TimeWindow(start=cursor, end=c.start))
            cursor = max(cursor, c.end)
        if cursor < BLOCKS_PER_DAY:
            windows.append(TimeWindow(start=cursor, end=BLOCKS_PER_DAY))
        return windows

    def convenience(self, task_window: TimeWindow) -> float:
        """How convenient is this task window? 1.0 = fully available, 0.0 = fully blocked."""
        if task_window.duration_blocks == 0:
            return 0.0
        available_blocks = 0
        for b in range(task_window.start, task_window.end):
            if self.is_available(b, b + 1):
                available_blocks += 1
        return available_blocks / task_window.duration_blocks

    @classmethod
    def workday_9to5(cls) -> DaySchedule:
        return cls(commitments=(TimeBlock.from_hours(9, 17, "work"),))

    @classmethod
    def free_day(cls) -> DaySchedule:
        return cls(commitments=())


class WeeklySchedule(BaseModel):
    """A member's weekly routine. Days not listed use default_day."""
    model_config = ConfigDict(frozen=True)

    days: dict[int, DaySchedule] = {}  # 0=Mon..6=Sun
    default_day: DaySchedule = Field(default_factory=DaySchedule.free_day)

    def get_day(self, weekday: int) -> DaySchedule:
        return self.days.get(weekday, self.default_day)

    def is_available(self, weekday: int, start: int, end: int) -> bool:
        return self.get_day(weekday).is_available(start, end)

    def convenience(self, weekday: int, task_window: TimeWindow) -> float:
        return self.get_day(weekday).convenience(task_window)

    @classmethod
    def from_preferred_hours(cls, start_h: float, end_h: float) -> WeeklySchedule:
        """Backward-compat: create a schedule where outside preferred hours is blocked."""
        s = hours_to_block(start_h)
        e = hours_to_block(end_h)
        blocks = []
        if s > 0:
            blocks.append(TimeBlock(start=0, end=s, label="unavailable"))
        if e < BLOCKS_PER_DAY:
            blocks.append(TimeBlock(start=e, end=BLOCKS_PER_DAY, label="unavailable"))
        day = DaySchedule(commitments=tuple(blocks))
        return cls(default_day=day)


# =====================================================================
# Task taxonomy
# =====================================================================

class EffortProfile(BaseModel):
    """Linked duration/price/skill for a specific task sub-type."""
    model_config = ConfigDict(frozen=True)

    duration_hours: float = Field(gt=0)
    credit_cost: float = Field(ge=0)       # bonding-curve price
    min_skill: float = Field(ge=0, le=1, default=0.3)
    min_trust: float = Field(ge=0, le=1, default=0.1)
    energy_cost: float = Field(ge=0, default=0.0)  # 0 = auto (duration-based)
    min_age: int = Field(ge=0, default=0)   # 0 = no minimum
    max_age: int = Field(ge=0, default=120) # 120 = no maximum
    required_tools: frozenset[str] = frozenset()  # empty = no tools needed
    required_vehicle: str = "none"               # none | bike | car | van


class TaskType(BaseModel):
    """A specific sub-type within a service category."""
    model_config = ConfigDict(frozen=True)

    name: str                   # "rewire_house", "fix_switch"
    category: str               # SkillCategory value or any string (schema evolution)
    effort: EffortProfile

    @property
    def energy_cost(self) -> float:
        """Effective energy cost. If 0, defaults to duration_hours."""
        e = self.effort.energy_cost
        return e if e > 0 else self.effort.duration_hours

    @classmethod
    def simple(cls, name: str, category: str,
               duration: float, price: float,
               min_skill: float = 0.3, min_trust: float = 0.1,
               energy_cost: float = 0.0,
               min_age: int = 0, max_age: int = 120,
               required_tools: frozenset[str] = frozenset(),
               required_vehicle: str = "none") -> TaskType:
        """Convenience constructor."""
        return cls(
            name=name,
            category=str(category),
            effort=EffortProfile(
                duration_hours=duration, credit_cost=price,
                min_skill=min_skill, min_trust=min_trust,
                energy_cost=energy_cost,
                min_age=min_age, max_age=max_age,
                required_tools=required_tools,
                required_vehicle=required_vehicle,
            ),
        )


class ServiceCategory(BaseModel):
    """A service category with its task sub-types and pricing."""
    model_config = ConfigDict(frozen=True)

    name: str  # SkillCategory value or dynamic
    base_price: float
    steepness: float
    task_types: dict[str, TaskType] = {}


# =====================================================================
# Member
# =====================================================================

class Member(BaseModel):
    """A community member — immutable snapshot.

    Behavioral methods (acceptance_probability, net_value) are pure functions
    of the member's state. To mutate (e.g. after a task), use
    member.model_copy(update={"credit_balance": new_balance}).
    """
    model_config = ConfigDict(frozen=True)

    member_id: str
    archetype: str = "general"
    skill_levels: dict[str, float]  # SkillCategory or any string (schema evolution)
    reputation: float = Field(ge=0, le=1, default=0.5)
    time_availability: float = Field(ge=0, le=1)
    social_capital: float = Field(ge=0, le=1, default=0.0)
    max_task_load: int = Field(ge=1, default=3)
    credit_balance: float = 50.0
    location: Location = Field(default_factory=Location)
    schedule: WeeklySchedule = Field(default_factory=WeeklySchedule)
    age: int = Field(ge=8, le=100, default=35)
    energy: float = Field(ge=0, default=-1.0)  # -1 = use energy_capacity
    household_id: str = ""              # empty = solo household (pool with self only)
    tools: frozenset[str] = frozenset()  # e.g. {"toolkit", "electrical_tools"}
    vehicle: str = "none"                # none | bike | car | van

    @model_validator(mode="after")
    def _validate_skills(self):
        for cat, level in self.skill_levels.items():
            if not 0.0 <= level <= 1.0:
                raise ValueError(
                    f"Skill level for {cat} must be in [0, 1], got {level}"
                )
        # Default energy to capacity if not set
        if self.energy < 0:
            object.__setattr__(self, "energy", self.energy_capacity)
        return self

    # --- Age and energy ---

    @computed_field
    @property
    def energy_capacity(self) -> float:
        """Maximum daily energy in hours, derived from age.

        Age-energy curve (piecewise linear):
          12-18:  4-7 hours  (growing, limited by law/school)
          18-30:  10 hours   (peak capacity)
          30-50:  10-7 hours (gradual decline)
          50-65:  7-4 hours  (accelerated decline)
          65-80:  4-2 hours  (limited endurance)
          80+:    1-2 hours
        """
        a = self.age
        if a < 12:
            return 2.0
        elif a <= 18:
            return 4.0 + (a - 12) * (3.0 / 6.0)    # 4 → 7
        elif a <= 30:
            return 10.0
        elif a <= 50:
            return 10.0 - (a - 30) * (3.0 / 20.0)  # 10 → 7
        elif a <= 65:
            return 7.0 - (a - 50) * (3.0 / 15.0)   # 7 → 4
        elif a <= 80:
            return 4.0 - (a - 65) * (2.0 / 15.0)   # 4 → 2
        else:
            return 1.0

    @computed_field
    @property
    def recovery_rate(self) -> float:
        """Overnight energy recovery rate (fraction of depleted energy restored).

        Younger members recover fully; older members accumulate fatigue.
          18-30: 100% (full reset)
          30-50: 100% -> 85%
          50-65: 85% -> 65%
          65-80: 65% -> 40%
          80+:   30%
        """
        a = self.age
        if a < 18:
            return 1.0
        elif a <= 30:
            return 1.0
        elif a <= 50:
            return 1.0 - (a - 30) * (0.15 / 20.0)   # 1.00 -> 0.85
        elif a <= 65:
            return 0.85 - (a - 50) * (0.20 / 15.0)  # 0.85 -> 0.65
        elif a <= 80:
            return 0.65 - (a - 65) * (0.25 / 15.0)  # 0.65 -> 0.40
        else:
            return 0.30

    def recover_overnight(self) -> Member:
        """Return new Member with energy recovered based on age.

        new_energy = prev + recovery_rate * (capacity - prev), capped at capacity.
        """
        cap = self.energy_capacity
        gap = cap - self.energy
        restored = min(cap, self.energy + self.recovery_rate * gap)
        return self.model_copy(update={"energy": restored})

    def can_do_task(self, task: Task) -> bool:
        """Check if member has enough energy AND meets age requirements."""
        effort = task.task_type.effort
        if self.age < effort.min_age or self.age > effort.max_age:
            return False
        if self.energy < task.task_type.energy_cost:
            return False
        return True

    def after_task(self, task: Task) -> Member:
        """Return new Member with energy depleted by task."""
        cost = task.task_type.energy_cost
        return self.model_copy(update={"energy": max(0.0, self.energy - cost)})

    # --- Backward compatibility ---

    @computed_field
    @property
    def preferred_hours(self) -> Optional[tuple[float, float]]:
        """Derive from schedule — first available window on default day."""
        windows = self.schedule.default_day.available_windows()
        if not windows:
            return None
        w = windows[0]
        return (block_to_hours(w.start), block_to_hours(w.end))

    # --- Behavioral methods ---

    def skill_in(self, category: str) -> float:
        """Get skill level in a category. Returns 0.0 if not listed."""
        return self.skill_levels.get(str(category), 0.0)

    def travel_cost_to(self, loc: Location, cost_per_km: float = 0.5) -> float:
        """Round-trip travel cost in credits, modified by vehicle type.

        Walking (no vehicle) is slow and tiring — effective cost is higher.
        A van is slightly more expensive than a car due to fuel.
        """
        mult = {"none": 2.0, "bike": 1.2, "car": 1.0, "van": 1.1}.get(self.vehicle, 1.0)
        return self.location.distance_to(loc) * cost_per_km * 2 * mult

    def net_value(self, task: Task, cost_per_km: float = 0.5) -> float:
        """Net credits earned: price minus travel cost."""
        return task.price - self.travel_cost_to(task.location, cost_per_km)

    def acceptance_probability(
        self,
        task: Task,
        weekday: int = 0,
        cost_per_km: float = 0.5,
        sensitivity: float = 1.0,
    ) -> float:
        """Probability the provider accepts this task.

        Uses the logistic function:
            P = 1 / (1 + e^{-k * (V_net_hourly - V_threshold)})

        where V_threshold is influenced by credit balance (desperation).
        A member with low credits has a lower threshold and accepts tasks
        they normally wouldn't.
        """
        # Net hourly rate (after travel)
        v_net = self.net_value(task, cost_per_km)
        hourly = v_net / max(task.duration, 0.25)

        # Threshold: baseline 2 cr/hr, lowered by desperation
        # Members with < 10 credits have threshold near 0
        # Members with > 100 credits (demurrage) have threshold near 1
        balance_ratio = min(self.credit_balance / 50.0, 2.0)
        v_threshold = 2.0 * min(balance_ratio, 1.0)

        # Schedule convenience [0, 1]
        conv = self.schedule.convenience(weekday, task.time_window)

        # Requester trust modulation
        req_trust = task.requester_reputation

        # Skill confidence — member is more willing if they're good at it
        skill = self.skill_in(task.required_skill)

        # Energy check — if task exceeds remaining energy, P drops sharply
        energy_cost = task.task_type.energy_cost
        if energy_cost > self.energy:
            return 0.0  # physically can't do it
        energy_ratio = (self.energy - energy_cost) / max(self.energy_capacity, 0.1)
        # energy_ratio near 0 = this task would exhaust me; near 1 = plenty left
        energy_factor = 0.3 + 0.7 * energy_ratio

        # Age eligibility — hard cutoff
        effort = task.task_type.effort
        if self.age < effort.min_age or self.age > effort.max_age:
            return 0.0

        # Tool requirement — hard cutoff
        if effort.required_tools and not effort.required_tools.issubset(self.tools):
            return 0.0

        # Vehicle requirement — hard cutoff
        if effort.required_vehicle != "none":
            if not vehicle_satisfies(self.vehicle, effort.required_vehicle):
                return 0.0

        # Urgency factor — urgent requesters get slightly more willing providers
        urgency_factor = 0.7 + 0.3 * task.urgency  # [0.7, 1.0]

        # Logistic: P = sigmoid(k * (hourly - threshold)) * modulations
        logistic = 1.0 / (1.0 + math.exp(-sensitivity * (hourly - v_threshold)))
        p = (logistic * (0.3 + 0.7 * conv) * (0.4 + 0.6 * req_trust)
             * (0.3 + 0.7 * skill) * energy_factor * urgency_factor)

        return max(0.0, min(1.0, p))

    def get_capability_embedding(self, categories: list[str]) -> np.ndarray:
        """Numeric vector of skill levels in the given category order."""
        return np.array([self.skill_levels.get(SkillCategory(c), 0.0) for c in categories])


# =====================================================================
# Task
# =====================================================================

class Task(BaseModel):
    """A service task — immutable snapshot."""
    model_config = ConfigDict(frozen=True)

    task_id: str
    task_type: TaskType
    time_window: TimeWindow
    location: Location = Field(default_factory=Location)
    requester_id: Optional[str] = None
    requester_reputation: float = Field(ge=0, le=1, default=0.5)
    weekday: int = Field(ge=0, le=6, default=0)  # 0=Mon
    urgency: float = Field(ge=0, le=1, default=0.5)   # priority/urgency [0, 1]
    deadline_block: int = Field(ge=0, le=96, default=96)  # hard deadline (96 = end of day)

    # --- Backward-compat computed fields ---

    @computed_field
    @property
    def required_skill(self) -> str:
        cat = self.task_type.category
        return cat.value if hasattr(cat, 'value') else str(cat)

    @computed_field
    @property
    def min_skill(self) -> float:
        return self.task_type.effort.min_skill

    @computed_field
    @property
    def min_trust(self) -> float:
        return self.task_type.effort.min_trust

    @computed_field
    @property
    def price(self) -> float:
        return self.task_type.effort.credit_cost

    @computed_field
    @property
    def duration(self) -> float:
        return self.task_type.effort.duration_hours


# =====================================================================
# EligiblePair and MatchingInstance
# =====================================================================

class EligiblePair(BaseModel):
    model_config = ConfigDict(frozen=True)

    member_idx: int
    task_idx: int
    var_idx: int


class MatchingInstance(BaseModel):
    """A complete matching problem — immutable snapshot."""
    model_config = ConfigDict(frozen=True)

    instance_id: str
    members: tuple[Member, ...]
    tasks: tuple[Task, ...]
    categories: tuple[str, ...]
    prices: dict[str, float]
    skill_threshold: float = 0.3
    description: str = ""
    pairs: tuple[EligiblePair, ...] = ()

    @model_validator(mode="after")
    def _build_pairs(self):
        """Auto-compute eligible pairs from members and tasks."""
        if self.pairs:
            return self  # already set (e.g. deserialized)
        pairs = []
        var_idx = 0
        for t_idx, task in enumerate(self.tasks):
            effort = task.task_type.effort
            for m_idx, member in enumerate(self.members):
                if task.requester_id and task.requester_id == member.member_id:
                    continue
                skill = member.skill_in(task.required_skill)
                if skill < self.skill_threshold:
                    continue
                # Age eligibility
                if member.age < effort.min_age or member.age > effort.max_age:
                    continue
                # Energy check (skip if task exceeds remaining energy)
                if member.energy < task.task_type.energy_cost:
                    continue
                # Tool requirement — hard pre-filter
                if effort.required_tools and not effort.required_tools.issubset(member.tools):
                    continue
                # Vehicle requirement — hard pre-filter
                if effort.required_vehicle != "none":
                    if not vehicle_satisfies(member.vehicle, effort.required_vehicle):
                        continue
                pairs.append(EligiblePair(
                    member_idx=m_idx, task_idx=t_idx, var_idx=var_idx,
                ))
                var_idx += 1
        # Use object.__setattr__ because model is frozen
        object.__setattr__(self, "pairs", tuple(pairs))
        return self

    @property
    def n_vars(self) -> int:
        return len(self.pairs)

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def member(self, pair: EligiblePair) -> Member:
        return self.members[pair.member_idx]

    def task(self, pair: EligiblePair) -> Task:
        return self.tasks[pair.task_idx]

    def summary(self) -> str:
        cats = sorted(set(t.required_skill for t in self.tasks))
        return (
            f"Instance '{self.instance_id}': "
            f"{self.n_members} members, {self.n_tasks} tasks, "
            f"{self.n_vars} eligible pairs, categories={cats}"
        )


# =====================================================================
# Weight functions for v2 models
# =====================================================================

def acceptance_weights(
    instance: MatchingInstance,
    weekday: int = 0,
    cost_per_km: float = 0.5,
) -> dict[int, float]:
    """Weight = probability the provider would accept the assignment.

    This is the weight function that encodes all nonlinear interactions:
    skill × net_hourly × schedule_convenience × requester_trust × desperation.
    A kernel should learn to approximate this.
    """
    weights = {}
    for pair in instance.pairs:
        m = instance.member(pair)
        t = instance.task(pair)
        weights[pair.var_idx] = m.acceptance_probability(
            t, weekday=weekday, cost_per_km=cost_per_km,
        )
    return weights


def linear_weights(
    instance: MatchingInstance,
    alpha: float = 0.4,
    beta: float = 0.2,
    gamma: float = 0.1,
    delta: float = 0.3,
) -> dict[int, float]:
    """Paper's linear formula for comparison."""
    weights = {}
    cats = list(instance.categories)
    for pair in instance.pairs:
        m = instance.member(pair)
        t = instance.task(pair)

        m_vec = np.array([m.skill_levels.get(str(c), 0.0) for c in cats])
        t_vec = np.array([1.0 if c == t.required_skill else 0.0 for c in cats])
        nm, nt = np.linalg.norm(m_vec), np.linalg.norm(t_vec)
        sim = float(np.dot(m_vec, t_vec) / (nm * nt)) if nm > 0 and nt > 0 else 0.0

        proximity = m.skill_in(t.required_skill)
        price_norm = min(t.price / 10.0, 1.0)

        weights[pair.var_idx] = (
            alpha * sim + beta * m.reputation
            + gamma * proximity + delta * price_norm
        )
    return weights


def skill_only_weights(instance: MatchingInstance) -> dict[int, float]:
    return {
        p.var_idx: instance.member(p).skill_in(instance.task(p).required_skill)
        for p in instance.pairs
    }


def uniform_weights(instance: MatchingInstance) -> dict[int, float]:
    return {p.var_idx: 1.0 for p in instance.pairs}


# =====================================================================
# Archetype profiles — fixture generation
# =====================================================================

class MemberProfile(BaseModel):
    """Archetype template — factory for generating Member instances.

    Skill levels, reputation, balance, and location are sampled from
    distributions defined by the profile. Schedule is a fixed template
    (all members of the same archetype share the same weekly routine).
    """
    model_config = ConfigDict(frozen=True)

    archetype: str
    weight: float = 0.2                    # fraction of population
    skill_categories: list[SkillCategory]  # categories this archetype has skill in
    skill_mean: float = 0.7
    skill_std: float = 0.1
    off_skill: float = 0.05               # skill level in non-primary categories
    reputation_range: tuple[float, float] = (0.4, 0.9)
    time_availability_range: tuple[float, float] = (0.3, 0.8)
    social_capital_range: tuple[float, float] = (0.1, 0.6)
    balance_range: tuple[float, float] = (20.0, 80.0)
    max_task_load: int = 2
    schedule_template: WeeklySchedule = Field(default_factory=WeeklySchedule)
    # Bounding box for location sampling
    location_min: Location = Field(default_factory=lambda: Location(x=0, y=0))
    location_max: Location = Field(default_factory=lambda: Location(x=20, y=20))

    def generate(self, member_id: str, rng: np.random.Generator,
                 all_categories: list[SkillCategory]) -> Member:
        """Sample a concrete Member from this profile."""
        skills = {}
        for cat in all_categories:
            if cat in self.skill_categories:
                raw = rng.normal(self.skill_mean, self.skill_std)
                skills[cat] = float(np.clip(raw, 0.1, 1.0))
            else:
                raw = rng.normal(self.off_skill, 0.03)
                skills[cat] = float(np.clip(raw, 0.0, 0.3))

        rep = float(rng.uniform(*self.reputation_range))
        avail = float(rng.uniform(*self.time_availability_range))
        social = float(rng.uniform(*self.social_capital_range))
        bal = float(rng.uniform(*self.balance_range))
        loc = Location(
            x=float(rng.uniform(self.location_min.x, self.location_max.x)),
            y=float(rng.uniform(self.location_min.y, self.location_max.y)),
        )

        return Member(
            member_id=member_id,
            archetype=self.archetype,
            skill_levels=skills,
            reputation=round(min(max(rep, 0.0), 1.0), 3),
            time_availability=round(min(max(avail, 0.0), 1.0), 3),
            social_capital=round(min(max(social, 0.0), 1.0), 3),
            max_task_load=self.max_task_load,
            credit_balance=round(bal, 1),
            location=loc,
            schedule=self.schedule_template,
        )


class CommunityConfig(BaseModel):
    """Full community configuration — profiles + categories + task types.

    Generate a complete MatchingInstance from config + seed.
    """
    model_config = ConfigDict(frozen=True)

    profiles: dict[str, MemberProfile]
    categories: dict[str, ServiceCategory]

    def generate_members(self, n: int, seed: int = 42) -> tuple[Member, ...]:
        """Generate n members distributed across archetypes by weight."""
        rng = np.random.default_rng(seed)
        all_cats = list(self.categories.keys())
        members = []
        # Distribute n members proportionally
        profile_list = list(self.profiles.items())
        weights = [p.weight for _, p in profile_list]
        total_w = sum(weights)
        counts = [max(1, round(n * w / total_w)) for w in weights]
        # Adjust to hit exactly n
        while sum(counts) > n:
            counts[counts.index(max(counts))] -= 1
        while sum(counts) < n:
            counts[counts.index(min(counts))] += 1

        for (arch_name, profile), count in zip(profile_list, counts):
            for i in range(count):
                mid = f"{arch_name[:3].upper()}{i+1:02d}"
                members.append(profile.generate(mid, rng, all_cats))
        return tuple(members)

    def generate_tasks(
        self,
        n: int,
        seed: int = 42,
        time_slots: list[TimeWindow] | None = None,
        location_bounds: tuple[Location, Location] | None = None,
        requester_ids: list[str | None] | None = None,
    ) -> tuple[Task, ...]:
        """Generate n tasks sampled from available task types."""
        rng = np.random.default_rng(seed + 1000)  # different stream from members
        if time_slots is None:
            time_slots = [
                TimeWindow.from_hours(6, 9),
                TimeWindow.from_hours(8, 12),
                TimeWindow.from_hours(11, 14),
                TimeWindow.from_hours(14, 18),
                TimeWindow.from_hours(17, 21),
            ]
        if location_bounds is None:
            location_bounds = (Location(x=0, y=0), Location(x=20, y=20))
        lo, hi = location_bounds

        # Collect all task types
        all_types = []
        for cat in self.categories.values():
            for tt in cat.task_types.values():
                all_types.append(tt)
        if not all_types:
            return ()

        tasks = []
        for i in range(n):
            tt = all_types[rng.integers(0, len(all_types))]
            slot = time_slots[rng.integers(0, len(time_slots))]
            loc = Location(
                x=float(rng.uniform(lo.x, hi.x)),
                y=float(rng.uniform(lo.y, hi.y)),
            )
            req_id = None
            if requester_ids and i < len(requester_ids):
                req_id = requester_ids[i]
            req_rep = float(rng.uniform(0.2, 0.9))
            weekday = int(rng.integers(0, 7))

            cat_str = tt.category.value if hasattr(tt.category, "value") else str(tt.category)
            tasks.append(Task(
                task_id=f"{cat_str[:4]}_{i+1:02d}",
                task_type=tt,
                time_window=slot,
                location=loc,
                requester_id=req_id,
                requester_reputation=round(req_rep, 2),
                weekday=weekday,
            ))
        return tuple(tasks)
