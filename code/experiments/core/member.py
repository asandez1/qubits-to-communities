"""Community member model with 7-axis capability profiles and archetype-driven behavior."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np


class MemberArchetype(Enum):
    MULTI_SKILLED_PROFESSIONAL = "multi_skilled_professional"
    STAY_AT_HOME_PARENT = "stay_at_home_parent"
    GIG_WORKER = "gig_worker"
    RETIREE = "retiree"
    ADVERSARIAL_EXTRACTOR = "adversarial_extractor"


@dataclass
class ServiceRequest:
    """A demand request from a member for a service category."""
    request_id: str
    requester_id: str
    category: str
    cycle: int
    fulfilled: bool = False
    assigned_to: Optional[str] = None


@dataclass
class ServiceOffer:
    """A supply offer from a member in a service category."""
    offer_id: str
    provider_id: str
    category: str
    cycle: int
    skill_level: float = 1.0


@dataclass
class Assignment:
    """Result of matching a request to a provider."""
    request: ServiceRequest
    provider_id: str
    category: str
    price: float
    weight: float  # composite quality score w_ij
    cycle: int


class CommunityMember:
    """
    A simulated community member with a 7-axis capability profile.

    Axes: professional skills, informal skills, physical assets,
    knowledge, time availability, network engagement, social capital.
    """

    def __init__(
        self,
        member_id: str,
        archetype: MemberArchetype,
        skills: List[str],
        assets: List[str],
        time_availability: float,
        strategy: str,
        bridge_propensity: float,
        demand_categories: List[str],
        demand_rate: float,
        rng: np.random.Generator,
    ):
        self.member_id = member_id
        self.archetype = archetype
        self.skills = skills
        self.assets = assets
        self.time_availability = time_availability
        self.strategy = strategy
        self.bridge_propensity = bridge_propensity
        self.demand_categories = demand_categories
        self.demand_rate = demand_rate
        self._rng = rng

        # 7-axis profile
        self.skill_levels: Dict[str, float] = {
            s: rng.uniform(0.5, 1.0) for s in skills
        }
        self.reputation: float = 0.5  # starts at neutral
        self.credit_balance: float = 0.0
        self.social_capital: float = 0.0
        self.network_engagement: float = 0.0

        # Tracking
        self.total_provided: int = 0
        self.total_consumed: int = 0
        self.total_earned: float = 0.0
        self.total_spent: float = 0.0
        self.last_transaction_cycle: int = 0
        self.active: bool = True
        self.joined_cycle: int = 0

        # Current cycle state
        self.current_tasks: int = 0
        self.max_task_load: int = max(1, int(time_availability * 3))

    def generate_requests(self, cycle: int, all_categories: List[str]) -> List[ServiceRequest]:
        """Generate demand requests based on archetype demand distribution."""
        requests = []
        if not self.active:
            return requests

        # Probability of generating a request this cycle
        if self._rng.random() < self.demand_rate:
            # Pick a category from demand preferences
            if self.demand_categories:
                cat = self._rng.choice(self.demand_categories)
            else:
                cat = self._rng.choice(all_categories)
            req = ServiceRequest(
                request_id=f"req_{self.member_id}_{cycle}_{uuid.uuid4().hex[:6]}",
                requester_id=self.member_id,
                category=cat,
                cycle=cycle,
            )
            requests.append(req)

        return requests

    def declare_availability(self, cycle: int, prices: Dict[str, float]) -> List[ServiceOffer]:
        """Declare service offers based on skills, time, and strategy."""
        offers = []
        if not self.active:
            return offers

        for skill in self.skills:
            # Base probability from time availability
            prob = self.time_availability

            # Strategy adjustments
            if self.strategy == "high_volume":
                prob *= 1.2
            elif self.strategy == "credit_accumulator":
                prob *= 0.8 if self.credit_balance > 100 else 1.1
            elif self.strategy == "community_builder":
                prob *= 1.0
            elif self.strategy == "bridge_arbitrage":
                # Only offer when price is high enough to be worth extracting
                base_prices = {"electrical": 10, "plumbing": 8, "tutoring": 5,
                               "transport": 6, "cooking": 4, "childcare": 7}
                base = base_prices.get(skill, 5)
                if prices.get(skill, base) < base * 1.5:
                    prob *= 0.3
                else:
                    prob *= 1.5

            # Price incentive: higher prices attract more supply
            base_prices = {"electrical": 10, "plumbing": 8, "tutoring": 5,
                           "transport": 6, "cooking": 4, "childcare": 7}
            base = base_prices.get(skill, 5)
            current = prices.get(skill, base)
            price_ratio = current / base
            prob *= min(1.5, max(0.5, price_ratio))

            # Task load check
            if self.current_tasks >= self.max_task_load:
                continue

            prob = min(1.0, prob)
            if self._rng.random() < prob:
                offers.append(ServiceOffer(
                    offer_id=f"off_{self.member_id}_{cycle}_{skill}",
                    provider_id=self.member_id,
                    category=skill,
                    cycle=cycle,
                    skill_level=self.skill_levels.get(skill, 0.5),
                ))

        return offers

    def update_reputation(self, success: bool, delta: float = 0.01):
        """Update reputation score after task completion."""
        if success:
            self.reputation = min(1.0, self.reputation + delta)
            self.network_engagement = min(1.0, self.network_engagement + 0.005)
        else:
            self.reputation = max(0.0, self.reputation - delta * 2)

    def decide_bridge_redeem(self, cycle: int) -> Optional[float]:
        """Decide whether to redeem credits via fiat bridge."""
        if not self.active or self.credit_balance < 10:
            return None

        if self._rng.random() < self.bridge_propensity * 0.05:
            # Redeem a fraction of balance
            if self.strategy == "bridge_arbitrage":
                amount = self.credit_balance * 0.8
            else:
                amount = self.credit_balance * self._rng.uniform(0.05, 0.2)
            return max(1.0, amount)
        return None

    def get_capability_embedding(self, all_categories: List[str]) -> np.ndarray:
        """
        Return a vector embedding of this member's capabilities.
        Used for computing sim(e_i, e_j) in the QUBO weight function.
        """
        embedding = []
        for cat in all_categories:
            embedding.append(self.skill_levels.get(cat, 0.0))
        embedding.append(self.reputation)
        embedding.append(self.time_availability)
        embedding.append(self.social_capital)
        return np.array(embedding, dtype=np.float64)

    @staticmethod
    def get_task_embedding(category: str, all_categories: List[str]) -> np.ndarray:
        """Return a vector embedding for a task/request."""
        embedding = []
        for cat in all_categories:
            embedding.append(1.0 if cat == category else 0.0)
        embedding.append(0.5)  # neutral reputation requirement
        embedding.append(0.5)  # neutral time
        embedding.append(0.0)  # no social capital requirement
        return np.array(embedding, dtype=np.float64)


def create_members(
    config: dict,
    num_members: int,
    rng: np.random.Generator,
    initial_credit: float = 50.0,
) -> List[CommunityMember]:
    """Create a population of members from archetype config."""
    archetypes = config["archetypes"]
    members = []

    for arch_name, arch_config in archetypes.items():
        count = max(1, int(num_members * arch_config["weight"]))
        archetype = MemberArchetype(arch_name)

        for i in range(count):
            member = CommunityMember(
                member_id=f"{arch_name}_{i:04d}",
                archetype=archetype,
                skills=list(arch_config["skills"]),
                assets=list(arch_config.get("assets", [])),
                time_availability=arch_config["time_availability"] + rng.normal(0, 0.05),
                strategy=arch_config["strategy"],
                bridge_propensity=arch_config["bridge_propensity"],
                demand_categories=list(arch_config.get("demand_categories", [])),
                demand_rate=arch_config.get("demand_rate", 0.3),
                rng=rng,
            )
            member.time_availability = np.clip(member.time_availability, 0.05, 1.0)
            member.credit_balance = initial_credit
            members.append(member)

    return members
