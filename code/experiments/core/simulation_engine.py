"""
Main simulation engine orchestrating the DCIN economy.

Each cycle:
1. Generate requests (members sample from demand distribution)
2. Declare availability (members declare based on capability + time + strategy)
3. Match (using configured strategy)
4. Execute transfers (credit flows via CreditEconomyService)
5. Update reputation (successful completion)
6. Update social graph (add edge between transacting members)
7. Apply demurrage (if past grace period)
8. Record metrics
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .credit_economy import CreditEconomyService, EconomyMetrics
from .matching_service import MatchingService, MatchingStrategy
from .member import (
    Assignment,
    CommunityMember,
    ServiceOffer,
    ServiceRequest,
    create_members,
)
from .topology_analyzer import TopologyAnalyzer, TopologyMetrics

logger = logging.getLogger(__name__)


@dataclass
class CycleRecord:
    cycle: int
    economy: EconomyMetrics
    topology: Optional[TopologyMetrics] = None
    num_requests: int = 0
    num_offers: int = 0
    num_matched: int = 0
    total_weight: float = 0.0
    bridge_redemptions: int = 0
    bridge_volume: float = 0.0


@dataclass
class SimulationResults:
    strategy: str
    num_members: int
    num_cycles: int
    seed: int
    wall_time: float
    cycle_records: List[CycleRecord] = field(default_factory=list)
    final_topology: Optional[TopologyMetrics] = None
    novel_connections: List[Tuple[str, str, int]] = field(default_factory=list)
    organic_transactions: List[Tuple[str, str, int]] = field(default_factory=list)
    social_graph: Optional[nx.Graph] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict (excluding graph)."""
        records = []
        for cr in self.cycle_records:
            d = {
                "cycle": cr.cycle,
                "num_requests": cr.num_requests,
                "num_offers": cr.num_offers,
                "num_matched": cr.num_matched,
                "total_weight": cr.total_weight,
                "bridge_redemptions": cr.bridge_redemptions,
                "bridge_volume": cr.bridge_volume,
            }
            # Economy metrics
            em = cr.economy
            d["gini"] = em.gini
            d["velocity"] = em.velocity
            d["total_supply"] = em.total_supply
            d["total_transactions"] = em.total_transactions
            d["match_rate"] = em.match_rate
            d["treasury_fiat"] = em.treasury_fiat
            d["treasury_credits"] = em.treasury_credits
            d["reserve_ratio"] = em.reserve_ratio
            d["prices"] = em.prices
            # Topology (sampled)
            if cr.topology:
                d["topology"] = asdict(cr.topology)
            records.append(d)

        result = {
            "strategy": self.strategy,
            "num_members": self.num_members,
            "num_cycles": self.num_cycles,
            "seed": self.seed,
            "wall_time": self.wall_time,
            "cycle_records": records,
        }
        if self.final_topology:
            result["final_topology"] = asdict(self.final_topology)
        return result

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class SimulationEngine:
    """Orchestrates the DCIN economy simulation."""

    def __init__(
        self,
        config: dict,
        matching_strategy: MatchingStrategy,
        num_members: int = 500,
        num_cycles: int = 1000,
        seed: int = 42,
        topology_sample_interval: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        self.config = config
        self.matching_strategy = matching_strategy
        self.num_members = num_members
        self.num_cycles = num_cycles
        self.seed = seed
        self.topology_sample_interval = topology_sample_interval
        self.progress_callback = progress_callback

        self.rng = np.random.default_rng(seed)

        # Categories
        self.categories = list(config["categories"].keys())

        # Members
        initial_credit = config["economy"].get("initial_credit_grant", 50.0)
        self.members_list = create_members(config, num_members, self.rng, initial_credit)
        self.members: Dict[str, CommunityMember] = {
            m.member_id: m for m in self.members_list
        }

        # Economy
        econ_cfg = config["economy"]
        self.economy = CreditEconomyService(
            categories=config["categories"],
            protocol_fee=econ_cfg.get("protocol_fee", 0.025),
            ewma_alpha=econ_cfg.get("ewma_alpha", 0.1),
            demurrage_max=econ_cfg.get("demurrage_max", 0.02),
            demurrage_velocity_target=econ_cfg.get("demurrage_velocity_target", 1.0),
            demurrage_grace_cycles=econ_cfg.get("demurrage_grace_cycles", 180),
            treasury_fiat=config.get("treasury", {}).get("initial_fiat_reserves", 10000.0),
            min_reserve_ratio=config.get("treasury", {}).get("min_reserve_ratio", 0.15),
            suspended_reserve_ratio=config.get("treasury", {}).get("suspended_reserve_ratio", 0.05),
            bridge_config=config.get("treasury", {}),
        )
        for m in self.members_list:
            self.economy.set_balance(m.member_id, m.credit_balance)

        # Matching
        match_cfg = config.get("matching", {})
        self.matcher = MatchingService(matching_strategy, match_cfg)

        # Social graph
        self.social_graph = nx.Graph()
        for m in self.members_list:
            self.social_graph.add_node(m.member_id, archetype=m.archetype.value)

        # Tracking for performativity
        self._prior_edges: set = set()
        self._novel_connections: List[Tuple[str, str, int]] = []
        self._organic_transactions: List[Tuple[str, str, int]] = []
        self._mediated_this_cycle: set = set()

        # Shock schedule
        self._shocks: List[Tuple[int, str, dict]] = []

    def inject_shock(self, cycle: int, shock_type: str, params: dict):
        """Schedule a shock for a specific cycle (Experiment 2)."""
        self._shocks.append((cycle, shock_type, params))

    def run(self) -> SimulationResults:
        """Run the full simulation."""
        t_start = time.time()
        cycle_records: List[CycleRecord] = []

        for cycle in range(self.num_cycles):
            record = self._run_cycle(cycle)
            cycle_records.append(record)

            if self.progress_callback and cycle % 50 == 0:
                self.progress_callback(cycle, self.num_cycles)

        wall_time = time.time() - t_start

        # Final topology
        analyzer = TopologyAnalyzer(self.social_graph)
        final_topo = analyzer.get_metrics()

        results = SimulationResults(
            strategy=self.matching_strategy.value,
            num_members=self.num_members,
            num_cycles=self.num_cycles,
            seed=self.seed,
            wall_time=wall_time,
            cycle_records=cycle_records,
            final_topology=final_topo,
            novel_connections=self._novel_connections,
            organic_transactions=self._organic_transactions,
            social_graph=self.social_graph,
        )

        logger.info(
            f"Simulation complete: {self.matching_strategy.value}, "
            f"{self.num_cycles} cycles, {wall_time:.1f}s"
        )
        return results

    def _run_cycle(self, cycle: int) -> CycleRecord:
        """Execute one simulation cycle."""
        self.economy.begin_cycle()
        self._mediated_this_cycle = set()

        # Apply any scheduled shocks
        self._apply_shocks(cycle)

        # 1. Generate requests
        all_requests: List[ServiceRequest] = []
        for member in self.members_list:
            reqs = member.generate_requests(cycle, self.categories)
            all_requests.extend(reqs)

        # 2. Declare availability
        prices = self.economy.get_all_prices()
        all_offers: List[ServiceOffer] = []
        for member in self.members_list:
            offers = member.declare_availability(cycle, prices)
            all_offers.extend(offers)

        # Count supply/demand per category for bonding curve update
        supply_counts: Dict[str, int] = {}
        demand_counts: Dict[str, int] = {}
        for offer in all_offers:
            supply_counts[offer.category] = supply_counts.get(offer.category, 0) + 1
        for req in all_requests:
            demand_counts[req.category] = demand_counts.get(req.category, 0) + 1

        # Update bonding curves
        self.economy.update_curves(supply_counts, demand_counts, cycle)
        prices = self.economy.get_all_prices()

        # 3. Match
        assignments = self.matcher.match(
            all_requests, all_offers, self.members,
            prices, self.categories, self.rng,
        )

        # 4. Execute transfers
        total_weight = 0.0
        for assignment in assignments:
            price = self.economy.get_price(assignment.category)
            result = self.economy.transfer(
                from_id=assignment.request.requester_id,
                to_id=assignment.provider_id,
                amount=price,
                category=assignment.category,
            )
            if result.success:
                total_weight += assignment.weight
                provider = self.members.get(assignment.provider_id)
                requester = self.members.get(assignment.request.requester_id)
                if provider:
                    provider.total_provided += 1
                    provider.total_earned += price
                    provider.last_transaction_cycle = cycle
                    provider.current_tasks += 1
                if requester:
                    requester.total_consumed += 1
                    requester.total_spent += price
                    requester.last_transaction_cycle = cycle

                # 5. Update reputation
                if provider:
                    provider.update_reputation(True)
                if requester:
                    requester.update_reputation(True, delta=0.005)

                # 6. Update social graph
                a_id = assignment.request.requester_id
                b_id = assignment.provider_id
                edge_key = (min(a_id, b_id), max(a_id, b_id))

                if self.social_graph.has_edge(a_id, b_id):
                    self.social_graph[a_id][b_id]["weight"] += 1
                else:
                    # Novel connection
                    if edge_key not in self._prior_edges:
                        self._novel_connections.append((a_id, b_id, cycle))
                    self.social_graph.add_edge(a_id, b_id, weight=1)

                self._prior_edges.add(edge_key)
                self._mediated_this_cycle.add(edge_key)

        # Track organic transactions (repeats NOT mediated this cycle)
        for a_id, b_id, intro_cycle in self._novel_connections:
            edge_key = (min(a_id, b_id), max(a_id, b_id))
            if edge_key not in self._mediated_this_cycle and self.social_graph.has_edge(a_id, b_id):
                # This pair transacted before via optimizer, and we check if
                # members autonomously choose to repeat. Simulate with probability
                # based on prior transaction weight and reputation.
                w = self.social_graph[a_id][b_id].get("weight", 0)
                if w > 1:
                    # Already transacted multiple times -> count as organic
                    prob = min(0.3, 0.05 * w)
                    if self.rng.random() < prob:
                        self._organic_transactions.append((a_id, b_id, cycle))

        # Reset current tasks
        for member in self.members_list:
            member.current_tasks = 0

        # Bridge redemptions
        bridge_redemptions = 0
        bridge_volume = 0.0
        for member in self.members_list:
            amount = member.decide_bridge_redeem(cycle)
            if amount:
                result = self.economy.bridge_redeem(member.member_id, amount, cycle)
                if result.success:
                    bridge_redemptions += 1
                    bridge_volume += result.credits_spent

        # 7. Apply demurrage + redistribution
        self.economy.apply_demurrage(cycle)
        self.economy.redistribute()

        # Update balances in member objects
        for member in self.members_list:
            member.credit_balance = self.economy.get_balance(member.member_id)

        # 8. Record metrics
        self.economy.record_match_stats(len(assignments), len(all_requests))
        econ_metrics = self.economy.get_metrics(cycle)

        # Topology (sampled for performance)
        topo_metrics = None
        if cycle % self.topology_sample_interval == 0 and self.social_graph.number_of_edges() > 0:
            try:
                analyzer = TopologyAnalyzer(self.social_graph.copy())
                topo_metrics = analyzer.get_metrics()
            except Exception:
                pass

        return CycleRecord(
            cycle=cycle,
            economy=econ_metrics,
            topology=topo_metrics,
            num_requests=len(all_requests),
            num_offers=len(all_offers),
            num_matched=len(assignments),
            total_weight=total_weight,
            bridge_redemptions=bridge_redemptions,
            bridge_volume=bridge_volume,
        )

    def _apply_shocks(self, cycle: int):
        """Apply scheduled shocks for this cycle."""
        for shock_cycle, shock_type, params in self._shocks:
            if cycle != shock_cycle:
                continue

            if shock_type == "demand_spike":
                # Temporarily increase demand rate for a category
                category = params.get("category", "electrical")
                multiplier = params.get("multiplier", 10)
                for member in self.members_list:
                    if category not in member.demand_categories:
                        member.demand_categories.append(category)
                    member.demand_rate = min(1.0, member.demand_rate * multiplier)
                logger.info(f"Cycle {cycle}: Demand spike ({multiplier}x) in {category}")

            elif shock_type == "supply_drop":
                category = params.get("category", "plumbing")
                fraction = params.get("fraction", 0.5)
                affected = [m for m in self.members_list if category in m.skills]
                n_deactivate = int(len(affected) * fraction)
                for m in self.rng.choice(affected, size=min(n_deactivate, len(affected)), replace=False):
                    m.active = False
                logger.info(f"Cycle {cycle}: Supply drop ({fraction*100}%) in {category}")

            elif shock_type == "demand_spike_end":
                # Restore demand rates
                for member in self.members_list:
                    archetype_cfg = self.config["archetypes"].get(member.archetype.value, {})
                    member.demand_rate = archetype_cfg.get("demand_rate", 0.3)

            elif shock_type == "supply_restore":
                for member in self.members_list:
                    member.active = True
