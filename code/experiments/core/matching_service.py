"""
Matching service with three pluggable strategies: Random, Greedy, QUBO.

QUBO Hamiltonian (Section 3.6.1):
    H = H_objective + H_constraints
    H_objective = -sum_{i,j} w_{ij} * x_{ij}
    w_{ij} = alpha * sim(e_i, e_j) + beta * rep_i + gamma * (1/d_ij) + delta * price_j

Constraints (penalty lambda = 10 * max(w)):
    - One-member-per-task:  sum_j (sum_i x_{ij} - 1)^2
    - Task-load limit:      sum_i max(0, sum_j x_{ij} - L_i)^2
    - Skill eligibility:    sum_{(i,j): skill_gap > tau} x_{ij}
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .member import Assignment, CommunityMember, ServiceOffer, ServiceRequest


class MatchingStrategy(Enum):
    RANDOM = "random"
    GREEDY = "greedy"
    QUBO = "qubo"


class MatchingService:
    """Dispatches matching to the configured strategy."""

    def __init__(
        self,
        strategy: MatchingStrategy,
        config: Optional[dict] = None,
    ):
        self.strategy = strategy
        self.config = config or {}
        self._weights = self.config.get("qubo_weights", {
            "alpha": 0.4, "beta": 0.2, "gamma": 0.1, "delta": 0.3
        })
        self._skill_gap_threshold = self.config.get("skill_gap_threshold", 0.3)
        self._penalty_multiplier = self.config.get("qubo_penalty_multiplier", 10)
        self._num_reads = self.config.get("qubo_num_reads", 1000)
        self._max_task_load = self.config.get("max_task_load", 3)

    def match(
        self,
        requests: List[ServiceRequest],
        offers: List[ServiceOffer],
        members: Dict[str, CommunityMember],
        prices: Dict[str, float],
        all_categories: List[str],
        rng: np.random.Generator,
    ) -> List[Assignment]:
        """Match requests to available providers."""
        if not requests or not offers:
            return []

        if self.strategy == MatchingStrategy.RANDOM:
            return self._match_random(requests, offers, members, prices, all_categories, rng)
        elif self.strategy == MatchingStrategy.GREEDY:
            return self._match_greedy(requests, offers, members, prices, all_categories, rng)
        elif self.strategy == MatchingStrategy.QUBO:
            return self._match_qubo(requests, offers, members, prices, all_categories, rng)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _compute_weight(
        self,
        member: CommunityMember,
        request: ServiceRequest,
        prices: Dict[str, float],
        all_categories: List[str],
    ) -> float:
        """
        Composite weight: w_ij = alpha * sim(e_i, e_j) + beta * rep_i + gamma * (1/d_ij) + delta * price_j
        Since we don't have geographic data, gamma term uses skill_level as proxy.
        """
        alpha = self._weights["alpha"]
        beta = self._weights["beta"]
        gamma = self._weights["gamma"]
        delta = self._weights["delta"]

        # Similarity between member capability embedding and task embedding
        mem_emb = member.get_capability_embedding(all_categories)
        task_emb = CommunityMember.get_task_embedding(request.category, all_categories)
        norm_m = np.linalg.norm(mem_emb)
        norm_t = np.linalg.norm(task_emb)
        if norm_m > 0 and norm_t > 0:
            sim = float(np.dot(mem_emb, task_emb) / (norm_m * norm_t))
        else:
            sim = 0.0

        # Reputation
        rep = member.reputation

        # Proximity proxy: skill level in the requested category (1/distance analogy)
        skill_level = member.skill_levels.get(request.category, 0.0)
        proximity = skill_level  # higher skill = closer match

        # Price signal: normalized by base price
        price = prices.get(request.category, 5.0) / 10.0

        return alpha * sim + beta * rep + gamma * proximity + delta * price

    def _get_eligible_offers(
        self,
        request: ServiceRequest,
        offers: List[ServiceOffer],
        members: Dict[str, CommunityMember],
    ) -> List[ServiceOffer]:
        """Filter offers to those eligible for a request."""
        eligible = []
        for offer in offers:
            if offer.category != request.category:
                continue
            if offer.provider_id == request.requester_id:
                continue
            member = members.get(offer.provider_id)
            if member is None:
                continue
            # Skill eligibility check
            skill = member.skill_levels.get(request.category, 0.0)
            if skill < self._skill_gap_threshold:
                continue
            eligible.append(offer)
        return eligible

    # ---- RANDOM MATCHER ----

    def _match_random(
        self,
        requests: List[ServiceRequest],
        offers: List[ServiceOffer],
        members: Dict[str, CommunityMember],
        prices: Dict[str, float],
        all_categories: List[str],
        rng: np.random.Generator,
    ) -> List[Assignment]:
        """Shuffle eligible providers, assign first match per request."""
        assignments = []
        provider_load: Dict[str, int] = {}

        shuffled_requests = list(requests)
        rng.shuffle(shuffled_requests)

        for req in shuffled_requests:
            eligible = self._get_eligible_offers(req, offers, members)
            eligible = [
                o for o in eligible
                if provider_load.get(o.provider_id, 0) < (members[o.provider_id].max_task_load if o.provider_id in members else self._max_task_load)
            ]
            if not eligible:
                continue

            rng.shuffle(eligible)
            chosen = eligible[0]
            price = prices.get(req.category, 5.0)
            member = members[chosen.provider_id]
            w = self._compute_weight(member, req, prices, all_categories)

            assignments.append(Assignment(
                request=req,
                provider_id=chosen.provider_id,
                category=req.category,
                price=price,
                weight=w,
                cycle=req.cycle,
            ))
            provider_load[chosen.provider_id] = provider_load.get(chosen.provider_id, 0) + 1
            req.fulfilled = True
            req.assigned_to = chosen.provider_id

        return assignments

    # ---- GREEDY MATCHER ----

    def _match_greedy(
        self,
        requests: List[ServiceRequest],
        offers: List[ServiceOffer],
        members: Dict[str, CommunityMember],
        prices: Dict[str, float],
        all_categories: List[str],
        rng: np.random.Generator,
    ) -> List[Assignment]:
        """Rank by composite score w_ij, assign top-1 per task."""
        assignments = []
        provider_load: Dict[str, int] = {}

        # Build all (request, offer, weight) triples
        triples = []
        for req in requests:
            eligible = self._get_eligible_offers(req, offers, members)
            for offer in eligible:
                member = members[offer.provider_id]
                w = self._compute_weight(member, req, prices, all_categories)
                triples.append((req, offer, w))

        # Sort by weight descending
        triples.sort(key=lambda t: t[2], reverse=True)

        assigned_requests = set()
        for req, offer, w in triples:
            if req.request_id in assigned_requests:
                continue
            pid = offer.provider_id
            member = members.get(pid)
            max_load = member.max_task_load if member else self._max_task_load
            if provider_load.get(pid, 0) >= max_load:
                continue

            price = prices.get(req.category, 5.0)
            assignments.append(Assignment(
                request=req,
                provider_id=pid,
                category=req.category,
                price=price,
                weight=w,
                cycle=req.cycle,
            ))
            provider_load[pid] = provider_load.get(pid, 0) + 1
            assigned_requests.add(req.request_id)
            req.fulfilled = True
            req.assigned_to = pid

        return assignments

    # ---- QUBO MATCHER ----

    def _match_qubo(
        self,
        requests: List[ServiceRequest],
        offers: List[ServiceOffer],
        members: Dict[str, CommunityMember],
        prices: Dict[str, float],
        all_categories: List[str],
        rng: np.random.Generator,
    ) -> List[Assignment]:
        """
        Build QUBO matrix from w_ij and constraints, solve via dwave-neal.
        Solves per-category sub-problems for tractability.
        Falls back to greedy if the problem is too small or solver fails.
        """
        import neal

        # Group requests and offers by category for per-category sub-problems
        req_by_cat: Dict[str, List[Tuple[int, ServiceRequest]]] = {}
        off_by_cat: Dict[str, List[Tuple[int, ServiceOffer]]] = {}
        for r_idx, req in enumerate(requests):
            req_by_cat.setdefault(req.category, []).append((r_idx, req))
        for o_idx, offer in enumerate(offers):
            off_by_cat.setdefault(offer.category, []).append((o_idx, offer))

        all_assignments = []
        provider_task_count: Dict[str, int] = {}  # track load per provider
        sampler = neal.SimulatedAnnealingSampler()

        for cat in set(req_by_cat.keys()) & set(off_by_cat.keys()):
            cat_requests = req_by_cat[cat]
            cat_offers = off_by_cat[cat]

            # Build eligible pairs within this category
            eligible_pairs: List[Tuple[int, int, float]] = []
            for r_idx, req in cat_requests:
                for o_idx, offer in cat_offers:
                    if offer.provider_id == req.requester_id:
                        continue
                    pid = offer.provider_id
                    member = members.get(pid)
                    if member is None:
                        continue
                    if provider_task_count.get(pid, 0) >= member.max_task_load:
                        continue
                    skill = member.skill_levels.get(cat, 0.0)
                    if skill < self._skill_gap_threshold:
                        continue
                    w = self._compute_weight(member, req, prices, all_categories)
                    eligible_pairs.append((r_idx, o_idx, w))

            if not eligible_pairs:
                continue

            # For tiny sub-problems, use greedy selection
            if len(eligible_pairs) < 4:
                eligible_pairs.sort(key=lambda t: t[2], reverse=True)
                assigned_r = set()
                for r_idx, o_idx, w in eligible_pairs:
                    if r_idx in assigned_r:
                        continue
                    offer = offers[o_idx]
                    pid = offer.provider_id
                    member = members.get(pid)
                    max_load = member.max_task_load if member else self._max_task_load
                    if provider_task_count.get(pid, 0) >= max_load:
                        continue
                    req = requests[r_idx]
                    all_assignments.append(Assignment(
                        request=req, provider_id=pid,
                        category=cat, price=prices.get(cat, 5.0),
                        weight=w, cycle=req.cycle,
                    ))
                    assigned_r.add(r_idx)
                    provider_task_count[pid] = provider_task_count.get(pid, 0) + 1
                    req.fulfilled = True
                    req.assigned_to = pid
                continue

            # Cap total pairs for tractability while preserving diversity
            # Ensure every unique provider AND every request are represented
            if len(eligible_pairs) > 400:
                from collections import defaultdict
                by_req = defaultdict(list)
                by_off = defaultdict(list)
                for r_idx, o_idx, w in eligible_pairs:
                    by_req[r_idx].append((r_idx, o_idx, w))
                    by_off[o_idx].append((r_idx, o_idx, w))
                kept = set()
                # Best candidate per request
                for r_idx in by_req:
                    best = max(by_req[r_idx], key=lambda t: t[2])
                    kept.add((best[0], best[1]))
                # Best request per provider (ensures all providers are candidates)
                for o_idx in by_off:
                    best = max(by_off[o_idx], key=lambda t: t[2])
                    kept.add((best[0], best[1]))
                # Fill remaining by weight
                remaining = [(r, o, w) for r, o, w in eligible_pairs if (r, o) not in kept]
                remaining.sort(key=lambda t: t[2], reverse=True)
                budget = 400 - len(kept)
                for r, o, w in remaining[:max(0, budget)]:
                    kept.add((r, o))
                eligible_pairs = [(r, o, w) for r, o, w in eligible_pairs if (r, o) in kept]

            # Build QUBO
            n_vars = len(eligible_pairs)
            var_map: Dict[Tuple[int, int], int] = {}
            weights_lookup: Dict[int, float] = {}
            for idx, (r_idx, o_idx, w) in enumerate(eligible_pairs):
                var_map[(r_idx, o_idx)] = idx
                weights_lookup[idx] = w

            Q: Dict[Tuple[int, int], float] = {}
            max_w = max(abs(w) for _, _, w in eligible_pairs)

            # Objective: -w_ij diagonal
            for (r_idx, o_idx), v in var_map.items():
                Q[(v, v)] = -weights_lookup[v]

            # Constraint: one-provider-per-request
            penalty = self._penalty_multiplier * max_w
            req_vars: Dict[int, List[int]] = {}
            for (r_idx, o_idx), v in var_map.items():
                req_vars.setdefault(r_idx, []).append(v)

            for v_list in req_vars.values():
                for vi in v_list:
                    Q[(vi, vi)] = Q.get((vi, vi), 0.0) - penalty
                for i in range(len(v_list)):
                    for j in range(i + 1, len(v_list)):
                        key = (min(v_list[i], v_list[j]), max(v_list[i], v_list[j]))
                        Q[key] = Q.get(key, 0.0) + 2.0 * penalty

            # Constraint: at-most-one request per provider
            # Uses (sum x)*(sum x - 1) penalty: only off-diagonal terms
            off_vars: Dict[int, List[int]] = {}
            for (r_idx, o_idx), v in var_map.items():
                off_vars.setdefault(o_idx, []).append(v)

            for v_list in off_vars.values():
                if len(v_list) <= 1:
                    continue
                for i in range(len(v_list)):
                    for j in range(i + 1, len(v_list)):
                        key = (min(v_list[i], v_list[j]), max(v_list[i], v_list[j]))
                        Q[key] = Q.get(key, 0.0) + 2.0 * penalty

            # Solve
            try:
                num_reads = max(100, min(self._num_reads, 200))
                response = sampler.sample_qubo(
                    Q, num_reads=num_reads,
                    seed=int(rng.integers(0, 2**31)),
                )
                best = response.first.sample
            except Exception:
                # Greedy fallback for this category
                eligible_pairs.sort(key=lambda t: t[2], reverse=True)
                best = {}

            # Extract assignments
            selected = []
            for (r_idx, o_idx), v in var_map.items():
                if best.get(v, 0) == 1:
                    selected.append((r_idx, o_idx, weights_lookup[v]))
            selected.sort(key=lambda t: t[2], reverse=True)

            assigned_r = set()
            for r_idx, o_idx, w in selected:
                if r_idx in assigned_r:
                    continue
                offer = offers[o_idx]
                pid = offer.provider_id
                member = members.get(pid)
                max_load = member.max_task_load if member else self._max_task_load
                if provider_task_count.get(pid, 0) >= max_load:
                    continue
                req = requests[r_idx]
                all_assignments.append(Assignment(
                    request=req, provider_id=pid,
                    category=cat, price=prices.get(cat, 5.0),
                    weight=w, cycle=req.cycle,
                ))
                assigned_r.add(r_idx)
                provider_task_count[pid] = provider_task_count.get(pid, 0) + 1
                req.fulfilled = True
                req.assigned_to = pid

        # Greedy fill: match remaining unmatched requests
        assigned_request_ids = {a.request.request_id for a in all_assignments}
        unmatched = [r for r in requests if r.request_id not in assigned_request_ids]
        if unmatched:
            unused_offers = [
                o for o in offers
                if provider_task_count.get(o.provider_id, 0) < (
                    members[o.provider_id].max_task_load
                    if o.provider_id in members else self._max_task_load
                )
            ]
            # Greedy fill: for each unmatched request, pick best available
            triples = []
            for req in unmatched:
                for offer in unused_offers:
                    if offer.category != req.category:
                        continue
                    if offer.provider_id == req.requester_id:
                        continue
                    member = members.get(offer.provider_id)
                    if member is None:
                        continue
                    skill = member.skill_levels.get(req.category, 0.0)
                    if skill < self._skill_gap_threshold:
                        continue
                    w = self._compute_weight(member, req, prices, all_categories)
                    triples.append((req, offer, w))
            triples.sort(key=lambda t: t[2], reverse=True)
            for req, offer, w in triples:
                if req.request_id in assigned_request_ids:
                    continue
                pid = offer.provider_id
                member = members.get(pid)
                max_load = member.max_task_load if member else self._max_task_load
                if provider_task_count.get(pid, 0) >= max_load:
                    continue
                all_assignments.append(Assignment(
                    request=req, provider_id=pid,
                    category=req.category, price=prices.get(req.category, 5.0),
                    weight=w, cycle=req.cycle,
                ))
                assigned_request_ids.add(req.request_id)
                provider_task_count[pid] = provider_task_count.get(pid, 0) + 1
                req.fulfilled = True
                req.assigned_to = pid

        return all_assignments
