"""
Cycle Engine — pure state transitions for the society model.

Takes a frozen CycleSnapshot + assignment results and produces the next
cycle's frozen state. No mutation anywhere — each step creates new
Member objects via model_copy(update={...}).

This is the skeleton for simulation logic. Current steps:
  1. Credit transfers (requester pays provider)
  2. Reputation updates (success → +rep, failure → -rep)
  3. Demurrage (floating rate on idle balances)
  4. Schema evolution (new categories)
  5. Add new members/tasks

Future steps (to be added after infrastructure is tested):
  - Provider rejection (acceptance_probability < threshold)
  - Partial completion / quality scoring
  - Bad ratings
  - Requester disputes

Usage:
    from core.cycle_engine import CycleEngine
    from core.history import CycleSnapshot

    engine = CycleEngine(config, seed=42)
    next_snapshot = engine.advance(current_snapshot, assignments)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.domain import Member, Task
from core.history import CycleSnapshot


class CycleEngine:
    """Pure state transition engine.

    advance() takes frozen state in and returns frozen state out.
    Deterministic: same seed + same inputs = same outputs.
    """

    def __init__(
        self,
        protocol_fee: float = 0.025,
        demurrage_max: float = 0.02,
        demurrage_velocity_target: float = 1.0,
        demurrage_grace_cycles: int = 180,
        initial_grant: float = 50.0,
        rep_success_delta: float = 0.01,
        rep_failure_delta: float = 0.02,
        req_rep_delta: float = 0.005,
        seed: int = 42,
    ):
        self.protocol_fee = protocol_fee
        self.demurrage_max = demurrage_max
        self.demurrage_velocity_target = demurrage_velocity_target
        self.demurrage_grace_cycles = demurrage_grace_cycles
        self.initial_grant = initial_grant
        self.rep_success_delta = rep_success_delta
        self.rep_failure_delta = rep_failure_delta
        self.req_rep_delta = req_rep_delta
        self._base_seed = seed

    def advance(
        self,
        current: CycleSnapshot,
        assignments: list[dict],
        new_tasks: Optional[tuple[Task, ...]] = None,
        new_members: Optional[tuple[Member, ...]] = None,
        new_categories: Optional[tuple[str, ...]] = None,
        reset_energy: bool = True,
    ) -> CycleSnapshot:
        """Pure transition: (current_state, assignments) -> next_state.

        Args:
            current: the current cycle's frozen snapshot
            assignments: list of dicts from SolutionResult.assignments
                         (must have member_id, task_id, category keys)
            new_tasks: tasks to add for the next cycle
            new_members: members joining the community
            new_categories: new skill categories being introduced
        """
        cycle = current.cycle + 1
        members = dict(zip(
            [m.member_id for m in current.members],
            current.members,
        ))

        # Economy state carries forward (velocity, treasury)
        eco = dict(current.economy_state)
        cycle_volume = 0.0

        # --- Step 0: Age-based overnight energy recovery (new day only) ---
        # Younger members fully recover; older members accumulate fatigue
        # (recovery_rate < 1 means they wake up tired after a depleting day).
        if reset_energy:
            for mid, m in list(members.items()):
                if hasattr(m, 'recover_overnight'):
                    members[mid] = m.recover_overnight()
                elif hasattr(m, 'energy_capacity'):
                    members[mid] = m.model_copy(update={"energy": m.energy_capacity})

        # --- Step 1: Credit transfers + energy depletion ---
        task_by_id = {t.task_id: t for t in current.tasks}
        for assign in assignments:
            task = task_by_id.get(assign.get("task_id"))
            if task is None:
                continue
            provider_id = assign.get("member_id")
            requester_id = task.requester_id
            price = task.price

            if provider_id and provider_id in members:
                provider = members[provider_id]
                fee = price * self.protocol_fee
                # Credit + energy depletion
                energy_cost = task.task_type.energy_cost if hasattr(task, 'task_type') else task.duration
                new_energy = max(0.0, provider.energy - energy_cost)
                members[provider_id] = provider.model_copy(
                    update={"credit_balance": provider.credit_balance + price,
                            "energy": new_energy}
                )
                cycle_volume += price

            if requester_id and requester_id in members:
                requester = members[requester_id]
                total_debit = price * (1 + self.protocol_fee)
                new_balance = max(0.0, requester.credit_balance - total_debit)
                members[requester_id] = requester.model_copy(
                    update={"credit_balance": new_balance}
                )

        # --- Step 1b: Intra-household partial equalization ---
        # Households share 30% of their "inequality gap" each cycle.
        # Not full pooling — preserves some individual incentive to earn.
        # Solo households (empty household_id) are skipped.
        POOL_FRACTION = 0.3
        households: dict[str, list[str]] = {}
        for mid, m in members.items():
            hid = getattr(m, 'household_id', '') or ''
            if hid:  # only pool members with explicit household
                households.setdefault(hid, []).append(mid)
        for hid, mids in households.items():
            if len(mids) <= 1:
                continue
            total = sum(members[mid].credit_balance for mid in mids)
            mean = total / len(mids)
            for mid in mids:
                m = members[mid]
                gap = mean - m.credit_balance
                new_bal = m.credit_balance + POOL_FRACTION * gap
                members[mid] = m.model_copy(update={"credit_balance": round(new_bal, 4)})

        # --- Step 2: Reputation updates ---
        for assign in assignments:
            provider_id = assign.get("member_id")
            if provider_id and provider_id in members:
                m = members[provider_id]
                new_rep = min(1.0, m.reputation + self.rep_success_delta)
                new_sc = min(1.0, m.social_capital + 0.005)
                members[provider_id] = m.model_copy(
                    update={"reputation": round(new_rep, 4),
                            "social_capital": round(new_sc, 4)}
                )

            task = task_by_id.get(assign.get("task_id"))
            if task and task.requester_id and task.requester_id in members:
                req = members[task.requester_id]
                new_rep = min(1.0, req.reputation + self.req_rep_delta)
                members[task.requester_id] = req.model_copy(
                    update={"reputation": round(new_rep, 4)}
                )

        # --- Step 3: Demurrage ---
        if cycle > self.demurrage_grace_cycles:
            # Velocity = cycle_volume / total_credits
            total_credits = sum(m.credit_balance for m in members.values())
            velocity = cycle_volume / total_credits if total_credits > 0 else 0
            d_rate = min(
                self.demurrage_max,
                self.demurrage_max * (1 - velocity / self.demurrage_velocity_target)
            )
            d_rate = max(0.0, d_rate)

            for mid, m in list(members.items()):
                if m.credit_balance > 0:
                    reduction = m.credit_balance * d_rate
                    members[mid] = m.model_copy(
                        update={"credit_balance": round(m.credit_balance - reduction, 4)}
                    )

            eco["demurrage_rate"] = round(d_rate, 6)
            eco["velocity"] = round(velocity, 6)

        # --- Step 4: Schema evolution ---
        categories = list(current.categories)
        if new_categories:
            for cat in new_categories:
                if cat not in categories:
                    categories.append(cat)
            # Existing members already return 0.0 for missing skill keys
            # via Member.skill_in() — no update needed

        # --- Step 5: Add new members/tasks ---
        if new_members:
            for nm in new_members:
                members[nm.member_id] = nm

        tasks = current.tasks
        if new_tasks:
            tasks = tasks + new_tasks

        # --- Build next snapshot ---
        eco["cycle_volume"] = round(cycle_volume, 4)
        eco["total_credits"] = round(
            sum(m.credit_balance for m in members.values()), 4
        )

        return CycleSnapshot(
            cycle=cycle,
            members=tuple(members.values()),
            tasks=tasks,
            categories=tuple(categories),
            prices=current.prices,
            assignments=tuple(assignments),
            economy_state=eco,
        )


def create_initial_snapshot(
    members: tuple[Member, ...],
    tasks: tuple[Task, ...],
    categories: tuple[str, ...],
    prices: dict[str, float],
) -> CycleSnapshot:
    """Create cycle-0 snapshot with initial grants applied."""
    return CycleSnapshot(
        cycle=0,
        members=members,
        tasks=tasks,
        categories=categories,
        prices=prices,
        economy_state={"cycle_volume": 0, "total_credits": sum(m.credit_balance for m in members)},
    )
