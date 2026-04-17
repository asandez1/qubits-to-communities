"""
Repository layer for OrquestIA domain models.

Provides read/write persistence for Members, Tasks, TaskTypes, and
MatchingInstances. The domain model (domain.py) stays pure — frozen
Pydantic objects with no I/O. The repository handles storage.

Current backend: JSON files.  Swap to SQLite/Mongo later by implementing
the same interface without touching domain.py or benchmark_model.py.

Usage:
    from core.repository import Repository

    repo = Repository("data/community")   # directory path
    repo.add_member(member)
    repo.add_task(task)
    repo.save()                           # writes to disk

    # Later:
    repo = Repository.load("data/community")
    members = repo.list_members()
    electricians = repo.members_by_category("electrical", min_skill=0.5)
    instance = repo.build_instance("cycle_42", skill_threshold=0.4)
"""

from __future__ import annotations

import json
import os
from typing import Optional

from core.domain import (
    CommunityConfig,
    EligiblePair,
    Location,
    MatchingInstance,
    Member,
    MemberProfile,
    ServiceCategory,
    SkillCategory,
    Task,
    TaskType,
    TimeWindow,
)


class Repository:
    """Read/write store for community members, tasks, and task types.

    All data lives in a single directory as JSON files:
        members.json   — list of Member dicts
        tasks.json     — list of Task dicts
        catalog.json   — dict of ServiceCategory (task type catalog)
        meta.json      — metadata (categories, prices, description)

    The repository is the ONLY place that reads/writes files.
    Domain objects are frozen — the repo creates new versions on update.
    """

    def __init__(self, path: str):
        self._path = path
        self._members: dict[str, Member] = {}          # member_id -> Member
        self._tasks: dict[str, Task] = {}               # task_id -> Task
        self._catalog: dict[str, ServiceCategory] = {}  # category name -> ServiceCategory
        self._prices: dict[str, float] = {}
        self._description: str = ""

    # -----------------------------------------------------------------
    # Members
    # -----------------------------------------------------------------

    def add_member(self, member: Member) -> None:
        """Add or replace a member."""
        self._members[member.member_id] = member

    def add_members(self, members: list[Member] | tuple[Member, ...]) -> None:
        for m in members:
            self._members[m.member_id] = m

    def get_member(self, member_id: str) -> Optional[Member]:
        return self._members.get(member_id)

    def remove_member(self, member_id: str) -> bool:
        return self._members.pop(member_id, None) is not None

    def update_member(self, member_id: str, **updates) -> Member:
        """Update fields on a frozen member (creates a new object)."""
        old = self._members[member_id]
        new = old.model_copy(update=updates)
        self._members[member_id] = new
        return new

    def list_members(self) -> list[Member]:
        return list(self._members.values())

    def members_by_archetype(self, archetype: str) -> list[Member]:
        return [m for m in self._members.values() if m.archetype == archetype]

    def members_by_category(
        self, category: str, min_skill: float = 0.0
    ) -> list[Member]:
        """Find members with skill >= min_skill in the given category."""
        cat = SkillCategory(category)
        return [
            m for m in self._members.values()
            if m.skill_levels.get(cat, 0.0) >= min_skill
        ]

    def members_near(
        self, location: Location, max_distance_km: float
    ) -> list[Member]:
        """Find members within max_distance_km of a location."""
        return [
            m for m in self._members.values()
            if m.location.distance_to(location) <= max_distance_km
        ]

    def members_available(
        self, weekday: int, start_block: int, end_block: int
    ) -> list[Member]:
        """Find members available during a time window on a given day."""
        return [
            m for m in self._members.values()
            if m.schedule.is_available(weekday, start_block, end_block)
        ]

    @property
    def n_members(self) -> int:
        return len(self._members)

    # -----------------------------------------------------------------
    # Tasks
    # -----------------------------------------------------------------

    def add_task(self, task: Task) -> None:
        self._tasks[task.task_id] = task

    def add_tasks(self, tasks: list[Task] | tuple[Task, ...]) -> None:
        for t in tasks:
            self._tasks[t.task_id] = t

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def remove_task(self, task_id: str) -> bool:
        return self._tasks.pop(task_id, None) is not None

    def list_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def tasks_by_category(self, category: str) -> list[Task]:
        return [t for t in self._tasks.values() if t.required_skill == category]

    def tasks_near(
        self, location: Location, max_distance_km: float
    ) -> list[Task]:
        return [
            t for t in self._tasks.values()
            if t.location.distance_to(location) <= max_distance_km
        ]

    @property
    def n_tasks(self) -> int:
        return len(self._tasks)

    # -----------------------------------------------------------------
    # Task type catalog
    # -----------------------------------------------------------------

    def set_catalog(self, categories: dict[str, ServiceCategory]) -> None:
        self._catalog = dict(categories)

    def get_task_type(self, category: str, type_name: str) -> Optional[TaskType]:
        cat = self._catalog.get(category)
        if cat is None:
            return None
        return cat.task_types.get(type_name)

    def list_task_types(self, category: str) -> list[TaskType]:
        cat = self._catalog.get(category)
        return list(cat.task_types.values()) if cat else []

    def set_prices(self, prices: dict[str, float]) -> None:
        self._prices = dict(prices)

    # -----------------------------------------------------------------
    # Build a MatchingInstance (snapshot for QUBO)
    # -----------------------------------------------------------------

    def build_instance(
        self,
        instance_id: str,
        skill_threshold: float = 0.3,
        member_filter: Optional[list[str]] = None,
        task_filter: Optional[list[str]] = None,
        description: str = "",
    ) -> MatchingInstance:
        """Create an immutable MatchingInstance from the current repository state.

        Optionally filter to specific member_ids or task_ids.
        """
        if member_filter:
            members = tuple(
                self._members[mid] for mid in member_filter
                if mid in self._members
            )
        else:
            members = tuple(self._members.values())

        if task_filter:
            tasks = tuple(
                self._tasks[tid] for tid in task_filter
                if tid in self._tasks
            )
        else:
            tasks = tuple(self._tasks.values())

        categories = tuple(
            sorted(set(t.required_skill for t in tasks))
        )
        prices = {
            c: self._prices.get(c, 5.0) for c in categories
        }

        return MatchingInstance(
            instance_id=instance_id,
            members=members,
            tasks=tasks,
            categories=categories,
            prices=prices,
            skill_threshold=skill_threshold,
            description=description or self._description,
        )

    # -----------------------------------------------------------------
    # Populate from CommunityConfig
    # -----------------------------------------------------------------

    def populate_from_config(
        self,
        config: CommunityConfig,
        n_members: int = 50,
        n_tasks: int = 25,
        seed: int = 42,
    ) -> None:
        """Generate members and tasks from archetype profiles."""
        members = config.generate_members(n_members, seed=seed)
        tasks = config.generate_tasks(n_tasks, seed=seed)
        self.add_members(members)
        self.add_tasks(tasks)
        self._catalog = {k.value if hasattr(k, 'value') else k: v
                         for k, v in config.categories.items()}
        self._prices = {
            cat.name.value if hasattr(cat.name, 'value') else cat.name: cat.base_price
            for cat in config.categories.values()
        }

    # -----------------------------------------------------------------
    # Persistence — JSON files
    # -----------------------------------------------------------------

    def save(self) -> None:
        """Write all data to the repository directory."""
        os.makedirs(self._path, exist_ok=True)

        # Members
        members_data = [m.model_dump(mode="json") for m in self._members.values()]
        with open(os.path.join(self._path, "members.json"), "w") as f:
            json.dump(members_data, f, indent=2)

        # Tasks
        tasks_data = [t.model_dump(mode="json") for t in self._tasks.values()]
        with open(os.path.join(self._path, "tasks.json"), "w") as f:
            json.dump(tasks_data, f, indent=2)

        # Catalog
        catalog_data = {
            k: v.model_dump(mode="json") for k, v in self._catalog.items()
        }
        with open(os.path.join(self._path, "catalog.json"), "w") as f:
            json.dump(catalog_data, f, indent=2)

        # Meta
        meta = {
            "prices": self._prices,
            "description": self._description,
            "n_members": self.n_members,
            "n_tasks": self.n_tasks,
        }
        with open(os.path.join(self._path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> Repository:
        """Load a repository from a directory."""
        repo = cls(path)

        # Members
        members_path = os.path.join(path, "members.json")
        if os.path.exists(members_path):
            with open(members_path) as f:
                for d in json.load(f):
                    repo._members[d["member_id"]] = Member.model_validate(d)

        # Tasks
        tasks_path = os.path.join(path, "tasks.json")
        if os.path.exists(tasks_path):
            with open(tasks_path) as f:
                for d in json.load(f):
                    repo._tasks[d["task_id"]] = Task.model_validate(d)

        # Catalog
        catalog_path = os.path.join(path, "catalog.json")
        if os.path.exists(catalog_path):
            with open(catalog_path) as f:
                for k, v in json.load(f).items():
                    repo._catalog[k] = ServiceCategory.model_validate(v)

        # Meta
        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
                repo._prices = meta.get("prices", {})
                repo._description = meta.get("description", "")

        return repo

    def summary(self) -> str:
        archetypes = {}
        for m in self._members.values():
            archetypes[m.archetype] = archetypes.get(m.archetype, 0) + 1
        arch_str = ", ".join(f"{k}: {v}" for k, v in sorted(archetypes.items()))
        cats = sorted(set(t.required_skill for t in self._tasks.values()))
        return (
            f"Repository '{self._path}': "
            f"{self.n_members} members ({arch_str}), "
            f"{self.n_tasks} tasks, categories={cats}"
        )
