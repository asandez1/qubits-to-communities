"""
Simulation History — full-snapshot state management.

Stores a complete CycleSnapshot at every cycle. Supports queries over
time (member history, balance series, Gini series) and schema evolution
(new categories mid-simulation without losing history).

Persistence: append-only JSONL file (one line per cycle).
50 members x 1000 cycles ≈ 25 MB — loads in < 1 second.

Usage:
    from core.history import CycleSnapshot, SimulationHistory

    history = SimulationHistory("runs/exp1", seed=42)
    history.append(initial_snapshot)
    # ... run cycles, appending after each ...
    history.save()

    # Queries
    history.gini_at(100)
    history.balance_series("alice")
    history.member_history("bob")
    inst = history.matching_instance_at(50)
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from core.domain import (
    MatchingInstance,
    Member,
    Task,
)


# =====================================================================
# CycleSnapshot — one complete state
# =====================================================================

class CycleSnapshot(BaseModel):
    """Complete, immutable state of the community at one cycle."""
    model_config = ConfigDict(frozen=True)

    cycle: int
    members: tuple[Member, ...]
    tasks: tuple[Task, ...]
    categories: tuple[str, ...]
    prices: dict[str, float]
    assignments: tuple[dict, ...] = ()   # SolutionResult.assignments
    economy_state: dict = {}             # velocity, treasury, demurrage rate, etc.

    def to_matching_instance(self, skill_threshold: float = 0.3) -> MatchingInstance:
        """Reconstruct a MatchingInstance for QUBO solving/replay."""
        return MatchingInstance(
            instance_id=f"cycle_{self.cycle:04d}",
            members=self.members,
            tasks=self.tasks,
            categories=self.categories,
            prices=self.prices,
            skill_threshold=skill_threshold,
        )

    def member_by_id(self, member_id: str) -> Optional[Member]:
        for m in self.members:
            if m.member_id == member_id:
                return m
        return None

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)


# =====================================================================
# SimulationHistory — append-only store with queries
# =====================================================================

class SimulationHistory:
    """Full history of a simulation run. One snapshot per cycle."""

    def __init__(self, path: str, seed: int = 42, config: Optional[dict] = None):
        self._path = path
        self._seed = seed
        self._config = config or {}
        self._snapshots: list[CycleSnapshot] = []
        self._catalog_versions: list[tuple[int, dict]] = []  # (cycle, catalog)

    # -----------------------------------------------------------------
    # Append
    # -----------------------------------------------------------------

    def append(self, snapshot: CycleSnapshot) -> None:
        """Append a cycle snapshot. Must be sequential."""
        expected = len(self._snapshots)
        if snapshot.cycle != expected:
            raise ValueError(
                f"Expected cycle {expected}, got {snapshot.cycle}. "
                f"Snapshots must be appended sequentially."
            )
        self._snapshots.append(snapshot)

    # -----------------------------------------------------------------
    # Queries — state at a point in time
    # -----------------------------------------------------------------

    def state_at(self, cycle: int) -> CycleSnapshot:
        """Return the full snapshot at a given cycle. O(1)."""
        if cycle < 0 or cycle >= len(self._snapshots):
            raise IndexError(f"Cycle {cycle} not in history (0-{len(self._snapshots)-1})")
        return self._snapshots[cycle]

    def matching_instance_at(self, cycle: int, skill_threshold: float = 0.3) -> MatchingInstance:
        """Reconstruct a MatchingInstance for QUBO replay at any cycle."""
        return self.state_at(cycle).to_matching_instance(skill_threshold)

    def member_at(self, member_id: str, cycle: int) -> Optional[Member]:
        """Single member lookup at a cycle."""
        return self.state_at(cycle).member_by_id(member_id)

    # -----------------------------------------------------------------
    # Queries — time series
    # -----------------------------------------------------------------

    def member_history(self, member_id: str) -> list[tuple[int, Member]]:
        """Full timeline for one member. Returns (cycle, Member) pairs."""
        result = []
        for snap in self._snapshots:
            m = snap.member_by_id(member_id)
            if m is not None:
                result.append((snap.cycle, m))
        return result

    def balance_series(self, member_id: str) -> list[tuple[int, float]]:
        """Credit balance over time for one member."""
        return [
            (cycle, m.credit_balance)
            for cycle, m in self.member_history(member_id)
        ]

    def reputation_series(self, member_id: str) -> list[tuple[int, float]]:
        """Reputation over time for one member."""
        return [
            (cycle, m.reputation)
            for cycle, m in self.member_history(member_id)
        ]

    # -----------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------

    @staticmethod
    def _gini(values: list[float]) -> float:
        """Compute Gini coefficient from a list of values."""
        arr = np.array(values, dtype=np.float64)
        if len(arr) == 0 or arr.sum() == 0:
            return 0.0
        arr = np.sort(arr)
        n = len(arr)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))

    def gini_at(self, cycle: int) -> float:
        """Gini coefficient of credit balances at a cycle."""
        snap = self.state_at(cycle)
        balances = [m.credit_balance for m in snap.members]
        return self._gini(balances)

    def gini_series(self) -> list[tuple[int, float]]:
        """Gini at every cycle."""
        return [(i, self.gini_at(i)) for i in range(len(self._snapshots))]

    def total_credits_at(self, cycle: int) -> float:
        """Sum of all member balances at a cycle."""
        return sum(m.credit_balance for m in self.state_at(cycle).members)

    # -----------------------------------------------------------------
    # Schema evolution
    # -----------------------------------------------------------------

    def evolve_catalog(self, cycle: int, catalog: dict) -> None:
        """Register a catalog change at a cycle."""
        self._catalog_versions.append((cycle, catalog))

    def active_categories_at(self, cycle: int) -> tuple[str, ...]:
        """Which categories existed at a given cycle."""
        return self.state_at(cycle).categories

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def current_cycle(self) -> int:
        return len(self._snapshots) - 1 if self._snapshots else -1

    @property
    def n_cycles(self) -> int:
        return len(self._snapshots)

    @property
    def seed(self) -> int:
        return self._seed

    # -----------------------------------------------------------------
    # Persistence — JSONL
    # -----------------------------------------------------------------

    def save(self) -> None:
        """Write history to disk."""
        os.makedirs(self._path, exist_ok=True)

        # Manifest
        manifest = {
            "seed": self._seed,
            "n_cycles": self.n_cycles,
            "config_hash": hashlib.md5(
                json.dumps(self._config, sort_keys=True).encode()
            ).hexdigest() if self._config else "",
            "catalog_versions": [
                {"cycle": c, "version": i + 1}
                for i, (c, _) in enumerate(self._catalog_versions)
            ],
        }
        with open(os.path.join(self._path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        # Config
        if self._config:
            with open(os.path.join(self._path, "config.json"), "w") as f:
                json.dump(self._config, f, indent=2)

        # Catalog versions
        for i, (cycle, catalog) in enumerate(self._catalog_versions):
            fname = f"catalog_v{i+1:03d}.json"
            with open(os.path.join(self._path, fname), "w") as f:
                json.dump(catalog, f, indent=2)

        # Snapshots — one JSON line per cycle
        with open(os.path.join(self._path, "snapshots.jsonl"), "w") as f:
            for snap in self._snapshots:
                f.write(snap.model_dump_json() + "\n")

    @classmethod
    def load(cls, path: str) -> SimulationHistory:
        """Load history from disk."""
        # Manifest
        with open(os.path.join(path, "manifest.json")) as f:
            manifest = json.load(f)

        # Config
        config = {}
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)

        history = cls(path, seed=manifest["seed"], config=config)

        # Catalog versions
        for cv in manifest.get("catalog_versions", []):
            fname = f"catalog_v{cv['version']:03d}.json"
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    history._catalog_versions.append((cv["cycle"], json.load(f)))

        # Snapshots
        snapshots_path = os.path.join(path, "snapshots.jsonl")
        if os.path.exists(snapshots_path):
            with open(snapshots_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        snap = CycleSnapshot.model_validate_json(line)
                        history._snapshots.append(snap)

        return history

    def summary(self) -> str:
        if not self._snapshots:
            return f"SimulationHistory '{self._path}': empty"
        last = self._snapshots[-1]
        return (
            f"SimulationHistory '{self._path}': "
            f"{self.n_cycles} cycles, {last.n_members} members, "
            f"{last.n_tasks} tasks, seed={self._seed}"
        )
