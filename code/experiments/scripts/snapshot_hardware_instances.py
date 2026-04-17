"""
Snapshot the 8 hardware benchmark instances from v1 simulation harvest into a
v2-native JSON file. This decouples the hardware-path experiments (Exp 6d, 7,
7d, 7e, 8) from `community_economy.json` and the v1 simulation engine.

Output: `results/exp6d/hardware_instances_v2.json`

The snapshot contains everything needed to reproduce the paper's hardware
results (§4, §5 + Phase 2) without running any simulation:
- 8 frozen MatchingInstance objects (v2 Pydantic)
- Each instance has members (with their 9 quantum-relevant attributes) and
  tasks (with category, price, requester_id)
- The pretrained θ (`theta.npy`) already lives at `results/exp6d/theta.npy`

Verification target: running the v2 paper_classical_weights + paper_qubo on
the snapshot must yield the same regret numbers as the existing Exp 6d/7
results computed from the v1 harvest.

Usage:
    python experiments/scripts/snapshot_hardware_instances.py
"""
import json
import logging
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from core.domain import (
    Member, Task, TaskType, EffortProfile, MatchingInstance,
    Location, WeeklySchedule, SkillCategory, TimeWindow,
)
from experiments.exp6b_hybrid_replication import harvest_instances


CONFIG_PATH = os.path.join(ROOT, "config", "community_economy.json")
OUTPUT_PATH = os.path.join(ROOT, "results", "exp6d", "hardware_instances_v2.json")


def v1_member_to_v2(v1_member, member_id_suffix: str = "") -> Member:
    """Convert a v1 CommunityMember into a v2 Member.

    Only the 9 quantum-relevant attributes are preserved with full fidelity.
    location, schedule, age, etc. are filled with sensible defaults — they are
    NOT used by member_embedding_9d() or by the v1 weight formula that
    produced the paper's hardware results.
    """
    # SkillCategory enum keys; v1 stores as strings already
    skill_levels = {}
    for cat_name, value in v1_member.skill_levels.items():
        # Map string → enum (SkillCategory is StrEnum so str==enum.value)
        try:
            cat = SkillCategory(cat_name)
        except ValueError:
            cat = cat_name  # unknown category, keep as string
        skill_levels[cat] = float(value)

    return Member(
        member_id=str(v1_member.member_id) + member_id_suffix,
        archetype=getattr(v1_member, "archetype", "unknown"),
        skill_levels=skill_levels,
        reputation=float(v1_member.reputation),
        time_availability=float(v1_member.time_availability),
        social_capital=float(getattr(v1_member, "social_capital", 0.0)),
        max_task_load=int(getattr(v1_member, "max_task_load", 3)),
        credit_balance=float(getattr(v1_member, "credit_balance", 50.0)),
        # Defaults for unused-in-hardware-path fields:
        location=Location(),
        schedule=WeeklySchedule(),
        age=35,
        energy=8.5,  # peak energy for age=35 (computed by energy_capacity)
    )


def v1_request_to_v2_task(v1_req, price: float, task_idx: int) -> Task:
    """Convert a v1 ServiceRequest into a v2 Task."""
    cat_name = v1_req.category
    try:
        cat = SkillCategory(cat_name)
    except ValueError:
        cat = cat_name

    effort = EffortProfile(
        duration_hours=1.0,
        credit_cost=float(price),
        min_skill=0.3,
        min_trust=0.3,
        energy_cost=1.0,
        min_age=0,
        max_age=120,
    )
    task_type = TaskType(
        name=f"{cat_name}_{task_idx:03d}",
        category=cat,
        effort=effort,
    )
    return Task(
        task_id=str(v1_req.request_id),
        task_type=task_type,
        time_window=TimeWindow(start=36, end=60),  # 09:00-15:00 default
        location=Location(),
        requester_id=str(v1_req.requester_id),
        requester_reputation=0.5,
        weekday=0,
        urgency=0.5,
    )


def snapshot_v1_instance(v1_inst, instance_idx: int) -> dict:
    """Convert one v1 instance dict into a v2-native dict snapshot.

    Structure saved:
      {
        "instance_id": "inst_0_electrical_cycle50",
        "category_label": "electrical",
        "cycle": 50,
        "prices": {cat: price, ...},
        "categories": [cat, ...],
        "matcher_weights": {alpha, beta, gamma, delta},
        "members": [Member.model_dump(), ...],
        "tasks": [Task.model_dump(), ...],
        "eligible_pair_refs": [
            {"req_idx": int, "offer_idx": int,
             "task_id": str, "member_id": str},
            ...
        ]
      }

    The eligible_pair_refs preserves the original v1 (request_idx, offer_idx)
    mapping so that downstream code can reproduce the v1 QUBO variable layout
    exactly.
    """
    cats_tuple = tuple(v1_inst["categories"])
    prices = dict(v1_inst["prices"])
    matcher = v1_inst["matcher"]
    matcher_weights = dict(matcher._weights)

    # Collect unique members and tasks (v1 has duplicates across pairs)
    member_map: dict[str, Member] = {}   # member_id -> v2 Member
    task_map: dict[str, Task] = {}       # task_id (= request_id) -> v2 Task
    pair_refs = []

    for r_idx, o_idx, req, offer, v1_mem in v1_inst["eligible_pairs"]:
        mid = str(v1_mem.member_id)
        if mid not in member_map:
            member_map[mid] = v1_member_to_v2(v1_mem)
        tid = str(req.request_id)
        if tid not in task_map:
            price = prices.get(req.category, 5.0)
            task_map[tid] = v1_request_to_v2_task(req, price, len(task_map))
        pair_refs.append({
            "req_idx": int(r_idx),
            "offer_idx": int(o_idx),
            "task_id": tid,
            "member_id": mid,
            "category": str(req.category),
        })

    return {
        "instance_id": f"inst_{instance_idx}_{v1_inst['category']}_cycle{v1_inst['cycle']}",
        "category_label": v1_inst["category"],
        "cycle": int(v1_inst["cycle"]),
        "prices": prices,
        "categories": list(cats_tuple),
        "matcher_weights": matcher_weights,
        "members": [m.model_dump(mode="json") for m in member_map.values()],
        "tasks": [t.model_dump(mode="json") for t in task_map.values()],
        "eligible_pair_refs": pair_refs,
    }


def main():
    logger.info("Harvesting v1 hardware instances (this runs the v1 simulation briefly)...")
    v1_instances = harvest_instances(CONFIG_PATH, 30)
    logger.info(f"Got {len(v1_instances)} instances from v1 harvest")

    snapshot = {
        "schema_version": "1.0",
        "description": (
            "Hardware-benchmark matching instances for Exp 6d / 7 / 7d / 7e / 8. "
            "Converted from v1 harvest (community_economy.json, seed=42, cycles "
            "[50,100,150,200,250,300,350,400,450,499]) to v2 Pydantic "
            "representation. The 9 quantum-relevant member attributes "
            "(6 skills + reputation + time_availability + social_capital) are "
            "preserved with full fidelity; location/schedule/age/etc. are "
            "defaults (not used by the hardware path)."
        ),
        "source_config": "experiments/config/community_economy.json",
        "source_harvest_function": "experiments.exp6b_hybrid_replication.harvest_instances",
        "source_simulation_seed": 42,
        "source_num_members": 30,
        "source_cycles_sampled": [50, 100, 150, 200, 250, 300, 350, 400, 450, 499],
        "n_instances": len(v1_instances),
        "instances": [snapshot_v1_instance(inst, i) for i, inst in enumerate(v1_instances)],
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)

    total_pairs = sum(len(inst["eligible_pair_refs"]) for inst in snapshot["instances"])
    total_members = sum(len(inst["members"]) for inst in snapshot["instances"])
    total_tasks = sum(len(inst["tasks"]) for inst in snapshot["instances"])
    logger.info(f"Snapshot saved to {OUTPUT_PATH}")
    logger.info(f"  {snapshot['n_instances']} instances, "
                f"{total_pairs} total eligible pairs, "
                f"{total_members} total member-objects (with dup across instances), "
                f"{total_tasks} total task-objects, "
                f"{os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
