# The OrquestIA Society Model

## Why This Exists

A community of 500 people generates hundreds of coordination problems every day. Alice needs her wiring fixed. Bob can tutor math but only after 3 PM because he picks up his daughter from school. Carol has a van sitting idle but won't drive 8 km across town for 3 credits when fuel costs more than that. David is an electrician with 130 credits losing value to demurrage — he *wants* to work but only if the task is nearby and the requester is trustworthy.

No spreadsheet captures this. No simple matching algorithm handles the *interactions* between skill, distance, schedule, price, reputation, and desperation. The decision "should Carol drive to that task?" depends on at least six variables simultaneously, and they multiply — not add.

The Society Model is the computational representation of this community. It encodes every member, every task, every schedule constraint, every economic incentive, and every trust relationship as structured data that an optimization engine (QUBO solver) and eventually an AI agent can reason about.

## What Problem It Solves

The core problem: **assign members to tasks** such that:

1. Each task gets exactly one qualified provider
2. No member is overloaded beyond their capacity
3. No member is assigned two tasks at the same time
4. Providers have adequate skill for the task
5. Requesters trust the assigned provider
6. Requesters can afford the task
7. The assignment is actually *worth it* for the provider (price minus travel cost minus inconvenience)

Constraints 1-3 are hard (physics — you can't be in two places at once). Constraints 4-7 are soft (economics — the system should prefer better matches but can compromise when necessary).

The naive approach — sort by skill, assign greedily — ignores distance, schedule, price, and the provider's willingness. The result: high-skill providers get sent on money-losing trips across town at 5 AM, while a perfectly adequate provider lives next door and is available right now.

The Society Model makes all these dimensions explicit so that the optimizer can find assignments where providers actually *want to show up*.

## Architecture

```
                     community_economy_v2.json
                              |
                              v
                      CommunityConfig
                     (archetype profiles)
                              |
                    generate_members(50)
                    generate_tasks(25)
                              |
                              v
    +--------------------------------------------------+
    |              Repository (persistence)             |
    |                                                  |
    |   members.json    tasks.json    catalog.json     |
    |                                                  |
    |   add_member()    add_task()    set_catalog()    |
    |   update_member() remove_task() get_task_type()  |
    |   members_near()  tasks_by_category()            |
    |   members_available(weekday, start, end)         |
    +--------------------------------------------------+
                              |
                     build_instance()
                              |
                              v
                     MatchingInstance
                    (frozen snapshot)
                              |
              +---------------+---------------+
              |               |               |
    acceptance_weights   linear_weights   skill_only_weights
              |               |               |
              v               v               v
                      QUBOBuilder
                   (7-constraint Hamiltonian)
                              |
                              v
                   solve_exhaustive() or
                   solve_simulated_annealing()
                              |
                              v
                      SolutionResult
                   (skill, net_value, distance,
                    convenience, violations)
```

Three layers, cleanly separated:

- **Domain Model** (`domain.py`) — what a member *is*, what a task *is*, what a schedule *means*. Pure data + behavioral methods. No I/O.
- **Repository** (`repository.py`) — where members and tasks are stored and queried. JSON persistence today, database tomorrow. The only layer that reads/writes files.
- **Solver** (`benchmark_model.py`) — how to build and solve the QUBO. Takes any object with the right attributes (duck typing). No knowledge of Pydantic or persistence.

## The Domain Model

### Members

A member is a frozen snapshot of a person's capabilities, constraints, and economic state at a point in time:

```python
Member(
    member_id="alice",
    archetype="professional",
    skill_levels={
        SkillCategory.ELECTRICAL: 0.90,
        SkillCategory.PLUMBING: 0.70,
    },
    reputation=0.85,           # [0, 1] — trust from the community
    time_availability=0.5,     # fraction of day available
    social_capital=0.6,        # network centrality
    max_task_load=2,           # max concurrent tasks
    credit_balance=80.0,       # current credits (demurrage eats this)
    age=42,
    location=Location(x=3, y=7),
    schedule=WeeklySchedule(
        default_day=DaySchedule(commitments=(
            TimeBlock.from_hours(9, 17, "work"),
        )),
        days={5: DaySchedule.free_day(), 6: DaySchedule.free_day()},
    ),
    household_id="hh_alice_bob",               # partners pool credits
    tools=frozenset({"toolkit", "electrical_tools"}),
    vehicle="van",                             # none | bike | car | van
)
```

**Frozen** means you cannot accidentally mutate Alice's balance while the optimizer is running. To update her state after a completed task:

```python
alice_after = alice.model_copy(update={"credit_balance": 70.0})
```

**Behavioral methods** on Member:

| Method | What it computes |
|--------|-----------------|
| `skill_in("electrical")` | Skill level in a category (0.0 if unlisted) |
| `travel_cost_to(location)` | Round-trip Manhattan distance x 0.5 cr/km |
| `net_value(task)` | Price minus travel cost (can be negative) |
| `acceptance_probability(task)` | Sigmoid: would this provider actually accept? |
| `get_capability_embedding(categories)` | Numeric vector for QUBO/kernel |

### The Acceptance Decision

The most important method. When the system proposes "Alice, fix wiring at 4th and Main for 10 credits," Alice's brain computes something like:

*"It's 2 km away (4 cr travel), I net 6 cr, it's during my lunch break (inconvenient but doable), the requester has 0.8 reputation (decent), and I have 80 credits so I'm not desperate. Probability I accept: 78%."*

The model encodes this as:

$$P_{accept} = \sigma(k \cdot (V_{hourly} - V_{threshold})) \times conv \times trust \times skill \times energy \times urgency$$

Where:
- $V_{hourly}$ = net value per hour after travel (travel cost scaled by vehicle: walking 2.0x, bike 1.2x, car 1.0x, van 1.1x)
- $V_{threshold}$ = baseline hourly rate, lowered by desperation (low balance)
- $conv$ = schedule convenience [0, 1]
- $trust$ = requester reputation [0, 1]
- $skill$ = provider's confidence in the task category [0, 1]
- $energy$ = remaining energy factor [0, 1] — drops to 0 if task exceeds capacity
- $urgency$ = 0.7 + 0.3 × task.urgency — urgent tasks attract slightly more willing providers

**Hard cutoffs** ($P = 0$ immediately):
- Member's age outside the task's [min_age, max_age] range
- Task's energy cost exceeds member's remaining energy
- Member lacks the task's `required_tools` (e.g., electrician without toolkit)
- Member's vehicle doesn't satisfy the task's `required_vehicle` (catering needs car+)

A broke member ($balance = 3$) has $V_{threshold} \approx 0$ and accepts almost anything. A wealthy member under demurrage pressure ($balance = 130$) has a higher threshold but is motivated to spend. A tired member (energy near zero after a full day) won't accept another task even if the pay is good.

### Age and Energy

Members have an **age** (8-100) that determines:

1. **Task eligibility** — legal and physical constraints. A 15-year-old can tutor another kid but can't rewire a house (min_age=18). A 70-year-old can teach for 2 hours but can't move furniture for 6 (max_age=65). These are hard filters — ineligible pairs never enter the QUBO.

2. **Energy capacity** — maximum daily working hours, derived from age:

| Age range | Capacity | Rationale |
|-----------|----------|-----------|
| 12-18 | 4-7 h | Growing, limited by school/law |
| 18-30 | 10 h | Peak physical capacity |
| 30-50 | 10-7 h | Gradual decline |
| 50-65 | 7-4 h | Accelerated decline |
| 65-80 | 4-2 h | Limited endurance |
| 80+ | 1-2 h | Minimal |

3. **Energy depletion** — each task consumes energy. A 40-year-old (8.5 h capacity) who does a 6-hour rewire has 2.5 h left — enough for a light switch fix (1 h) but not another heavy task. The QUBO's energy constraint prevents assigning two tasks whose combined energy exceeds the member's budget.

4. **Fatigue in acceptance** — a member at 0.5 h remaining energy has `energy_factor` near 0, sharply reducing acceptance probability for any new task. This naturally creates "afternoon fatigue" — morning tasks get higher acceptance than evening ones after a full day.

5. **Age-based overnight recovery** — younger members wake up fully restored; older members accumulate fatigue across days. Recovery rate:

| Age | Recovery rate | Behavior |
|-----|--------------|----------|
| 18-30 | 100% | Full overnight reset |
| 30-50 | 100% → 85% | Gradual decline |
| 50-65 | 85% → 65% | Noticeable lingering fatigue |
| 65-80 | 65% → 40% | Often starts next day tired |
| 80+ | 30% | Rarely fully recovered |

The formula: `new_energy = prev + recovery_rate × (capacity − prev)`. A 70-year-old who depletes fully (0 h) wakes up at 57% capacity — they need a full rest day to recover. A 25-year-old always wakes up at 100%.

```python
# A tired 55-year-old after a long task
worker = Member(member_id="carlos", age=55, ...)   # capacity 6.0h, rec_rate 0.78
after_task = worker.after_task(heavy_task)          # energy: 6.0 → 0.5h
rested = after_task.recover_overnight()             # energy: 0.5 + 0.78*5.5 = 4.8h
# Didn't fully recover — accumulated fatigue
rested.acceptance_probability(another_heavy_task)   # lower than yesterday
```

Within a day: energy depletes after each task. Overnight: age-based recovery applies. Multiple matching rounds per day (e.g., morning/afternoon/evening) deplete across rounds but only recover at the next day's start.

### Households, Tools, Vehicles

Real communities aren't atomic. Members share economic units (families, couples), have different material capacities (toolkit or no toolkit, van or bike), and tasks vary in urgency.

**Households** (`household_id`) — members sharing an ID form a household. After each cycle, 30% of the credit imbalance between household members is equalized. This captures family economic solidarity without full pooling (individual incentive to earn is preserved). Solo members have no household_id and are unaffected.

**Tools** (`tools: frozenset[str]`) — discrete capabilities. Electricians need `{"toolkit", "electrical_tools"}`. Cooks need `{"kitchen"}`. Tutors need nothing. Missing tools is a **hard filter** — the task doesn't even enter the QUBO as an eligible pair.

**Vehicles** (`vehicle: "none"|"bike"|"car"|"van"`) — travel cost multiplier + hard filter. A catering task that requires a car is not assignable to a member with only a bike. Walking (no vehicle) doubles the effective travel cost, making far tasks unprofitable for the poorest members.

**Task urgency** (`task.urgency: float ∈ [0, 1]`) — modulates acceptance upward. Urgent tasks (leaky pipe, emergency wiring) attract providers who might otherwise pass.

```python
# A retiree without tools won't be assigned electrical work,
# even if she has the skill
retiree = Member(..., age=68, tools=frozenset(), vehicle="car")
electrical_task = Task(task_type=TaskType.simple(
    "rewire", "electrical", duration=4, price=15,
    required_tools=frozenset({"toolkit", "electrical_tools"}),
))
retiree.acceptance_probability(electrical_task)  # 0.0 — hard cutoff
```

### Tasks

A task is not just "I need an electrician." It's a specific sub-type with linked effort and price:

```python
Task(
    task_id="fix_wiring",
    task_type=TaskType(
        name="install_fan",
        category=SkillCategory.ELECTRICAL,
        effort=EffortProfile(
            duration_hours=2.0,
            credit_cost=8.0,
            min_skill=0.5,
            min_trust=0.3,
        ),
    ),
    time_window=TimeWindow.from_hours(9, 11),   # 09:00-11:00
    location=Location(x=4, y=8),
    requester_id="bob",
    requester_reputation=0.80,
    weekday=1,  # Tuesday
)
```

The task taxonomy within each category matters:

| Category | Sub-type | Duration | Price | Min Skill | Energy | Age range |
|----------|----------|----------|-------|-----------|--------|-----------|
| Electrical | Rewire room | 6 h | 15 cr | 0.7 | 8 h | 18-65 |
| Electrical | Fix switch | 0.5 h | 3 cr | 0.3 | 1 h | 12-80 |
| Electrical | Install fan | 2 h | 8 cr | 0.5 | 3 h | 18-70 |
| Cooking | Meal prep | 2 h | 5 cr | 0.4 | 2 h | 12-80 |
| Cooking | Catering event | 5 h | 12 cr | 0.7 | 6 h | 18-65 |
| Cooking | Baking order | 1.5 h | 4 cr | 0.3 | 1.5 h | 12-80 |
| Transport | Furniture move | 3 h | 10 cr | 0.5 | 5 h | 18-55 |
| Transport | Local delivery | 1 h | 4 cr | 0.3 | 1 h | 16-75 |
| Childcare | Afternoon care | 3 h | 8 cr | 0.5 | 3 h | 18-70 |
| Childcare | Homework help | 1.5 h | 5 cr | 0.3 | 1.5 h | 14-80 |

Each sub-type has a linked energy cost (how physically draining it is) and age range (who can legally/physically do it). A "furniture move" costs 5 energy hours on a 3-hour task — it's more exhausting than its duration suggests. A 60-year-old with 4 h energy capacity can't take it.

### Schedules

Real people have fixed commitments. The model captures this as blocked periods per day of the week:

```python
# Bob: works 9-5, picks up daughter at 3:15, free on weekends
bob_weekday = DaySchedule(commitments=(
    TimeBlock.from_hours(9, 17, "work"),
    TimeBlock.from_hours(15.25, 15.75, "school_pickup"),
))
bob_schedule = WeeklySchedule(
    default_day=bob_weekday,
    days={5: DaySchedule.free_day(), 6: DaySchedule.free_day()},
)
```

Available time is computed by inverting blocked periods:
- Monday: 00:00-09:00 and 17:00-24:00
- Saturday: all day

When the optimizer considers assigning Bob to a 10 AM task on Wednesday, `bob_schedule.convenience(wednesday, TimeWindow(10, 12))` returns **0.0** — he's at work. The same task on Saturday returns **1.0**.

### Time Representation

All time uses **15-minute blocks** (integers 0-96 per day). Block 36 = 09:00, block 68 = 17:00. This eliminates floating-point comparison bugs (is 9.0000001 before or after 9.0?) while providing enough granularity for real scheduling.

Convenience constructor: `TimeWindow.from_hours(9, 17)` creates `TimeWindow(start=36, end=68)`.

### Distance and Travel Cost

Locations use **Manhattan distance** — realistic for grid-based urban communities where people follow streets, not straight lines.

Travel cost = distance x 0.5 cr/km x 2 (round trip). A task 8 km away costs 8 credits in travel. If the task pays 10 credits, the net value is only 2 credits. If the task pays 5 credits, the provider **loses** 3 credits going there.

## The Five Archetypes

The community is not homogeneous. Five archetypes represent the diversity of a real neighborhood:

| Archetype | Pop% | Age | Energy | Primary Skills | Schedule | Balance |
|-----------|------|-----|--------|---------------|----------|---------|
| Professional | 25% | 28-55 | 7-10 h | Electrical, Plumbing | 9-5 worker, free weekends | 30-130 cr |
| Parent | 30% | 25-45 | 7-9 h | Cooking, Childcare, Tutoring | School pickup 3pm, evenings | 8-80 cr |
| Gig Worker | 25% | 18-40 | 9-10 h | Transport | Fully flexible | 3-25 cr |
| Retiree | 15% | 62-78 | 2-4 h | Tutoring, Cooking | Mornings, rest evenings | 65-140 cr |
| Newcomer | 5% | 16-25 | 7-10 h | Low in everything | Fully flexible | 2-10 cr |

Each archetype creates specific tensions through the intersection of age, energy, and economics:
- The **professional** (age 35, energy 9.2h) has high skill but narrow availability — the only window for a Tuesday electrical task is before 9 AM or after 5 PM. Can handle heavy tasks but not after a full workday.
- The **parent** (age 35, energy 9.2h) has moderate skills and moderate availability — great for midday cooking or tutoring, blocked during school hours. Energy matches task demands well.
- The **gig worker** (age 25, energy 10h) will go anywhere anytime but has low reputation — high-trust tasks (childcare, electrical) are out of reach. Peak energy means they can stack multiple tasks per day.
- The **retiree** (age 70, energy 3.3h) has high credits burning under demurrage — motivated to work, high trust, but energy limits them to 1-2 light tasks per day. Furniture moving is age-blocked entirely.
- The **newcomer** (age 20, energy 10h) has peak energy but qualifies for almost nothing — low skill, low reputation, limited to tasks with min_age < 18 until they build trust.

## The QUBO Hamiltonian

The matching problem is formulated as a QUBO (Quadratic Unconstrained Binary Optimization):

$$H = -\sum_{i,j} w_{ij} \cdot x_{ij} + \sum_{k=1}^{7} \lambda_k \cdot H_k$$

Where $x_{ij} = 1$ if member $i$ is assigned to task $j$, and the seven constraint terms are:

| # | Constraint | Type | What it prevents |
|---|-----------|------|------------------|
| 1 | One-member-per-task | Hard | Two people assigned to the same task |
| 2 | Task-load limit | Hard | Overloading a member beyond capacity |
| 3 | Temporal non-overlap | Hard | Same member doing two tasks at once |
| 4 | Skill eligibility | Soft | Under-qualified assignment |
| 5 | Trust threshold | Soft | Low-reputation provider on sensitive task |
| 6 | Credit affordability | Soft | Requester can't afford the task |
| 7 | Effort convenience | Soft | Bad time + low pay combination |
| 8 | Energy capacity | Hard | Two tasks whose combined energy exceeds member's budget |

Hard constraints use $\lambda = 10 \times \max(w)$ — violation always costs more than any benefit. Soft constraints use $\lambda = 5 \times \max(w)$ — the solver can override them if no better option exists. Age eligibility is enforced in pre-filtering (ineligible pairs never enter the QUBO).

## Weight Functions

The weight $w_{ij}$ determines how desirable it is to assign member $i$ to task $j$. Four functions are available:

**`acceptance_weights`** (recommended) — models the provider's actual willingness:
$$w_{ij} = P_{accept}(i, j) = \sigma(k \cdot (V_{hourly} - V_{threshold})) \times conv \times trust \times skill$$

This captures nonlinear interactions: "far + cheap + bad time = near zero" even if each factor alone seems acceptable. A kernel trained on community data should learn to approximate this function.

**`linear_weights`** — the paper's original formula:
$$w_{ij} = \alpha \cdot sim(e_i, e_j) + \beta \cdot rep_i + \gamma \cdot skill_i + \delta \cdot price_j$$

Treats all factors as independent additive terms. Works when skill dominates, fails when distance/time/price interact.

**`skill_only_weights`** — just the provider's skill level in the task category. Baseline.

**`uniform_weights`** — all pairs equal weight. Tests constraint-only behavior.

## Evaluation Metrics

After solving, every assignment is evaluated on external quality metrics that don't depend on which weight function was used:

| Metric | What it measures |
|--------|-----------------|
| `mean_provider_skill` | Average skill of assigned providers in their task category |
| `mean_net_value` | Average credits earned after travel costs |
| `mean_distance_km` | Average provider-to-task distance |
| `mean_convenience` | Average schedule fit [0=fully blocked, 1=fully available] |
| `money_losing_assignments` | Count where travel cost exceeds payment |
| `total_violations` | Constraint violations (should be 0) |

These metrics are what matter for a real community. A "good" matching has high skill, positive net value, short distance, high convenience, and zero money-losing assignments.

External macro factors (regional economy, security, weather) are intentionally scoped out of this pipeline study; they are treated as uniform scaling factors in the parent OrquestIA model and left for future coupled-community experiments.

## Benchmark Results

Three tiers of fixed test instances:

| Tier | Members | Tasks | Vars | Key finding |
|------|---------|-------|------|-------------|
| S | 4 | 3 | 12 | Paper's worked example. All methods agree. |
| M | 12 | 6 | 36 | Methods diverge: acceptance sacrifices skill (0.70 vs 0.89) for 46% higher net value and 59% better schedule fit |
| L | 50 | 25 | 428 | 15-20 money-losing assignments across all methods — the nonlinear decision space a kernel must learn |

The M-tier result is the key validation: `acceptance_weights` produces fundamentally different assignments than `linear_weights` because it models the provider's actual decision. Linear assigns the highest-skill provider regardless of distance or schedule. Acceptance assigns the provider most likely to *show up*.

## How to Use It

**Add a member to the community:**
```python
from core.repository import Repository
from core.domain import Member, SkillCategory, Location, WeeklySchedule

repo = Repository.load("data/my_community")
repo.add_member(Member(
    member_id="diana",
    archetype="parent",
    skill_levels={SkillCategory.COOKING: 0.85, SkillCategory.CHILDCARE: 0.70},
    reputation=0.75,
    time_availability=0.6,
    credit_balance=45.0,
    location=Location(x=6, y=8),
    schedule=WeeklySchedule.from_preferred_hours(9, 15),
))
repo.save()
```

**Query the community:**
```python
cooks = repo.members_by_category("cooking", min_skill=0.5)
nearby = repo.members_near(Location(x=5, y=7), max_distance_km=3)
available = repo.members_available(weekday=1, start_block=36, end_block=48)  # Tue 9-12
```

**Run the optimizer:**
```python
from core.domain import acceptance_weights
from core.benchmark_model import QUBOBuilder, solve_simulated_annealing, evaluate_solution

instance = repo.build_instance("tuesday_morning", skill_threshold=0.3)
weights = acceptance_weights(instance, weekday=1)
builder = QUBOBuilder(instance, weights)
Q = builder.build()
bits, energy = solve_simulated_annealing(Q, builder.n_vars)
result = evaluate_solution(instance, builder, bits)

for a in result.assignments:
    print(f"{a['member_id']} -> {a['task_id']} "
          f"skill={a['skill']:.2f} net={a['net_value']:+.1f}cr "
          f"dist={a['distance_km']:.1f}km")
```

## State Evolution: The Cycle Engine

The domain model is frozen — a snapshot at one point in time. But a real community evolves: providers earn credits, requesters spend them, reputations change, inequality emerges. The **Cycle Engine** bridges this gap.

### How It Works

Each coordination cycle follows this flow:

```
CycleSnapshot (cycle N)
       |
       v
  to_matching_instance()  →  QUBOBuilder  →  Solver  →  SolutionResult
       |                                                       |
       v                                                       v
  CycleEngine.advance(snapshot, assignments)  →  CycleSnapshot (cycle N+1)
       |
       v
  SimulationHistory.append(snapshot)  →  snapshots.jsonl
```

`advance()` is a **pure function** — frozen state in, frozen state out. No mutation. Each step creates new Member objects via `model_copy(update={...})`:

1. **Credit transfers** — requester pays provider at task price, minus protocol fee (2.5%)
2. **Reputation updates** — successful completion: provider +0.01, requester +0.005
3. **Demurrage** — floating rate `d = d_max × (1 - V/V_target)` applied after grace period
4. **Schema evolution** — new categories added, existing members get skill=0.0
5. **New members/tasks** — community grows over time

### Running a Simulation

```python
from core.domain import acceptance_weights
from core.history import SimulationHistory
from core.cycle_engine import CycleEngine, create_initial_snapshot
from core.benchmark_model import QUBOBuilder, solve_simulated_annealing, evaluate_solution

# Setup
snap = create_initial_snapshot(members, tasks, categories, prices)
engine = CycleEngine(seed=42)
history = SimulationHistory("runs/my_sim", seed=42)
history.append(snap)

# Run 100 cycles
for cycle in range(100):
    instance = snap.to_matching_instance()
    weights = acceptance_weights(instance)
    builder = QUBOBuilder(instance, weights)
    Q = builder.build()
    bits, energy = solve_simulated_annealing(Q, builder.n_vars)
    result = evaluate_solution(instance, builder, bits)

    snap = engine.advance(snap, result.assignments)
    history.append(snap)

history.save()  # persists to JSONL
```

### Querying History

Every cycle's complete state is stored. You can query any point in time:

```python
# What was Alice's balance at cycle 50?
alice_50 = history.member_at("alice", 50)
print(alice_50.credit_balance)

# How did Alice's balance evolve?
for cycle, balance in history.balance_series("alice"):
    print(f"  Cycle {cycle}: {balance:.1f} cr")

# Gini coefficient at any cycle
print(history.gini_at(50))      # 0.254

# Gini over time (tracks inequality emergence)
for cycle, gini in history.gini_series():
    print(f"  Cycle {cycle}: Gini={gini:.4f}")

# Replay any cycle through the QUBO solver
instance_50 = history.matching_instance_at(50)
# ... run QUBOBuilder, compare methods, compute regret ...
```

### Schema Evolution

Adding a new category mid-simulation doesn't lose history:

```python
from core.domain import Task, TaskType, TimeWindow

gardening_task = Task(
    task_id="garden_1",
    task_type=TaskType.simple("weeding", "gardening", duration=2, price=4),
    time_window=TimeWindow.from_hours(8, 10),
)

# At cycle 200, add "gardening" category
snap = engine.advance(snap, assignments,
                      new_categories=("gardening",),
                      new_tasks=(gardening_task,))

# History before cycle 200 has 6 categories
# History after cycle 200 has 7 categories
# Existing members: skill_in("gardening") = 0.0
```

### Persistence Format

```
runs/my_sim/
  manifest.json       # seed, n_cycles, catalog versions
  config.json         # original config (immutable)
  snapshots.jsonl     # one JSON line per cycle (append-only)
```

50 members × 1000 cycles ≈ 25 MB. Loads in under 1 second.

### What the Simulation Reveals

Running 20 cycles on the S-tier fixture (4 members, 3 tasks):

| Cycle | Member A Balance | Gini |
|-------|-----------------|------|
| 0 | 50.0 cr | 0.000 |
| 5 | 100.0 cr | 0.127 |
| 10 | 150.0 cr | 0.189 |
| 15 | 183.4 cr | 0.227 |
| 20 | 213.7 cr | 0.254 |

Inequality emerges naturally from the bonding curve pricing — members with scarce skills (electrical) earn more per task than those with common skills. This matches the paper's Experiment 1 finding (Gini 0.73 at 1000 cycles with 500 members). Progressive demurrage (Experiment 1b) would reduce this.

## What This Enables

The Society Model is the foundation for four downstream systems:

1. **QUBO optimization** — the immediate use case. The model provides the data, the solver finds assignments.

2. **AI agent interaction** — when LLM agents replace rule-based decisions, they need to query "who's available near this location Tuesday afternoon with cooking skill above 0.5?" The repository provides this.

3. **Kernel/quantum learning** — the `acceptance_probability` function is the target a kernel ridge regressor or quantum circuit should learn to approximate. The model provides training data (member features + task features + acceptance label) and evaluation metrics (did the learned weights produce better coordination?).

4. **Multi-cycle simulation** — the history + cycle engine enable running full simulations: tracking Gini evolution, testing redistribution mechanisms, measuring how reputation builds over time, and comparing coordination strategies across hundreds of cycles.

## Pipeline Architecture Comparison (Experiment 9)

Running 30 days on the M-tier fixture (12 members aged 17-68, grouped into 8 households, varied tools and vehicles, 2 urgent tasks) under five weight functions, in two modes: **1x/day** (one matching round per day, energy fully resets) and **3x/day** (three rounds per day, energy depletes across rounds, age-based overnight recovery).

### Mode A: 1x/day (30 cycles)

| Architecture | Gini | HH Gini | AvgBal | MinBal | Fill% | EDef |
|-------------|------|---------|--------|--------|-------|------|
| kc_9d | **0.375** | 0.412 | 106.2 | 2.0 cr | 93.3% | 1.08 h |
| acceptance | 0.381 | **0.407** | 106.2 | 2.0 cr | 93.3% | 1.15 h |
| kc_12d | 0.408 | 0.427 | 106.2 | 2.0 cr | 93.3% | 1.17 h |
| linear | 0.413 | 0.431 | 106.2 | 2.0 cr | 93.3% | 1.18 h |
| skill_only | 0.413 | 0.431 | 106.2 | 2.0 cr | 93.3% | 1.18 h |

### Mode B: 3x/day (90 cycles, 30 days)

| Architecture | Gini | HH Gini | AvgBal | MinBal | Fill% | EDef |
|-------------|------|---------|--------|--------|-------|------|
| kc_12d | **0.412** | **0.450** | 144.4 | 0.9 cr | 86.7% | 2.27 h |
| acceptance | 0.415 | 0.452 | 144.5 | 0.9 cr | 86.7% | 2.25 h |
| skill_only | 0.416 | 0.453 | 144.5 | 0.9 cr | 86.7% | 2.25 h |
| linear | 0.416 | 0.453 | 144.5 | 0.9 cr | 86.7% | 2.25 h |
| kc_9d | 0.422 | 0.459 | 143.6 | 0.9 cr | 86.7% | 2.19 h |

### What Tier A realism revealed

**Fulfillment rate drops below 100%.** Mode A fills 93.3% of tasks; Mode B fills 86.7%. The tool and age hard filters bite: when the only electrician with a toolkit is already booked or exhausted, some electrical tasks go undone. This is realism — **not every posted task gets fulfilled**. Fulfillment rate becomes a first-class quality metric the system must balance against skill and equity.

**Household Gini > Individual Gini.** Household inequality is higher than individual inequality because pooling equalizes within families, making family members nearly identical, but widens the gap between rich and poor *families*. This is sociologically accurate — Piketty's wealth concentration operates at the household level.

**Energy deficit emerges.** Members are measurably tired at end of day: 1.08-1.18 h/member in Mode A, 2.19-2.27 h/member in Mode B. Multi-cycle days create real fatigue that the 1x/day mode hides.

**Architecture differences compress.** In Mode B, all five methods produce Gini 0.412-0.422 (0.010 range). Before Tier A, the range was 0.017. The structural features — household pooling, tool filters, age cutoffs — now dominate the weight function's algorithmic choices. **The society itself enforces redistribution; the optimizer has less room to move.**

**kc_9d still wins Mode A** (0.375 Gini) but the gap to `acceptance` (0.381) is marginal. The kernel's implicit equity advantage shrinks as explicit structural equity mechanisms take over.

**Mode matters as much as architecture.** A community that runs matchmaking 3x/day accepts lower fulfillment (86.7% vs 93.3%) and higher fatigue (2.25 h vs 1.15 h deficit) in exchange for richer coordination (more total assignments, faster reputation building: average 0.82 vs 0.74). Temporal frequency is itself a policy lever.

## Future Functionality

What the current model does well: members have realistic biology (age, energy, recovery), geography (Manhattan distance, travel costs by vehicle), schedules (15-min blocks with blocked commitments), economics (credits, demurrage, household pooling), and material constraints (tools, vehicles). The QUBO enforces 8 constraints including the 5 paper constraints plus credit affordability, effort convenience, and energy capacity. The cycle engine tracks full snapshots per cycle with age-based overnight recovery.

What's still missing, grouped by priority:

### Tier B — Planned next round

**Behavioral richness in CycleEngine.advance():** the engine currently assumes every assignment succeeds. The next layer adds realistic member decisions:
- **Provider rejection** — `acceptance_probability` below a threshold means the task goes unfulfilled and rolls to the next cycle (deadline permitting)
- **Partial completion** — quality score based on skill match and remaining energy; low quality triggers partial payment
- **Bad ratings** — failed or low-quality tasks reduce provider reputation (currently reputation only goes up)
- **Requester disputes** — credit refund path when tasks aren't delivered; dispute flag on the member

These interact with Tier A in expected ways: a tired 55-year-old with 0.3 h energy left is more likely to fail a 2-hour task than a rested 25-year-old, so bad ratings concentrate on exhausted seniors who didn't decline in time.

**Relational (directed) trust:** replace scalar `reputation` with a social graph where Alice trusts Bob 0.9 but Carol only 0.3 (regardless of Carol's community-wide reputation). Acceptance probability becomes `P × relational_trust(provider, requester)`. QUBO prefers known providers even at slightly lower skill, producing small-world clustering in the emergent graph. Implementation: add `social_graph: dict[(MemberID, MemberID), float]` to repository, seeded by households (family members trust each other at 1.0) and updated by successful task completions.

**Skill learning and depreciation:** skills are static today. Realistically:
- After successful task: `skill_levels[category] += learning_rate × (1 - current_skill)` (diminishing returns)
- Unused skills decay slowly toward 0
- Long simulations would show "skill polarization" (specialists) or "broadening" (generalists) depending on architecture

Low code cost; only matters for runs of 100+ cycles.

**Shared household energy and schedule constraints:** currently households pool credits but not time. A more realistic model: if one parent does a 4-hour task during childcare hours, the other parent becomes schedule-blocked. This captures the "who picks up the kids?" coordination that families actually negotiate.

### Tier C — Deferred, probably needed for publication-grade simulation

**Health shocks and burnout:** stochastic events (probability scales with age × recent energy depletion) that temporarily reduce a member's capacity for several cycles. Older members suffer longer shocks. Requires a separate RNG stream to preserve reproducibility.

**Non-credit exchanges (barter):** when both parties are below some balance threshold, allow task-for-task or task-for-goods exchanges. Prevents the pure credit model from forcing starvation on newcomers. New TransactionType enum and constraint.

**External world leakage:** the community isn't closed. Members earn outside income, buy goods outside the network, occasionally migrate in or out. Add `external_income_per_cycle` per archetype and `migration_probability`. Prevents unrealistic credit collapse in long runs and tests resilience under external shocks.

**LLM-driven agent decisions:** currently the cycle engine uses deterministic rules (provider always accepts if filter passes). The parent project has infrastructure for Phi-2/Mistral agents via Ollama. Replacing the rule-based acceptance with an LLM that reads the task description, the provider's history, and the requester's message would produce emergent behavior the sigmoid can't capture — but at ~200ms per decision × 50 members × 30 cycles, a single simulation run becomes ~5 minutes instead of 30 seconds.

### Beyond this round — speculative

**Multi-archetype life stages:** members age across cycles. A gig worker at 25 becomes a parent at 32, a professional at 40, eventually a retiree at 65. This makes the society truly longitudinal and lets you measure whether the current coordination rules are sustainable across a human lifetime.

**Community genesis and collapse:** start with 5 members and watch the community grow through archetype-triggered migration. Measure when the coordination system bootstraps vs collapses. This is the ultimate stress test for the paper's "parallel layer" thesis.

**Coupled communities:** two neighborhoods that can trade with each other through bridge members. Tests whether the coordination system scales without becoming fragile. Requires the fiat bridge mechanics from the parent paper.

**Emergent services:** members identify unmet demand patterns and propose new task types that don't yet exist in the catalog. The Emergence Agent from the DCIN architecture would drive this — currently just a stub.

**Physical resources and depletion:** a kitchen is not infinite — hosting 3 catering events in a row depletes the pantry. Tools wear out. Vehicles need maintenance. Adds a second resource axis orthogonal to credits.

### The point

Everything above is additive. The current model is already realistic enough to produce the findings in Experiment 9. But the evaluator is right that it's still a simplification — a real neighborhood has relational trust, skill evolution, health shocks, and members who sometimes just say no. Each addition makes the model more like a living community and less like a matching problem.

The architecture is designed to grow. Adding a new dimension (e.g., `owns_car: bool` or `tools: set[str]` as we already did) means adding a field to `Member` or `Task`, updating the JSON config, and the entire pipeline — repository, instance building, QUBO solving, evaluation, simulation — works unchanged. The frozen Pydantic foundation makes extension cheap; the only discipline required is to keep new features additive and defaults backward-compatible.
