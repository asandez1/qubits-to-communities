# Package Manifest

**Total**: 70 files, 2.3 MB uncompressed.

## Top level

| File | Purpose |
|------|---------|
| `README.md` | Overview, headline result, reproduction guide |
| `LICENSE` | MIT |
| `CITATION.cff` | Structured citation metadata (GitHub / Zenodo auto-renders) |
| `requirements.txt` | Python deps pinned to the tested version triplet |
| `MANIFEST.md` | This file — file-level inventory |

## `paper/`

| File | Purpose |
|------|---------|
| `paper.md` | **Primary manuscript** — quantum pipeline paper (the Zenodo target) |
| `orquestia_context_paper.md` | Parent DCIN paper (OrquestIA v3 unified) — full sociological + governance context |
| `SOCIETY_MODEL.md` | Society model architecture + Tier A features (v2 Pydantic domain) |
| `figures/fig1_pipeline.png` | Three-stage pipeline architecture diagram |
| `figures/fig2_v3_circuit.png` | 19-qubit V3-extended circuit schematic |
| `figures/fig3_stratification.png` | **Three-tier regret stratification** — the headline figure |
| `figures/fig4_society.png` | Society model overview (members + cycle engine) |
| `figures/fig5_exp9.png` | Exp 9 dual-mode Gini evolution (5 architectures, 1×/day + 3×/day) |
| `figures/fig6_topology.png` | Social topology: greedy vs QUBO emergent structure |
| `figures/generate_figures.py` | Regenerate figs 1–3, 5–6 from JSON results |
| `figures/generate_society_visualization.py` | Regenerate fig 4 from society-model state |

## `code/`

### `code/core/` — domain model + solvers

| File | Purpose |
|------|---------|
| `domain.py` | Frozen Pydantic v2 domain: `Member`, `Task`, `MatchingInstance`, weight functions |
| `benchmark_model.py` | 7-constraint QUBO Hamiltonian, exhaustive + SA solvers, evaluate_solution |
| `benchmark_fixtures.py` | S/M/L-tier benchmark fixtures |
| `cycle_engine.py` | Pure state-transition function: credit, household pool, reputation, demurrage, energy recovery |
| `history.py` | Append-only JSONL simulation history + Gini/balance queries |
| `repository.py` | JSON persistence for Members/Tasks |
| `member.py`, `llm_member.py` | Rule-based and LLM-driven member implementations |
| `simulation_engine.py` | Full-cycle simulation driver for parent-paper experiments |
| `matching_service.py` | Random / greedy / QUBO matching strategies |
| `topology_analyzer.py` | Louvain communities, bridges, PI / TPM metrics |
| `credit_economy.py` | Bonding curves, demurrage, treasury reserve monitoring |

### `code/experiments/` — scripts that produced reported results

**Quantum pipeline (primary):**
| Script | Experiment |
|--------|-----------|
| `exp6d_v3_extended.py` | 19q V3-extended hardware benchmark (Exp 6d) — supports `--backend` and `--theta-path` |
| `exp6_hybrid_pipeline.py`, `exp6b_hybrid_replication.py`, `exp6c_scaled_training.py` | Earlier 7q V3 variants for context |
| `exp7_mechanism_isolation.py` | Four-arm mechanism isolation (Arms A, B, C, D) |
| `exp7_multi_trajectory.py` | 30-trajectory logical-depth noise (Arm B) |
| `exp7c_routing_depth.py` | Routing-depth depolarizing noise (Arm B') |
| `exp7d_multi_seed_theta.py` | **Phase 2A**: 10-seed SPSA θ training — reveals bimodal noiseless distribution |
| `exp7e_classical_baselines.py` | **Phase 2F**: LightGBM + GNN-MLP baselines vs K_c |
| `exp8_kernel_qubo.py` | K_c on 12D full constraint vector (Exp 8) |
| `regen_theta_seed3.py` | Regenerate seed-3 θ for ibm_kingston validation |

**Society model (pipeline comparison + ablation):**
| Script | Experiment |
|--------|-----------|
| `exp9_pipeline_comparison.py` | Exp 9: 5-architecture 30-day M/L-tier comparison |
| `exp9b_factorial_ablation.py` | **Phase 2B**: 2⁴ Tier-A factorial (age/tools/HH/urgency) |
| `exp9c_quantum_bridge.py` | **Phase 2D**: V3 noiseless as 6th architecture in Exp 9 |
| `exp9d_longitudinal.py` | **Phase 2G**: 180-cycle run with age advancement |
| `exp2e_ltier_comparison.py` | **Phase 2E**: L-tier scaling wrapper around Exp 9 |

**Performativity + parent-paper context:**
| Script | Experiment |
|--------|-----------|
| `exp3_performativity.py` | Exp 3: random/greedy/QUBO bridges, PI, TPM |
| `exp3b_n10_runner.py` | **Phase 2C**: Exp 3 at n=10 seeds (unlocks p < 0.001) |
| `exp1_credit_dynamics.py`, `exp1b_redistribution.py`, `exp4_bridge_resistance.py` | Parent-paper context (cited in §7) |

| File | Purpose |
|------|---------|
| `code/config/community_economy.json` | Community configuration (5 archetypes, 6 categories, bonding-curve params). **Only used to regenerate the frozen instance snapshot** and for generative parent-paper experiments (Exp 1–4). Not needed for any hardware-path experiment. |
| `code/activate_gpu.sh` | Helper: sources venv and fixes cusparse/nvjitlink LD_LIBRARY_PATH shadowing |

### `code/scripts/` — v2 unification + verification utilities

| File | Purpose |
|------|---------|
| `snapshot_hardware_instances.py` | Regenerate `hardware_instances_v2.json` from the v1 simulation (runs once to freeze the 8 benchmark instances as v2-native data) |
| `verify_v2_equivalence.py` | Byte-for-byte numerical comparison of v1 and v2 weights / QUBO / optima |
| `verify_bridge_equivalence.py` | Drop-in test of `hardware_instances_v1_compat()` vs v1 `harvest_instances()` |
| `verify_hardware_json_equivalence.py` | Replay all saved hardware JSONs through the v2 pipeline and confirm recorded regrets |

## `results/`

All raw results are included for transparency. Every number reported in the
paper is reproducible from these JSONs.

### Hardware (Heron R2)

| File | Content |
|------|---------|
| `exp6d/theta.npy` | **Pretrained seed-42 θ** (57 params, SPSA, used across all Exp 6/7/8 and §4 hardware runs) |
| `exp6d/theta_seed3.npy` | Regenerated seed-3 θ for ibm_kingston validation |
| `exp6d/exp6d_v3_extended.json` | ibm_fez hardware result (2026-04-10): 0.61% regret, 6/8 co-optimal |
| `exp6d/exp6d_v3_extended_ibm_marrakesh.json` | ibm_marrakesh (2026-04-11): 1.48%, 5/8 |
| `exp6d_kingston/exp6d_v3_extended_ibm_kingston.json` | **ibm_kingston + seed-42 θ** (2026-04-16): 0.61%, 6/8 (n=2 chip replication) |
| `exp6d_kingston_seed3/exp6d_v3_extended_ibm_kingston.json` | **ibm_kingston + seed-3 θ** (2026-04-16): 1.48%, 5/8 (within-chip bimodal) |

### Mechanism isolation + baselines

| File | Content |
|------|---------|
| `exp7/exp7_mechanism_isolation.json` | Four-arm controls (A, B, C, D) |
| `exp7/exp7b_multi_trajectory.json` | Arm B: 30 MC trajectories of ibm_fez NoiseModel at logical depth (1.48% ± 0.00%) |
| `exp7/exp7c_routing_depth.json` | Arm B': 5 trajectories of routing-depth depolarizing noise (1.48% ± 0.00%) |
| `exp7d/multi_seed_theta.json` | **Phase 2A**: 10 SPSA seeds, bimodal distribution (8/10 at 1.48%, 2/10 at 0.61%) |
| `exp7e/classical_baselines.json` | **Phase 2F**: LightGBM 1.48%, GNN-MLP 1.48%, K_c 18D 1.74%; K_c 9D remains best at 0.86% |
| `exp8/exp8_kernel_qubo.json` | Kernel-weighted QUBO (K_c 9D + 12D variants) |

### Society model

| File | Content |
|------|---------|
| `exp9b/factorial.json` | **Phase 2B**: 80 runs (2⁴ factorial × 5 architectures). Age Δ=−0.09, HH Δ=−0.08, Tools Δ=+0.03 |
| `exp9c/bridge.json` | **Phase 2D**: Exp 9 with 6th V3-noiseless architecture (Gini 0.396) |
| `exp9d/longitudinal.json` | **Phase 2G**: 180-cycle Gini trajectories (kc_9d reversal from 0.375 to 0.531) |
| `exp2e_ltier/exp9_l.json` | **Phase 2E**: L-tier (50×25, 428 vars). Acceptance wins at scale |

### Performativity

| File | Content |
|------|---------|
| `exp3_n10/exp3_validation.json` | **Phase 2C**: n=10 runs. QUBO > Greedy, p = 9.1 × 10⁻⁵, A₁₂ = 1.0, d ≈ 27 |
