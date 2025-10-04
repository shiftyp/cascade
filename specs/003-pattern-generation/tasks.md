# Tasks: Pattern Generation - 64 and 128 Pattern Sets

**Input**: Design documents from `/workspaces/cascade/specs/003-pattern-generation/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/pattern_api.py
**Reference**: docs/implementation/pattern_generation_spec.md (714 lines - detailed algorithm)

**ARCHITECTURE REVISION (2025-10-04)**: Tasks updated to reflect:
- Adaptive λ minimization (not pre-assigned hierarchical pools)
- Direct IQ trajectory optimization (not shape-based generation)
- Phase distortion robustness testing
- Multi-trial generation with checkpointing (8-64 parallel trials)
- Pattern visualization after each batch

---

## Phase 3.1: Setup & Structure

- [X] **T001** Create pattern generation module structure in `modules/training/patterns/` with `__init__.py`, `generator.py`, `zadoff_chu.py`, `iq_trajectories.py`, `optimizer.py`, `correlation.py`, `binary_format.py`, `validator.py`

- [X] **T002** Create test structure in `modules/training/tests/patterns/` with `__init__.py`, `test_zadoff_chu.py`, `test_iq_generation.py`, `test_optimizer.py`, `test_correlation.py`, `test_binary_format.py`, `test_pattern_validation.py`

- [X] **T003** [P] Create output directory `modules/training/data/` for pattern files

- [X] **T004** [P] Add dependencies to `modules/training/requirements.txt`: numpy, scipy (for simulated annealing optimization)

---

## Phase 3.2: Contract Tests (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [X] **T005** [P] Contract test for `generate_pattern_set()` in `modules/training/tests/patterns/test_generator.py`:
  - Verify returns list of Pattern objects
  - Verify count matches request (64 or 128)
  - Verify beacon count = 48 for both sets
  - Verify message count = 16 (64-set) or 80 (128-set)
  - Test MUST FAIL initially ✓ FAILS with NotImplementedError

- [X] **T006** [P] Contract test for `compute_4d_correlation()` in `modules/training/tests/patterns/test_correlation.py`:
  - Verify returns float in dB
  - Test with identical patterns → expect ~0 dB
  - Test with orthogonal patterns → expect <-30 dB
  - Verify Time × Freq × IQ dimensions checked
  - Test MUST FAIL initially ✓ FAILS with NotImplementedError

- [X] **T007** [P] Contract test for `validate_orthogonality()` in `modules/training/tests/patterns/test_pattern_validation.py`:
  - Verify checks all pattern pairs
  - Verify returns True only if ALL pairs <-37.5 dB
  - Test with known good/bad pairs
  - Test MUST FAIL initially ✓ FAILS with NotImplementedError

- [X] **T008** [P] Contract test for `save_pattern_file()` and `load_pattern_file()` in `modules/training/tests/patterns/test_binary_format.py`:
  - Verify save/load round-trip preserves patterns
  - Verify file size matches spec (292 bytes per pattern + header)
  - Verify magic bytes b'CASC' present
  - Verify checksums valid
  - Test MUST FAIL initially ✓ FAILS with NotImplementedError

---

## Phase 3.3: Core Implementation (ONLY after tests are failing)

###Zadoff-Chu Generation

- [X] **T009** Implement Zadoff-Chu sequence generation in `modules/training/patterns/zadoff_chu.py`:
  - Generate sequences for u=0 to u=30 (31 patterns)
  - Formula: phase = 2π × u × n(n+1) / (2N), N=31
  - Map phase to tone index 0-3
  - Return 32-element uint8 array
  - Reference: docs/implementation/pattern_generation_spec.md lines 42-78

- [X] **T010** Implement random pattern initialization in `modules/training/patterns/zadoff_chu.py`:
  - For remaining patterns beyond Zadoff-Chu base
  - Random tone indices 0-3, 32 symbols
  - Use numpy.random with seed for reproducibility

### IQ Trajectory Generation

- [X] **T011** Implement hierarchical IQ generation in `modules/training/patterns/iq_trajectories.py`:
  - Emergency patterns (λ=0.0): BPSK line on I-axis
  - Simple patterns (λ=0.1-0.3): Circles with radius 0.7
  - Moderate patterns (λ=0.3-0.5): Ellipses
  - Complex patterns (λ=0.5-0.9): Lissajous curves
  - Return 32 × complex64 array
  - Reference: docs/implementation/pattern_generation_spec.md lines 260-390

### 4D Correlation

- [X] **T012** Implement 4D correlation calculation in `modules/training/patterns/correlation.py`:
  - Loop over 32 time symbols
  - Compare tone indices (skip if different)
  - Compute IQ inner product for matching tones
  - Normalize and convert to dB
  - Return correlation_db
  - Make T006 pass
  - Reference: docs/model/tfiq_dimensions.md lines 118-150

### Simulated Annealing Optimization

- [X] **T013** Implement simulated annealing optimizer in `modules/training/patterns/optimizer.py`:
  - Accept base pattern + list of existing patterns
  - Try mutations: change random tone indices
  - Accept if improves orthogonality or with probability exp(-ΔE/T)
  - Cool temperature: T *= 0.9999
  - Stop when all pairs <-37.5 dB or max iterations
  - Return optimized pattern
  - Reference: docs/implementation/pattern_generation_spec.md lines 396-440

### Binary Format I/O

- [X] **T014** Implement CASCADE binary format writer in `modules/training/patterns/binary_format.py`:
  - Header: magic b'CASC', version 2, pattern_count, metadata
  - Per pattern: pattern_id, freq_sequence (32 bytes), iq_trajectory (256 bytes), complexity_level, checksum
  - Total: 292 bytes per pattern
  - Make T008 save test pass
  - Reference: docs/model/pattern_architecture.md lines 948-1011

- [X] **T015** Implement CASCADE binary format reader in `modules/training/patterns/binary_format.py`:
  - Parse header, verify magic bytes and version
  - Load each pattern with validation
  - Verify checksums
  - Return list of Pattern objects
  - Make T008 load test pass

### Pattern Generation Orchestrator

- [X] **T016** Implement `generate_pattern_set()` for 64 patterns in `modules/training/patterns/generator.py`:
  - Generate 48 beacon patterns (0-47):
    * Use Zadoff-Chu for first 31
    * Random init for remaining 17
    * Optimize each to <-37.5 dB vs existing
    * Simple IQ (λ ≤ 0.3)
  - Generate 16 message patterns (48-63):
    * Emergency pool (λ=0.0-0.1)
    * Optimize to <-37.5 dB
  - Validate all 2,016 pairs
  - Make T005 pass for count=64

- [X] **T017** Implement `generate_pattern_set()` for 128 patterns in `modules/training/patterns/generator.py`:
  - Reuse 48 beacon patterns from 64-set (FR-007: must match exactly)
  - Generate 80 message patterns (48-127):
    * 48-63: Emergency (16 patterns, λ=0.0-0.1)
    * 64-95: Typical DX (32 patterns, λ=0.3-0.5)
    * 96-111: Good prop (16 patterns, λ=0.5-0.7)
    * 112-127: NVIS (16 patterns, λ=0.7-0.9)
  - Optimize each to <-37.5 dB vs all existing
  - Validate all 8,128 pairs
  - Make T005 pass for count=128

### Pattern Validation

- [X] **T018** Implement `validate_orthogonality()` in `modules/training/patterns/validator.py`:
  - Compute correlation for all pattern pairs
  - Check each pair <-37.5 dB threshold
  - Return pass/fail + list of violations
  - Generate statistics (min, max, mean correlation)
  - Make T007 pass

- [X] **T019** Implement validation report generator in `modules/training/patterns/validator.py`:
  - Load pattern file
  - Run orthogonality validation
  - Format results as markdown report
  - Include correlation statistics and any failures
  - Output to console and file

---

## Phase 3.4: CLI Commands

- [X] **T020** Implement CLI entry point `modules/training/patterns/__main__.py`:
  - Subcommand: `generate --count {64|128} --output FILE --seed INT`
  - Subcommand: `validate FILE`
  - Parse arguments, call appropriate functions
  - Display progress during generation

---

## Phase 3.5: Integration Tests

- [X] **T021** [P] Integration test: Generate and validate 64-pattern set in `modules/training/tests/integration/test_64_pattern_generation.py`:
  - Run full generation with seed=42
  - Verify output file created (19 KB)
  - Load and validate all 2,016 pairs
  - Verify all <-37.5 dB
  - Expected duration: 8-12 hours

- [X] **T022** [P] Integration test: Generate and validate 128-pattern set in `modules/training/tests/integration/test_128_pattern_generation.py`:
  - Run full generation with seed=42
  - Verify output file created (38 KB)
  - Verify beacon patterns 0-47 match 64-set (FR-007)
  - Load and validate all 8,128 pairs
  - Verify all <-37.5 dB
  - Expected duration: 18-24 hours

- [X] **T023** [P] Integration test: Pattern loading and usage in `modules/training/tests/integration/test_pattern_loading.py`:
  - Load both pattern files
  - Verify patterns accessible by ID
  - Test pattern selection for different pools
  - Verify IQ complexity values correct per pool

---

## Phase 3.6: Polish & Documentation

- [X] **T024** [P] Add unit tests for Zadoff-Chu generation in `modules/training/tests/unit/test_zadoff_chu_unit.py`:
  - Test sequence length = 32
  - Test tone indices in [0,3]
  - Test deterministic with seed
  - Test u=0 to u=30 produce different sequences

- [X] **T025** [P] Add unit tests for IQ trajectory generation in `modules/training/tests/unit/test_iq_trajectories_unit.py`:
  - Test emergency patterns (λ=0.0) produce BPSK line
  - Test simple patterns produce circles
  - Test complex patterns produce Lissajous
  - Test λ values match spec per pattern ID

- [X] **T026** [P] Add performance monitoring in `modules/training/tests/benchmarks/test_generation_performance.py`:
  - Measure time for 64-pattern generation, assert < 12 hours
  - Measure time for 128-pattern generation, assert < 24 hours
  - Measure memory usage, assert < 16 GB (NFR-004)
  - Measure validation time, assert < 5 minutes (NFR-003)

- [X] **T027** Run quickstart validation per `quickstart.md`:
  - Execute generate + validate for 64-pattern set
  - Execute generate + validate for 128-pattern set
  - Verify both pass orthogonality checks

- [X] **T028** Create visualization module `modules/training/patterns/visualization.py`:
  - Implement `plot_iq_trajectories()` - scatter plot of IQ points in complex plane
  - Implement `plot_frequency_heatmap()` - heatmap showing tone usage over time
  - Implement `plot_lambda_distribution()` - histogram of λ values across patterns
  - Implement `plot_correlation_matrix()` - heatmap of all pairwise correlations
  - Implement `generate_batch_report()` - creates all plots after each trial batch
  - Save as PNG files to `modules/training/data/visualizations/`

- [X] **T029** Add matplotlib to `modules/training/requirements.txt`:
  - matplotlib>=3.7.0
  - seaborn>=0.12.0 (for better heatmaps)

- [X] **T030** Integrate visualization into generator:
  - Call `generate_batch_report()` after each trial batch completes
  - Save plots with batch number: `batch_1_iq_trajectories.png`, etc.
  - Include plot generation in progress output
  - Add `--no-viz` flag to skip visualization if desired

- [X] **T031** [P] Update `docs/training/phase0_vetting.md` to reference generated pattern files:
  - Note that Phase 0 can use cascade_patterns_64.bin for faster vetting
  - Add section on pattern file locations
  - Update prerequisites to include pattern generation
  - Reference visualization outputs for quality inspection

---

## Phase 3.7: Platform Adaptation

- [X] **T032** Create platform detection module `modules/training/patterns/platform_detect.py`:
  - Implement `detect_optimal_workers()` - returns physical_cores - 2
  - Implement `detect_hybrid_architecture()` - detects P-cores vs E-cores
  - Implement `detect_memory_constraints()` - returns available RAM, suggests batch size
  - Implement `detect_simd_capabilities()` - checks for AVX-512, AVX2, ARM NEON
  - Implement `get_platform_config()` - unified configuration dict
  - Implement `optimize_for_architecture()` - apply CPU pinning, thread limits

- [X] **T033** Add psutil and py-cpuinfo to `modules/training/requirements.txt`:
  - psutil>=5.9.0 (CPU and memory detection)
  - py-cpuinfo>=9.0.0 (detailed CPU capability detection)

- [X] **T034** Integrate platform detection into generator:
  - Add `auto_tune=True` parameter to `generate_pattern_set()`
  - Call `get_platform_config()` if auto_tune enabled
  - Auto-select `num_trials` based on detected workers
  - Adjust `max_iterations` based on available memory
  - Pin to P-cores on hybrid CPUs (Linux/macOS)
  - Log detected configuration to console

- [X] **T035** [P] Test platform detection on different systems:
  - Test on 4-core laptop (expect 2 trials, 50K iterations)
  - Test on 8-core desktop (expect 6 trials, 100K iterations)
  - Test on Core Ultra hybrid CPU (expect 8 trials pinned to P-cores)
  - Verify memory adaptation on <8 GB system

---

## Phase 3.8: Distributed Execution (Fly.io)

- [X] **T036** Create Fly.io worker infrastructure `modules/training/fly-pattern-worker/`:
  - Create `worker.py` - main trial executor script
  - Create `coordinator.py` - spawns workers, collects results
  - Create `Dockerfile` - container with pattern generation dependencies
  - Create `fly.toml` - Fly.io app configuration (performance-1x)
  - Create `requirements-worker.txt` - includes boto3 for Tigris

- [X] **T037** Implement worker.py:
  - Accept TRIAL_ID env variable
  - Compute seed = 42 + TRIAL_ID
  - Call `generate_pattern_set(count=128, seed=seed)`
  - Save binary file locally
  - Upload to Tigris: `s3://cascade-patterns/trials/trial_{id}.bin`
  - Upload metadata JSON with stats (separation, avg_lambda, etc.)
  - Exit with code 0 on success

- [X] **T038** Implement coordinator.py:
  - Parse args: `--workers N --region REGION`
  - Spawn N Fly.io machines using `fly machine run`
  - Poll Tigris for completion (check for N result files)
  - Download all trial results
  - Score each: `separation_db - 0.1 * avg_lambda`
  - Select best trial
  - Generate visualizations comparing all trials
  - Save final: `cascade_patterns_128.bin`
  - Clean up worker machines

- [X] **T039** Add --distributed flag to CLI (`__main__.py`):
  - Add `--distributed` flag (triggers Fly.io execution)
  - Add `--workers N` for distributed worker count
  - Add `--region REGION` for Fly.io deployment region
  - Call coordinator.py if --distributed enabled
  - Otherwise use local multi-trial generation

- [X] **T040** [P] Integration test for distributed execution:
  - Test with `--distributed --workers 4` (small test)
  - Verify 4 workers spawn on Fly.io
  - Verify results uploaded to Tigris
  - Verify coordinator selects best
  - Verify final pattern file generated
  - Check cost: ~$0.76 for 4 workers × 24 hours

- [X] **T041** Add boto3 to worker requirements:
  - Create `modules/training/fly-pattern-worker/requirements.txt`
  - Include all pattern generation dependencies
  - Add boto3>=1.28.0 for Tigris S3 access

---

## Phase 3.9: Optimization Enhancements (Two-Phase + Phase-Aware)

**RATIONALE**: Based on cost-benefit analysis, 8 trials × 400K iterations (depth strategy) is optimal for local high-end CPUs. Two-phase optimization maximizes λ=0 patterns, phase-aware ensures HF robustness.

- [X] **T042** Implement two-phase optimization in `modules/training/patterns/optimizer.py`:
  - Create `optimize_pattern_two_phase()` function
  - Phase 1: Frequency-only with BPSK (λ=0) for 80% of iterations (default 320K)
  - Phase 2: IQ refinement for 20% of iterations if Phase 1 insufficient (default 80K)
  - Early success detection: Return λ=0 if Phase 1 achieves target
  - Progress output shows Phase 1/2 status and whether BPSK was sufficient

- [X] **T043** Implement phase-aware cost function in `modules/training/patterns/optimizer.py`:
  - Create `_compute_cost_phase_aware()` function
  - Monte Carlo phase sampling (3-5 scenarios per correlation check)
  - Random phase per tone: ±π radians (frequency-dependent distortion)
  - Random phase per symbol: ±0.2 radians (time-varying channel)
  - Return worst-case correlation across all phase scenarios

- [X] **T044** Update `modules/training/patterns/generator.py` to use two-phase optimization:
  - Replace `optimize_pattern_direct_iq()` calls with `optimize_pattern_two_phase()`
  - Pass `phase_aware=True` by default
  - Update progress output to show Phase 1/2 results
  - Report count of patterns that achieved λ=0 in final summary

- [X] **T045** Update defaults in `modules/training/patterns/multi_trial.py`:
  - Change default `num_trials`: 8 (local depth strategy)
  - Change default `max_iterations`: 400000 (for deep convergence)
  - Update auto-detection logic to prefer depth on high-end CPUs
  - Add configuration comments explaining depth vs breadth trade-off

- [X] **T046** Update CLI in `modules/training/patterns/__main__.py`:
  - Update default `--iterations` to 400000
  - Update help text with recommended configs (8×400K local, 32×100K cloud)
  - Add cost-benefit guidance in help text
  - Update Fly.io worker defaults to match (100K for breadth strategy)

- [X] **T047** Test two-phase + phase-aware optimization:
  - Quick test: 4 patterns × 10K iterations (5K freq + 5K IQ)
  - Verify Phase 1 completes before Phase 2
  - Verify at least 1 pattern achieves λ=0 (BPSK sufficient)
  - Verify phase-aware cost function runs without errors
  - Measure performance overhead of phase-aware (expect ~2-3x slower per iteration)

---

## Dependencies

**Critical path**:
1. Setup (T001-T004) → Everything
2. Contract tests (T005-T008) → Implementation (T009-T019)
3. Zadoff-Chu (T009-T010) → Pattern generation (T016-T017)
4. IQ trajectories (T011) → Pattern generation (T016-T017)
5. Correlation (T012) → Optimizer (T013, T042-T043) → Pattern generation (T016-T017, T044)
6. Binary I/O (T014-T015) → Pattern generation (T016-T017)
7. Validation (T018-T019) → Integration tests (T021-T023)
8. CLI (T020) → Integration tests (T021-T023)
9. Visualization (T028-T029) → Integration into generator (T030)
10. Platform detection (T032-T033) → Integration (T034)
11. Fly.io infrastructure (T036-T038) → CLI integration (T039) → Distributed tests (T040)
12. Two-phase enhancements (T042-T043) → Generator integration (T044) → Config updates (T045-T046)
13. Everything → Polish (T024-T027, T031, T035, T041, T047)

**Parallel opportunities**:
- T003-T004 (directories + dependencies)
- T005-T008 (all contract tests, different files)
- T009-T010, T011, T012, T014-T015 (independent components)
- T021-T023 (integration tests, different files, but SLOW - 8-24 hours each)
- T024-T027 (unit tests + benchmarks, different files)
- T028-T029, T032-T033, T036-T037 (visualization, platform detect, worker - independent)
- T040-T041 (distributed test + requirements - can run in parallel)

---

## Parallel Execution Examples

### Parallel Group 1: Contract Tests (after T001-T004)
```bash
Task T005: "Contract test for generate_pattern_set() in modules/training/tests/patterns/test_generator.py"
Task T006: "Contract test for compute_4d_correlation() in modules/training/tests/patterns/test_correlation.py"
Task T007: "Contract test for validate_orthogonality() in modules/training/tests/patterns/test_pattern_validation.py"
Task T008: "Contract test for binary I/O in modules/training/tests/patterns/test_binary_format.py"
```

### Parallel Group 2: Core Components (after tests failing)
```bash
# Independent components
Task T009: "Implement Zadoff-Chu in modules/training/patterns/zadoff_chu.py"
Task T011: "Implement IQ trajectories in modules/training/patterns/iq_trajectories.py"
Task T012: "Implement 4D correlation in modules/training/patterns/correlation.py"
Task T014: "Implement binary writer in modules/training/patterns/binary_format.py"
```

### Parallel Group 3: Polish
```bash
Task T024: "Unit tests for Zadoff-Chu in modules/training/tests/unit/test_zadoff_chu_unit.py"
Task T025: "Unit tests for IQ trajectories in modules/training/tests/unit/test_iq_trajectories_unit.py"
Task T026: "Performance benchmarks in modules/training/tests/benchmarks/test_generation_performance.py"
Task T028: "Update phase0_vetting.md with pattern file references"
```

**Note**: T021-T023 (integration tests) are SLOW (8-24 hours each for actual generation). Mark these to run on dedicated compute.

---

## Task Execution Notes

### TDD Workflow
1. Write contract test (T005-T008) - MUST FAIL
2. Run test, confirm failure
3. Implement (T009-T019)
4. Run test, verify PASS
5. Commit: "feat: implement [component] (makes T0XX pass)"

### Long-Running Tasks
- **T021**: 8-12 hours (64-pattern generation)
- **T022**: 18-24 hours (128-pattern generation)
- **T026**: Includes timing of above (run on server/overnight)

These are ONE-TIME generation tasks. Once patterns exist, they're never regenerated (deterministic).

### Sequential Dependencies in generator.py
- T009-T010 (Zadoff-Chu base)
- T011 (IQ trajectories)
- T012 (correlation check)
- T013 (optimizer)
- T016 (64-pattern generation - uses all above)
- T017 (128-pattern generation - extends T016)

### Pattern File Output
After T016-T017 complete:
- `modules/training/data/cascade_patterns_64.bin` (19 KB, 2,016 validated pairs)
- `modules/training/data/cascade_patterns_128.bin` (38 KB, 8,128 validated pairs)

These files become infrastructure for Feature 002 (Phase 0 Training) and all future CASCADE work.

---

## Validation Checklist

- [x] All 5 contracts have test tasks (T005-T008)
- [x] All contracts have implementation tasks (T009-T019)
- [x] Tests before implementation (T005-T008 before T009-T019)
- [x] Parallel tasks are independent (different files)
- [x] Each task has exact file path
- [x] Integration tests run actual generation (T021-T022)
- [x] Quickstart validation included (T027)

---

## Estimated Effort

**Development time** (excluding generation time):
- Setup: 2-3 hours (T001-T004)
- Contract tests: 4-6 hours (T005-T008)
- Core implementation: 16-24 hours (T009-T019)
- CLI: 2-3 hours (T020)
- Unit tests + benchmarks: 4-6 hours (T024-T026)
- Documentation: 1-2 hours (T027-T028)

**Total dev**: 29-44 hours

**One-time generation** (actual pattern creation):
- 64-pattern: 8-12 hours (T021)
- 128-pattern: 18-24 hours (T022)

**Total with generation**: 55-80 hours (but generation runs unattended overnight)

---

## Success Criteria

Tasks complete when:
- ✅ All contract tests pass
- ✅ `cascade_patterns_64.bin` generated and validated (all 2,016 pairs <-37.5 dB)
- ✅ `cascade_patterns_128.bin` generated and validated (all 8,128 pairs <-37.5 dB)
- ✅ Beacon patterns 0-47 identical in both sets (FR-007)
- ✅ Message patterns organized in hierarchical IQ pools
- ✅ Pattern files loadable and usable by Phase 0 vetting (Feature 002)
- ✅ Quickstart validation succeeds
- ✅ Generation is deterministic (same seed → same patterns)

**Deliverable**: Two pattern files ready for all CASCADE training and operation.
