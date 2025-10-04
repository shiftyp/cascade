# Tasks: Phase 0 Training - Ideal Conditions Vetting

**Input**: Design documents from `/workspaces/cascade/specs/002-phase0-training/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/vetting_api.py

---

## Phase 3.1: Setup & Structure

- [ ] **T001** Create vetting module structure in `modules/training/src/vetting/` with `__init__.py`, `signal_generator.py`, `awgn_channel.py`, `test_scenarios.py`, `metrics.py`, `validator.py`

- [ ] **T002** Create test structure in `modules/training/tests/vetting/` with `__init__.py`, `test_signal_generation.py`, `test_awgn_channel.py`, `test_vetting_scenarios.py`

- [ ] **T003** [P] Add dependencies to `modules/training/requirements.txt`: torch>=2.0, numpy, scipy, pytest, galois (for GF(256) Reed-Solomon encoding)

- [ ] **T004** [P] Create vetting output directory structure: `modules/training/vetting_results/` with subdirs for checkpoints, reports, metrics

---

## Phase 3.2: Contract Tests (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [ ] **T005** [P] Contract test for `generate_cascade_signal()` in `modules/training/tests/vetting/test_signal_generation.py`:
  - Verify output shape (76800 complex samples for 1.6s @ 48kHz)
  - Verify RS(32,20) structure (20 info + 12 parity symbols)
  - Verify pattern_id encoded in symbol 0
  - Test MUST FAIL initially

- [ ] **T006** [P] Contract test for `apply_awgn_channel()` in `modules/training/tests/vetting/test_awgn_channel.py`:
  - Verify output shape matches input
  - Verify measured SNR within 0.5 dB of target
  - Verify reproducibility with seed
  - Test MUST FAIL initially

- [ ] **T007** [P] Contract test for `mix_multi_user_signals()` in `modules/training/tests/vetting/test_signal_generation.py`:
  - Verify correct number of ground truths returned
  - Verify async start time offsets applied correctly
  - Verify clock drift applied per user
  - Test MUST FAIL initially

- [ ] **T008** [P] Contract test for `calculate_shannon_efficiency()` in `modules/training/tests/vetting/test_metrics.py`:
  - Verify result in [0.0, 1.0] range
  - Verify matches theoretical Shannon formula
  - Test with known values (2500 Hz, +15 dB → 12,570 bps)
  - Verify confidence intervals returned (3-tuple: efficiency, CI_lower, CI_upper)
  - Verify CI bounds are reasonable (CI_lower < efficiency < CI_upper)
  - Test MUST FAIL initially

- [ ] **T009** [P] Contract test for `calculate_decode_accuracy()` in `modules/training/tests/vetting/test_metrics.py`:
  - Verify all correct decodes → 1.0
  - Verify no correct decodes → 0.0
  - Verify partial decodes give proportional score
  - Test MUST FAIL initially

- [ ] **T010** [P] Contract test for `run_vetting_test()` in `modules/training/tests/vetting/test_vetting_scenarios.py`:
  - Verify TestResult returned with all required fields
  - Verify pass/fail logic works (compare achieved vs target)
  - Test MUST FAIL initially

- [ ] **T011** [P] Contract test for `run_full_vetting()` in `modules/training/tests/vetting/test_vetting_scenarios.py`:
  - Verify 7 TestResults produced
  - Verify overall_pass determined by Test 5
  - Verify recommendation generated
  - Test MUST FAIL initially

---

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Signal Generation & Channel

- [ ] **T012** [P] Implement `apply_awgn_channel()` in `modules/training/src/vetting/awgn_channel.py`:
  - Calculate noise power from signal power and SNR
  - Generate white Gaussian noise with numpy.random.normal
  - Support reproducibility with random seeds
  - Make T006 pass

- [ ] **T013** Implement RS(32,20) encoder in `modules/training/src/vetting/signal_generator.py`:
  - Use reedsolo or galois library for GF(256) Reed-Solomon
  - Encode: pattern_id (1 byte) + checksum (1 byte) + data (18 bytes) = 20 info symbols
  - Generate 12 parity symbols
  - Return 32-symbol codeword

- [ ] **T014** Implement simplified pattern loader in `modules/training/src/vetting/signal_generator.py`:
  - Generate 128 pseudo-random tone index sequences (Zadoff-Chu based)
  - Each pattern: 32 symbols, tone indices 0-3
  - Simple IQ trajectories (circles, not full Lissajous)
  - Store in memory (don't need full 38 KB pattern file for vetting)

- [ ] **T015** Implement 4D signal mapping in `modules/training/src/vetting/signal_generator.py`:
  - Map each RS symbol (8 bits) to Time × Freq × IQ
  - Tone index (2 bits) selects which of 4 tones
  - IQ index (6 bits) selects 64-QAM constellation point
  - Generate 32 symbols × 50ms = 1.6s signal @ 48kHz

- [ ] **T016** Implement `generate_cascade_signal()` in `modules/training/src/vetting/signal_generator.py`:
  - Compose: RS encode → 4D mapping → IQ signal generation
  - Apply clock drift (frequency shift)
  - Apply start time offset (zero-pad beginning)
  - Make T005 pass

- [ ] **T017** Implement `mix_multi_user_signals()` in `modules/training/src/vetting/signal_generator.py`:
  - Generate signal for each UserConfig
  - Sum overlapping IQ samples (complex addition)
  - Create GroundTruth for each user
  - Make T007 pass

### Metrics & Evaluation

- [ ] **T018** [P] Implement `calculate_shannon_efficiency()` in `modules/training/src/vetting/metrics.py`:
  - Shannon capacity: B × log₂(1 + 10^(SNR/10))
  - Efficiency: achieved_bps / shannon_capacity
  - Use bandwidth = 2,500 Hz (CASCADE standard)
  - Calculate 95% confidence interval using bootstrap method (1000 resamples) (NFR-006)
  - Return tuple: (efficiency, confidence_interval_lower, confidence_interval_upper)
  - Make T008 pass

- [ ] **T019** [P] Implement `calculate_decode_accuracy()` in `modules/training/src/vetting/metrics.py`:
  - Compare decoded outputs to ground truth
  - Count correct pattern_id + data_bytes matches
  - Return fraction: correct_count / total_users
  - Calculate 95% confidence interval using Wilson score interval (NFR-006)
  - Return tuple: (accuracy, confidence_interval_lower, confidence_interval_upper)
  - Make T009 pass

### Test Scenarios

- [ ] **T020** Implement 7 vetting test configurations in `modules/training/src/vetting/test_scenarios.py`:
  - Test 1: 1 user, pattern 64, +15 dB, target 99.9% accuracy, 95% Shannon
  - Test 2: 10 users, 10 patterns, +15 dB, target 98% accuracy, 92% Shannon
  - Test 3: 20 users, 10 patterns (2× reuse), target 95% accuracy, 90% Shannon
  - Test 4: 30 users, 20 patterns, async starts, target 93% accuracy, 88% Shannon
  - Test 5: 45 users, 128 patterns, full chaos, target 90% accuracy, 85% Shannon (CRITICAL)
  - Test 6: 45 users with kernel coordination (3 rounds: random → prokernel → antikernel per research.md), target 92% accuracy, 87% Shannon
  - Test 7: 45 users, SNR sweep +15 to -22 dB, graceful degradation
  - Return list of VettingConfig objects

### Vetting Runner

- [ ] **T021** Implement `run_vetting_test()` in `modules/training/src/vetting/validator.py`:
  - Generate training samples from VettingConfig
  - Train model for specified hours
  - Evaluate on test set
  - Calculate accuracy and Shannon efficiency
  - Return TestResult
  - Make T010 pass

- [ ] **T022** Implement `run_full_vetting()` in `modules/training/src/vetting/validator.py`:
  - Load 7 test configs from test_scenarios.py
  - Run tests 1-7 in sequence
  - Determine overall pass (Test 5 >= 85% Shannon)
  - Generate recommendation based on results
  - Return VettingResult
  - Make T011 pass

- [ ] **T023** Implement `generate_validation_report()` in `modules/training/src/vetting/validator.py`:
  - Format VettingResult as markdown
  - Include executive summary (pass/fail)
  - Detail each of 7 tests
  - Highlight Test 5 (critical)
  - Provide recommendations (Path A/B/C or fix)
  - Write to output file

---

## Phase 3.4: Integration Tests

- [ ] **T024** [P] Integration test: Full vetting suite executes all 7 tests in `modules/training/tests/integration/test_full_vetting.py`:
  - Verify all 7 TestResults generated
  - Verify Test 5 result present
  - Verify overall pass/fail determined correctly

- [ ] **T025** [P] Integration test: Critical threshold detection in `modules/training/tests/integration/test_threshold_detection.py`:
  - Mock Test 5 at 85.5% Shannon → expect PASS
  - Mock Test 5 at 84.5% Shannon → expect FAIL (marginal)
  - Mock Test 5 at 78% Shannon → expect FAIL (clear failure)

- [ ] **T026** [P] Integration test: Recommendation engine in `modules/training/tests/integration/test_recommendations.py`:
  - Test 5 >= 85% → recommend paths A/B/C
  - Test 5 < 80% → recommend FIX_ARCHITECTURE
  - Verify recommendation strings match enum values

- [ ] **T027** Integration test: End-to-end vetting with simple model in `modules/training/tests/integration/test_e2e_vetting.py`:
  - Create minimal CASCADE model (random weights)
  - Run full vetting
  - Verify completes without errors
  - Verify report generated

---

## Phase 3.5: Model Preparation

- [ ] **T028** Create minimal CASCADE model stub for vetting in `modules/training/src/models/cascade_minimal.py`:
  - Pattern correlation (Time × Freq × IQ matching)
  - RS decoder (soft decoding with erasure handling)
  - Multi-user separation (successive cancellation)
  - Basic kernel interpretation (read available_tones)
  - Note: This is NOT the full expert network, just enough for vetting
  - See research.md Section 7 for "minimal" definition: No noise/propagation experts needed for AWGN

- [ ] **T029** Implement training loop for vetting in `modules/training/src/vetting/trainer.py`:
  - Accept VettingConfig and model
  - Generate training samples
  - Train for specified hours
  - Track metrics (loss, accuracy, Shannon)
  - Return trained model + metrics

---

## Phase 3.6: Polish & Validation

- [ ] **T030** [P] Add unit tests for UserConfig validation in `modules/training/tests/unit/test_user_config.py`:
  - Verify tone_selection has exactly 4 indices
  - Verify clock_drift in [-50, +50] range
  - Verify data_payload is 18 bytes

- [ ] **T031** [P] Add unit tests for RS encoding in `modules/training/tests/unit/test_rs_encoding.py`:
  - Verify 20 info + 12 parity symbols
  - Test erasure recovery (lose 12 symbols, still decode)
  - Verify pattern_id in symbol 0

- [ ] **T032** [P] Add performance benchmarks in `modules/training/tests/benchmarks/test_vetting_performance.py`:
  - Verify Test 1 completes in ~1 hour
  - Verify Test 5 completes in ~12 hours
  - Verify total vetting <= 60 hours
  - Monitor peak memory usage during Test 5 (45 users), assert < 64 GB RAM (NFR-003)
  - Log memory usage throughout all tests for optimization insights

- [ ] **T033** Run full vetting quickstart validation per `quickstart.md`:
  - Execute all steps
  - Verify expected outputs
  - Confirm validation report generated
  - Verify pass/fail logic works

- [ ] **T034** [P] Update `docs/training/phase0_vetting.md` with actual implementation details:
  - Add "Implementation Status" section
  - Document actual file locations
  - Add usage examples

---

## Dependencies

**Critical path**:
1. Setup (T001-T004) → Everything
2. Contract tests (T005-T011) → Implementation (T012-T023)
3. Signal generation (T012-T017) → Vetting runner (T021-T023)
4. Metrics (T018-T019) → Vetting runner (T021-T023)
5. All implementation → Integration tests (T024-T027)
6. Model stub (T028-T029) → Full vetting (T022)
7. Everything → Polish (T030-T034)

**Parallel opportunities**:
- T003-T004 (linting + structure)
- T005-T011 (all contract tests, different files)
- T012, T018-T019 (AWGN channel, metrics - independent)
- T024-T026 (integration tests, different files)
- T030-T032 (unit tests + benchmarks, different files)

---

## Parallel Execution Examples

### Parallel Group 1: Contract Tests (after T001-T004)
```bash
# All contract tests can run in parallel (different files)
Task T005: "Contract test for generate_cascade_signal() in modules/training/tests/vetting/test_signal_generation.py"
Task T006: "Contract test for apply_awgn_channel() in modules/training/tests/vetting/test_awgn_channel.py"
Task T007: "Contract test for mix_multi_user_signals() in modules/training/tests/vetting/test_signal_generation.py"
Task T008: "Contract test for calculate_shannon_efficiency() in modules/training/tests/vetting/test_metrics.py"
Task T009: "Contract test for calculate_decode_accuracy() in modules/training/tests/vetting/test_metrics.py"
Task T010: "Contract test for run_vetting_test() in modules/training/tests/vetting/test_vetting_scenarios.py"
Task T011: "Contract test for run_full_vetting() in modules/training/tests/vetting/test_vetting_scenarios.py"
```

### Parallel Group 2: Core Implementation (after tests failing)
```bash
# Independent components can be implemented in parallel
Task T012: "Implement apply_awgn_channel() in modules/training/src/vetting/awgn_channel.py"
Task T018: "Implement calculate_shannon_efficiency() in modules/training/src/vetting/metrics.py"
Task T019: "Implement calculate_decode_accuracy() in modules/training/src/vetting/metrics.py"
Task T020: "Implement 7 test scenario configs in modules/training/src/vetting/test_scenarios.py"
```

### Parallel Group 3: Integration Tests
```bash
# Different integration test files
Task T024: "Integration test for full vetting suite in modules/training/tests/integration/test_full_vetting.py"
Task T025: "Integration test for threshold detection in modules/training/tests/integration/test_threshold_detection.py"
Task T026: "Integration test for recommendation engine in modules/training/tests/integration/test_recommendations.py"
```

### Parallel Group 4: Polish
```bash
# Unit tests and docs can be done in parallel
Task T030: "Unit tests for UserConfig in modules/training/tests/unit/test_user_config.py"
Task T031: "Unit tests for RS encoding in modules/training/tests/unit/test_rs_encoding.py"
Task T032: "Performance benchmarks in modules/training/tests/benchmarks/test_vetting_performance.py"
Task T034: "Update docs/training/phase0_vetting.md with implementation details"
```

---

## Task Execution Notes

### TDD Workflow
1. Write contract test (T005-T011) - test MUST FAIL
2. Run test, confirm failure
3. Implement functionality (T012-T023)
4. Run test again, verify it PASSES
5. Commit with message: "feat: implement [function] (makes T0XX pass)"

### Sequential Dependencies
- **T013-T017** must be sequential (all modify signal_generator.py):
  - T013: RS encoder
  - T014: Pattern loader
  - T015: 4D mapping
  - T016: generate_cascade_signal (uses T013-T015)
  - T017: mix_multi_user_signals (uses T016)

- **T021-T023** must be sequential (all modify validator.py):
  - T021: run_vetting_test
  - T022: run_full_vetting (uses T021)
  - T023: generate_validation_report (uses T022)

### Critical Test (T010 → T021 → T022)
Test 5 (45-user chaos, 85% Shannon threshold) is implemented through T021-T022 and validated by T024-T025

---

## Validation Checklist

Before marking complete, verify:

- [x] All 7 contracts from vetting_api.py have corresponding test tasks (T005-T011)
- [x] All contracts have implementation tasks (T012-T023)
- [x] All tests come before implementation (T005-T011 before T012-T023)
- [x] Parallel tasks are truly independent (different files or no shared state)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task (checked: signal_generator.py tasks are sequential)
- [x] Integration tests (T024-T027) come after core implementation
- [x] Quickstart validation (T033) comes last

---

## Estimated Effort

**Setup**: 2-4 hours (T001-T004)
**Contract tests**: 4-6 hours (T005-T011, can parallelize)
**Implementation**: 16-24 hours (T012-T023, some parallel)
**Integration**: 4-6 hours (T024-T027, can parallelize)
**Model stub**: 8-12 hours (T028-T029)
**Polish**: 4-6 hours (T030-T034, can parallelize)

**Total**: 38-52 hours development time (excluding 60 hours GPU training time for actual vetting run)

---

## Success Criteria

Tasks complete when:
- ✅ All contract tests pass
- ✅ Full vetting suite runs without errors
- ✅ Validation report generates correctly
- ✅ Quickstart validation succeeds
- ✅ Test 5 (45-user chaos) executes and produces Shannon efficiency metric
- ✅ Pass/fail determination works (85% threshold)

**Note**: Actual vetting RESULTS (whether Test 5 achieves 85%) depend on CASCADE model performance and are the PURPOSE of this feature, not a task completion criteria.
