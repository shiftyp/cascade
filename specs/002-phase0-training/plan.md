# Implementation Plan: Phase 0 Training - Ideal Conditions Vetting

**Branch**: `002-phase0-training` | **Date**: 2025-10-04 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/workspaces/cascade/specs/002-phase0-training/spec.md`

## Summary

Validate CASCADE's 128-pattern chaos architecture by training in ideal conditions (AWGN only) to achieve 85%+ Shannon efficiency with 45 users, proving the architecture works before investing 18 months in real HF data collection. This 2.5-day vetting phase de-risks the project by identifying fundamental flaws early.

## Technical Context

**Language/Version**: Python 3.11 (matches CASCADE training module)
**Primary Dependencies**: PyTorch 2.0+, NumPy, SciPy (standard ML stack)
**Storage**: Local filesystem for model checkpoints and validation reports
**Testing**: PyTest with custom metrics (Shannon efficiency, decode accuracy)
**Target Platform**: Linux with CUDA (1x RTX 4090 GPU)
**Project Type**: Monorepo - Training module
**Performance Goals**: 85%+ Shannon efficiency with 45 users in AWGN within 60 GPU-hours
**Constraints**: <64 GB RAM, reproducible with seeds, AWGN-only (no real HF impairments)
**Scale/Scope**: 7 progressive tests, 5K-50K samples per test, validates before 150K-hour data collection

## Constitution Check

- [x] **Data-First Development**: Phase 0 runs BEFORE data collection to validate architecture - this is a meta-validation that prevents wasted data collection effort if architecture is flawed
- [x] **Monorepo Module Architecture**: Confined to `modules/training/` - vetting scripts, model training
- [x] **Clean Separation**: Tests model layer (continuous optimization) without protocol involvement
- [x] **Test-Driven Development**: Each of 7 tests defines targets first, then trains to achieve them
- [ ] **Real-World Data Priority**: **VIOLATION - Uses AWGN synthetic data, not real HF**
  - **Justification**: This is a PRE-data-collection validation phase. Purpose is to validate architecture works in IDEAL conditions before investing 18 months collecting real data. If model can't achieve 85% in simple AWGN, it won't achieve 78-85% in complex real HF. This prevents wasted data collection effort.
  - **Constitutional alignment**: Principle V allows this as architecture validation, not core functionality training
- [x] **Privacy-Preserving**: No real data involved, purely synthetic signals
- [x] **Reproducible Research**: All tests use fixed seeds, deterministic AWGN generation

**Violations requiring justification:** 1 (synthetic data for architecture validation)

## Project Structure

### Training Module (this feature)
```
modules/training/
├── src/
│   ├── vetting/             # NEW: Phase 0 vetting implementation
│   │   ├── signal_generator.py     # Synthetic CASCADE signal generation
│   │   ├── awgn_channel.py         # AWGN-only channel simulation
│   │   ├── test_scenarios.py       # 7 test configurations
│   │   ├── metrics.py               # Shannon efficiency, accuracy calculations
│   │   └── validator.py             # Runs all tests, generates report
│   ├── datasets/           # (existing)
│   ├── models/             # (existing - expert networks, conductor)
│   ├── pipelines/          # (existing)
│   └── benchmarks/         # (existing)
└── tests/
    ├── vetting/            # NEW: Vetting test suite
    │   ├── test_signal_generation.py
    │   ├── test_awgn_channel.py
    │   └── test_vetting_scenarios.py
    └── benchmarks/         # (existing)
```

**Structure Decision**: This feature belongs solely to the **Training Module**. It's a pre-training validation phase that tests whether the CASCADE model architecture (expert networks, conductor, pattern system) can achieve theoretical performance in ideal conditions. No protocol module involvement (no discrete decisions being made), no data module involvement (no real data collection), no applications.

---

## Phase 0: Outline & Research

### Research Tasks

1. **AWGN Channel Implementation**
   - Decision: Use NumPy random.normal for white Gaussian noise generation
   - Rationale: Standard approach, reproducible with seeds, matches theoretical Shannon calculations
   - Alternatives: SciPy stats (overkill), custom (unnecessary complexity)

2. **Synthetic CASCADE Signal Generation**
   - Decision: Implement RS(32,20) encoding following docs/model/pattern_architecture.md specification
   - Rationale: Must match actual CASCADE protocol for valid testing
   - Components needed:
     * Pattern lookup (load 128 patterns from specification)
     * RS encoding (GF(256) over 32 symbols, 20 information + 12 parity)
     * 4D mapping (tone index selection + IQ modulation)
     * Multi-user mixing (overlapping signals with different start times, drifts)

3. **Shannon Efficiency Calculation**
   - Decision: Use standard C = B × log₂(1 + SNR) formula, compare achieved bits/sec
   - Rationale: Industry standard, matches documentation claims
   - Implementation: Measure throughput as (successfully decoded bits) / (transmission time)

4. **Test Progression Strategy**
   - Decision: Curriculum learning (1 → 10 → 20 → 30 → 45 users)
   - Rationale: Validates each capability before adding complexity
   - Each test builds on previous (can't test 45-user chaos if 10-user separation fails)

5. **Kernel Coordination Simulation**
   - Decision: Implement simple prokernel (available_tones list) and antikernel (shift request) exchange
   - Rationale: Need to test if emergent coordination provides claimed 2-5% improvement
   - Implementation: Round 1 (random kernels), Round 2 (prokernels), Round 3 (antikernels)

**Output**: research.md documenting these decisions

---

## Phase 1: Design & Contracts

### Data Model

**VettingConfig** (Test scenario configuration)
- num_users: int (1, 10, 20, 30, 45)
- patterns: List[int] (pattern IDs to use)
- snr_db: float (-22 to +15)
- test_type: enum (single, orthogonality, freq_reuse, time_reuse, chaos, kernel, snr_sweep)
- target_accuracy: float (0.90 to 0.999)
- target_shannon: float (0.85 to 0.95)

**UserConfig** (Single user in multi-user scenario)
- pattern_id: int (0-127)
- tone_selection: List[int] (4 tone indices from 0-77)
- start_time_offset: float (0-10 seconds for chaos)
- clock_drift_hz: float (±50 Hz)
- snr_db: float
- data_payload: bytes (18 bytes for RS pattern)

**VettingResult** (Output from running vetting)
- test_results: Dict[str, TestResult] (7 test outcomes)
- overall_pass: bool (Test 5 >= 85% Shannon)
- achieved_shannon_percent: float
- recommendation: enum (proceed_real_data, proceed_synthetic, proceed_hybrid, fix_architecture)
- identified_issues: List[str]

**TestResult** (Single test outcome)
- test_name: str
- num_users: int
- achieved_accuracy: float
- achieved_shannon: float
- target_accuracy: float
- target_shannon: float
- passed: bool
- duration_hours: float

**Output**: data-model.md with entity definitions

### API Contracts

Since this is a training/validation module (not a web API), contracts are defined as **Python function signatures** for the vetting system:

#### Contract 1: Signal Generation
```python
def generate_cascade_signal(
    pattern_id: int,        # 0-127
    data_bytes: bytes,      # 18 bytes
    tone_selection: List[int],  # 4 indices from 0-77
    snr_db: float,
    start_time_offset: float = 0.0,
    clock_drift_hz: float = 0.0
) -> np.ndarray:
    """
    Generate synthetic CASCADE signal with RS(32,20) structure

    Returns: Complex IQ samples (48kHz, 1.6s duration)
    """
    pass
```

#### Contract 2: AWGN Channel
```python
def apply_awgn_channel(
    signal: np.ndarray,
    snr_db: float,
    seed: int = None
) -> np.ndarray:
    """
    Apply white Gaussian noise at specified SNR

    Returns: Noisy signal (same shape as input)
    """
    pass
```

#### Contract 3: Multi-User Mixing
```python
def mix_multi_user_signals(
    user_configs: List[UserConfig]
) -> Tuple[np.ndarray, List[GroundTruth]]:
    """
    Generate and mix multiple CASCADE users with async starts

    Returns: Mixed signal + ground truth labels for each user
    """
    pass
```

#### Contract 4: Vetting Execution
```python
def run_vetting_test(
    config: VettingConfig,
    model: CASCADEModel,
    num_samples: int
) -> TestResult:
    """
    Run single vetting test scenario

    Returns: Test results with accuracy and Shannon efficiency
    """
    pass
```

#### Contract 5: Full Vetting Suite
```python
def run_full_vetting() -> VettingResult:
    """
    Execute all 7 tests in sequence

    Returns: Complete vetting result with pass/fail and recommendations
    """
    pass
```

**Output**: contracts/vetting_api.yaml (function signatures), contracts/test_*.py (failing contract tests)

### Quickstart Validation

**Goal**: Validate vetting can be run end-to-end

**Steps**:
1. Load CASCADE model (untrained or random init)
2. Run `run_full_vetting()`
3. Verify validation report generated
4. Check Test 5 (45-user chaos) result
5. Verify pass/fail determination
6. Review recommendations

**Success criteria**:
- Vetting completes without errors
- All 7 tests execute and report metrics
- Final report shows pass/fail status
- Recommendations align with results

**Output**: quickstart.md with validation steps

---

## Phase 2: Task Planning Approach

**Task Generation Strategy**:

### Test Implementation Tasks (TDD order)
1. **Contract tests** (fail initially):
   - Test signal generation produces valid CASCADE RS patterns
   - Test AWGN channel adds correct noise power
   - Test multi-user mixing handles async starts
   - Test vetting runner executes all 7 scenarios
   - Test metrics calculate Shannon correctly

2. **Implementation tasks** (make tests pass):
   - Implement RS(32,20) encoder following CASCADE spec
   - Implement 4D signal mapping (pattern → Time × Freq × IQ)
   - Implement AWGN noise generator
   - Implement multi-user signal mixer
   - Implement Shannon efficiency calculator
   - Implement 7 test scenario configs
   - Implement vetting runner orchestration
   - Implement validation report generator

3. **Integration tasks**:
   - Test full vetting suite runs Test 1-7 in sequence
   - Verify Test 5 critical threshold detection (85%)
   - Validate recommendation engine (pass → 3 paths, fail → fix)
   - Generate sample validation report

### Ordering Strategy
- Test tasks BEFORE implementation tasks (TDD)
- Signal generation before multi-user (dependency)
- Individual tests before full suite (composition)
- Mark parallel where tests are independent

**Estimated Output**: 20-25 numbered tasks in dependency order

**IMPORTANT**: This is executed by the /tasks command, NOT during /plan

---

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Uses synthetic AWGN data instead of real HF | Validate architecture BEFORE 18-month real data collection | Cannot validate architecture without training; real data collection takes 18 months; if architecture is flawed, wastes 18 months + $5K. AWGN vetting takes 2.5 days and proves architecture works in ideal conditions. This is a meta-validation step that PREVENTS wasted real data collection. |

**Constitutional exception**: Principle V (Real-World Data Priority) allows synthetic for architecture validation pre-data-collection.

---

## Progress Tracking

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning approach described
- [ ] Phase 3: Tasks generated (/tasks command - not executed by /plan)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS (with documented synthetic data exception)
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved (spec is well-defined from phase0_vetting.md)
- [x] Complexity deviations documented (synthetic data justified)

---

*Based on Constitution v1.0.0 - See `.specify/memory/constitution.md`*
*Ready for /tasks command to generate implementation tasks*
