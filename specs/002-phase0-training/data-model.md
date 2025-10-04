# Data Model: Phase 0 Training

**Feature**: 002-phase0-training
**Date**: 2025-10-04

## Entity Definitions

### VettingConfig

**Purpose**: Configuration for a single vetting test scenario

**Attributes**:
- `test_name`: str - Human-readable test name (e.g., "Test 5: Full Chaos")
- `num_users`: int - Number of simultaneous users (1, 10, 20, 30, 45)
- `patterns`: List[int] - Pattern IDs to use (subset of 0-127)
- `snr_db`: float - Signal-to-noise ratio in dB (-22 to +15)
- `test_type`: str - Test category (single, orthogonality, freq_reuse, time_reuse, chaos, kernel, snr_sweep)
- `target_accuracy`: float - Required decode accuracy (0.90 to 0.999)
- `target_shannon`: float - Required Shannon efficiency (0.85 to 0.95)
- `num_samples`: int - Training samples to generate (5,000 to 50,000)
- `training_hours`: float - Allocated GPU time (1 to 16 hours)

**Validation Rules**:
- num_users must be positive integer
- patterns must be valid CASCADE pattern IDs (0-127)
- snr_db typically +15 for high-SNR tests, varies for Test 7
- target_shannon must be realistic for AWGN (≤0.95)

**Relationships**:
- Tests build on each other (Test N+1 requires Test N to pass)
- Test 5 is critical threshold (overall pass/fail)

---

### UserConfig

**Purpose**: Configuration for a single user in multi-user scenario

**Attributes**:
- `user_id`: int - Unique identifier (0 to num_users-1)
- `pattern_id`: int - CASCADE pattern to use (0-127)
- `tone_selection`: List[int] - Four tone indices from 78-tone grid (0-77)
- `start_time_offset`: float - Async start time in seconds (0-10s for chaos)
- `clock_drift_hz`: float - Frequency drift (±50 Hz)
- `snr_db`: float - User's SNR (can vary per user)
- `data_payload`: bytes - 18 bytes data to encode
- `num_patterns`: int - Multi-pattern transmission (1-4)

**Validation Rules**:
- tone_selection must contain exactly 4 unique indices from 0-77
- clock_drift_hz must be in [-50, +50] range
- data_payload must be exactly 18 bytes (RS requirement)
- num_patterns must be 1-4

**Relationships**:
- Multiple UserConfigs combine to create multi-user test scenario
- Frequency reuse: Multiple users can share pattern_id if tone_selections differ
- Time reuse: Multiple users can share pattern_id and tones if start_time_offsets differ

---

### TrainingSample

**Purpose**: Single training sample (multi-user scenario with ground truth)

**Attributes**:
- `sample_id`: int - Unique identifier
- `users`: List[UserConfig] - All users in this sample
- `mixed_signal`: np.ndarray - Combined IQ signal (complex64)
- `ground_truth`: List[GroundTruth] - Expected decodes for each user
- `snr_db`: float - Overall SNR level
- `sample_rate_hz`: int - 48,000 Hz (CASCADE standard)
- `duration_sec`: float - Signal duration (typically 1.6-3.2s)

**Validation Rules**:
- mixed_signal must be complex-valued IQ samples
- ground_truth must have one entry per user
- sample_rate_hz must be 48,000 (CASCADE spec)

**Relationships**:
- Thousands of TrainingSamples comprise one VettingConfig test
- Model trains on samples, evaluated on held-out test set

---

### GroundTruth

**Purpose**: Expected decode output for one user in multi-user sample

**Attributes**:
- `user_id`: int
- `pattern_id`: int - Expected pattern (0-127)
- `data_bytes`: bytes - Expected decoded data (18 bytes)
- `tone_indices`: List[int] - Which tones user transmitted on
- `start_symbol`: int - Which symbol user started at in the mixed signal

**Validation Rules**:
- data_bytes must be exactly 18 bytes
- pattern_id must match UserConfig

**Relationships**:
- One GroundTruth per user in TrainingSample
- Used for loss calculation during training

---

### TestResult

**Purpose**: Outcome of running one vetting test

**Attributes**:
- `test_name`: str
- `num_users`: int
- `num_samples_trained`: int
- `achieved_accuracy`: float - Fraction of users correctly decoded (0-1)
- `achieved_shannon`: float - Measured Shannon efficiency (0-1)
- `target_accuracy`: float - Required accuracy
- `target_shannon`: float - Required Shannon efficiency
- `passed`: bool - Whether test met both targets
- `duration_hours`: float - Actual GPU time used
- `per_user_throughput_bps`: float - Average bits/sec per user
- `total_capacity_bps`: float - Sum across all users

**Validation Rules**:
- achieved_accuracy must be in [0, 1]
- achieved_shannon must be in [0, 1] and ≤ 1.0 (can't exceed Shannon limit)
- passed = (achieved_accuracy >= target_accuracy) AND (achieved_shannon >= target_shannon)

**Relationships**:
- One TestResult per VettingConfig
- Seven TestResults comprise one VettingResult

---

### VettingResult

**Purpose**: Overall outcome of Phase 0 vetting (all 7 tests)

**Attributes**:
- `test_results`: Dict[str, TestResult] - Results keyed by test name
- `overall_pass`: bool - True if Test 5 achieved ≥85% Shannon
- `best_shannon_achieved`: float - Highest efficiency across all tests
- `recommendation`: str - Next step (proceed_real_data, proceed_synthetic, proceed_hybrid, fix_architecture)
- `identified_issues`: List[str] - Problems found (if any)
- `total_duration_hours`: float - Sum of all test durations
- `timestamp`: datetime - When vetting completed

**Validation Rules**:
- overall_pass determined by Test 5 specifically (45-user chaos ≥85% Shannon)
- recommendation must be one of 4 valid values
- total_duration_hours should be ≈60 hours per NFR-001

**Relationships**:
- Contains all 7 TestResults
- Used for project go/no-go decision

**State Machine**:
```
NOT_STARTED → RUNNING → COMPLETE
                      ↓
            [overall_pass == true] → ARCHITECTURE_VALIDATED
            [overall_pass == false] → ARCHITECTURE_NEEDS_REVISION
```

---

### ValidationReport

**Purpose**: Human-readable report document for project leads

**Attributes**:
- `summary`: str - Executive summary (pass/fail, key findings)
- `test_summaries`: List[str] - One paragraph per test
- `critical_test_detail`: str - Deep dive on Test 5 (45-user chaos)
- `shannon_efficiency_analysis`: str - Comparison to theoretical limits
- `next_steps`: str - Detailed recommendations based on results
- `risk_assessment`: str - What results mean for data collection investment
- `appendix_metrics`: Dict - Raw numbers, charts, detailed metrics

**Format**: Markdown document

**Relationships**:
- Generated from VettingResult
- Includes all TestResults formatted for readability
- Primary deliverable for project decision-making

---

## Entity Relationships

```
VettingConfig (7 configs, one per test)
    ↓ generates
TrainingSample (5K-50K samples per test)
    ↓ contains
UserConfig (1-45 configs per sample) + GroundTruth (labels)
    ↓ trains model, produces
TestResult (one per VettingConfig)
    ↓ aggregates into
VettingResult (overall outcome)
    ↓ generates
ValidationReport (decision document)
```

## Data Flow

1. Define 7 VettingConfigs (Test 1-7)
2. For each VettingConfig:
   a. Generate TrainingSamples (thousands)
   b. Each sample creates UserConfigs and mixes signals
   c. Train model on samples
   d. Evaluate to produce TestResult
3. Aggregate 7 TestResults into VettingResult
4. Generate ValidationReport for stakeholders
5. Make go/no-go decision based on Test 5 (85% threshold)

## State Transitions

**VettingResult State Machine**:
```
NOT_STARTED
    ↓
RUNNING (tests 1-7 executing)
    ↓
COMPLETE
    ├─ Test 5 >= 85% Shannon → ARCHITECTURE_VALIDATED
    │                           ↓
    │                      [Choose Path A, B, or C]
    │
    └─ Test 5 < 80% Shannon → ARCHITECTURE_NEEDS_REVISION
                                ↓
                           [Fix issues, re-vet]
```
