# Tasks: Signal Generator

**Input**: Design documents from `/workspaces/cascade/specs/004-signal-generator/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/, quickstart.md
**Module**: `modules/training/` (Training Module)
**Tech Stack**: Python 3.11, NumPy, SciPy, commpy, pytest, hypothesis

---

## Overview

This implementation creates a two-part CASCADE V2 signal generator:
1. **Core Signal Generator**: Produces clean V2-compliant IQ signals
2. **Synthetic Data Orchestrator**: Generates expert-specific training datasets for 5 neural network experts

**Critical Requirements**:
- Must generate **5 separate expert datasets** (QRN, Signal, Timing, Channel, QRM)
- Performance: <100ms single signal, 100 signals in <30s
- V2 compliance: GMSK BT=0.3, 200 sym/s, 135-tone grid
- Test-driven development (TDD): Tests before implementation

---

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- File paths are absolute from repository root

---

## Phase 3.1: Setup (Week 1, Days 1-2)

- [x] **T001** Create project structure for signal generator
  - Create `modules/training/src/signal_generator/` directory
  - Create `modules/training/src/channel_simulator/` directory
  - Create `modules/training/tests/signal_generator/` directory
  - Create `modules/training/tests/channel_simulator/` directory
  - Add `__init__.py` files to all package directories

- [x] **T002** Initialize Python project dependencies
  - Create `modules/training/pyproject.toml` if not exists
  - Add dependencies: `numpy>=1.24.0`, `scipy>=1.10.0`, `scikit-dsp-comm>=2.0.0` (provides commpy)
  - Add dev dependencies: `pytest>=7.0.0`, `pytest-cov>=4.0.0`, `hypothesis>=6.0.0`, `matplotlib>=3.5.0`
  - Install dependencies: `pip install -e modules/training/`

- [x] **T003 [P]** Configure linting and formatting
  - Add `.flake8` config for Python 3.11 (if not exists)
  - Add `pyproject.toml` black/isort config (if not exists)
  - Verify pattern files exist: `ls modules/training/patterns/tournament/pattern_*.pkl`

---

## Phase 3.2: Tests First (TDD) - Week 1, Days 3-5 ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Core Generator Contract Tests

- [ ] **T004 [P]** Contract test: SignalGeneratorInterface.generate()
  - File: `modules/training/tests/signal_generator/test_contract_generator.py`
  - Test `generate()` method signature and return types
  - Test parameter validation (pattern_id 0-7, frequency_pair 0-66, etc.)
  - Test deterministic output with seeds
  - MUST FAIL (no implementation yet)

- [ ] **T005 [P]** Contract test: SignalGeneratorInterface.verify_v2_compliance()
  - File: `modules/training/tests/signal_generator/test_contract_compliance.py`
  - Test V2 compliance checks (symbol rate, GMSK bandwidth, tone spacing)
  - Test pattern orthogonality validation
  - MUST FAIL (no implementation yet)

- [ ] **T006 [P]** Contract test: PatternLoaderInterface
  - File: `modules/training/tests/signal_generator/test_contract_pattern_loader.py`
  - Test `load_pattern()` with all 8 patterns × 6 lengths
  - Test pattern caching behavior
  - Test FileNotFoundError for invalid patterns
  - MUST FAIL (no implementation yet)

- [ ] **T007 [P]** Contract test: GMSKModulatorInterface
  - File: `modules/training/tests/signal_generator/test_contract_gmsk.py`
  - Test `generate_gmsk_fsk()` output format and dimensions
  - Test constant envelope property (|I² + Q²| ≈ 1)
  - Test BT=0.3 parameter
  - MUST FAIL (no implementation yet)

- [ ] **T008 [P]** Contract test: ConstellationMapperInterface
  - File: `modules/training/tests/signal_generator/test_contract_modulation.py`
  - Test `map_to_constellation()` for BPSK/QPSK/8-PSK/16-APSK
  - Test Gray coding verification
  - Test unit power normalization
  - MUST FAIL (no implementation yet)

- [ ] **T009 [P]** Contract test: PolarCodecInterface
  - File: `modules/training/tests/signal_generator/test_contract_polar.py`
  - Test `encode()` with rates 1/2, 2/3, 3/4, 4/5, 5/6, 7/8
  - Test systematic encoding property
  - Test block length validation (must be power of 2)
  - MUST FAIL (no implementation yet)

### Orchestrator Contract Tests

- [ ] **T010 [P]** Contract test: ChannelOrchestratorInterface.generate_qrn_expert_data()
  - File: `modules/training/tests/channel_simulator/test_contract_qrn_expert.py`
  - Test pure noise generation (NO signal)
  - Test QRN types: crackling, static, lightning, power_line
  - Test label structure (burst_times, intensity, noise_floor_db)
  - MUST FAIL (no implementation yet)

- [ ] **T011 [P]** Contract test: ChannelOrchestratorInterface.generate_signal_expert_data()
  - File: `modules/training/tests/channel_simulator/test_contract_signal_expert.py`
  - Test clean signal output (bit-identical to input)
  - Test labels include pattern_id, modulation, polar_codeword
  - Test NO noise added
  - MUST FAIL (no implementation yet)

- [ ] **T012 [P]** Contract test: ChannelOrchestratorInterface.generate_timing_expert_data()
  - File: `modules/training/tests/channel_simulator/test_contract_timing_expert.py`
  - Test collision scenario generation (1-3 signals)
  - Test time offset quantization to sample boundaries
  - Test separated signals in labels (ground truth)
  - MUST FAIL (no implementation yet)

- [ ] **T013 [P]** Contract test: ChannelOrchestratorInterface.generate_channel_expert_data()
  - File: `modules/training/tests/channel_simulator/test_contract_channel_expert.py`
  - Test channel distortion (Rayleigh, Rician, multipath, Doppler)
  - Test impulse response in labels
  - Test kernel_parameters requirement (NEW from CLAUDE.md update)
  - MUST FAIL (no implementation yet)

- [ ] **T014 [P]** Contract test: ChannelOrchestratorInterface.generate_qrm_expert_data()
  - File: `modules/training/tests/channel_simulator/test_contract_qrm_expert.py`
  - Test pure interference (NO CASCADE signal)
  - Test interference types: CW, SSB, FT8, digital, radar, power_line
  - Test frequency offset accuracy
  - MUST FAIL (no implementation yet)

- [ ] **T015 [P]** Contract test: ChannelOrchestratorInterface.generate_batch()
  - File: `modules/training/tests/channel_simulator/test_contract_batch.py`
  - Test batch generation for all 5 expert types
  - Test deterministic output with seeds
  - Test performance: 100 examples in <30s
  - MUST FAIL (no implementation yet)

### Integration Tests (From Quickstart)

- [ ] **T016 [P]** Integration test: Core Generator end-to-end
  - File: `modules/training/tests/signal_generator/test_integration.py`
  - Test quickstart Part 1 scenarios (clean signal generation)
  - Test V2 compliance validation passes
  - Test spectrum visualization (optional, requires matplotlib)

- [ ] **T017 [P]** Integration test: Expert dataset generation
  - File: `modules/training/tests/channel_simulator/test_integration_experts.py`
  - Test quickstart Part 2 scenarios (all 5 experts)
  - Verify separation: QRN=pure noise, Signal=clean, Timing=collisions
  - Test kernel_parameters passed to Channel expert

- [ ] **T018 [P]** Integration test: Batch generation
  - File: `modules/training/tests/channel_simulator/test_integration_batch.py`
  - Test quickstart Part 3 scenarios (batch datasets)
  - Test dataset save/load (NPZ format)
  - Test balanced class distribution

- [ ] **T019 [P]** Integration test: SNR sweep
  - File: `modules/training/tests/channel_simulator/test_integration_snr.py`
  - Test quickstart Part 4 scenario (SNR sweep -30 to +10 dB)
  - Verify SNR accuracy (±1 dB)
  - Test different noise realizations per SNR

- [ ] **T020 [P]** Integration test: Full pipeline
  - File: `modules/training/tests/test_quickstart_validation.py`
  - Test quickstart Part 5 (end-to-end pipeline)
  - Generate clean signal → All 5 expert examples
  - Verify all expert_type fields correct

---

## Phase 3.3: Core Generator Implementation - Weeks 2-3 (ONLY after tests are failing)

### Pattern Loader

- [x] **T021 [P]** Implement PatternLoader class
  - File: `modules/training/src/signal_generator/pattern_loader.py`
  - Implement `load_pattern(pattern_id, length)` → loads .pkl files
  - Implement pattern caching (dict keyed by (pattern_id, length))
  - Implement `load_all_patterns()` → pre-load all 48 files
  - Validate pattern bits (shape, dtype, values 0/1)
  - Make T006 contract test pass

### GMSK Pulse Shaping

- [x] **T022 [P]** Implement GMSK modulator
  - File: `modules/training/src/signal_generator/gmsk.py`
  - Implement `generate_gmsk_fsk(pattern_bits, frequency_pair, sample_rate=48000)`
  - Implement `generate_gaussian_filter(BT=0.3, span_symbols=4)`
  - Apply Gaussian filter to NRZ pattern
  - Integrate to get phase, convert to IQ
  - Ensure constant envelope (±1% tolerance)
  - Make T007 contract test pass

### Constellation Mapping

- [x] **T023 [P]** Implement constellation mapper
  - File: `modules/training/src/signal_generator/modulation.py`
  - Implement `map_to_constellation(bits, modulation)` for BPSK/QPSK/8-PSK/16-APSK
  - Implement Gray coding for QPSK and 8-PSK
  - Implement 16-APSK ring configuration (4+12 rings)
  - Normalize all constellations to unit average power
  - Implement `get_bits_per_symbol(modulation)` helper
  - Make T008 contract test pass

### Polar Codec Integration

- [x] **T024 [P]** Implement Polar codec wrapper
  - File: `modules/training/src/signal_generator/polar_codec.py`
  - Use commpy library: `from commpy.channelcoding import Polar`
  - Implement `encode(data_bits, code_rate, block_length)`
  - Implement rate validation (k < n, n is power of 2)
  - Implement `get_supported_rates()` → [(1,2), (2,3), (3,4), (4,5), (5,6), (7,8)]
  - Handle padding if data_bits < K
  - Make T009 contract test pass
  - **Source**: CASCADE V2 spec (CLAUDE.md:65, 186) - Polar codes with adaptive rates

### Main Signal Generator

- [x] **T025** Implement SignalGenerator class (Part 1: Core methods)
  - File: `modules/training/src/signal_generator/generator.py`
  - Implement `__init__()` → initialize PatternLoader, load all patterns
  - Implement `validate_parameters(pattern_id, frequency_pair, modulation, polar_rate)`
  - Implement `get_tone_frequencies(frequency_pair)` → (tone_a, tone_b) using 135-channel grid
  - Implement `estimate_message_capacity(pattern_length, polar_rate)` → max data bits
  - Implement `get_required_pattern_length(message_bits, polar_rate)` → 64/128/256/512/1024/2048

- [x] **T026** Implement SignalGenerator class (Part 2: Generation)
  - File: `modules/training/src/signal_generator/generator.py`
  - Implement `generate(pattern_id, frequency_pair, modulation, polar_rate, message, seed)`
  - Convert message → bits
  - Determine pattern length from message size
  - Encode with Polar codec
  - Generate GMSK 2-FSK (Layer 1)
  - Map to constellation (Layer 2)
  - Combine layers (IQ multiplication)
  - Return CleanIQSignal + metadata
  - Implement `generate_from_params(kernel_params, message_data, seed)` wrapper
  - Make T004 contract test pass

- [x] **T027** Implement V2 compliance validator
  - File: `modules/training/src/signal_generator/generator.py`
  - Implement `verify_v2_compliance(signal)` method in SignalGenerator
  - Check symbol rate: 200 ± 0.1 symbols/second
  - Check GMSK bandwidth: < 30 Hz at -40 dB (FFT analysis)
  - Check tone spacing: 20 Hz ± 0.5 Hz
  - Check sample rate: exactly 48000 Hz
  - Check pattern orthogonality: < -20 dB (cross-correlation)
  - Return compliance dict with pass/fail per check
  - Make T005 contract test pass

### CLI Interface (Core Generator)

- [x] **T028 [P]** Implement Core Generator CLI
  - File: `modules/training/src/signal_generator/cli.py`
  - Implement `cascade-signal generate` command (argparse)
  - Args: --pattern-id, --freq-pair, --modulation, --polar-rate, --message, --output, --seed
  - Implement `cascade-signal generate-batch` command with config YAML
  - Implement `cascade-signal verify` command (V2 compliance check)
  - Output IQ to .npy file, metadata to .json
  - Entry point in pyproject.toml: `cascade-signal = signal_generator.cli:main`

---

## Phase 3.4: Orchestrator Implementation - Weeks 3-4

### AWGN Generator

- [ ] **T029 [P]** Implement AWGN generator
  - File: `modules/training/src/channel_simulator/awgn.py`
  - Implement `add_awgn(signal, snr_db, seed)` → complex AWGN noise
  - Calculate noise power from SNR
  - Generate I/Q noise (decorrelated)
  - Verify measured SNR within ±0.5 dB

### QRN Generators (Atmospheric Noise)

- [ ] **T030 [P]** Implement QRN generators
  - File: `modules/training/src/channel_simulator/qrn.py`
  - Implement `generate_crackling_noise(duration, burst_rate, intensity, sample_rate, seed)` → Poisson bursts
  - Implement `generate_static_noise(duration, intensity, sample_rate, seed)` → 1/f spectrum
  - Implement `generate_lightning_noise(duration, strike_rate, intensity, sample_rate, seed)` → impulse noise
  - Implement `generate_power_line_noise(duration, intensity, sample_rate, seed)` → 50/60 Hz harmonics
  - Return (noise_iq, labels_dict) with burst_times/strike_times

### Multipath Generator

- [ ] **T031 [P]** Implement multipath fading
  - File: `modules/training/src/channel_simulator/multipath.py`
  - Implement `apply_multipath(signal, delay_spread_ms, num_taps, fading_type, sample_rate, seed)`
  - Generate tapped delay line (Rayleigh or Rician gains)
  - Apply convolution with impulse response
  - Return (faded_signal, channel_params_dict) with impulse_response, tap_delays, tap_gains

### QRM Generators (Interference)

- [ ] **T032 [P]** Implement QRM generators
  - File: `modules/training/src/channel_simulator/qrm.py`
  - Implement `generate_cw_interference(duration, freq_offset, strength_db, sample_rate, seed)` → single tone
  - Implement `generate_ft8_interference(duration, freq_offset, strength_db, sample_rate, seed)` → 50 Hz GFSK
  - Implement `generate_ssb_interference(duration, freq_offset, strength_db, sample_rate, seed)` → voice spectrum
  - Return complex IQ interference signals

### Collision Generator

- [ ] **T033 [P]** Implement collision scenario generator
  - File: `modules/training/src/channel_simulator/collision.py`
  - Implement `generate_collision_scenario(clean_signals, time_offsets_ms, snr_db_list, relative_powers, base_noise_floor_db, seed)`
  - Time-shift signals to sample boundaries
  - Scale signals by SNR and relative power
  - Sum overlapping signals + base noise
  - Return (combined_iq, labels_dict) with signal_boundaries, individual_signals, kernels

### Main Orchestrator

- [ ] **T034** Implement ChannelOrchestrator class (Part 1: Expert generators)
  - File: `modules/training/src/channel_simulator/orchestrator.py`
  - Implement `__init__()`
  - Implement `generate_qrn_expert_data(duration, qrn_type, intensity, sample_rate, seed)` → ExpertTrainingExample
  - Implement `generate_signal_expert_data(clean_iq, seed)` → pass-through, labels only
  - Implement `generate_qrm_expert_data(duration, interference_type, freq_offset, strength_db, sample_rate, seed)` → ExpertTrainingExample
  - Make T010, T011, T014 contract tests pass

- [ ] **T035** Implement ChannelOrchestrator class (Part 2: Timing expert)
  - File: `modules/training/src/channel_simulator/orchestrator.py`
  - Implement `generate_timing_expert_data(collision_scenario, clean_signals, base_noise_floor_db, seed)` → ExpertTrainingExample
  - Use collision generator from T033
  - Include separated signals in labels (ground truth)
  - Make T012 contract test pass

- [ ] **T036** Implement ChannelOrchestrator class (Part 3: Channel expert)
  - File: `modules/training/src/channel_simulator/orchestrator.py`
  - Implement `generate_channel_expert_data(clean_iq, channel_type, channel_params, kernel_parameters, seed)` → ExpertTrainingExample
  - **CRITICAL**: Accept kernel_parameters argument (NEW from CLAUDE.md update)
  - Use multipath generator from T031
  - Include kernel_parameters in labels
  - Make T013 contract test pass

- [ ] **T037** Implement ChannelOrchestrator class (Part 4: Batch and traditional methods)
  - File: `modules/training/src/channel_simulator/orchestrator.py`
  - Implement `generate_batch(expert_type, num_examples, config, seed)` → List[ExpertTrainingExample]
  - Implement `add_channel_effects(clean_iq, channel_conditions, seed)` → RealisticIQSignal (traditional combined effects)
  - Implement `generate_snr_sweep(clean_iq, snr_start_db, snr_stop_db, snr_step_db, seed)` → List[RealisticIQSignal]
  - Make T015 contract test pass

### Dataset Export

- [ ] **T038 [P]** Implement dataset save/load
  - File: `modules/training/src/channel_simulator/batch.py`
  - Implement `save_expert_dataset(examples, output_dir, dataset_name, format='npz')` → dict
  - Support NPZ, HDF5, Zarr formats
  - Save IQ samples and labels separately
  - Include metadata JSON with generation parameters
  - Implement load functions for PyTorch/TensorFlow compatibility

### CLI Interface (Orchestrator)

- [ ] **T039 [P]** Implement Orchestrator CLI
  - File: `modules/training/src/channel_simulator/cli.py`
  - Implement `cascade-orchestrator simulate` command (add channel effects)
  - Implement `cascade-orchestrator sweep` command (SNR sweep)
  - Implement `cascade-orchestrator generate-expert-dataset` command (batch for specific expert)
  - Args: --expert-type (qrn/signal/timing/channel/qrm), --num-examples, --config
  - Entry point: `cascade-orchestrator = channel_simulator.cli:main`

---

## Phase 3.5: Integration & Validation - Week 5-6

### Integration Tests Passing

- [ ] **T040** Make integration tests pass
  - Run T016-T020 integration tests
  - Fix any issues discovered
  - Ensure all quickstart scenarios work end-to-end
  - Verify expert separation (QRN=pure noise, Signal=clean, etc.)

### Performance Validation

- [ ] **T041** Performance benchmarking
  - File: `modules/training/tests/test_performance.py`
  - Benchmark Core Generator: 512-symbol signal in <100ms
  - Benchmark Orchestrator: 100 signals in <30s
  - Profile with cProfile, identify bottlenecks
  - Optimize if needed (pattern caching, vectorization)

### Property-Based Tests

- [ ] **T042 [P]** Add property-based tests with hypothesis
  - File: `modules/training/tests/signal_generator/test_properties.py`
  - Test signal generation never crashes for any valid kernel parameters
  - Test V2 compliance holds for all pattern/frequency/modulation combinations
  - Test SNR measurement accuracy across wide range
  - Use hypothesis strategies for random parameter generation

### Documentation

- [ ] **T043 [P]** Update module documentation
  - Add docstrings to all public methods (Google style)
  - Create `modules/training/README.md` with signal generator usage
  - Document expert dataset format for NN training
  - Add examples from quickstart.md

---

## Phase 3.6: Polish - Week 6

### Code Quality

- [ ] **T044** Remove duplication and refactor
  - Extract common validation logic
  - Consolidate noise generation utilities
  - Ensure DRY principle throughout

- [ ] **T045** Final test coverage check
  - Run `pytest --cov=modules/training/src --cov-report=html`
  - Ensure >90% coverage for Core Generator
  - Ensure >85% coverage for Orchestrator
  - Add tests for uncovered branches

- [ ] **T046** Run quickstart validation
  - Execute all steps from `specs/004-signal-generator/quickstart.md`
  - Verify all expected outputs match
  - Confirm success criteria met
  - Document any deviations

### Final Validation

- [ ] **T047** Run full test suite
  - Execute `pytest modules/training/tests/ -v`
  - Ensure all contract tests pass
  - Ensure all integration tests pass
  - Fix any failing tests

- [ ] **T048** Generate example datasets
  - Generate 100 examples for each of 5 expert types (500 total)
  - Save to `output/expert_datasets/`
  - Verify file sizes and format
  - Confirm ready for NN training (Phase 2 of CASCADE roadmap)

---

## Dependencies

### Critical Path
```
Setup (T001-T003)
  ↓
Contract Tests (T004-T015, T016-T020) [ALL PARALLEL]
  ↓
Core Generator Implementation (T021-T028) [MOSTLY PARALLEL]
  ├─ T021 (PatternLoader) → T025
  ├─ T022 (GMSK) → T026
  ├─ T023 (Modulation) → T026
  ├─ T024 (Polar) → T026
  ├─ T025 → T026 → T027
  └─ T028 (CLI, independent)
  ↓
Orchestrator Implementation (T029-T039) [MOSTLY PARALLEL]
  ├─ T029-T033 (Noise/Interference generators) → T034-T037
  ├─ T034-T037 (Orchestrator class, sequential)
  ├─ T038 (Dataset export, independent)
  └─ T039 (CLI, independent)
  ↓
Integration & Validation (T040-T048)
  ├─ T040 (Integration tests)
  ├─ T041-T043 [PARALLEL]
  ├─ T044-T046 (sequential refinement)
  └─ T047-T048 (final validation)
```

### Detailed Dependencies
- **T001-T003**: No dependencies (setup)
- **T004-T020**: Depend on T001-T003 (structure exists)
- **T021-T024**: Depend on T004-T009 (contract tests exist)
- **T025**: Depends on T021-T024 (all components ready)
- **T026**: Depends on T025 (core methods implemented)
- **T027**: Depends on T026 (generator working)
- **T028**: Depends on T004, T021-T027 (Core Generator complete)
- **T029-T033**: Depend on T010-T014 (contract tests exist)
- **T034**: Depends on T029-T033 (noise generators ready)
- **T035**: Depends on T033, T034 (collision generator + base orchestrator)
- **T036**: Depends on T031, T034 (multipath + base orchestrator)
- **T037**: Depends on T034-T036 (all expert generators ready)
- **T038**: Depends on T010-T015 (dataset format defined)
- **T039**: Depends on T034-T038 (Orchestrator complete)
- **T040**: Depends on T028, T039 (both CLIs working)
- **T041-T043**: Depend on T040 (integration working)
- **T044-T046**: Depend on T041-T043 (testing complete)
- **T047-T048**: Depend on T044-T046 (polished code)

---

## Parallel Execution Examples

### Phase 3.2: All Contract Tests (T004-T020)
```bash
# Launch all 17 contract tests in parallel (different files, independent)
# Week 1, Day 3
pytest modules/training/tests/signal_generator/test_contract_generator.py &
pytest modules/training/tests/signal_generator/test_contract_compliance.py &
pytest modules/training/tests/signal_generator/test_contract_pattern_loader.py &
pytest modules/training/tests/signal_generator/test_contract_gmsk.py &
pytest modules/training/tests/signal_generator/test_contract_modulation.py &
pytest modules/training/tests/signal_generator/test_contract_polar.py &
pytest modules/training/tests/channel_simulator/test_contract_qrn_expert.py &
pytest modules/training/tests/channel_simulator/test_contract_signal_expert.py &
pytest modules/training/tests/channel_simulator/test_contract_timing_expert.py &
pytest modules/training/tests/channel_simulator/test_contract_channel_expert.py &
pytest modules/training/tests/channel_simulator/test_contract_qrm_expert.py &
pytest modules/training/tests/channel_simulator/test_contract_batch.py &
pytest modules/training/tests/signal_generator/test_integration.py &
pytest modules/training/tests/channel_simulator/test_integration_experts.py &
pytest modules/training/tests/channel_simulator/test_integration_batch.py &
pytest modules/training/tests/channel_simulator/test_integration_snr.py &
pytest modules/training/tests/test_quickstart_validation.py &
wait
```

### Phase 3.3: Core Generator Components (T021-T024)
```bash
# Week 2, Days 1-2: Implement all components in parallel
# T021: PatternLoader
# T022: GMSK
# T023: Modulation
# T024: Polar
# All independent, different files
```

### Phase 3.4: Noise Generators (T029-T033)
```bash
# Week 3, Days 1-2: Implement all noise generators in parallel
# T029: AWGN
# T030: QRN
# T031: Multipath
# T032: QRM
# T033: Collision
# All independent, different files
```

### Phase 3.5: Polish Tasks (T041-T043)
```bash
# Week 6, Days 1-2: Parallel polishing
# T041: Performance benchmarking
# T042: Property-based tests
# T043: Documentation
# All independent
```

---

## Notes

### TDD Workflow
1. **Week 1, Days 3-5**: Write ALL tests (T004-T020)
2. Verify tests FAIL (no implementation yet)
3. **Weeks 2-4**: Implement code to make tests pass
4. Run tests continuously during implementation
5. **Weeks 5-6**: Integration, validation, polish

### Expert Dataset Separation (Critical!)
- **QRN Expert (T010, T030, T034)**: ONLY noise, NO signal
- **Signal Expert (T011, T034)**: ONLY clean signal, NO noise
- **Timing Expert (T012, T033, T035)**: Collision scenarios, includes separated ground truth
- **Channel Expert (T013, T031, T036)**: Channel distortion + kernel_parameters (NEW!)
- **QRM Expert (T014, T032, T034)**: ONLY interference, NO CASCADE signal

### Performance Targets
- Single signal generation: <100ms (T041)
- Batch of 100 signals: <30s (T041)
- Pattern loading (all 48): <1s (T021)

### V2 Compliance (Critical!)
- Symbol rate: 200 ± 0.1 sym/s (T027)
- GMSK bandwidth: <30 Hz at -40 dB (T027)
- Tone spacing: 20 Hz ± 0.5 Hz (T027)
- Pattern orthogonality: <-20 dB (T027)

### File Paths (All Absolute)
- **Core Generator**: `modules/training/src/signal_generator/`
- **Orchestrator**: `modules/training/src/channel_simulator/`
- **Tests**: `modules/training/tests/`
- **Patterns**: `modules/training/patterns/tournament/pattern_*.pkl`
- **Output**: `output/expert_datasets/`

---

## Validation Checklist

Before marking tasks.md complete, verify:

- [x] All 2 contract files have corresponding test tasks (T004-T015)
- [x] All entities from data-model.md have creation tasks (covered in T021-T027, T034-T037)
- [x] All tests come before implementation (T004-T020 before T021-T048)
- [x] Parallel tasks [P] are truly independent (different files)
- [x] Each task specifies exact file path
- [x] No [P] task modifies same file as another [P] task
- [x] Dependencies correctly identified
- [x] Critical requirements captured (expert separation, kernel_parameters, V2 compliance)
- [x] TDD workflow enforced (tests MUST fail before implementation)
- [x] Performance targets specified (T041)

---

**Total Tasks**: 48 tasks over 6 weeks
**Estimated Timeline**:
- Week 1: Setup + All Tests (T001-T020)
- Weeks 2-3: Core Generator (T021-T028)
- Weeks 3-4: Orchestrator (T029-T039)
- Weeks 5-6: Integration, Validation, Polish (T040-T048)

**Status**: Ready for execution
**Next Step**: Begin with T001 (project structure)
