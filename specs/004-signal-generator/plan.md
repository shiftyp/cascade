# Implementation Plan: Signal Generator

**Branch**: `004-signal-generator` | **Date**: 2025-10-07 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/workspaces/cascade/specs/004-signal-generator/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path ✅
2. Fill Technical Context ✅
3. Fill Constitution Check ✅
4. Evaluate Constitution Check → In Progress
5. Execute Phase 0 → research.md
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md update
7. Re-evaluate Constitution Check
8. Plan Phase 2 → Describe task generation approach
9. STOP - Ready for /tasks command
```

## Summary
Signal generator for CASCADE V2 testing with two-part architecture: (1) **Core Signal Generator** produces clean V2-compliant IQ signals using kernel parameters (pattern_id, frequency_pair, modulation, polar_rate) and generates GMSK-modulated 2-FSK patterns with IQ data overlay, (2) **Synthetic Data Orchestrator** generates **expert-specific training datasets** for CASCADE's 5 specialized neural networks (QRN Expert, Signal Expert, Timing Expert, Channel Expert, QRM Expert) plus integration scenarios with combined impairments.

**Critical Requirement**: Orchestrator must generate **5 separate datasets** for expert pre-training, NOT just general noisy signals. Each expert requires isolated training data (e.g., pure noise for QRN Expert, clean signals for Signal Expert, collision scenarios for Timing Expert).

## Technical Context
**Language/Version**: Python 3.11
**Primary Dependencies**: numpy, scipy (DSP), pickle (pattern loading), commpy (Polar codes), matplotlib (validation plots)
**Storage**: IQ sample arrays (complex float), metadata JSON files, pattern files (.pkl from genetic algorithm)
**Testing**: pytest with property-based testing (hypothesis) for signal validation
**Target Platform**: Linux (development), cross-platform (library usage)
**Project Type**: single (Python library + CLI in training module)
**Performance Goals**: Generate 512-symbol signal in <100ms (Core Generator), process batch of 100 signals in <30s (Orchestrator)
**Constraints**: Must match CASCADE V2 specification exactly (GMSK BT=0.3, 200 sym/s, 135-tone grid, 8 patterns, -21.19 dB orthogonality)
**Scale/Scope**: 8 patterns × 67 frequency pairs × 6 nested lengths × 4 modulations × 7 polar rates = thousands of test configurations

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Data-First Development**: ✅ Training module feature - pattern generation already complete (prerequisite met)
- [x] **Monorepo Module Architecture**: ✅ Confined to `modules/training/` (signal generation for model training/validation)
- [x] **Clean Separation**: ✅ Generates physical layer signals only (no protocol logic, no model inference)
- [x] **Test-Driven Development**: ✅ Tests planned in Phase 1 (contract tests for V2 compliance, property tests for signal integrity)
- [x] **Real-World Data Priority**: ✅ Generates synthetic signals for testing, but orchestrator mimics real-world conditions from KiwiSDR data characteristics
- [x] **Privacy-Preserving**: ✅ N/A (generates test signals, no user data collection)
- [x] **Reproducible Research**: ✅ Deterministic generation with explicit seeds, versioned pattern files, documented parameters

**No violations detected** - Feature aligns with constitutional principles

## Project Structure

### Documentation (this feature)
```
specs/004-signal-generator/
├── plan.md              # This file
├── spec.md              # Feature specification (complete)
├── research.md          # Phase 0 output (to be created)
├── data-model.md        # Phase 1 output (to be created)
├── quickstart.md        # Phase 1 output (to be created)
├── contracts/           # Phase 1 output (to be created)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (CASCADE monorepo - training module)
```
modules/training/
├── src/
│   ├── signal_generator/       # NEW: Core Signal Generator
│   │   ├── __init__.py
│   │   ├── generator.py        # Main signal generation logic
│   │   ├── gmsk.py             # GMSK pulse shaping (BT=0.3)
│   │   ├── modulation.py       # BPSK/QPSK/8-PSK/16-APSK constellations
│   │   ├── polar_codec.py      # Polar encoding (rates 1/2 to 7/8)
│   │   ├── pattern_loader.py   # Load patterns from .pkl files
│   │   └── cli.py              # Command-line interface
│   ├── channel_simulator/      # NEW: Synthetic Data Orchestrator
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # Main orchestration logic
│   │   ├── awgn.py             # Additive White Gaussian Noise
│   │   ├── qrn.py              # Atmospheric noise model (QRN)
│   │   ├── multipath.py        # Frequency-selective fading
│   │   ├── qrm.py              # Interference from other stations
│   │   ├── batch.py            # Batch generation management
│   │   └── cli.py              # Command-line interface
│   ├── patterns/               # EXISTING: Pattern generator
│   │   └── tournament/         # Genetic algorithm patterns
│   │       └── *.pkl           # Pre-generated pattern files
│   └── ...
└── tests/
    ├── signal_generator/       # NEW: Core Generator tests
    │   ├── test_v2_compliance.py    # V2 spec validation
    │   ├── test_gmsk.py             # GMSK pulse shaping
    │   ├── test_modulation.py       # Constellation mapping
    │   ├── test_polar.py            # Polar encoding
    │   ├── test_integration.py      # End-to-end generation
    │   └── test_cli.py              # CLI interface
    ├── channel_simulator/      # NEW: Orchestrator tests
    │   ├── test_awgn.py             # AWGN application
    │   ├── test_qrn.py              # QRN model
    │   ├── test_multipath.py        # Multipath simulation
    │   ├── test_qrm.py              # QRM interference
    │   ├── test_batch.py            # Batch processing
    │   └── test_cli.py              # CLI interface
    └── ...
```

**Structure Decision**: This feature belongs to the **training module** because it generates synthetic test signals for decoder validation and model training. Per the constitution's module structure, the training module handles "dataset versioning, model architecture, training pipelines, and **benchmarking suite**". Signal generation enables the benchmarking suite by providing controlled test cases. It does NOT belong to the protocol module (which handles message encoding/beacons/priority) or data module (which collects real KiwiSDR recordings).

---

## Phase 0: Outline & Research
**Status**: ✅ COMPLETE

**Output**: `research.md` (14 technical decisions documented)

**Key Decisions**:
1. **Dual-layer modulation**: GMSK 2-FSK pattern + BPSK/QPSK/8-PSK/16-APSK data (IQ domain multiplication)
2. **GMSK pulse shaping**: BT=0.3 via SciPy Gaussian filter + phase integration
3. **Frequency plan**: 135 channels at 20 Hz spacing (300-3000 Hz), 67 frequency pairs
4. **Constellation mapping**: Standard PSK/APSK with Gray coding
5. **Polar encoding**: commpy library (rates 1/2 to 7/8)
6. **Pattern loading**: Pickle format from genetic algorithm output (48 files cached at startup)
7. **Sample rate**: 48 kHz complex baseband IQ
8. **Channel simulation**: Separate models for AWGN, QRN (Poisson bursts), multipath (Rayleigh taps), QRM (synthetic interferers)
9. **Metadata**: JSON sidecar files with ground truth for decoder validation
10. **Performance**: <100ms single signal, 100 signals in <30s batch
11. **API design**: CLI (argparse) + Library (Python classes)
12. **Testing**: Multi-level (unit, contract, integration, property-based with hypothesis)
13. **Dependencies**: NumPy, SciPy, commpy (minimal)
14. **Future work**: Diversity modes, real-time streaming, hardware integration deferred to post-V1

**No blocking issues identified** - Ready for Phase 1

---

## Phase 1: Design & Contracts
**Status**: ✅ COMPLETE

**Output**: data-model.md, contracts/, quickstart.md

**Key Deliverables**:
1. **data-model.md**: 7 core entities + 1 metadata format defined
   - KernelParameters, MessageData, Pattern, CleanIQSignal (Core Generator)
   - ExpertTrainingExample (5 expert types), ChannelConditions, RealisticIQSignal, GroundTruth (Orchestrator)
   - **Updated**: Channel Expert labels now include kernel_parameters (per CLAUDE.md architecture update)

2. **contracts/signal_generator_interface.py**: Core Generator API contract
   - SignalGeneratorInterface: Main generation methods
   - PatternLoaderInterface: Pattern caching
   - GMSKModulatorInterface: GMSK pulse shaping
   - ConstellationMapperInterface: IQ modulation
   - PolarCodecInterface: Error correction encoding

3. **contracts/channel_orchestrator_interface.py**: Orchestrator API contract
   - ChannelOrchestratorInterface: Expert-specific dataset generation
   - 5 expert data generation methods (QRN, Signal, Timing, Channel, QRM)
   - Batch generation, SNR sweep, dataset export
   - **Updated**: Channel expert generator now requires kernel_parameters argument

4. **quickstart.md**: End-to-end validation guide
   - Part 1: Core Generator validation (clean signals)
   - Part 2: Expert-specific data generation (5 experts)
   - Part 3: Batch generation for training datasets
   - Part 4: SNR sweep for decoder validation
   - Part 5: Integration test (full pipeline)
   - **Updated**: Channel expert examples now pass kernel_parameters

**CLAUDE.md Architecture Updates Incorporated**:
- **Embedding encoder inputs** (lines 505, 610, 682, 742): Now takes channel observations + kernel parameters (not just channel alone)
- **Rationale**: Encoder needs kernel context (pattern_id, frequency_pair, modulation) for context-aware embedding compression
- **Integration decoder outputs** (lines 605, 674-678): Outputs soft decisions (log-likelihood ratios) for Polar decoder, not hard bits
- **TX vs RX kernel** (lines 623, 708): Clarified terminology - signal generator produces TX kernels

**No constitution violations** - Ready for Phase 2

---

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:

The /tasks command will generate implementation tasks based on Phase 1 design artifacts (contracts, data model, quickstart). Tasks will follow TDD principles with contract tests written before implementation.

**Task Categories**:

1. **Contract Tests** (Week 1, 15-20 tasks):
   - Core Generator contract tests (signal_generator_interface.py)
   - Orchestrator contract tests (channel_orchestrator_interface.py)
   - Each contract method → 1-2 test tasks
   - Tests must fail initially (no implementation yet)

2. **Core Generator Implementation** (Weeks 2-3, 20-25 tasks):
   - Pattern loader with caching
   - GMSK pulse shaping (BT=0.3)
   - Constellation mapping (BPSK/QPSK/8-PSK/16-APSK)
   - Polar codec integration
   - Dual-layer signal generation
   - V2 compliance validation
   - CLI interface

3. **Orchestrator - Expert Data Generators** (Weeks 3-4, 25-30 tasks):
   - QRN generators (crackling, static, lightning, power line)
   - Signal expert generator (clean signals)
   - Timing expert generator (collision scenarios)
   - Channel expert generator (multipath, Rayleigh, Rician, Doppler) + kernel parameters
   - QRM generators (CW, SSB, FT8, digital modes)
   - Batch generation framework
   - Dataset export (NPZ/HDF5/Zarr)

4. **Orchestrator - Traditional Channel Effects** (Week 5, 10-15 tasks):
   - AWGN generator
   - SNR sweep generator
   - Integration with expert generators

5. **Integration & Validation** (Week 6, 10-15 tasks):
   - End-to-end pipeline tests
   - Quickstart validation
   - Performance benchmarking
   - Documentation

**Ordering Strategy**:
- TDD order: Contract tests → Implementation → Integration tests
- Dependency order: Pattern loader → GMSK → Modulation → Generator → Orchestrator
- Parallelizable tasks marked [P] for independent execution

**Estimated Task Count**: 80-105 tasks total

**Key Considerations**:
- **Expert-based architecture**: Orchestrator must generate 5 separate datasets (QRN, Signal, Timing, Channel, QRM)
- **Kernel parameters**: Channel expert generator must accept and include kernel_parameters in labels
- **Collision scenarios**: Timing expert requires complex multi-signal generation with ground truth separation
- **Performance targets**: <100ms single signal, 100 signals in <30s batch

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

---

## Progress Tracking

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning approach described (/plan command)
- [ ] Phase 3: Tasks generated (/tasks command - NOT YET RUN)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] CLAUDE.md architecture updates incorporated
- [x] All technical unknowns resolved
- [ ] Complexity deviations documented: NONE

---

**Plan Status**: ✅ COMPLETE - Ready for /tasks command
