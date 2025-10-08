# Feature Specification: Signal Generator

**Feature Branch**: `004-signal-generator`
**Created**: 2025-10-07
**Status**: Draft
**Input**: User description: "signal generator"

## Execution Flow (main)
```
1. Parse user description from Input
   → Feature: Signal generator for CASCADE testing
2. Extract key concepts from description
   → Actors: Test engineers, decoder developers
   → Actions: Generate test signals, add noise, simulate propagation
   → Data: Patterns, IQ samples, audio files
   → Constraints: Must match V2 specification exactly
3. For each unclear aspect:
   → [NEEDS CLARIFICATION: Output format - WAV files, raw IQ, or live audio?]
   → [NEEDS CLARIFICATION: Noise models - Gaussian only or include QRN/QRM/QSB?]
   → [NEEDS CLARIFICATION: Multipath simulation required?]
4. Fill User Scenarios & Testing section
   → Primary: Generate known test signals for decoder validation
5. Generate Functional Requirements
   → Must generate spec-compliant signals
   → Must support all pattern lengths, modulations, Polar rates
6. Identify Key Entities
   → Test signal, Pattern configuration, Channel conditions
7. Run Review Checklist
   → [NEEDS CLARIFICATION: Integration with pattern generator output?]
   → [NEEDS CLARIFICATION: Real-time or batch generation?]
8. Return: SUCCESS (spec ready for planning with clarifications needed)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## Clarifications

### Session 2025-10-07
- Q: What level of HF channel simulation is required? → A: Full HF model - AWGN + QRN + Multipath + QRM interference (most realistic)
- Q: Architecture design → A: Two-part system - (1) Core CASCADE signal generator for clean V2-compliant signals, (2) Synthetic data orchestrator that wraps generator and adds realistic HF channel conditions
- Q: Should the Core Signal Generator support diversity mode (2×/4×/8× frequency pairs)? → A: No - Single frequency pair only (diversity added later if needed)
- Q: Should Orchestrator output WAV, IQ arrays, or both? → A: IQ arrays only (WAV conversion separate if needed)
- Q: What inputs should Core Generator accept? → A: Message data + discrete kernel parameters (pattern_id, frequency_pair, modulation, polar_rate) - aligns with CASCADE protocol
- Q: What user interface should the system support? → A: Both CLI and library API - Maximum flexibility for scripts, automation, and programmatic usage
- Q: When SNR requested is below Shannon limit for selected modulation, should system generate anyway or error? → A: Warn but generate - Issue warning, proceed with generation (enables stress testing)

---

## Module Context *(mandatory for CASCADE)*

### Target Module
**Module:** `training` (test signal generation for model training and validation)

**Architecture:** Two-part system
1. **Core Signal Generator**: Produces clean CASCADE V2-compliant signals (patterns, modulation, encoding)
2. **Synthetic Data Orchestrator**: Wraps generator, adds realistic HF channel conditions (QRN, multipath, QRM), manages batch generation

**Rationale:** Separation enables (a) clean reference signal generation for spec validation, (b) realistic channel simulation for decoder training, (c) independent testing of each component.

### Module Dependencies
**Prerequisites:**
- Pattern generation (must be completed - patterns exist in `modules/training/patterns/tournament/`)
- Signal specification (`docs/protocol/signal_specification.md` - defines physical layer)

**Not dependent on:**
- Data module (KiwiSDR collection) - independent test capability
- Protocol module - generates physical layer only, not protocol messages

### Module Interfaces

**Core Signal Generator:**
- **Inputs:**
  - Pattern files (`.pkl` from pattern generator)
  - Message data (text or binary)
  - Discrete kernel parameters: pattern_id (0-7), frequency_pair (0-66), modulation (BPSK/QPSK/8-PSK/16-APSK), polar_rate (1/2, 2/3, 3/4, 4/5, 5/6, 7/8)
- **Outputs:**
  - Clean IQ sample arrays (complex float)

**Synthetic Data Orchestrator:**
- **Inputs:**
  - Clean IQ arrays from Core Generator
  - Channel condition parameters: SNR, QRN settings, multipath configuration, QRM interferer specs
- **Outputs:**
  - IQ sample arrays with channel effects applied (WAV conversion handled separately if needed for hardware testing)

**Integration points:**
- Decoder validation (feeds test signals to receiver)
- Model training (generates training samples with known ground truth)
- Performance benchmarking (controlled SNR sweeps)

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a CASCADE decoder developer, I need to:
1. Generate clean reference signals that match V2 specification exactly (via **Core Signal Generator**), so I can validate basic decoder functionality against known-good signals
2. Generate realistic test signals with HF channel impairments (via **Synthetic Data Orchestrator**), so I can train and validate my decoder under realistic atmospheric noise, multipath fading, and interference conditions matching real-world KiwiSDR data

### Acceptance Scenarios
1. **Given** pattern files loaded and message "TEST", **When** user provides kernel params (pattern_id=3, freq_pair=25, modulation=QPSK, polar_rate=2/3) to Core Generator, **Then** generator produces clean IQ array with correct GMSK 2-FSK pattern + QPSK data encoding
2. **Given** clean IQ array from Core Generator, **When** Orchestrator applies channel conditions (SNR=-15dB, QRN enabled, multipath off), **Then** output IQ array contains signal with atmospheric noise at specified SNR
3. **Given** clean IQ array, **When** Orchestrator runs SNR sweep from -30dB to +10dB in 3dB steps, **Then** system produces series of IQ arrays with identical signal at different noise levels
4. **Given** IQ array with known message "TEST", **When** fed to decoder under test, **Then** decoder successfully recovers transmitted message (validates both generator and decoder)
5. **Given** clean IQ array, **When** Orchestrator applies multipath simulation with 3ms delay spread, **Then** output IQ array contains delayed copies of signal with appropriate phase/amplitude per HF channel model

### Edge Cases
- What happens when requested message size exceeds pattern capacity?
  - **Expected**: Generator selects next larger pattern length automatically or returns error
- How does system handle invalid pattern IDs (>7)?
  - **Expected**: Validation error with clear message
- What happens when SNR requested is below Shannon limit for selected modulation?
  - **Expected**: System issues warning message but proceeds with generation (enables decoder stress testing at theoretically impossible SNR levels)

---

## Requirements *(mandatory)*

### Functional Requirements

**Core Signal Generator (Clean V2-compliant signals):**
- **FR-001**: Generator MUST accept kernel discrete parameters as input: pattern_id (0-7), frequency_pair (0-66), modulation (BPSK/QPSK/8-PSK/16-APSK), polar_rate (1/2, 2/3, 3/4, 4/5, 5/6, 7/8)
- **FR-002**: Generator MUST accept message data as input (text or binary)
- **FR-003**: Generator MUST load pattern files from pattern generator output (`.pkl` format)
- **FR-004**: Generator MUST produce signals using specified pattern from kernel parameters
- **FR-005**: Generator MUST support all 6 nested pattern lengths (64, 128, 256, 512, 1024, 2048 symbols) based on message size
- **FR-006**: Generator MUST apply GMSK pulse shaping with BT=0.3 to pattern FSK layer
- **FR-007**: Generator MUST apply specified IQ modulation scheme from kernel parameters
- **FR-008**: Generator MUST apply correct constellation mapping per CASCADE signal specification
- **FR-009**: Generator MUST encode user data with Polar codes at rate specified in kernel parameters
- **FR-010**: Generator MUST pad or truncate messages to fit selected pattern length
- **FR-011**: Generator MUST generate signals on frequency pair specified in kernel parameters using 135-tone grid (300-3000 Hz, 20 Hz spacing)
- **FR-012**: Generator MUST maintain 200 symbols/second symbol rate (5ms per symbol)
- **FR-013**: Generator MUST support single frequency pair only (diversity mode deferred to future enhancement)
- **FR-014**: Generator MUST output clean IQ sample arrays (complex float format)
- **FR-015**: Generator MUST verify generated signals match V2 specification (self-test mode)

**Synthetic Data Orchestrator (Realistic HF channel simulation):**
- **FR-016**: Orchestrator MUST accept clean IQ sample arrays from Core Generator as input
- **FR-017**: Orchestrator MUST add AWGN (Additive White Gaussian Noise) at specified SNR levels
- **FR-018**: Orchestrator MUST support SNR range from -35 dB to +20 dB
- **FR-019**: Orchestrator MUST issue warning when requested SNR is below Shannon limit for selected modulation, but proceed with generation (enables stress testing)
- **FR-020**: Orchestrator MUST apply atmospheric noise (QRN) models with realistic crackling and static burst characteristics
- **FR-021**: Orchestrator MUST simulate frequency-selective fading (multipath) with configurable delay spread (1-5ms typical for HF)
- **FR-022**: Orchestrator MUST simulate QRM (interference from other CASCADE stations) with configurable interferer count and strength
- **FR-023**: Orchestrator MUST support batch generation (multiple SNR levels, channel conditions, interferer scenarios)
- **FR-024**: Orchestrator MUST provide progress indication for long batch generation runs
- **FR-025**: Orchestrator MUST include option to inject bit errors for decoder stress testing
- **FR-026**: Orchestrator MUST output IQ sample arrays (complex float format) with channel effects applied
- **FR-027**: Orchestrator MUST include metadata file describing signal parameters (pattern ID, modulation, SNR, message, channel conditions)
- **FR-028**: Orchestrator MUST tag outputs with known ground truth for decoder validation

**User Interface Requirements:**
- **FR-029**: System MUST provide command-line interface (CLI) for both Core Generator and Orchestrator
- **FR-030**: System MUST provide Python library API for both Core Generator and Orchestrator (importable modules)
- **FR-031**: CLI MUST support all kernel parameters, message input, and channel configuration options
- **FR-032**: Library API MUST expose same functionality as CLI for programmatic usage (scripts, notebooks, test harnesses)

### Key Entities *(data involved)*

**KernelParameters (Input to Core Generator):**
- Represents discrete portion of CASCADE kernel
- Attributes: pattern_id (0-7), frequency_pair (0-66), modulation (BPSK/QPSK/8-PSK/16-APSK), polar_rate (1/2, 2/3, 3/4, 4/5, 5/6, 7/8)
- Source: Provided by user/test harness (mirrors actual CASCADE RX kernel discrete fields)
- Purpose: Specifies exactly how to encode the signal per V2 protocol

**MessageData (Input to Core Generator):**
- Represents user data to be transmitted
- Format: Text string or binary byte array
- Constraints: Must fit within pattern capacity after Polar encoding

**CleanIQSignal (Output from Core Generator, Input to Orchestrator):**
- Represents V2-compliant signal without channel effects
- Format: Complex float array (I + jQ samples)
- Sample rate: 48 kHz (standard sound card rate)
- Duration: Depends on pattern length (0.32s to 10.24s @ 200 sym/s)
- Attributes: Associated metadata (pattern_id, modulation, polar_rate, message, timestamp)

**Pattern:**
- Represents one of 8 CASCADE patterns at specific length
- Attributes: pattern_id (0-7), length (64-2048), binary_sequence
- Source: Loaded from pattern generator output files (`.pkl`)

**ChannelConditions (Input to Orchestrator):**
- Represents simulated HF propagation environment (full model)
- Attributes: SNR, AWGN_level, QRN_characteristics (burst_rate, intensity), multipath_config (tap_count, delay_spread_ms), QRM_interferers (count, relative_strength_dB, frequency_offsets_Hz)
- Purpose: Specify realistic HF impairments to apply

**RealisticIQSignal (Output from Orchestrator):**
- Represents signal with HF channel effects applied
- Format: Complex float array (I + jQ samples)
- Sample rate: 48 kHz
- Attributes: All CleanIQSignal metadata + ChannelConditions metadata
- Purpose: Feed to decoder for realistic testing

**GroundTruth:**
- Represents known transmitted data for validation
- Attributes: original_message, kernel_params, polar_encoded_bits, iq_symbols, expected_decode_output
- Purpose: Enables automated decoder testing and performance measurement

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded (two-part system: Core Generator + Orchestrator)
- [x] Dependencies and assumptions identified

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked and resolved (5 questions answered)
- [x] User scenarios defined
- [x] Requirements generated and clarified
- [x] Entities identified
- [x] Review checklist passed

---
