# Research: Phase 0 Training Implementation

**Feature**: 002-phase0-training
**Date**: 2025-10-04

## Research Questions & Decisions

### 1. AWGN Channel Simulation

**Question**: How to implement ideal AWGN channel for vetting?

**Decision**: Use NumPy's `random.normal()` with calculated noise power

**Rationale**:
- Standard in communications research
- Reproducible with fixed random seeds (NFR-004)
- Matches theoretical Shannon capacity calculations
- Simple implementation (no external dependencies)

**Implementation approach**:
```python
# Calculate noise power from SNR
signal_power = np.mean(np.abs(signal)**2)
noise_power = signal_power / (10 ** (snr_db / 10))
noise_std = np.sqrt(noise_power)

# Generate AWGN
awgn = np.random.normal(0, noise_std, signal.shape)
noisy_signal = signal + awgn
```

**Alternatives considered**:
- SciPy stats.norm: Overkill for simple Gaussian
- Custom noise generator: Unnecessary complexity
- MATLAB-style awgn(): Not needed in Python

---

### 2. Synthetic CASCADE Signal Generation

**Question**: How to generate valid CASCADE signals for testing without real transmitters?

**Decision**: Implement RS(32,20) encoder following CASCADE specification in docs/model/pattern_architecture.md

**Components required**:

**A. Pattern Storage**
- Load 128 patterns from specification (will use placeholder/simplified patterns for vetting)
- Each pattern: 32-symbol tone index sequence (indices 0-3)
- Each pattern: Single IQ trajectory for baked-in complexity

**B. RS(32,20) Encoding**
- Use Reed-Solomon over GF(256)
- 20 information symbols: pattern_id (1 byte) + checksum (1 byte) + data (18 bytes)
- 12 parity symbols
- Standard RS implementation (e.g., reedsolo library or custom)

**C. 4D Signal Mapping**
- Map each RS symbol (8 bits) to Time × Frequency × IQ point
- Tone index (2 bits) selects which of pattern's 4 tones
- IQ index (6 bits) selects 64-QAM constellation point
- 32 symbols × 50ms = 1.6s pattern duration

**D. Multi-User Mixing**
- Generate N users with independent configurations
- Each starts at random time offset (chaos)
- Each has random clock drift (±50 Hz)
- Mix by summing overlapping IQ samples

**Rationale**:
- Must match actual CASCADE protocol for valid architecture test
- RS(32,20) is critical to chaos tolerance claim (37.5% erasure)
- Multi-user mixing tests actual interference, not idealized separation

**Alternatives considered**:
- Simplified non-RS signals: Would not test RS erasure tolerance
- Coordinated (non-chaos) mixing: Would not test chaos handling
- Fixed patterns: Would not test pattern reuse mechanisms

---

### 3. Shannon Efficiency Measurement

**Question**: How to accurately measure Shannon efficiency for validation?

**Decision**: Compare achieved throughput against theoretical Shannon capacity

**Formula**:
```python
# Theoretical Shannon capacity
shannon_capacity_bps = bandwidth_hz * log2(1 + 10^(snr_db/10))

# Achieved throughput
successfully_decoded_bits = sum(decoded_user.data_bits for user in users if user.correct)
transmission_duration_sec = pattern_duration_sec  # 1.6s

achieved_throughput_bps = successfully_decoded_bits / transmission_duration_sec

# Shannon efficiency
efficiency = achieved_throughput_bps / shannon_capacity_bps
```

**Rationale**:
- Matches industry standard definition
- Directly tests claims in documentation (78-85% target)
- Accounts for multi-user capacity (45 users × 218 bps should ≈ 9,805 bps total)

**Edge cases**:
- Bandwidth calculation: Use 2,500 Hz (CASCADE's full bandwidth)
- Multi-pattern users: Count all decoded bits across all patterns
- Failed decodes: Don't count toward achieved throughput (realistic)

---

### 4. Test Progression & Curriculum Learning

**Question**: In what order should tests run to efficiently validate architecture?

**Decision**: Progressive complexity from 1 user → 45 users

**Test sequence**:
1. **1 user** (1 hour): Validate encode/decode basics
2. **10 users** (4 hours): Validate pattern orthogonality
3. **20 users** (6 hours): Validate frequency reuse (critical)
4. **30 users** (8 hours): Validate time reuse
5. **45 users chaos** (12 hours): Validate full system (CRITICAL)
6. **45 users + kernels** (16 hours): Validate coordination
7. **SNR sweep** (12 hours): Validate degradation

**Rationale**:
- Each test builds on previous (fail early if basics broken)
- Critical Test 5 happens mid-sequence (not wasted if earlier tests fail)
- Kernel test (6) runs after chaos baseline (5) for comparison
- Total: 59 hours ≈ 60 hours (NFR-001)

**Stopping criteria**:
- If Test 1-2 fail: Basic architecture broken, stop immediately
- If Test 3 fails: Frequency reuse broken, invalidates capacity claims
- If Test 5 fails: Architecture cannot meet targets, stop and fix

---

### 5. Kernel Coordination Simulation

**Question**: How to test prokernel/antikernel mechanism without real network?

**Decision**: Simulate 3-round kernel exchange in training

**Round 1: Random kernels (baseline)**
```python
# Each user gets random 64-bit kernel (meaningless)
kernels = [random.randint(0, 2**64) for _ in range(45)]
# Model must decode without kernel help
# Establishes baseline performance
```

**Round 2: Prokernels (frequency guidance)**
```python
# Each user announces their available_tones
prokernels = [
    encode_kernel(
        available_tones=[0,1,2,...,77],  # Simplified: all tones available
        max_patterns=4 if user.hardware=='coral' else 2,
        hardware_tier=user.hardware
    )
    for user in users
]
# Model uses kernels to select compatible tones
# Should improve efficiency (fewer collisions)
```

**Round 3: Antikernels (interference coordination)**
```python
# Interfered users request shifts
antikernels = []
for user in users:
    if user.experiencing_interference:
        antikernel = create_antikernel(
            request_shift_hz=50,  # Shift +50 Hz
            or_reduce_power_db=3   # Or reduce 3 dB
        )
        antikernels.append(antikernel)

# Transmitters adapt based on antikernels
# Should further improve efficiency (emergent disjoint allocation)
```

**Expected progression**:
- Round 1: 85% Shannon (baseline, no kernels)
- Round 2: 86% Shannon (+1% from frequency guidance)
- Round 3: 87% Shannon (+2% from antikernel coordination)

**Validation target**: Round 3 must show ≥2% improvement over Round 1 (FR-011)

**Rationale**:
- Tests claimed kernel coordination benefit (2-5% improvement)
- Simpler than full network simulation
- Focuses on mechanism, not implementation details

---

### 6. Pattern Representation for Vetting

**Question**: Do we need actual 128 patterns or can vetting use simplified versions?

**Decision**: Use simplified orthogonal patterns for vetting, don't require full pattern generation

**Simplified approach**:
- Generate 128 pseudo-random tone sequences that are approximately orthogonal
- Use simple IQ trajectories (circles, not full Lissajous)
- Focus: Test that model CAN separate patterns, not that THESE EXACT patterns work

**Rationale**:
- Full pattern generation takes 18-24 hours (expensive prerequisite)
- Vetting purpose: Validate separation mechanism, not specific pattern quality
- Can use Zadoff-Chu sequences directly (good enough orthogonality)
- If vetting passes with simplified patterns, real patterns will work better

**Alternative rejected**:
- Require full 128-pattern generation first: Delays vetting, not necessary for architecture validation

**Note**: If vetting passes and project proceeds, THEN generate real patterns before production training

---

### 7. Model Architecture for Vetting

**Question**: Does vetting need full expert network implementation?

**Decision**: Use simplified model architecture for vetting (proves concept, not production quality)

**Minimum viable model**:
- Pattern correlation (Time × Freq × IQ matching)
- RS decoder (symbol-level, can handle erasures)
- Multi-user separation (successive cancellation or parallel decode)
- Basic kernel interpretation (read available_tones, adjust tone selection)

**Full expert network NOT required for vetting**:
- Noise expert: Not needed (AWGN is simple)
- Propagation expert: Not needed (no multipath/fading)
- Simplified signal expert: Sufficient for pattern separation

**Rationale**:
- Vetting tests architecture viability, not implementation quality
- Full expert network is implementation detail
- Can validate chaos, reuse, coordination with simpler model
- Saves development time (weeks of expert network tuning not needed for vetting)

**Implication**: If vetting passes with simple model, full expert network will improve results

---

## Research Findings Summary

### Technical Decisions
1. **AWGN simulation**: NumPy random.normal (standard, reproducible)
2. **Signal generation**: RS(32,20) encoder + 4D mapping (matches CASCADE spec)
3. **Shannon measurement**: Achieved throughput vs theoretical capacity
4. **Test progression**: Curriculum learning 1→10→20→30→45 users
5. **Kernel simulation**: 3-round exchange (random → prokernel → antikernel)
6. **Patterns**: Simplified orthogonal sequences (Zadoff-Chu based)
7. **Model**: Minimum viable for separation testing (not full expert network)

### No Remaining Unknowns
All technical approaches defined. No NEEDS CLARIFICATION markers remain.

### Dependencies Identified
- PyTorch (model training framework)
- NumPy/SciPy (signal processing, AWGN)
- Reed-Solomon library (reedsolo or galois)
- Pattern specification (simplified Zadoff-Chu generation)

### Ready for Phase 1
All research complete. Can proceed to data model and contracts.
