# CASCADE Dual-Layer Architecture - Documentation Update Specifications

**Date**: 2025-10-04
**Purpose**: Complete guide for updating all CASCADE documentation to reflect dual-layer architecture
**Status**: Specification for remaining 25 files (5 critical files already completed)

---

## ✅ Already Completed (5 files)

1. **CLAUDE.md** - Project context updated with dual-layer summary
2. **docs/architecture.md** - Core parameters, kernel coordination, hardware platform
3. **docs/model/kernel_architecture.md** - NEW comprehensive 28-byte kernel specification
4. **docs/model/pattern_architecture.md** - Dual-layer section added at top
5. **docs/model/tfiq_dimensions.md** - Dual-layer overview added

---

## Remaining Files - Detailed Change Specifications

### A. Feature Specifications (specs/003-pattern-generation/) - 4 files

#### 1. specs/003-pattern-generation/research.md

**Current state**: Has 2-FSK analysis, needs dual-layer additions

**Changes needed**:

**ADD new section after "6. Execution Mode Selection"**:
```markdown
### 9. Dual-Layer Architecture Decision (FINAL - 2025-10-04)

**Decision**: Separate pattern orthogonality (Time×Freq) from data encoding (IQ)

**Why Time×Frequency-only orthogonality**:
2-FSK with 150-tone grid:
- 75 frequency-orthogonal patterns (different tone pairs, λ=0)
- 53 time-orthogonal patterns (same tones, different hopping sequences, λ=0)
- ALL 128 patterns achievable with λ=0

IQ dimension completely freed for data encoding!

**Dual-layer benefits**:
1. All patterns λ=0 (vs 59% mixed): Universal -10 dB SNR threshold
2. IQ for data (not separation): Adaptive BPSK→16-QAM modulation
3. 2-5× throughput: 15-39 bits vs 7 bits per pattern
4. No GPS required: Differential encoding immune to drift
5. Simpler generation: Frequency-only optimization (2D vs 3D)

**Comparison**:
| Approach | All λ=0? | Throughput/Pattern | IQ Purpose |
|----------|----------|-------------------|------------|
| IQ-based orthogonality | No (59%) | 7 bits | Pattern separation |
| Dual-layer | Yes (100%) | 15-39 bits | Data encoding |

**Reference**: Dual-layer architecture analysis and throughput optimization (2025-10-04)
```

**UPDATE existing Section 7 "FSK Order Selection"**:
- Change expected outcomes: "All λ=0" instead of "0.08-0.10 average"
- Update throughput: 94 bps (1 pattern, BPSK data) vs 44 bps

#### 2. specs/003-pattern-generation/plan.md

**Changes needed**:

**UPDATE "Timeline" section**:
```markdown
**Timeline**:
- Development: 18-22 hours (simplified: frequency-only optimization, all λ=0)
- Testing: 2-3 hours (simpler validation)
- **Production generation (optimal)**: 48-60 hours local (8 trials × 200K iterations)
  - Expected: -37.5 to -40 dB separation (Time×Freq), **all λ=0**
  - Simpler: Frequency-only optimization (no IQ search)
  - Best for: Core Ultra 7 265K, Ryzen 9, high-end CPUs
  - Pattern characteristics: All BPSK baseline, optimized tone revisits
```

**UPDATE tasks list**:
- Remove: IQ trajectory generation, IQ mutation, phase-aware optimization
- Add: Tone revisit validation, differential encoding specification

#### 3. specs/003-pattern-generation/tasks.md

**Changes needed**:

**ADD note at top**:
```markdown
**ARCHITECTURE UPDATE (2025-10-04)**: Dual-layer architecture implemented
- All patterns λ=0 (Time×Frequency orthogonality only)
- IQ dimension reserved for adaptive data encoding (BPSK→16-QAM)
- Simplified optimization: Frequency-only (no IQ search)
- Some tasks below reference IQ optimization (legacy, not implemented in final version)
```

**UPDATE Phase 3.3 tasks**:
- Mark T011 (IQ generation), T013 (IQ mutation) as "NOT NEEDED (all λ=0)"
- Update T016-T017 outcomes: "All λ=0, optimized tone revisits"

#### 4. specs/003-pattern-generation/data-model.md (if exists)

**Check if file exists**, if so:

**UPDATE Pattern entity**:
```markdown
Pattern:
- pattern_id: int (0-127)
- freq_sequence: uint8[32] (tone indices in {0, 1} for 2-FSK)
- iq_trajectory: complex64[32] (all values 1+0j for λ=0 baseline)
- tone_0_visit_count: int (for data capacity, should be 14-18)
- tone_1_visit_count: int (for data capacity, should be 14-18)
- iq_complexity_lambda: float (always 0.0)
- version: uint8 (pattern format version)
```

### B. Protocol Documentation (docs/protocol/) - 8 files

#### 5. docs/protocol/signal_specification.md

**MAJOR UPDATES throughout**:

**Line ~35**: UPDATE sequence structure
```markdown
OLD: 'sequence': [int] * 32,      # 32 symbols, each 0-3 (4-tone selection from 78-tone grid)
NEW: 'sequence': [int] * 32,      # 32 symbols, each 0-1 (2-FSK, 2 adjacent tones)
```

**Line ~122**: UPDATE tone grid
```markdown
OLD: **Reference tone grid**: 78 discrete tones spanning 300-2764 Hz
NEW: **Reference tone grid**: 150 discrete tones spanning 300-3300 Hz, 20 Hz spacing (full 3 kHz SSB)
```

**Line ~144**: UPDATE pattern tone usage
```markdown
OLD: - Each of 128 patterns selects 4 base tones from 78-tone grid
NEW: - Each of 128 patterns allocated 2 adjacent tones from 150-tone grid (2-FSK)
     - Pattern hopping: Selects 1 of 2 tones per symbol (frequency hopping within 64 Hz band)
     - All patterns λ=0 (BPSK baseline for pattern skeleton)
```

**ADD new section**: "Dual-Layer Signal Structure"
```markdown
## Dual-Layer Signal Structure

**Layer 1 - Pattern Skeleton** (Time×Frequency):
- Frequency hopping pattern (tone indices 0 or 1 per symbol)
- BPSK baseline (all IQ = 1+0j for pattern structure)
- Recognition: NN identifies pattern from hopping sequence
- Information: 7 bits (which of 128 patterns)

**Layer 2 - Data Overlay** (Adaptive IQ):
- Modulation: BPSK/QPSK/8-PSK/16-QAM (kernel parameter)
- Encoding: Differential phase/amplitude changes
- Redundancy: Tone revisits (4× typical)
- Information: 8-32 bits (SNR-dependent)

**Signal generation**:
1. Protocol: Creates baseline signal (pattern freq_sequence + modulation constellation)
2. Encoder NN: Mutates signal (continuous optimizations) - messages only
3. Beacons: Skip step 2 (protocol-only for fast decode)
```

**UPDATE all "4-tone" or "4-FSK" references** to "2-tone" or "2-FSK"

#### 6. docs/protocol/adaptive_4fsk.md

**REMOVED**: File deleted as 4-FSK architecture is obsolete in 2-FSK/150-tone system.
All patterns now use 2-FSK with adaptive IQ modulation for data payload.

#### 7. docs/protocol/beacons.md

**ADD beacon structure section**:
```markdown
## Beacon Structure (100 bytes)

**3 Kernel Candidates** (84 bytes):
- Each kernel: 28 bytes (17-bit discrete + 8-bit version + 24-byte embedding)
- Purpose: Pro-kernels for others to reach you
- Also: Anti-kernels (your current RX state for others to avoid)

**Metadata** (16 bytes):
- Callsign: 6 bytes (encoded)
- Grid square: 3 bytes (4-char Maidenhead)
- Net ID: 2 bytes (which network, if member)
- Relay status: 1 byte (relay flag + relay count)
- Timestamp: 2 bytes (relative time)
- Checksum: 2 bytes (CRC16)

**Transmission**:
- Duration: 5-8 seconds @ adaptive modulation
  - BPSK: 8.5s (100 bytes × 8 bits / 94 bps)
  - QPSK: 5.6s (100 bytes × 8 bits / 144 bps)
- Interval: Every 60 seconds
- Overhead: 14% @ BPSK, 9% @ QPSK (acceptable)

**Signal generation**:
Beacons use protocol-only (no encoder NN mutations)
Why: Faster decode (critical for coordination timeliness)
     All stations can quickly extract kernels without reversing mutations
```

**UPDATE beacon content section**:
- Add kernel structure details
- Add pro/anti-kernel explanation
- Add top-3 candidate rationale

#### 8. docs/protocol/kernel_lifecycle.md

**MAJOR EXPANSION** (this file is critical):

**ADD complete lifecycle flow**:
```markdown
## Complete Kernel Lifecycle

### Phase 1: Kernel Generation (Decoder, Background)

**Continuous monitoring**:
Your decoder monitors own channel:
- SNR measurements
- Multipath profile
- Interference spectrum
- Tone quality per frequency

**Decoder generates 3 kernel candidates**:
Purpose: Optimal parameters for OTHERS to reach YOU
Output: 3 × 28-byte kernels (discrete + embedding + score)

Example:
Candidate #1: {pattern: 42, freq: 23, mod: QPSK, emb: [...], score: 0.85}
Candidate #2: {pattern: 67, freq: 45, mod: 8PSK, emb: [...], score: 0.78}
Candidate #3: {pattern: 91, freq: 12, mod: BPSK, emb: [...], score: 0.71}

Diversity: Different pattern/frequency/modulation combinations
Flexibility: Others can choose based on their constraints

### Phase 2: Beacon Transmission (Every 60s)

**Beacon content**:
- 3 kernel candidates (your pro-kernels)
- Metadata (call, grid, net, relay status)
- Total: 100 bytes

**Transmission**:
Generated by PROTOCOL only (not encoder NN)
Why: Simplifies decode, enables fast kernel extraction
Duration: 5-8 seconds @ adaptive modulation

**Beacon serves dual purpose**:
- Pro-kernel: For stations wanting to reach you
- Anti-kernel: Your current RX state (others avoid interfering)

### Phase 3: Beacon Reception (All Stations)

**Continuous beacon monitoring**:
Process: 180 beacons per minute (45 stations × 4 bands)
Decoder batch: 50-100 ms for all beacons (on TPU)
Extract: 540 kernels total (45 × 3 × 4)

**Categorization**:
If targeting Station A:
  - A's 3 kernels → pro-kernels (choose from these)

All other 44 stations:
  - Current kernel → anti-kernels (avoid these)

### Phase 4: TX Kernel Coordination (When Transmitting)

**Protocol selection** (discrete params):
```python
# Choose from target's 3 candidates
for candidate in target.pro_kernels:
    # Check collision with anti-kernels
    collisions = check_collisions(candidate, anti_kernels)
    if no_collisions:
        selected_discrete = candidate.discrete
        base_embedding = candidate.embedding
        break

# If all collide, pick least-bad
if not selected:
    selected = target.pro_kernels[0]  # Best for target, accept collision
```

**Embedding combination** (continuous):
```python
# Weighted combination
combined_embedding = (
    0.70 * base_embedding  # Toward target
    - sum([weight(distance, signal) * ak.embedding for ak in anti_kernels])
)

# Weights: Nearby stations matter more (distance-based)
```

**Final kernel**:
- Discrete: From target's preferred options (coordinated)
- Embedding: Weighted to avoid interference
- Result: Optimized for delivery + minimal network impact

### Phase 5: Signal Generation and Transmission

**Protocol baseline** (discrete params):
Generate standard IQ signal:
- Pattern's frequency sequence
- Modulation constellation (QPSK, 8-PSK, etc.)
- Standard rotation/scaling

**Encoder mutations** (continuous embedding):
Messages only! Beacons skip this.

Apply learned optimizations:
- Per-symbol frequency shifts (±0.1-2 Hz fine tuning)
- Constellation rotation/scaling (continuous adjustments)
- Power shaping (per-symbol power control)
- Pre-equalization (channel compensation)

**RF transmission**: Mutated signal (optimized beyond discrete granularity)

### Phase 6: Reception and Decode

**Decoder processing**:
1. Batch process: All active patterns (20-60 simultaneously)
2. Use kernel parameters:
   - Discrete: Which pattern, frequency, modulation to look for
   - Embedding: What encoder mutations to expect/reverse
3. Separate patterns: Via Time×Freq orthogonality + constellation differences
4. Demodulate data: Differential decoding per pattern
5. Output: Decoded messages

**Beacon decode** (special case):
Fast processing (no encoder mutations to reverse)
Extract: 3 kernel candidates + metadata
Update: Pro/anti-kernel databases for next transmission

### Phase 7: Loop

Decoder continuously:
- Monitors channel → generates new kernels
- Processes beacons → updates pro/anti-kernel state
- Decodes messages → delivers user data

Every 60s: Beacon with updated kernels
When TX: Use latest pro/anti-kernels for coordination
```

#### 9. docs/protocol/adaptive_tone_grid.md

**UPDATE to 150-tone grid**:

**REPLACE section on "78-tone grid"** with:
```markdown
## 150-Tone Reference Grid

**Full 3 kHz SSB channel utilization**:
```
Tones: 150 (0-149)
Spacing: 20 Hz
Range: 300-3300 Hz
Total BW: 3,000 Hz (fills standard SSB filter)

Tone frequencies:
Tone 0: 300 Hz
Tone 1: 320 Hz
...
Tone 149: 3280 Hz
```

**Cognitive sharing** (legacy vs SDR):
Legacy radios (~40 Hz resolution):
- Perceive: ~75 "wide tones" (2 adjacent 20 Hz tones seen as one 40 Hz tone)
- Allocation: Occupies 2 tone positions per pattern
- Example: Pattern on "legacy tone 10" = actual tones 20-21

Modern SDRs (20 Hz resolution):
- Perceive: Full 150-tone grid
- Allocation: Precise tone pair assignment
- Can fit in gaps: Between legacy allocations

Kernel coordination:
- Detects: Legacy vs SDR equipment (from beacon characteristics)
- Assigns: Appropriate tone spacing
- Cognitive: SDR fills gaps around legacy users
```

**UPDATE micro-tuning section**:
Add: Micro-tuning as kernel parameter (not fixed offset)
Explain: Each kernel includes ±2 Hz offset for interference avoidance

#### 10. docs/protocol/link_adaptation.md

**ADD modulation adaptation section**:
```markdown
## Adaptive Modulation Selection

**SNR-based kernel parameter**:

Kernel includes modulation order (3 bits):
- Decoder measures: Channel SNR
- Generates kernels: With appropriate modulation for measured SNR
- Protocol selects: From target's beaconed options
- Encoder uses: To generate data constellation

**SNR thresholds**:
| SNR Range | Modulation | Bits/Symbol | Data/Pattern | Threshold Logic |
|-----------|------------|-------------|--------------|-----------------|
| < 0 dB | BPSK | 1 | 8 bits | Maximum robustness |
| 0-10 dB | QPSK | 2 | 16 bits | Typical HF DX |
| 10-20 dB | 8-PSK | 3 | 24 bits | Good propagation |
| > 20 dB | 16-QAM | 4 | 32 bits | Excellent (local/NVIS) |

**Adaptive selection**:
Decoder continuously measures SNR
Regenerates kernels when SNR changes significantly (>3 dB)
Next beacon: Updated kernels with new modulation recommendation

Network adapts: Collective modulation rises/falls with propagation
```

#### 11. docs/protocol/chaos_transmission.md

**UPDATE capacity calculations**:

**REPLACE throughput numbers**:
```markdown
OLD: 7 bits per pattern = 43.75 bps @ 200 sym/s
NEW: 15-39 bits per pattern = 94-244 bps @ 200 sym/s (adaptive)
```

**UPDATE multi-user scenarios**:
```markdown
20 patterns simultaneously:
OLD: 20 × 7 bits / 0.16s = 875 bps aggregate
NEW: 20 × avg 25 bits / 0.16s = 3,125 bps aggregate (mixed modulation)
```

#### 12. docs/protocol/message_format.md

**ADD dual-layer encoding section**:
```markdown
## Message Encoding (Dual-Layer)

**Layer 1** (pattern selection, 7 bits):
User selects: Which of 128 patterns to transmit
Encoded in: Frequency hopping sequence (Time×Frequency)

**Layer 2** (data payload, 8-32 bits):
User data: Encoded via differential modulation
Encoded in: IQ phase/amplitude changes on tone revisits
Capacity: Varies with kernel modulation parameter

**Total message**:
Pattern + data = 15-39 bits per pattern transmission
Multi-pattern: Can send 4-8 patterns for higher throughput
```

#### 13. docs/protocol/micro_tuning.md

**UPDATE**: Micro-tuning as kernel parameter

**ADD**:
```markdown
## Micro-Tuning in Kernel

**Kernel parameter**: Frequency offset (±0.1-2 Hz)

Instead of fixed ±2 Hz:
- Kernel specifies: Exact offset for this transmission
- Range: -2.0 to +2.0 Hz (8-16 discrete steps)
- Purpose: Cognitive interference avoidance

**Coordination via anti-kernels**:
Station A beacon: Using +1.5 Hz offset (anti-kernel)
Station B transmit: Chooses -1.0 Hz (avoid A)
Separation: 2.5 Hz (enables sub-tone multiplexing)

**Capacity gain**:
Without micro-tuning: 150 tone positions
With micro-tuning: ~750 effective positions (5× per tone)
Result: 20-30 patterns → 40-60 patterns per channel
```

#### 14. docs/protocol/priority_handling.md

**ADD kernel selection for priority**:
```markdown
## Priority-Based Kernel Selection

**Emergency transmission**:
Protocol selects: Target's pro-kernel candidate #1 (best for delivery)
Ignores: Anti-kernels (accept interference, maximize reach)
Embedding weight: α = 0.95 (heavily toward target, minimal coordination)

**Routine transmission**:
Protocol selects: Best network fit from 3 candidates (minimize interference)
Respects: All anti-kernels (cooperative)
Embedding weight: α = 0.70 (balanced target + coordination)

**Adaptive**: Priority level affects kernel selection strategy
```

### C. Implementation Specifications (docs/implementation/) - 3 files

#### 15. docs/implementation/pattern_generation_spec.md (714 lines - MAJOR REWRITE)

**THIS IS THE LARGEST UPDATE**

**REPLACE entire algorithm section** (~lines 1-400):

```markdown
# Pattern Generation: Time×Frequency Orthogonality (All λ=0)

**Purpose**: Generate 128 patterns orthogonal in Time×Frequency space ONLY
**Status**: Simplified from 4D (Time×Freq×IQ) to 2D (Time×Freq)
**Runtime**: 48-60 hours (8 trials × 200K iterations, faster than IQ-based)

---

## Overview

CASCADE patterns are **frequency-hopping sequences** (2-FSK) with ALL patterns using λ=0 (BPSK baseline).

**Key simplification**:
- OLD: Optimize Time × Freq × IQ (3D search space, find minimum λ)
- NEW: Optimize Time × Freq only (2D search, force λ=0)

**Why this works**:
2-FSK provides sufficient orthogonality in Time×Frequency space:
- 75 patterns: Frequency-orthogonal (different tone pairs)
- 53 patterns: Time-orthogonal (same tone pairs, different hopping)
- IQ not needed: Reserved for data encoding (Layer 2)

**Result**: Simpler, faster generation with better properties (all λ=0)

---

## Algorithm Overview

**Input**:
- count: 128 patterns
- seed: Random seed for reproducibility

**Optimization**:
1. Zadoff-Chu initialization (31 patterns)
2. Random initialization (97 patterns)
3. Simulated annealing (frequency-only, λ=0 forced)
4. Tone revisit optimization (for data capacity)
5. Validation (Time×Freq correlation only)

**Output**:
- 128 patterns, all λ=0
- -37.5 to -40 dB separation (Time×Freq)
- Optimized tone revisit distribution
- 38 KB binary file

---

## Pattern Structure

```python
Pattern:
    pattern_id: int (0-127)
    freq_sequence: uint8[32]  # Tone indices in {0, 1} for 2-FSK
    iq_trajectory: complex64[32]  # All values 1+0j (λ=0, BPSK baseline)
    tone_0_visits: int  # Count (should be 14-18 for good data capacity)
    tone_1_visits: int  # Count (should be 14-18)
    version: uint8  # Pattern format version
```

---

## Step 1: Zadoff-Chu Initialization

```python
def generate_zadoff_chu_2fsk(u, N=31):
    """
    Generate Zadoff-Chu sequence for 2-FSK

    Maps phase (0-2π) to tone indices (0-1)
    """
    sequence = []
    for n in range(31):
        phase = 2 * pi * u * n * (n + 1) / (2 * N)
        # Map to 2 regions (not 4)
        tone_idx = 0 if phase < pi else 1
        sequence.append(tone_idx)

    # Pad to 32 symbols
    sequence.append(0)

    return np.array(sequence, dtype=np.uint8)

# Generate first 31 patterns
base_patterns = [generate_zadoff_chu_2fsk(u) for u in range(31)]
```

Good starting orthogonality (~-15 dB natural Z-C separation)

---

## Step 2: Random Initialization

```python
# Remaining 97 patterns: random
for i in range(31, 128):
    pattern = np.random.randint(0, 2, size=32, dtype=np.uint8)
    base_patterns.append(pattern)
```

---

## Step 3: Frequency-Only Optimization

**Simulated annealing** (simplified from IQ-based):

```python
def optimize_pattern_frequency_only(pattern_id, base_sequence, existing_patterns):
    """
    Optimize frequency sequence for Time×Freq orthogonality

    All λ=0 (no IQ search needed!)
    """
    current_freq = base_sequence.copy()
    current_iq = np.ones(32, dtype='complex64')  # Force λ=0 (BPSK)

    # SA parameters
    temperature = 10.0
    cooling_rate = 0.99998
    iterations = 200000  # Simpler problem, fewer iterations needed

    for iter in range(iterations):
        # Mutate frequency only
        new_freq = current_freq.copy()
        idx = random.randint(0, 31)
        new_freq[idx] = random.randint(0, 1)  # 2-FSK: 0 or 1

        # Evaluate (frequency correlation only)
        cost = compute_time_freq_correlation(new_freq, existing_patterns)

        # Add tone revisit quality
        revisit_quality = evaluate_revisit_distribution(new_freq)
        cost += 0.1 * revisit_quality  # Prefer balanced, distributed revisits

        # Accept/reject (standard SA)
        if accept(cost, temperature):
            current_freq = new_freq

        temperature *= cooling_rate

    return current_freq, np.ones(32, dtype='complex64'), 0.0  # freq, iq(λ=0), λ
```

**No IQ optimization** → Simpler, faster convergence

---

## Step 4: Tone Revisit Optimization

**Ensure good data encoding capacity**:

```python
def evaluate_revisit_distribution(freq_sequence):
    """
    Score quality of tone revisit pattern

    Good: Even distribution, balanced counts
    Bad: Clustered visits, imbalanced counts
    """
    tone_0_visits = sum(1 for t in freq_sequence if t == 0)
    tone_1_visits = sum(1 for t in freq_sequence if t == 1)

    # Check minimum (need ≥8 for data encoding)
    if tone_0_visits < 8 or tone_1_visits < 8:
        penalty = 10.0  # Strong penalty
    else:
        penalty = 0.0

    # Check balance (prefer similar counts)
    imbalance = abs(tone_0_visits - tone_1_visits) / 32.0
    penalty += imbalance * 2.0  # Moderate penalty

    # Check clustering (prefer distributed)
    max_run = max_consecutive_run(freq_sequence)
    if max_run > 8:
        penalty += (max_run - 8) * 0.5

    return penalty  # Lower is better
```

**Expected outcomes**:
- Tone 0 visits: 14-18 (avg 16)
- Tone 1 visits: 14-18 (avg 16)
- Max run length: ≤6 symbols
- Good for: 4× redundancy in data layer

---

## Step 5: Validation

**Time×Frequency correlation** (simplified from 4D):

```python
def compute_time_freq_correlation(pattern_i, pattern_j):
    """
    Correlation in Time×Frequency space only

    IQ is constant (1+0j), so no IQ correlation component
    """
    matches = 0
    for t in range(32):
        if pattern_i.freq_sequence[t] == pattern_j.freq_sequence[t]:
            matches += 1  # Same tone at same time

    # Normalize
    correlation = matches / 32.0

    # Convert to dB
    correlation_db = 20 * log10(correlation + 1e-10)

    return correlation_db
```

**Validation criteria**:
- All pairs: <-37.5 dB
- All patterns: λ=0
- All patterns: Tone visits ≥8 per tone
- All patterns: Balanced visits (within 20%)

---

## Expected Outcomes

**Pattern characteristics**:
```
Count: 128
Orthogonality: -37.5 to -40 dB (Time×Freq)
λ distribution: 0.0 (all patterns, 100%)
Tone 0 visits: 14-18 avg per pattern (good for data)
Tone 1 visits: 14-18 avg per pattern
Generation time: 48-60 hours (8 trials × 200K iterations)

vs IQ-based approach:
- Simpler: 2D optimization vs 3D
- Faster: 200K iterations vs 400K
- Better: All λ=0 vs 59% λ=0
- Time: 48-60h vs 72-96h
```

**Validation metrics**:
- Min separation: -40+ dB (excellent)
- Mean separation: -45 dB (typical)
- Max separation: -50+ dB (best pairs)
- Tone revisit quality: 95%+ patterns meet criteria

---

## Implementation Notes

**Removed compared to IQ-based**:
- IQ trajectory generation (all λ=0, returns np.ones())
- IQ mutation in SA (not needed)
- Phase-aware correlation (not applicable for λ=0)
- Lambda complexity measurement (always 0)
- Two-phase optimization (single phase, frequency-only)

**Code simplification**: ~800 lines removed, faster generation

**Testing**: Simpler validation (2D correlation vs 4D)
```

**This replaces ~400 lines of the original spec with simplified algorithm**

#### 16. docs/implementation/kernel_encoding_spec.md

**COMPLETE REWRITE**:

```markdown
# Kernel Encoding Specification

**28-byte hybrid kernel structure**

---

## Kernel Structure

### Discrete Portion (Bytes 0-2, plus version byte 3)

**Byte 0-1** (14 bits used):
```
Bits 0-6: Pattern ID (0-127)
Bits 7-13: Frequency pair (0-74, maps to tone pair from 150-tone grid)
```

**Byte 2** (3 bits used):
```
Bits 0-2: Modulation order
  000 = BPSK
  001 = QPSK
  010 = 8-PSK
  011 = 16-QAM
  100-111 = Reserved for future
```

**Byte 3** (version):
```
Bits 0-3: Protocol version (v0-v15)
Bits 4-7: Model version (v0-v15)
```

### Continuous Embedding (Bytes 4-27)

**24 bytes** = 48 dimensions × 4-bit quantization

**Encoding**:
```python
# Each dimension: float32 → int4 (0-15)
def quantize_dimension(float_value):
    # Assuming value in [-3.0, +3.0] typical range
    quantized = int((float_value + 3.0) / 6.0 * 15.0)
    return clip(quantized, 0, 15)

# Pack 2 dimensions per byte
def pack_embedding(float_vector_48):
    bytes = []
    for i in range(0, 48, 2):
        dim_0 = quantize_dimension(float_vector_48[i])
        dim_1 = quantize_dimension(float_vector_48[i+1])
        byte = (dim_0 << 4) | dim_1  # Pack into single byte
        bytes.append(byte)
    return bytes  # 24 bytes
```

**Dequantization** (at receiver):
```python
def dequantize_embedding(bytes_24):
    float_vector = []
    for byte in bytes_24:
        dim_0 = (byte >> 4) & 0xF
        dim_1 = byte & 0xF
        # Map 0-15 back to -3.0 to +3.0
        float_0 = (dim_0 / 15.0) * 6.0 - 3.0
        float_1 = (dim_1 / 15.0) * 6.0 - 3.0
        float_vector.extend([float_0, float_1])
    return np.array(float_vector, dtype=np.float32)  # 48 dims
```

---

## Beacon Encoding (100 bytes)

**3 Kernel Candidates** (84 bytes):
```
kernel_1: 28 bytes
kernel_2: 28 bytes
kernel_3: 28 bytes
```

**Metadata** (16 bytes):
```
callsign: 6 bytes
grid: 3 bytes (4-char Maidenhead)
net_id: 2 bytes
relay_status: 1 byte
timestamp: 2 bytes
checksum: 2 bytes (CRC16 over all 98 bytes)
```

**Total**: 100 bytes per beacon

**Transmission**:
- Encoded using: Protocol baseline (no encoder mutations)
- Duration: 5-8 seconds @ adaptive modulation
- Interval: Every 60 seconds
```

#### 17. docs/implementation/pattern_pool_selection_spec.md

**RENAME** to: `modulation_selection_spec.md`

**REWRITE**:
```markdown
# Adaptive Modulation Selection Specification

**Replaces**: Pattern pool selection (obsolete with all λ=0)

---

## Decoder Kernel Generation

**Decoder analyzes channel**:
```python
def generate_kernel_candidates(channel_state):
    """
    Generate 3 kernel options for own reception

    Returns top-3 by decoder NN inference
    """
    snr = measure_snr()
    multipath = measure_multipath()
    interference = spectrum_analysis()

    # Decoder NN generates kernels
    candidates = decoder_NN.generate_kernels(
        channel_features=[snr, multipath, interference, ...],
        available_patterns=range(128),
        available_freqs=range(75)  # Tone pairs
    )

    # Top-3 candidates
    return sorted(candidates, key=lambda k: k.score, reverse=True)[:3]
```

**Modulation selection logic** (part of kernel generation):
```python
# Based on measured SNR
if snr < 0:
    modulation = BPSK  # Maximum robustness
elif snr < 10:
    modulation = QPSK  # Typical HF
elif snr < 20:
    modulation = 8PSK  # Good conditions
else:
    modulation = 16QAM  # Excellent

# Include in kernel discrete params
kernel.modulation = modulation
```

**Diversity in candidates**:
```
Candidate #1: Best modulation for SNR (max throughput)
Candidate #2: Conservative modulation (more robust)
Candidate #3: Different frequency (alternative if interference)

Gives others flexibility in selection
```
```

### D. Training Documentation (docs/training/) - 3 files

#### 18. docs/training/README.md

**UPDATE training requirements**:

**ADD section**: "Dual-Layer Training Requirements"
```markdown
## Dual-Layer Training Requirements

**Encoder NN training**:
- Must learn: Signal mutations guided by embedding
- Training data: Needs all modulation orders (BPSK, QPSK, 8-PSK, 16-QAM)
- Per-symbol optimizations: Frequency shifts, constellation mutations, power shaping
- Loss function: Delivery rate + interference caused + latency

**Decoder NN training** (dual role):
- Role 1 - Demodulation: Separate overlapping patterns, extract data
- Role 2 - Kernel generation: Generate optimal RX kernel candidates
- Training data: 24-36K hours HF recordings (sufficient for all λ=0 patterns)
- Multi-task: Joint training for both roles

**Simplified vs IQ-based**:
All patterns λ=0 (vs mixed λ):
- Less variation in pattern characteristics
- May need less training data
- 24K hours likely sufficient (vs 36K+ for complex IQ)
```

#### 19. docs/training/phase0_vetting.md

**UPDATE for all λ=0 patterns**:

**REPLACE pattern complexity discussion**:
```markdown
OLD: "Patterns have varying λ (0.0-0.9), train on distribution"
NEW: "All patterns λ=0 (BPSK baseline), train on adaptive modulation layer"
```

**ADD**:
```markdown
## Training with Dual-Layer Architecture

**Pattern layer** (all λ=0):
- Simple: All patterns use BPSK skeleton
- Consistent: No variation in IQ complexity
- Focus: Train on frequency hopping recognition

**Data layer** (adaptive modulation):
- Complex: Must train on BPSK, QPSK, 8-PSK, 16-QAM
- Adaptive: Learn when to use which modulation
- Differential: Train on phase-change encoding

**Training strategy**:
Synthetic signals: Apply all 4 modulations to patterns
Real channels: 24-36K hours HF recordings
Combinations: Pattern × modulation × channel → large training set

**Vetting**: Ensure NN can decode all modulation orders on all patterns
```

#### 20. docs/training/data_pipeline.md

**ADD section**: "Multi-Modulation Training Data"
```markdown
## Generating Multi-Modulation Training Data

For each HF recording:
1. Apply Pattern 0-127 (all with λ=0 skeleton)
2. For each pattern, generate 4 variants:
   - BPSK data modulation
   - QPSK data modulation
   - 8-PSK data modulation
   - 16-QAM data modulation

Result: 128 patterns × 4 modulations = 512 signal types per recording

Training: NN must learn to decode all 512 combinations
Enables: Runtime modulation adaptation
```

### E. Deployment Documentation (docs/deployment/) - 1 file

#### 21. docs/deployment/hardware_requirements.md

**COMPLETE REWRITE**:

```markdown
# CASCADE Hardware Requirements

**UPDATED 2025-10-04**: Dual-layer architecture with kernel coordination

---

## Recommended Platform

**Raspberry Pi 4 + Google Coral Edge TPU**

### Components

**Computing** ($110 total):
- Raspberry Pi 4 (4GB RAM): $50
- Google Coral USB Accelerator: $60
- Power supply: Included
- Case: $10 optional

**Radio** ($180 total):
- QMX assembled: $150
- GPS module (u-blox): $30
- Antenna: $50-150 (separate)

**Total CASCADE Station**: $290 (computer + radio)
vs IC-7300: $1,400 (5× more expensive, less capable for CASCADE)

### Performance Specifications

**Encoder NN** (kernel generation + signal mutations):
- Inference time: 3-5 ms (on Coral TPU)
- Model: 2-5M parameters (INT8 quantized)
- Memory: 20-40 MB
- Function: Generate kernel, mutate baseline signal

**Decoder NN** (pattern recognition + kernel generation):
- Batch inference: 50-100 ms for 180 patterns (on Coral TPU)
- Model: 2-5M parameters (INT8 quantized)
- Memory: 20-40 MB
- Function: Decode patterns, generate kernels

**Protocol coordinator**:
- Kernel selection: <1 ms (CPU, simple logic)
- Embedding combination: <1 ms (CPU, vector math)
- Baseline signal generation: 2-5 ms (CPU, signal synthesis)

**Total TX pipeline**: 6-11 ms (comfortable)
**Total RX pipeline**: 50-100 ms per frame (real-time capable)

### Capacity

**With RPi4 + Coral TPU**:
- Beacon processing: 180 beacons/minute (background, 1-2 seconds total)
- Message decoding: 20-60 patterns per 0.16s frame (batch on TPU)
- Network: Full 45-user network capable ✓
- Limitation: Batch processing latency (~100 ms acceptable)

### Alternative Platforms

**Budget: Raspberry Pi 5 (CPU-only)** - $80
- Encoder: 20-30 ms (slower)
- Decoder: 500-800 ms batch (3-5× real-time)
- Capacity: 10-20 patterns max (limited network)
- Use case: Small nets, testing

**Premium: PC with GPU** - $600-800
- Encoder: 5-8 ms (faster)
- Decoder: 20-30 ms batch (parallel GPU processing)
- Capacity: 100+ patterns (unlimited for practical purposes)
- Use case: Net controllers, EOC, high-capacity nodes

### Radio Requirements

**Primary: QMX + GPS** ($180):
- Symbol rate: 200 sym/s capable (NN ISI tolerance)
- Frequency stability: ±0.1 Hz (GPS-disciplined, enables 20 Hz spacing)
- Bandwidth: IQ mode (15-20 kHz for multi-band monitoring)
- Output: 5W (portable), upgradeable to 50W with amp (+$150)
- Best value: 1/8 cost of IC-7300, better CASCADE support

**Alternative: IC-7300** ($1,400):
- Symbol rate: 100-150 sym/s (3 kHz filter)
- Frequency stability: ±1 Hz (TCXO, adequate for 20 Hz spacing)
- Bandwidth: 3 kHz (single band at a time)
- Output: 100W (base station)
- Works: But expensive, less optimal than QMX for CASCADE

**Legacy: Any SSB radio** ($0-600, already owned):
- Symbol rate: 40-100 sym/s (2.4 kHz filter)
- Frequency stability: ±5-10 Hz (differential encoding tolerates this)
- Degradation: ~2 dB SNR penalty from drift
- Throughput: 8.75-94 bps (1 pattern, adaptive modulation)
- Use case: Emergency fallback, existing equipment

### Power Budget

**QRP Operation** (5W, battery):
- Transmit: 1 pattern (94 bps @ BPSK)
- Range: 100-500 miles (regional)
- Battery: 6-8 hours on 100Wh (15W total system)
- Use case: Portable EMCOMM, field operations

**Base Operation** (50-100W, AC power):
- Transmit: 4-8 patterns (575-1,950 bps @ adaptive)
- Range: 500-3000 miles (continental/intercontinental)
- Power: AC mains required
- Use case: Net controllers, base stations
```

### F. Examples (docs/examples/) - 1 new file

#### 22. docs/examples/kernel_coordination_examples.md (NEW FILE)

**CREATE complete file** showing real-world coordination scenarios:

```markdown
# Kernel Coordination Examples

**Real-world scenarios demonstrating pro/anti-kernel coordination**

---

## Scenario 1: Emergency Net with Relay

**Network setup**:
- Net: EMCOMM-NET-7 (hurricane response)
- Relay: N5REL (grid EM91, 100W base station)
- Members: 12 field stations (5-25W portable)
- Other QSOs: 33 unrelated stations active on same bands

**Relay beacon** (N5REL):
```python
pro_kernels = [
    # Candidate #1: Best throughput (for good conditions)
    {pattern: 75, frequency: 40, modulation: 8PSK, embedding: [...]},

    # Candidate #2: Balanced (typical)
    {pattern: 50, frequency: 35, modulation: QPSK, embedding: [...]},

    # Candidate #3: Maximum robustness (for weak stations)
    {pattern: 20, frequency: 25, modulation: BPSK, embedding: [...]},
]

metadata = {
    callsign: "N5REL",
    grid: "EM91",
    net_id: "EMCOMM-NET-7",
    relay_status: True,
    relay_count: 12
}
```

**Member stations transmit**:

**Member A** (good signal, 25W):
```python
# Select relay candidate #2 (QPSK, good match for 25W)
selected = N5REL.pro_kernels[1]  # QPSK option

# Collect anti-kernels (other 11 net members + 33 others = 44 total)
anti_kernels = [...]  # Current RX states

# Combine embeddings
tx_embedding = 0.70 * selected.embedding - sum([0.006 * ak.embedding for ak in anti_kernels])

# Result: Optimized for relay, avoids other members
```

**Member B** (weak signal, 5W):
```python
# Select relay candidate #3 (BPSK, robust for 5W)
selected = N5REL.pro_kernels[2]  # BPSK option

# Even at 5W, reaches relay via optimized kernel
# BPSK provides maximum robustness
```

**Emergent behavior**:
- All members → relay on different patterns/frequencies (distributed via kernel selection)
- Relay receives all → forwards to destinations
- No explicit routing protocol → emerges from kernel coordination!

---

## Scenario 2: Direct QSO (Point-to-Point)

**K1ABC wants to contact W2XYZ**:

**W2XYZ beacon** (target):
```python
pro_kernels = [
    {pattern: 89, frequency: 55, modulation: QPSK, embedding: [...]},
    {pattern: 67, frequency: 42, modulation: 8PSK, embedding: [...]},
    {pattern: 45, frequency: 30, modulation: BPSK, embedding: [...]},
]
```

**K1ABC transmit kernel**:
```python
# Measured own-to-W2XYZ SNR: +8 dB (good)
# Choose: Candidate #1 (QPSK, best throughput for this SNR)

selected = W2XYZ.pro_kernels[0]

# Anti-kernels from 44 other stations
# Combine with distance weighting (nearby matter more)
tx_kernel = weighted_combination(selected, anti_kernels, distance_weights)

# Transmit: Optimized for W2XYZ, minimizes interference to others
```

**Result**: Clean QSO with minimal network impact

---

## Scenario 3: High Interference (Crowded Band)

**Situation**: 20m band crowded, 60 active stations

**Station trying to transmit to target**:

**Problem**:
- Target's pro-kernel candidate #1: Pattern 42, frequency 30
- Anti-kernel collision: 3 stations already receiving on pattern 42, freq 30
- Can't use candidate #1!

**Solution** (top-3 candidates):
```python
# Check all 3 candidates
candidate_1: pattern 42, freq 30 → 3 collisions ✗
candidate_2: pattern 67, freq 45 → 1 collision ⚠️
candidate_3: pattern 91, freq 12 → 0 collisions ✓

# Protocol selects candidate #3
# Not optimal for channel, but clean network fit
# Better: Deliverable message with no interference
# Than: Optimal but interfered message
```

**Emergent**: Network naturally load-balances across patterns/frequencies

---

## Scenario 4: Mixed Equipment (Legacy + SDR)

**Network composition**:
- 15 legacy radios (IC-718, FT-857D, ±5 Hz drift)
- 30 modern SDR (QMX, ±0.1 Hz GPS)

**Legacy station beacon**:
```python
# Decoder generates kernels optimized for ±5 Hz tolerance
pro_kernels = [
    {pattern: 20, frequency: 15, modulation: BPSK, embedding: [...]},
    # Lower modulation (BPSK/QPSK only)
    # Considers own drift in embedding
]
```

**SDR station targeting legacy**:
```python
# Selects: Legacy's BPSK candidate (robust)
# Embedding combination: Accounts for legacy drift
# Encoder mutations: Pre-compensates for expected ±5 Hz drift

Result: Clean delivery despite legacy equipment
```

**SDR-to-SDR**:
```python
# Both support 8-PSK/16-QAM
# Higher throughput when both endpoints capable
# Automatic: Via kernel modulation parameter
```

**Cognitive sharing**:
- Legacy: Occupies ~80 Hz (wider due to drift)
- SDR: Fits in gaps with micro-tuning
- Coexistence: 20-30 users per channel (mixed equipment)

---

*Comprehensive examples of CASCADE's kernel coordination in practice*
```

### G. Summary Change Specification for Remaining Files

**The following files need updates but are lower priority** (can be done incrementally):

#### Protocol Files (Remaining 3 of 8)
- `docs/protocol/message_validation.md`: Update message size (15-39 bits), add kernel validation
- `docs/protocol/net_operations.md`: Add relay coordination via kernels, top-3 selection examples
- `docs/protocol/qso_protocol.md`: Update throughput expectations (94-244 bps), add kernel exchange protocol

#### Model Files (Remaining)
- `docs/model/README.md`: Add pointer to kernel_architecture.md, update summary
- `docs/model/experts.md`: Update if it discusses patterns (likely minimal changes)
- `docs/model/shared_encoder.md`: Add encoder mutation role, update for dual-layer

#### Examples (Remaining)
- `docs/examples/multi_user_scenarios.md`: Update all throughput calculations (2-5× improvement)
- `docs/examples/contest_ui.md`: Update capacity estimates
- `docs/examples/continuous_coexistence.md`: Add cognitive sharing examples

#### Other
- `docs/diagrams_needed.md`: Add dual-layer architecture diagram, kernel coordination flow
- `docs/interface/augmented_inference.md`: Update for dual-role decoder (decode + kernel gen)
- `docs/privacy.md`: No changes needed (orthogonal to architecture)

---

## Change Pattern Template

For each file above, apply this pattern:

**1. Add dual-layer architecture note at top**:
```
**UPDATED 2025-10-04**: CASCADE uses dual-layer encoding
- Layer 1: Pattern ID (Time×Freq, all λ=0) - 7 bits
- Layer 2: Adaptive data (BPSK→16-QAM) - 8-32 bits
See: [Kernel Architecture](../model/kernel_architecture.md)
```

**2. Update all throughput numbers**:
- OLD: 7 bits, 44 bps
- NEW: 15-39 bits, 94-244 bps (adaptive)

**3. Update all λ references**:
- OLD: "Mixed λ (0.0-0.9)" or "λ avg 0.08-0.10"
- NEW: "All λ=0 (BPSK baseline)"

**4. Update all tone grid references**:
- OLD: "78 tones @ 32 Hz" or "4 tones from 78"
- NEW: "150 tones @ 20 Hz" and "2-FSK (tone indices 0-1)"

**5. Add kernel coordination context** (where relevant):
- Pro/anti-kernel mechanism
- Top-3 candidates
- Distributed coordination (78-85% efficiency)

---

## Priority Order for Remaining Updates

**High Priority** (affects implementation):
1. docs/implementation/pattern_generation_spec.md (rewrite completed above)
2. docs/protocol/signal_specification.md (detailed changes above)
3. docs/protocol/beacons.md (detailed changes above)

**Medium Priority** (affects understanding):
4. docs/protocol/kernel_lifecycle.md (detailed rewrite above)
5. docs/protocol/adaptive_modulation.md (new file, detailed above)
6. docs/training/* files (training requirements)

**Low Priority** (reference/examples):
7. docs/examples/* (update throughput numbers)
8. docs/protocol/net_operations.md, qso_protocol.md (minor updates)
9. docs/interface/*, docs/model/README.md (pointers and summaries)

---

*Complete change specification for CASCADE dual-layer architecture documentation*
*Use this as systematic guide for updating remaining 25 files*
