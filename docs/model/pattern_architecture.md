# CASCADE 4D Pattern Architecture

CASCADE uses a two-tier pattern system with 256 total patterns (perfect 8-bit encoding): 64 beacon patterns optimized for 4-FSK coordination, and 192 message patterns optimized for 70-tone traffic. Both are 4-dimensional trajectories through Time × Discrete Frequency × Continuous I × Continuous Q space. This comprehensive document covers pattern generation, 4D mathematics, discrete frequency-hopping (FHSS), continuous IQ modulation, and multi-pattern transmission.

## Table of Contents

1. [Overview](#overview)
2. [Pattern Design Principles](#pattern-design-principles)
3. [4D Mathematical Foundation](#4d-mathematical-foundation)
4. [Pattern Generation Algorithm](#pattern-generation-algorithm)
5. [Discrete Frequency Hopping](#discrete-frequency-hopping)
6. [Continuous IQ Trajectories](#continuous-iq-trajectories)
7. [Model-Driven Tone Shifting](#model-driven-tone-shifting)
8. [Multi-Pattern Transmission](#multi-pattern-transmission)
9. [Per-Receiver Tone Adaptation](#per-receiver-tone-adaptation)
10. [Pattern Storage Format](#pattern-storage-format)
11. [Implementation & Training](#implementation--training)

---

## Overview

CASCADE uses a **two-tier pattern architecture:**

**Tier 1: Beacon Patterns (64 patterns, IDs 0-63)**
- **Tones:** 4-FSK only [1490, 1520, 1580, 1610] Hz
- **IQ hierarchy:**
  - 0-15: Minimal (BPSK line, emergency)
  - 16-63: Simple (circles/ellipses, normal beacons)
- **Purpose:** Kernel exchange, emergency negotiation, channel probing
- **Storage per pattern:** 288 bytes (freq + single IQ trajectory)
- **Total storage:** 18 KB (64 × 288), **Generation:** 6-8 hours

**Tier 2: Message Patterns (192 patterns, IDs 64-255)**
- **Tones:** 70 discrete message tones (35 lower + 35 upper)
- **IQ hierarchy (baked-in, HF-realistic pools):**
  - 64-79: Minimal (BPSK, emergency/disturbed, 16 patterns)
  - 80-207: Simple-Moderate (circles/ellipses, typical HF DX, **128 patterns - LARGEST POOL**)
  - 208-239: Moderate-Complex (ellipses, good single-hop, 32 patterns)
  - 240-255: Complex (Lissajous, NVIS exceptional, 16 patterns - rarely used)
- **Purpose:** All message traffic, organized by propagation conditions
- **Storage per pattern:** 292 bytes (freq + single IQ trajectory + metadata)
- **Total storage:** 56 KB (192 × 292), **Generation:** 30-40 hours

**Total System: 74 KB storage** (18 KB + 56 KB), 47% savings vs dynamic collapse, perfect 8-bit encoding

**Why hierarchical?** Matches HF reality (most operation at λ=0.3-0.5), eliminates redundant storage (no need for both full and collapsed trajectories), pattern ID indicates complexity (lower = simpler/robust, higher = complex/rare), better anti-kernel resilience, beacons probe channel to inform message pattern pool selection.

**Key distinction from LoRa CSS:** CASCADE uses discrete frequency hops (not continuous chirps), making it fundamentally frequency-hopping spread spectrum (FHSS) like Bluetooth, not chirp spread spectrum (CSS) like LoRa.

---

## Pattern Design Principles

### Common to Both Tiers
1. **4D Orthogonality**: <-30 dB cross-correlation across Time × Freq × I × Q within each tier
2. **Discrete Frequency**: Hops among exact reference tones (no interpolation)
3. **Hierarchical IQ**: Pattern ID indicates baked-in IQ complexity (lower = simpler)
4. **Multi-Pattern Capable**: 1-4 patterns simultaneously (kernel-driven)
5. **FHSS (not CSS)**: Patent-safe discrete frequency hopping
6. **Single IQ trajectory**: Each pattern stores one IQ trajectory (not two)

### Beacon Patterns (0-63) - Complexity Hierarchy
- **4-tone alphabet**: Optimized for [1490, 1520, 1580, 1610] Hz only
- **IQ organization:**
  - Patterns 0-15: BPSK line (emergency, maximum robustness)
  - Patterns 16-63: Simple circles/ellipses (normal beacons, anti-kernel pool)
- **Channel probing**: Measurements inform message pattern pool selection
- **Storage:** 288 bytes per pattern (no redundant IQ)

### Message Patterns (64-255) - Propagation-Matched Pools
- **70-tone alphabet**: Optimized for full message tone grid
- **IQ pools (HF-realistic, baked-in):**
  - **64-79** (16 patterns): BPSK/minimal, emergency/disturbed
  - **80-207** (128 patterns): Simple/moderate IQ (λ=0.3-0.5), **TYPICAL HF DX - LARGEST POOL**
  - **208-239** (32 patterns): Moderate/complex IQ (λ=0.5-0.7), good single-hop
  - **240-255** (16 patterns): Complex Lissajous (λ=0.7-0.9), NVIS exceptional (rarely used)
- **Pattern selection**: Model picks from pool matching measured multipath (most use 80-207)
- **Storage:** 292 bytes per pattern (47% savings vs dynamic)
- **High capacity**: 176 normal patterns (80-255) support 280+ users

---

## 4D Mathematical Foundation

### Trajectory Definition

Each pattern P(t, λ) is a parametric function through 4D space:

```python
class Pattern4D:
    """
    Pattern as 4D trajectory

    Dimensions:
      1. Time: t ∈ [0, 31] (32 discrete symbols)
      2. Frequency: Discrete tone index ∈ {0, 1, ..., 69}
      3. I: Continuous ∈ [-1.5, +1.5]
      4. Q: Continuous ∈ [-1.5, +1.5]

    Parameters:
      λ (complexity): [0.0, 1.0]
        1.0 = complex trajectory (high SNR)
        0.0 = simple trajectory (low SNR)
    """

    def trajectory(self, t, complexity_lambda):
        # Dimension 1: TIME (implicit via parameter t)

        # Dimension 2: FREQUENCY (discrete hop)
        tone_idx = self.freq_sequence[t]  # Integer 0-69
        freq_hz = REFERENCE_TONES[tone_idx]  # Exact discrete frequency

        # Dimensions 3 & 4: I and Q (continuous)
        iq = self.iq_path(t, complexity_lambda)

        return {
            'tone_index': tone_idx,  # Discrete
            'frequency_hz': freq_hz,  # Discrete
            'iq_basis': iq,  # Continuous complex
        }
```

### 4D Orthogonality

Patterns must be <-30 dB orthogonal across ALL dimensions:

```python
def pattern_correlation_4d(pattern_i, pattern_j, complexity_lambda):
    """
    Compute correlation in 4D space
    Must be < -30 dB for all λ ∈ [0, 1]
    """
    correlation = 0

    for t in range(32):  # Dimension 1: TIME
        # Dimension 2: FREQUENCY (discrete)
        tone_i = pattern_i.freq_sequence[t]
        tone_j = pattern_j.freq_sequence[t]

        # Different discrete tones → orthogonal in frequency
        if tone_i != tone_j:
            continue

        # Same tone → check IQ orthogonality
        # Dimensions 3 & 4: I and Q
        iq_i = complex(pattern_i.i_traj[t], pattern_i.q_traj[t])
        iq_j = complex(pattern_j.i_traj[t], pattern_j.q_traj[t])

        # Inner product in IQ space
        correlation += abs(iq_i * iq_j.conjugate())

    # Normalize and convert to dB
    normalized = correlation / 32
    return 20 * np.log10(normalized + 1e-10)

# Requirement: correlation_4d(i, j, λ) < -30 dB
# For all i ≠ j, all λ ∈ [0, 1]
```

---

## Pattern Generation Algorithm

CASCADE generates two separate pattern sets optimized for their respective tone grids.

### Beacon Pattern Generation (64 patterns, 4 tones)

**Phase 1: Zadoff-Chu for 4-Tone Alphabet**

```python
def generate_beacon_pattern_freq_sequence(pattern_id, num_symbols=32):
    """
    Generate beacon pattern for 4-FSK tones
    Optimized for [1490, 1520, 1580, 1610] Hz

    This is EASIER than 70-tone generation (less crowded space)
    """
    u = pattern_id  # Zadoff-Chu root (0-63)
    N = 31  # Prime

    sequence = []
    for n in range(31):
        phase = 2 * np.pi * u * n * (n + 1) / (2 * N)

        # Map to 4-tone index (0-3)
        tone_idx = int((phase % (2 * np.pi)) / (2 * np.pi) * 4)
        sequence.append(tone_idx)

    sequence.append(0)  # Pad to 32

    return np.array(sequence, dtype=np.uint8)

# Maps to: [1490, 1520, 1580, 1610][tone_idx]
```

**Phase 2: Simple IQ Trajectories for Beacons**

```python
def generate_beacon_iq_trajectories(pattern_id):
    """
    Beacon IQ: Simple (λ max 0.3)
    Prioritizes robustness over throughput
    """

    iq_full = []  # λ=0.3 maximum (not 1.0!)
    iq_collapsed = []  # λ=0.0

    for t in range(32):
        # Full complexity: Simple circle (NOT complex Lissajous)
        angle = 2 * np.pi * pattern_id * t / 64
        radius = 0.7  # Moderate radius
        iq_full.append(radius * np.exp(1j * angle))

        # Collapsed: BPSK line
        iq_collapsed.append(np.exp(1j * (2 * np.pi * pattern_id * t / 32)))

    return {
        'full': np.array(iq_full),  # Simple circle (λ=0.3)
        'collapsed': np.array(iq_collapsed),  # BPSK
        'max_lambda': 0.3,  # Beacon limit
    }
```

**Phase 3: Optimize to <-30 dB (Faster for 4 Tones)**

```python
# Optimization is FASTER with only 4 tones
# 6-8 hours for 64 beacon patterns (vs 24-30 hours for 128 70-tone patterns)
# Better orthogonality achievable (less crowded space)
```

### Message Pattern Generation (128 patterns, 70 tones)

**Phase 1: Zadoff-Chu for 70-Tone Alphabet**

```python
def generate_message_pattern_freq_sequence(pattern_id, num_symbols=32):
    """
    Generate message pattern for 70-tone grid
    Maps to message tone indices (0-69)

    Can use transfer learning from beacon patterns
    """
    u = pattern_id  # Zadoff-Chu root (0-127)
    N = 31  # Prime closest to 32

    sequence = []
    for n in range(31):
        # Zadoff-Chu quadratic phase
        phase = 2 * np.pi * u * n * (n + 1) / (2 * N)

        # Map to discrete tone index (0-69)
        tone_idx = int((phase % (2 * np.pi)) / (2 * np.pi) * 70)
        sequence.append(tone_idx)

    sequence.append(0)  # Pad to 32 symbols

    return np.array(sequence, dtype=np.uint8)

# Generate 31 base patterns
base_freq_sequences = [
    generate_zadoff_chu_frequency_sequence(u)
    for u in range(31)
]

# Properties:
# - Perfect autocorrelation (zero sidelobes)
# - Cross-correlation ~-15 dB (not sufficient, needs optimization)
# - Constant amplitude
```

### Phase 2: IQ Trajectory Generation

Extend Zadoff-Chu to IQ dimensions for additional orthogonality:

```python
def generate_iq_trajectories(pattern_id, num_symbols=32):
    """
    Generate continuous IQ trajectories

    Returns:
      full_complexity: Lissajous curve (λ=1.0, theoretical maximum)
      collapsed: Simple circle/line (λ=0.0)

    Note: On HF, typical λ is 0.3-0.6 due to multipath phase distortion.
          Full complexity (λ>0.7) realistic only for NVIS (80m/40m single-hop).
          Multi-hop DX typically uses λ=0.3-0.5 (simplified ellipses).
    """

    iq_full = []
    iq_collapsed = []

    for t in range(32):
        # FULL COMPLEXITY (λ=1.0): Lissajous curve
        # Theoretical maximum - realistic only for NVIS propagation
        # Different patterns use different frequency ratios
        freq_a = (pattern_id % 7) + 1  # 1-7
        freq_b = (pattern_id % 5) + 1  # 1-5

        angle_a = 2 * np.pi * freq_a * t / 32
        angle_b = 2 * np.pi * freq_b * t / 32
        offset = 2 * np.pi * pattern_id / 128  # Adjusted for 128 patterns

        i_full = np.cos(angle_a + offset)
        q_full = np.sin(angle_b + offset)
        iq_full.append(complex(i_full, q_full))

        # COLLAPSED (λ=0.0): Simple rotation (BPSK circle)
        # Used for severe multipath, emergency traffic
        simple_angle = 2 * np.pi * pattern_id * t / 32
        iq_collapsed.append(np.exp(1j * simple_angle))

    return {
        'full': np.array(iq_full),     # Theoretical max (rarely used on HF)
        'collapsed': np.array(iq_collapsed),  # Emergency/disturbed
        'typical_hf': 0.4,  # Recommend λ=0.4 for multi-hop DX
    }
```

### Phase 3: Optimization to <-30 dB

Simulated annealing optimizes frequency sequences to achieve exactly -30 dB:

```python
def optimize_to_30db(base_freq_seq, existing_patterns, iterations=100000):
    """
    Optimize frequency sequence to <-30 dB with existing patterns
    Uses simulated annealing
    """

    best_pattern = base_freq_seq.copy()
    best_max_corr = float('inf')

    temperature = 1.0
    cooling_rate = 0.9999

    for i in range(iterations):
        # Mutate: Change one random symbol to different tone
        candidate = best_pattern.copy()
        idx = np.random.randint(32)
        candidate[idx] = np.random.randint(70)  # 0-69 (70 tones)

        # Check correlation with all existing patterns
        max_corr = max(
            cross_correlation_4d(candidate, p)
            for p in existing_patterns
        )

        # Simulated annealing acceptance
        if max_corr < best_max_corr:
            best_pattern = candidate
            best_max_corr = max_corr
        elif np.random.random() < np.exp(-(max_corr - best_max_corr) / temperature):
            best_pattern = candidate
            best_max_corr = max_corr

        temperature *= cooling_rate

        # Success criterion
        if best_max_corr < -30:
            return best_pattern

    if best_max_corr < -30:
        return best_pattern
    else:
        raise ValueError(f"Could not achieve -30 dB (got {best_max_corr:.1f} dB)")

# Generate all 128 patterns
patterns_4d = []

for pattern_id in range(128):
    # Frequency sequence (optimized)
    if pattern_id < 31:
        base_freq = base_freq_sequences[pattern_id]
    else:
        base_freq = np.random.randint(0, 70, size=32)

    freq_seq = optimize_to_30db(base_freq, patterns_4d)

    # IQ trajectories
    iq_trajs = generate_iq_trajectories(pattern_id)

    patterns_4d.append({
        'id': pattern_id,
        'freq_sequence': freq_seq,
        'iq_full': iq_trajs['full'],
        'iq_collapsed': iq_trajs['collapsed']
    })
```

### Phase 4: Validation

```python
def validate_4d_patterns(patterns):
    """
    Verify <-30 dB orthogonality across all dimensions
    Test multiple complexity values
    """

    for lambda_test in [0.0, 0.5, 1.0]:  # Test collapsed, medium, full
        max_corr = -100

        for i, p1 in enumerate(patterns):
            for j, p2 in enumerate(patterns):
                if i >= j:
                    continue

                corr = pattern_correlation_4d(p1, p2, lambda_test)
                max_corr = max(max_corr, corr)

                if corr > -30:
                    print(f"WARNING: Patterns {i},{j} at λ={lambda_test}: {corr:.1f} dB")

        print(f"λ={lambda_test}: Max correlation = {max_corr:.1f} dB")
        assert max_corr < -30

    print("✓ All patterns meet -30 dB in 4D space")
```

---

## Discrete Frequency Hopping

### 70 Reference Tone Grid

See [Adaptive Tone Grid](../protocol/adaptive_tone_grid.md) for comprehensive tone grid specification.

**Summary:**
- Lower band: 35 tones (300-1388 Hz, indices 0-34)
- Beacon reservation: 150 Hz (1475-1625 Hz, centered at 1550 Hz)
- Upper band: 35 tones (1700-2788 Hz, indices 35-69)
- Spacing: 32 Hz (optimized for HF: 5-20 Hz multipath + 0.8-4 Hz/s drift)

### Discrete Hopping (NOT Continuous Chirping)

```python
# CASCADE frequency behavior:

Time:  0ms   50ms  100ms 150ms 200ms 250ms
Tone:  12    34    5     18    29    7
Freq:  684   2789  460   908   1989  524  (Hz - EXACT discrete values)

# Each symbol at constant frequency (50ms dwell)
# Instantaneous hop to next discrete tone
# NO frequency sweeps, NO chirps
# This is FHSS (Bluetooth-like), not CSS (LoRa-like)
```

**Patent safety:** Discrete frequency selection fundamentally different from LoRa's continuous chirp modulation.

---

## Continuous IQ Trajectories

### IQ as Lissajous Curves (HF-Adapted)

While frequency hops discretely, IQ trajectories are continuous. **Important:** HF multipath phase distortion limits practical IQ complexity to λ=0.3-0.6 for most propagation modes. Full complexity (λ>0.7) is theoretical maximum, realistic only for NVIS single-hop paths.

```python
def get_iq_basis(pattern_id, t, complexity_lambda):
    """
    Continuous IQ trajectory at time t
    Smoothly adapts via λ parameter
    """

    # Full complexity (λ=1.0): Lissajous curve
    freq_a = (pattern_id % 7) + 1
    freq_b = (pattern_id % 5) + 1

    angle_a = 2 * np.pi * freq_a * t / 32
    angle_b = 2 * np.pi * freq_b * t / 32
    offset = 2 * np.pi * pattern_id / 64

    i_full = np.cos(angle_a + offset)
    q_full = np.sin(angle_b + offset)
    iq_full = complex(i_full, q_full)

    # Collapsed (λ=0.0): Simple circle
    simple_angle = 2 * np.pi * pattern_id * t / 32
    iq_collapsed = np.exp(1j * simple_angle)

    # Linear interpolation (continuous)
    iq = (1 - complexity_lambda) * iq_collapsed + complexity_lambda * iq_full

    return iq  # Continuous complex value
```

### IQ Collapse Visualization (HF-Realistic)

```
NVIS (λ=0.7-0.9):              Multi-hop DX (λ=0.3-0.5):   Emergency (λ=0.0):
80m/40m single-hop             20m typical                 Disturbed/weak
    Q ↑                            Q ↑                          Q ↑
      │   ╱──╲                       │   ╱─╲                      │
    1 │  ╱ ●  ╲                    1 │  ╱   ╲                   1 │   ●
      │ │   ↓  │                     │ │  ●  │                    │  ╱ ╲
    0 ├─┼──────┼─ I               0 ├─┼─────┼─ I              0 ├─●───●─ I
      │ │   ●  │                     │ │  ●  │                    │  ╲ ╱
   -1 │  ╲    ╱                   -1 │  ╲   ╱                  -1 │   ●
      └──────────                     └───────                     └──────

 Moderate Lissajous            Simple ellipse/circle        BPSK line
 (Realistic for NVIS)          (MOST COMMON on HF)          (Emergency)
 4-6 IQ directions             2-4 IQ directions            1-2 directions
 2-3 bits/symbol               1-2 bits/symbol              0.5-1 bit/symbol

 Note: λ=1.0 (full complexity) is THEORETICAL maximum
       λ=0.4 is TYPICAL for HF DX (multipath phase distortion)
       λ=0.7-0.9 realistic only for NVIS (80m/40m single-hop)
```

### HF-Realistic IQ Complexity Limits

**Multipath phase distortion constrains practical IQ complexity:**

```python
def hf_realistic_lambda_limits(propagation_mode, multipath_delay_ms):
    """
    Determine realistic λ based on HF propagation characteristics

    HF multipath causes phase distortion that smears IQ constellation.
    Complex Lissajous curves (λ>0.7) only work on clean single-hop paths.
    """

    propagation_limits = {
        'nvis_80m_40m': {
            'multipath_delay': 0.5,  # ms (single-hop, minimal)
            'lambda_max': 0.9,  # Near-full Lissajous OK
            'iq_directions': 6,  # Can use 6-8 directions
            'typical_lambda': 0.7,
            'note': 'Clean propagation, IQ coherent'
        },

        'single_hop_f2_20m': {
            'multipath_delay': 2,  # ms (one F2 reflection)
            'lambda_max': 0.6,  # Moderate complexity
            'iq_directions': 4,  # QPSK-level
            'typical_lambda': 0.5,
            'note': 'Some phase distortion, keep IQ simple'
        },

        'multi_hop_dx_20m': {
            'multipath_delay': 5,  # ms (2-3 hops typical)
            'lambda_max': 0.5,  # Simplified ellipses
            'iq_directions': 4,  # QPSK max
            'typical_lambda': 0.4,  # RECOMMENDED DEFAULT
            'note': 'Significant multipath, rely on time-freq separation'
        },

        'long_path_15m': {
            'multipath_delay': 10,  # ms (short+long path)
            'lambda_max': 0.3,  # Nearly collapsed
            'iq_directions': 2,  # BPSK-level
            'typical_lambda': 0.2,
            'note': 'Severe phase distortion, IQ nearly linear'
        },

        'disturbed_aurora': {
            'multipath_delay': 20,  # ms (chaotic)
            'lambda_max': 0.1,  # Collapsed
            'iq_directions': 1-2,  # BPSK only
            'typical_lambda': 0.0,
            'note': 'Phase incoherent, use frequency-time only'
        },
    }

    # Rule of thumb:
    # lambda_max ≈ 1.0 / (1 + multipath_delay_ms / 2)
    # For 5ms delay: λ_max ≈ 1.0 / (1 + 2.5) = 0.29 ≈ 0.3

    return propagation_limits[propagation_mode]

# CRITICAL INSIGHT:
# Most HF operation uses λ = 0.3-0.5 (simplified ellipses)
# NOT λ = 1.0 (full Lissajous)
# Full complexity reserved for exceptional NVIS conditions
```

### Continuous Collapse Mechanism (HF-Bounded)

```python
def continuous_collapse_hf_realistic(snr_db, propagation_mode):
    """
    Complexity λ varies continuously with SNR
    BUT capped by propagation mode (HF reality)
    """

    # Determine SNR-based lambda (as before)
    if snr_db > 15:
        lambda_snr = 1.0
    elif snr_db > -10:
        lambda_snr = (snr_db + 10) / 25  # 0.0 to 1.0
    else:
        lambda_snr = 0.0

    # Determine propagation-based maximum
    prop_limits = hf_realistic_lambda_limits(propagation_mode)
    lambda_max_propagation = prop_limits['lambda_max']

    # Actual lambda: Limited by BOTH SNR and propagation
    lambda_actual = min(lambda_snr, lambda_max_propagation)

    # Add hysteresis (prevent oscillation)
    lambda_actual = apply_hysteresis(lambda_actual, window=2)  # ±2 dB

    return {
        'lambda': lambda_actual,  # Bounded by HF propagation
        'iq_directions': prop_limits['iq_directions'],
        'limited_by': 'propagation' if lambda_snr > lambda_max_propagation else 'snr'
    }

# Example:
# SNR = +15 dB (excellent), multi-hop DX:
#   lambda_snr = 1.0 (SNR says "use full complexity")
#   lambda_max_prop = 0.5 (propagation says "too much multipath")
#   lambda_actual = 0.5 (LIMITED BY PROPAGATION)
#
# This is HF reality! ✓
```

---

## Model-Driven Tone Shifting

Model can shift ±3 tones to avoid interference, always staying on discrete grid:

```python
def model_select_discrete_tone(base_tone_idx, interference_state, available_tones):
    """
    Select optimal discrete tone from ±3 candidates

    Args:
        base_tone_idx: Pattern's nominal tone (0-69)
        interference_state: Measured power at each tone
        available_tones: Which tones RX can decode

    Returns:
        selected_tone_idx: Discrete index (0-69)
    """

    # Generate discrete candidates (±3 tones = ±96 Hz)
    candidates = [
        base_tone_idx + shift
        for shift in range(-3, 4)
        if (base_tone_idx + shift) in available_tones
        and 0 <= (base_tone_idx + shift) < 70
    ]

    if not candidates:
        # No nearby tones available - search farther
        candidates = available_tones

    # Model scores each discrete option
    scores = model.score_tones(
        candidates,
        base_tone=base_tone_idx,
        interference=interference_state
    )

    # Select best (discrete)
    best_idx = np.argmax(scores)
    selected_tone = candidates[best_idx]

    return selected_tone  # Integer: 0-69

# Example:
# Pattern wants tone 23 (1036 Hz)
# Interference HIGH at 1036 Hz
# Candidates: [20,21,22,23,24,25,26] (±3 tones)
# Interference: [LOW, MED, HIGH, HIGH, HIGH, LOW, MED]
# Selects: 20 (972 Hz) - clearest
# Transmits at exactly 972 Hz (not 970.5 Hz!)
```

### Training Discrete Selection

```python
def train_tone_selection_discrete():
    """
    Train model using Gumbel-softmax for differentiability
    """

    # Model outputs tone scores
    tone_logits = model.tone_selection_head(features)  # [70 values]

    # Gumbel-softmax (differentiable discrete sampling)
    tone_probs = gumbel_softmax(tone_logits, temperature=1.0)

    # During training: soft selection (continuous)
    if training:
        soft_tone_selection = sum(
            tone_probs[i] * REFERENCE_TONES[i]
            for i in range(70)
        )
        freq_selected = soft_tone_selection  # Weighted average

    # During inference: hard selection (discrete)
    else:
        tone_idx = argmax(tone_probs)
        freq_selected = REFERENCE_TONES[tone_idx]  # Exact discrete

    # Loss gradient flows during training ✓
    # Discrete selection during inference ✓
```

---

## Multi-Pattern Transmission

Users transmit 1-4 patterns simultaneously based on receiver capability:

```python
def multi_pattern_transmission(data, assigned_patterns, target_kernel):
    """
    Kernel-driven multi-pattern transmission
    Strong receivers get 4×, weak receivers get 1×
    """

    # Kernel announces capacity
    max_patterns = target_kernel['max_patterns_simultaneous']
    # RPi: 1-2 patterns
    # RPi+Coral: 2-4 patterns
    # Desktop: 3-4 patterns

    # Select best patterns from assigned pool
    pattern_scores = model.evaluate_patterns(
        assigned_patterns,
        link_snr=estimate_snr(target_kernel),
        available_tones=target_kernel['available_tones']
    )

    selected = top_k(pattern_scores, k=max_patterns)

    # Split data across patterns
    for i, pattern in enumerate(selected):
        chunk = data[i::max_patterns]  # Interleaved
        transmit_pattern(pattern, chunk, target_kernel)

    # Throughput scales with pattern count
    # 1 pattern: 80 bps
    # 2 patterns: 160 bps
    # 4 patterns: 320 bps
```

### Multi-Pattern Example

```
User A → User B (strong link, +15 dB):

B's kernel: max_patterns=4, available_tones=[0-69]

A transmits on 4 patterns simultaneously:
  Pattern 5:  Hops through tones [12, 34, 5, 18, ...]
  Pattern 12: Hops through tones [45, 7, 29, 3, ...]
  Pattern 19: Hops through tones [22, 61, 14, 38, ...]
  Pattern 26: Hops through tones [8, 52, 31, 9, ...]

At t=0: 4 different discrete tones active (12, 45, 22, 8)
Each carries 1/4 of data
4× throughput = 320 bps
```

---

## Per-Receiver Tone Adaptation

### Receiver Measures Available Tones

```python
def measure_available_tones():
    """
    Receiver measures which of 70 discrete tones are usable
    Accounts for selective fading, local QRM
    """

    available = []

    for tone_idx in range(70):
        freq_hz = REFERENCE_TONES[tone_idx]

        # Measure SNR at exact discrete frequency
        snr = measure_snr(freq_hz)
        qrm = detect_interference(freq_hz)

        if snr > -10 and not qrm:
            available.append(tone_idx)

    return available

# Examples:
# Excellent: [0-69] (all 70)
# Selective fading: [0-34, 40-69] (60 tones, notch at 35-39)
# Heavy QRM: [5-12, 25-35, 50-69] (34 tones)
# Extreme: [10, 25, 40, 55] (4 tones only - system still works!)
```

### Kernel Encoding (40 bits)

```python
def encode_available_tones_to_kernel(available_tones):
    """
    Run-length encoding of tone ranges (40 bits)

    Format:
      - 4 bits: Number of ranges (0-15)
      - 36 bits: Up to 4 ranges (9 bits each)
    """

    ranges = find_contiguous_ranges(available_tones)
    # [0-34, 40-69] → [(0,35), (40,30)]

    num_ranges = min(len(ranges), 4)
    encoded = num_ranges << 36

    for i, (start, length) in enumerate(ranges[:4]):
        # 7-bit start + 2-bit length encoding
        if length == 1:
            range_bits = (start << 2) | 0b00
        elif length <= 127:
            range_bits = ((start << 2) | 0b01) << 7 | length
        else:
            range_bits = (start << 2) | 0b10  # "Remaining"

        encoded |= (range_bits << (i * 9))

    return encoded  # 40 bits
```

### Transmitter Uses RX's Tone Subset

```python
def transmit_with_rx_tone_subset(pattern, target_kernel):
    """
    Use only tones receiver can decode
    """

    rx_tones = decode_available_tones(target_kernel)
    # e.g., [0-34, 40-69] (tone 35-39 blocked by QRM at RX)

    for t in range(32):
        base_tone = pattern.freq_sequence[t]

        if base_tone in rx_tones:
            selected = base_tone  # Use as-is
        else:
            # Base tone unavailable at RX - shift to nearest available
            selected = find_nearest(base_tone, rx_tones, max_dist=3)

        freq = REFERENCE_TONES[selected]
        transmit(frequency=freq, ...)  # Discrete frequency

# Graceful degradation:
# - 70 tones → 320 bps (4 patterns)
# - 40 tones → 280 bps (still works!)
# - 10 tones → 160 bps (minimal but functional)
```

---

## Pattern Storage Format

### File Specification (cascade_patterns_v1.bin)

```python
# Storage requirements per pattern (HIERARCHICAL - single IQ):
# - Frequency sequence: 32 bytes (tone indices)
#   * Beacon patterns (0-63): indices 0-3 (4-FSK)
#   * Message patterns (64-191): indices 0-69 (70 tones)
# - IQ trajectory: 256 bytes (32 × complex64, SINGLE baked-in complexity)
# - Metadata: 4 bytes (includes baked-in complexity level)
# Total: 292 bytes per pattern (47% savings vs dynamic collapse)

# 192 total patterns: 56,064 bytes ≈ 55 KB

FILE_FORMAT = {
    'header': {
        'magic': b'CASC',  # 4 bytes
        'version': 1,  # 2 bytes
        'pattern_count': 192,  # 2 bytes (64 beacon + 128 message)
        'beacon_pattern_count': 64,  # 2 bytes
        'message_pattern_count': 128,  # 2 bytes
        'pattern_length': 32,  # 2 bytes (symbols)
        'num_beacon_tones': 4,  # 2 bytes
        'num_message_tones': 70,  # 2 bytes
        'tone_spacing_hz': 32,  # 2 bytes
        'reserved': 14,  # bytes
        'total': 32  # bytes
    },

    'beacon_pattern_data': [  # 64 beacon patterns (IDs 0-63)
        {
            'id': 1,  # byte (0-63)
            'freq_sequence': 32,  # bytes (tone indices 0-3 for 4-FSK)
            'iq_trajectory': 256,  # bytes (32 complex64, SINGLE trajectory)
            'complexity_level': 1,  # byte (0=BPSK, 1=simple, 2=moderate)
            'checksum': 2,  # bytes (CRC16)
            'reserved': 1,  # byte
            'total': 292  # bytes
        }
    ],

    'message_pattern_data': [  # 128 message patterns (IDs 64-191)
        {
            'id': 1,  # byte (64-191)
            'freq_sequence': 32,  # bytes (tone indices 0-69)
            'iq_trajectory': 256,  # bytes (SINGLE baked-in complexity)
            'complexity_level': 1,  # byte (encoded IQ complexity for this pattern)
            'checksum': 2,  # bytes (CRC16)
            'reserved': 1,  # byte
            'total': 292  # bytes
        }
    ],

    'total_size': 56096  # bytes (≈55 KB, 47% savings)
}

# Complexity levels:
COMPLEXITY_ENCODING = {
    0: 'minimal (BPSK line)',  # Patterns 0-15, 64-79
    1: 'simple (circle)',  # Patterns 16-47, 80-95
    2: 'simple-moderate (ellipse)',  # Patterns 48-63, 96-111
    3: 'moderate (ellipse)',  # Patterns 112-127
    4: 'moderate-complex (ellipse)',  # Patterns 128-143
    5: 'complex (moderate Lissajous)',  # Patterns 144-159
    6: 'very-complex (Lissajous)',  # Patterns 160-175
    7: 'maximum (complex Lissajous)',  # Patterns 176-191 (rarely used)
}
```

---

## Implementation & Training

### Pattern Lookup (Runtime)

```python
# Patterns loaded at startup
PATTERN_TABLE = load_patterns('cascade_patterns_v1.bin')

def get_pattern_envelope(pattern_id, t, complexity_lambda, rx_available_tones):
    """
    Get 4D envelope for transmission
    """

    pattern = PATTERN_TABLE[pattern_id]

    # Discrete frequency
    base_tone_idx = pattern['freq_sequence'][t]

    # Shift if needed (interference or unavailable at RX)
    selected_tone = model.select_tone(
        base_tone_idx,
        rx_available_tones,
        interference_state
    )

    # Continuous IQ
    iq_full = pattern['iq_full'][t]
    iq_collapsed = pattern['iq_collapsed'][t]
    iq = (1 - complexity_lambda) * iq_collapsed + complexity_lambda * iq_full

    return {
        'frequency_hz': REFERENCE_TONES[selected_tone],  # Discrete
        'iq_basis': iq,  # Continuous
    }
```

### Training Strategy

```python
def train_4d_patterns():
    """
    Train model to optimize within 4D pattern structure
    """

    for batch in training_data:
        # Generate multi-user scenario
        users = generate_users(num=random.randint(10, 80))

        for user in users:
            # Pattern assignment (protocol layer)
            assigned = assign_pattern_pool(user.id)

            # Model selects (continuous layer):
            # - Which discrete tone (classification - Gumbel-softmax)
            # - What IQ basis (regression - MSE loss)
            # - Complexity λ (regression)
            # - How many patterns (classification)

            encoding = model.encode_4d(
                data=user.data,
                patterns=assigned,
                target_kernel=user.target.kernel,
                snr=user.link_snr
            )

            # Transmit and decode
            transmitted = transmit_4d(encoding)
            decoded = receive_4d(transmitted, user.pattern)

            # Loss
            loss = cross_entropy(decoded, user.data)

            # Additional losses:
            # - Tone selection (discrete) via Gumbel-softmax
            # - IQ optimization (continuous) via MSE
            # - Orthogonality preservation penalty

            optimize(loss)
```

---

## Alternative Approaches (Why Zadoff-Chu + 4D?)

### Gold Codes
- Cross-correlation: ~-15 dB (insufficient)
- **Verdict:** ❌ Need -30 dB

### Kasami Sequences
- Cross-correlation: ~-17 dB (insufficient)
- **Verdict:** ❌ Need -30 dB

### Walsh-Hadamard
- Requires perfect synchronization
- Poor with frequency/timing offsets
- **Verdict:** ❌ CASCADE is asynchronous

### Zadoff-Chu + Optimization + 4D Extension
- Base correlation: ~-15 dB
- With optimization: Exactly -30 dB ✓
- Extended to IQ: Additional orthogonality ✓
- Patent-free (1970s mathematics) ✓
- **Verdict:** ✅ Chosen approach

---

## See Also

- **[Adaptive Tone Grid](../protocol/adaptive_tone_grid.md)** - 70 discrete reference tones specification
- **[TFIQ Dimensions](tfiq_dimensions.md)** - Multi-user separation in 4D space
- **[Beacon Reservation](../protocol/beacon_reservation.md)** - 175 Hz center-band for coordination
- **[Emergency Relay Network](../protocol/emergency_relay_network.md)** - Ad-hoc emergency system
- **[Model README](README.md)** - Overall model architecture
- **[Signal Specification](../protocol/signal_specification.md)** - Physical layer details
