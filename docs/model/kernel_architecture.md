# CASCADE Kernel Architecture

**Date**: 2025-10-04
**Status**: Final Architecture

---

## Executive Summary

CASCADE uses a **4-kernel beacon structure** totaling 112 bytes, with each kernel being 28 bytes of hybrid discrete/continuous data. Stations beacon three RX kernels (indicating how others should reach them) plus one TX kernel (indicating current transmission state for collision avoidance).

**Key Innovation**: The combination of RX pro-kernels and TX anti-kernels enables distributed coordination achieving 78-85% efficiency without central control. Multi-pattern support (1-8 patterns) is encoded efficiently using pattern ranges.

---

## Kernel Structure (28 Bytes)

### Discrete Portion (3 bytes = 24 bits)

The discrete portion encodes all protocol-critical parameters needed for multi-pattern transmission and reception. Rather than encoding individual pattern IDs, it uses a range encoding (start + count) to efficiently represent 1-8 consecutive patterns.

**Pattern encoding (10 bits):**
- Pattern start: 7 bits (first pattern ID in range, 0-127)
- Pattern count: 3 bits (number of consecutive patterns, 1-8)

This allows encoding patterns like [67, 68, 69, 70] as start=67, count=4, which is much more efficient than listing all patterns individually.

**Channel parameters (10 bits):**
- Frequency pair: 7 bits (which tone pair from 75 available)
- Modulation: 3 bits (BPSK/QPSK/8-PSK/16-APSK)

**Version tracking (4 bits):**
- Protocol version: 2 bits (protocol compatibility, 0-3)
- Model version: 2 bits (NN model generation, 0-3)

**Purpose:**
The discrete portion enables deterministic protocol decisions including pattern selection, frequency allocation, modulation choice, and version compatibility. For TX kernels, it precisely indicates which patterns are actively being transmitted for collision avoidance.

### Continuous Embedding (24 bytes)

**48 dimensions × 4-bit quantization = 192 bits**:

```python
embedding_vector: int4[48]  # Quantized to 0-15 per dimension

Semantic structure (learned, not hardcoded):
- dims 0-7: Modulation fine-tuning
- dims 8-15: Constellation mutation hints
- dims 16-31: Per-symbol frequency adjustments (2 dims per symbol avg)
- dims 32-43: Power shaping guidance
- dims 44-47: Channel pre-equalization

Note: Exact semantics learned by NN during training
Above is approximate conceptual mapping
```

**Purpose**:
- Encoder NN mutation guidance
- Pro/anti-kernel weighted combination
- Continuous signal optimization beyond discrete granularity

**Quantization**:
```
Training: NN works with float32 (continuous)
Deployment: Quantized to INT4 for TPU efficiency and beacon bandwidth
De-quantization: Learned mapping (part of NN, minimal information loss)
```

---

## 4-Kernel Beacon Structure

Each station beacons 4 kernels adaptively (2-10 minutes based on SNR), totaling 112 bytes:

**Three RX Kernels (Pro-Kernels)**

The decoder generates three RX kernel options based on observed channel conditions from raw IQ samples. These serve as "pro-kernels" telling other stations the best ways to reach this station. Having three options provides flexibility - if the primary option has interference, transmitters can fall back to secondary or tertiary options.

Each RX kernel indicates:
- Preferred pattern range for incoming transmissions
- Maximum number of patterns this station can decode (1-8 based on current SNR and processing load)
- Preferred frequency pair that appears clear from this station's perspective
- Suggested modulation based on observed channel quality

**One TX Kernel (Anti-Kernel)**

The protocol generates one TX kernel indicating current transmission state. This serves as an "anti-kernel" for collision avoidance - other stations know to avoid these patterns and frequencies.

The TX kernel specifies:
- Pattern range currently being transmitted (start + count)
- Frequency pair in active use
- Current modulation
- When count=0, indicates idle state (not transmitting)

**Why This Structure Works**

The separation between RX and TX kernels provides precise network state information:

- RX kernels announce receive capability, updated based on real-time channel observations
- TX kernel provides ground truth on actual transmissions, eliminating guesswork
- Together they enable distributed coordination without central control
- Stations independently avoid collisions while optimizing for their targets

The multi-pattern encoding via ranges keeps the kernel compact while supporting variable throughput from 94 bps (1 pattern, BPSK) up to 1,950 bps (8 patterns, 16-APSK).
- Candidate #3: Acceptable for channel, zero interference

Enables: Network-aware selection without rerunning decoder
```

**Priority handling**:
```
Emergency: Select candidate #1 (ignore anti-kernels, maximize delivery)
Routine: Select candidate with best network score (minimize interference)
```

---

## Pro-Kernel and Anti-Kernel

### Pro-Kernel (Invitation)

**Station A beacons**:
```python
pro_kernels_A = [kernel_1, kernel_2, kernel_3]  # 3 RX candidates
tx_kernel_A = kernel_4  # Current TX state

Meaning: "To reach Station A optimally, use ONE of these 3 RX kernels; A is transmitting on TX kernel"
Purpose: Tells network how to transmit TO Station A and what A is currently transmitting
Broadcast: Via beacon adaptively (2-10 min based on SNR)
Content: 4 × 28 bytes = 112 bytes (3 RX + 1 TX kernel data)
```

**Selecting from pro-kernels**:
```python
# You want to contact Station A
# Pick from A's 3 candidates based on your constraints

if your_power < 25W:
    # Pick BPSK candidate (most robust)
    selected = [k for k in pro_kernels_A if k.modulation == BPSK][0]
elif interference_detected(your_channel):
    # Pick candidate with cleanest frequency
    selected = min(pro_kernels_A, key=lambda k: interference_at(k.frequency))
else:
    # Pick highest throughput
    selected = max(pro_kernels_A, key=lambda k: k.modulation_bits)
```

### Anti-Kernel (Warning)

**Same beacon serves as anti-kernel for non-targets**:
```python
# Station B receives Station A's beacon
# B is NOT trying to contact A
# For B, this is an anti-kernel

anti_kernel_A = station_A_beacon.current_kernel  # The one A is currently using for RX

Meaning: "Station A is currently receiving on THIS kernel, avoid interfering"
Purpose: Distributed interference avoidance
Used by: All stations transmitting to someone else
```

**Network-wide anti-kernel collection**:
```
45 stations beacon simultaneously:
Your station collects:
- 1 pro-kernel (your target): Use this
- 44 anti-kernels (everyone else): Avoid these

Coordination: Implicit, no negotiation needed
```

---

## TX Kernel Generation (Weighted Combination)

**When transmitting to Target Station**:

```python
def generate_tx_kernel(target_beacon, all_beacons, message, channel):
    """
    Generate optimal TX kernel via weighted combination

    Args:
        target_beacon: Target station's 3 pro-kernel candidates
        all_beacons: All 44 other stations' beacons
        message: User data to transmit
        channel: Own channel measurements

    Returns:
        final_kernel: 28-byte kernel for transmission
    """

    # Step 1: Protocol selects discrete parameters
    # Choose from target's 3 candidates
    target_options = target_beacon.kernels  # 3 candidates
    anti_kernels = [b.current_kernel for b in all_beacons if b != target_beacon]

    selected_discrete = None
    for candidate in target_options:
        # Check for collisions
        collisions = [ak for ak in anti_kernels
                      if ak.pattern == candidate.pattern
                      and ak.frequency == candidate.frequency]

        if len(collisions) == 0:
            selected_discrete = candidate.discrete
            selected_embedding_base = candidate.embedding
            break

    if selected_discrete is None:
        # All candidates have collisions, pick least bad
        selected_discrete = target_options[0].discrete
        selected_embedding_base = target_options[0].embedding

    # Step 2: Combine continuous embeddings (weighted)
    # Start with selected candidate's embedding
    combined_embedding = 0.70 * selected_embedding_base

    # Subtract anti-kernel embeddings (avoid interference)
    for i, anti_kernel in enumerate(anti_kernels):
        # Weight by: Distance (closer = higher weight), signal strength, priority
        distance_km = calculate_distance(own_grid, anti_kernel.grid)
        weight = 0.30 / len(anti_kernels) * exp(-distance_km / 1000)  # Nearby matters more

        combined_embedding -= weight * anti_kernel.embedding

    # Step 3: Normalize embedding
    combined_embedding = normalize(combined_embedding)  # Keep in valid range

    # Final kernel
    final_kernel = {
        'discrete': selected_discrete,
        'embedding': combined_embedding
    }

    return final_kernel
```

**Result**:
- Discrete params: From target's preferred options
- Embedding: Weighted to reach target while avoiding others
- Emergent coordination: Each station optimizes locally, network converges globally

---

## Signal Generation and Mutation

### Baseline Signal Generation (Protocol)

```python
def generate_baseline_signal(kernel_discrete, pattern, data_bits):
    """
    Protocol generates standard IQ signal

    Uses only discrete kernel parameters
    No encoder NN involved (fast, deterministic)
    """

    freq_sequence = pattern.freq_sequence  # [0,1,0,0,1,1,0,1,...]
    tone_pair = kernel_discrete.frequency_pair  # Which 2 tones
    modulation = kernel_discrete.modulation_order  # BPSK/QPSK/8-PSK/16-APSK

    # Map tone indices to physical frequencies
    tone_0_hz = 300 + tone_pair * 40  # Example mapping
    tone_1_hz = tone_0_hz + 20

    # Generate IQ samples per symbol
    iq_signal = []
    for symbol_idx in range(32):
        tone_idx = freq_sequence[symbol_idx]  # 0 or 1
        freq_hz = tone_0_hz if tone_idx == 0 else tone_1_hz

        # Get data bits for this symbol (with differential encoding)
        data_bits_for_symbol = get_differential_bits(symbol_idx, data_bits, tone_idx)

        # Modulate onto constellation
        if modulation == BPSK:
            iq_point = 1.0 if data_bits_for_symbol == 0 else -1.0  # Real axis
        elif modulation == QPSK:
            iq_point = QPSK_constellation[data_bits_for_symbol]  # 4 points
        elif modulation == 8PSK:
            iq_point = PSK8_constellation[data_bits_for_symbol]  # 8 points
        elif modulation == 16QAM:
            iq_point = QAM16_constellation[data_bits_for_symbol]  # 16 points

        # Generate samples for this symbol (at freq_hz with iq_point)
        symbol_samples = generate_tone(freq_hz, iq_point, sample_rate=48000)
        iq_signal.extend(symbol_samples)

    return iq_signal  # Standard signal, ready for encoder mutations
```

**For beacons**: Use baseline signal directly (no encoder step)
**For messages**: Pass to encoder for mutations

### Encoder NN Mutations (Messages Only)

```python
def encoder_mutate_signal(baseline_iq, kernel_embedding, channel_state):
    """
    Encoder NN applies continuous optimizations

    Mutations guided by kernel embedding (24 bytes)
    Learns optimal adjustments from training data
    """

    # NN forward pass
    mutations = encoder_NN(
        baseline_signal=baseline_iq,
        embedding=kernel_embedding,
        channel=channel_state
    )

    # Apply continuous mutations
    mutated_iq = baseline_iq.copy()

    for symbol_idx in range(32):
        # Frequency micro-shifts (sub-Hz precision)
        freq_shift_hz = mutations.frequency[symbol_idx]  # ±0.1-2 Hz
        mutated_iq[symbol_idx] = apply_frequency_shift(
            mutated_iq[symbol_idx],
            freq_shift_hz
        )

        # Constellation mutations (rotation, scaling)
        rotation_deg = mutations.rotation[symbol_idx]  # ±0-45° fine adjustment
        scale_factor = mutations.scale[symbol_idx]     # 0.5-1.5× adaptive scaling
        mutated_iq[symbol_idx] *= exp(1j * rotation_deg * pi/180) * scale_factor

        # Power shaping (per-symbol power control)
        power_factor = mutations.power[symbol_idx]  # 0.5-1.5× per symbol
        mutated_iq[symbol_idx] *= sqrt(power_factor)

    return mutated_iq  # Optimized signal beyond protocol granularity
```

**What encoder learns**:
- Interference nulling: Rotate constellation away from nearby patterns
- Channel pre-equalization: Compensate for predicted multipath
- Power efficiency: Use less power on redundant symbols
- Micro-coordination: Sub-Hz frequency steering (beyond ±2 Hz protocol steps)

---

## Decoder Operations

### Role 1: Message Demodulation

```python
def decode_message(rx_signal, kernel):
    """
    Extract message from received signal

    Uses both discrete and continuous kernel info
    """

    # Use discrete params to focus search
    pattern_id = kernel.discrete.pattern
    frequency_pair = kernel.discrete.frequency
    modulation = kernel.discrete.modulation

    # Use embedding to understand encoder mutations
    expected_mutations = dequantize_embedding(kernel.embedding)

    # NN decode
    decoded_data = decoder_NN(
        signal=rx_signal,
        pattern_template=patterns[pattern_id],
        frequency_hint=frequency_pair,
        modulation=modulation,
        expected_mutations=expected_mutations,  # What encoder probably did
        channel_estimate=measure_channel()
    )

    return decoded_data
```

### Role 2: Kernel Generation for Own RX

```python
def generate_own_kernels(channel_state):
    """
    Decoder generates 3 kernel candidates for own reception

    These become pro-kernels in your beacon
    """

    # Analyze own channel
    snr = measure_snr()
    multipath = measure_multipath()
    interference = measure_interference()

    # Decoder NN generates optimal kernels
    kernel_candidates = decoder_NN.generate_kernels(
        channel_state={snr, multipath, interference},
        available_patterns=patterns[0:128],
        available_frequencies=frequency_pairs[0:75]
    )

    # Output: 3 candidates (diversity for others to choose from)
    return [
        {discrete: {...}, embedding: [...], score: 0.85},
        {discrete: {...}, embedding: [...], score: 0.78},
        {discrete: {...}, embedding: [...], score: 0.71},
    ]
```

**These kernels broadcast in beacon** → become pro-kernels for network

---

## Beacon Structure

**123-byte beacon message**:

```
4 kernels: 3 RX + 1 TX = 4 × 28 bytes = 112 bytes
Metadata:
  - Callsign: 6 bytes
  - Grid square: 3 bytes
  - Net ID: 2 bytes
  - Relay status: 1 byte
  - Timestamp: 2 bytes
  - Checksum: 2 bytes
Total: 84 + 16 = 100 bytes

Transmission @ BPSK (94 bps): 8.5 seconds
Transmission @ QPSK (144 bps): 5.6 seconds

Beacon interval: Adaptive (2-10 minutes based on SNR)
Overhead: 14% @ BPSK, 9% @ QPSK (acceptable)
```

**Beacon encoding**:
```
Beacon uses: Protocol-generated signal (no encoder mutations)
Why: Simplifies decode (critical for fast coordination)
     All stations can quickly extract kernels
     Enables real-time pro/anti-kernel updates
```

---

## Distributed Coordination Mechanism

**Emergent coordination without central control**:

### Network State Discovery

```
Each station monitors all beacons:
- 45 stations × 4 HF bands = 180 beacons per minute
- Decodes: 180 × 4 kernels = 720 kernels total (540 RX + 180 TX)
- Categorizes:
  - If targeting station X: Use X's 3 pro-kernels
  - All others: 44 × current kernel = anti-kernels
```

### Relay Coordination

**Relay station beacon**:
```
Relay announces broad pro-kernels:
kernel_candidates = [
    {pattern: 50-80 (wide range), modulation: QPSK (conservative), freq: 30-60},
    {pattern: 48-95 (very wide), modulation: BPSK (maximum compat), freq: 20-70},
    {pattern: 64-95 (moderate), modulation: 8PSK (good cond), freq: 40-60},
]

Inclusive: Accepts wide range of patterns/frequencies
Purpose: Net members optimize to relay, relay receives all easily
```

**Net member selection**:
```
Each member picks from relay's candidates:
- Selects: Based on own channel quality
- Member A: Picks QPSK candidate (good conditions)
- Member B: Picks BPSK candidate (weak signal)
- Coordinated: All members reach relay on different subsets

Traffic flows: Members → Relay → Destination (emergent routing!)
```

### Weighted Combination Math

```python
# Target: Station A (relay)
pro_embedding = station_A.kernel_candidates[1].embedding  # Selected candidate

# Anti-kernels: 44 other stations
anti_embeddings = [b.current_kernel.embedding for b in other_stations]

# Weights (distance-based)
alpha = 0.70  # Pro-kernel weight (strong toward target)
betas = [calculate_weight(distance_to(s)) for s in other_stations]
# Nearby: β=0.05, Medium: β=0.01, Far: β=0.001

# Combination
tx_embedding = alpha * pro_embedding - sum([betas[i] * anti_embeddings[i]
                                             for i in range(44)])

# Normalize to valid range
tx_embedding = clip(tx_embedding, -15, +15)  # Keep in INT4 range after de-quant
```

**Emergent properties**:
- Stations naturally coordinate toward common frequencies (pro-kernel attraction)
- Nearby stations avoid each other (anti-kernel repulsion, weighted by distance)
- Network self-organizes without central control
- 78-85% packing efficiency achieved through distributed optimization

---

## Hardware Requirements

**For kernel operations**:

**Decoder (kernel generation)**:
- Generate 3 RX kernels: 5-10 ms (on Coral TPU)
- Generate 1 TX kernel: Protocol-determined (current state)
- Complexity: 2-5M parameter model
- Memory: 20-40 MB

**Protocol (coordination)**:
- Select discrete params: <1 ms (simple logic)
- Combine embeddings: <1 ms (48-dim vector math)
- Complexity: Minimal (C code or tiny NN)

**Encoder (signal mutation)**:
- Mutate signal: 10-20 ms (on Coral TPU)
- Complexity: 2-5M parameter model
- Memory: 20-40 MB

**Total TX pipeline**: 16-31 ms (comfortable for transmit preparation)

**Decoder (beacon processing)**:
- Batch decode 180 beacons: 1-2 seconds (background, over 60s period)
- Extract 540 kernels: <100 ms (parallel processing)
- Categorize pro/anti: <10 ms (vector operations)

**Total RX pipeline**: 50-100 ms per frame (real-time capable on RPi4+TPU)

---

## Comparison to Traditional Approaches

**TDMA/FDMA** (traditional):
```
Coordination: Central controller assigns slots
Efficiency: 60-70% (guard times, slot overhead)
Complexity: Central authority, synchronization required
Failure mode: Controller down = network down
```

**CASCADE kernel coordination**:
```
Coordination: Distributed (pro/anti-kernel beacons)
Efficiency: 78-85% (emergent optimization, minimal overhead)
Complexity: Each station autonomous, no central authority
Failure mode: Graceful degradation (network continues without failed station)
```

**CDMA** (spread spectrum):
```
Coordination: Pre-assigned PN codes
Efficiency: 40-60% (near-far problem, code cross-correlation)
Orthogonality: PN code correlation (~-20 to -25 dB)
Adaptability: Fixed codes, no runtime optimization
```

**CASCADE**:
```
Coordination: Dynamic kernel selection + embedding optimization
Efficiency: 78-85% (kernel coordination) × 55-70% (adaptive modulation)
Orthogonality: Pattern correlation (-37.5 dB) + constellation diversity
Adaptability: Runtime optimization (encoder mutations, modulation selection)
```

**CASCADE achieves TDMA-like efficiency with CDMA-like asynchronous operation**

---

## Version Compatibility

**Protocol version** (4 bits):
```
v1: 2-FSK, differential encoding, 28-byte kernel
v2: Potential future enhancements (e.g., hierarchical modulation)
v3-v15: Reserved for protocol evolution

Backward compatibility:
- v2 transmitter: Can fall back to v1 if receiver only supports v1
- v1 receiver: Ignores unknown protocol features in v2 transmission
```

**Model version** (4 bits):
```
v1: Trained on 24-36K hours HF data
v2: Retrained with 50K+ hours (better performance)
v3: New architecture (e.g., transformer vs CNN)

Compatibility:
- Newer model can decode older model transmissions
- Older model may have degraded performance on newer transmissions
- Version mismatch: Display warning, attempt decode
```

---

*Complete kernel specification for CASCADE dual-layer architecture*
*Enables 78-85% distributed coordination efficiency*
*Hardware: RPi4 + Coral TPU ($110, 15W, portable)*
