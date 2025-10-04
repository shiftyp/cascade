# Adaptive 4-FSK Control Channel

The 4-FSK interstitial channel serves as CASCADE's universal control plane, providing robust fallback communication and network coordination. Through adaptive pattern mutations and microsecond timing separation, the 4-FSK channel supports 10-15 simultaneous transmissions while maintaining -22 dB weak-signal capability.

## Overview

**4-FSK channel characteristics:**
- Frequencies: [1490, 1520, 1580, 1610] Hz (centered at 1550 Hz, symmetric)
- Symbol duration: 100ms normal, 200ms emergency (robust)
- Modulation: 4-FSK (2 bits/symbol)
- Base throughput: 20 bps per transmission (100ms symbols)
- Patterns: Uses 64 dedicated beacon patterns (IDs 0-63) optimized for 4-FSK
- Multi-user: 48 simultaneous transmissions via pattern orthogonality (anti-kernel resilient)

**Primary uses:**
1. Network discovery (beacons)
2. Initial contact (ACKs before kernel exchange)
3. Kernel exchange (target and anti-kernels)
4. Emergency coordination (universal channel)
5. Fallback communication (if message mode fails)

## Pattern Orthogonality in 4-FSK

**4-FSK uses CASCADE's 48 beacon patterns** (IDs 0-47, each selecting 4 tones from 78-tone grid):

```python
# Multiple stations beacon simultaneously
Station A: Pattern 20 (beacon), 4-FSK on [1490, 1520, 1580, 1610], 100ms symbols
Station B: Pattern 35 (beacon), 4-FSK on [1490, 1520, 1580, 1610], 100ms symbols  # Same frequencies!
Station C: Pattern 48 (beacon), 4-FSK on [1490, 1520, 1580, 1610], 100ms symbols

# All overlap in frequency and time
# Separated by: Beacon pattern orthogonality (<-30 dB in 4D space)
# Beacon patterns: Each uses 4 tones from 78-tone grid (adaptive selection)
# Model decodes all three independently
```

**4-FSK channel capacity:**
```
Symbol rate: 6.25 symbols/sec (160ms symbols)
Modulation: 4-FSK (2 bits/symbol)
Raw rate: 6.25 × 2 = 12.5 bps per transmission
Multiple transmissions: 10-15 simultaneous via pattern orthogonality

Aggregate at typical SNR (0 dB):
- Shannon limit (2 kHz 4-FSK bandwidth): ~4,300 bps
- 4-FSK actual: ~800 bps (conservative, 18% Shannon)
- Intentionally inefficient for maximum robustness

vs Message channel:
- Shannon limit @ +15 dB (2.5 kHz): ~12,500 bps
- Message actual: ~9,427 bps (75% Shannon - chaos-optimized)

4-FSK trades efficiency for robustness (-22 dB capability)
```

## Ideal 4-FSK Kernel

**Beacons include "ideal 4-FSK kernel"** - how to reach this station on 4-FSK:

```python
# Included in beacon (32 bits)
ideal_4fsk_kernel = {
    'base_pattern': 6 bits,              # One of 48 beacon patterns (my base for 4-FSK, IDs 0-47)
    'mutation_seed': 8 bits,             # Seed for pattern variation
    'preferred_time_offset_ms': 8 bits,  # When I listen best (0-255 × 10ms = 0-2.55s)
    'constellation_tolerance': 4 bits,   # IQ variance I can decode
    'timing_tolerance_ms': 4 bits,       # Symbol timing variance (0-15ms)
    'reserved': 2 bits
}
# Total: 32 bits (fits in beacon payload alongside callsign hash)
```

**Purpose**: Enables transmitters to adapt their 4-FSK transmission for optimal reception.

## Pattern Mutation Mechanism

**Transmitter generates mutated pattern from kernel:**

```python
def generate_mutated_4fsk_pattern(target_ideal_kernel, my_random_input):
    """Create pattern variant for collision avoidance"""

    # Base pattern from target's kernel
    base_pattern = PATTERNS[target_ideal_kernel.base_pattern]  # e.g., Pattern 5

    # Deterministic mutation from seed
    mutation_rng = seed_rng(target_ideal_kernel.mutation_seed)

    # Add my random input for uniqueness
    my_rng = seed_rng(my_random_input)  # Station-specific

    # Generate mutated pattern
    mutated = []
    for i, symbol in enumerate(base_pattern):
        # Apply mutations
        tone_offset = (mutation_rng.next() + my_rng.next()) % 4  # Stay within 4 tones
        timing_jitter = my_rng.next_gaussian() * target_ideal_kernel.timing_tolerance

        mutated.append({
            'tone': (symbol.tone + tone_offset) % 4,  # Shifted tone
            'time': symbol.time + timing_jitter        # Jittered timing
        })

    return mutated  # Unique variant of base pattern
```

**Properties:**
- **Deterministic from seed**: Receiver can predict mutations
- **Unique per transmitter**: My random input ensures differentiation
- **Bounded variance**: Within tolerance specified by kernel
- **Maintains correlation**: Still correlates with base pattern (~-15 to -20 dB vs -30 dB pure orthogonal)

## Microsecond Timing Offset Separation

**Multiple transmitters with random start times:**

```python
def transmit_4fsk_with_timing(message, target_kernel):
    """Transmit with microsecond-granularity timing"""

    # Base timing from kernel
    preferred_offset_ms = target_kernel.preferred_time_offset_ms * 10  # 0-2550ms

    # Add random jitter (microsecond level)
    my_jitter_us = random.randint(0, 10000)  # 0-10ms in microseconds

    # Calculate start time
    start_time = beacon_end_time + preferred_offset_ms + (my_jitter_us / 1000)

    # Transmit at precise time
    schedule_transmission(
        signal=encoded_4fsk,
        start_time_us=start_time * 1000,  # Convert to microseconds
        duration_ms=5100                   # 4-FSK transmission duration
    )
```

**Sound card captures offsets:**
```
48 kHz sampling: 21 microsecond resolution

Radio A starts: t=0
Radio B starts: t=234μs (11 samples later)
Radio C starts: t=891μs (43 samples later)

Model receives distinct onsets in sample stream
Can separate by analyzing amplitude envelope transitions
```

## Envelope-Based Separation

**Model learns from envelope perturbations:**

```python
class EnvelopeSeparator(nn.Module):
    """Separate overlapping 4-FSK via amplitude envelope analysis"""

    def __init__(self):
        # High-resolution temporal convolution
        self.onset_detector = nn.Conv1d(2, 64, kernel_size=128)  # 2.7ms window

        # Envelope extractor
        self.envelope_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, max_users)  # Output: number of detected users
        )

    def forward(self, iq_samples_48khz):
        """
        Input: IQ samples at 48 kHz (2400 samples for 50ms symbol)
        Output: Separated user signals
        """

        # Detect onsets (when new signals start)
        onset_features = self.onset_detector(iq_samples_48khz)

        # Extract envelope (amplitude variations from beating)
        envelope = abs(iq_samples_48khz)  # Magnitude

        # Analyze envelope features
        # - Amplitude steps (new signal starts)
        # - Beating frequency (phase offset between overlapping tones)
        # - Gradual changes (clock drift differences)

        envelope_features = self.envelope_net(envelope)

        # Combine onset timing + envelope analysis → separate signals
        separated = self.separate_by_timing_and_envelope(
            onset_features,
            envelope_features,
            iq_samples_48khz
        )

        return separated  # List of individual user signals
```

**Features model uses:**
1. **Onset timing**: Amplitude step when new transmission starts
2. **Beating patterns**: Amplitude modulation from phase offsets
3. **Clock drift**: Each radio's unique frequency error creates signature
4. **Envelope trajectory**: How amplitude evolves over 1.6s pattern

## Clock Drift as Signal Fingerprint

**Every radio has unique clock error:**

```python
# At 14 MHz carrier frequency:
Radio A: +30 ppm drift → +420 Hz error
Radio B: -15 ppm drift → -210 Hz error
Radio C: +5 ppm drift → +70 Hz error

# On 4-FSK tone at 312 Hz:
Radio A transmits: 312 + 0.42 = 312.42 Hz
Radio B transmits: 312 - 0.21 = 311.79 Hz
Radio C transmits: 312 + 0.07 = 312.07 Hz

# Over 1.6s pattern:
Phase accumulation:
Radio A: 0.42 Hz × 1.6s = 0.67 cycles = 241° phase rotation
Radio B: -0.21 Hz × 1.6s = -0.34 cycles = -122° rotation
Radio C: 0.07 Hz × 1.6s = 0.11 cycles = 40° rotation

# Each radio creates unique phase trajectory
# Model learns to track and separate by drift signature!
```

## Dedicated Target Kernel Slots

**Beacon includes dedicated receive window:**

```python
# My beacon structure:
beacon_transmission_with_slot = {
    # Part 1: Beacon announcement (1.3s)
    'callsign_hash': 16 bits,
    'emergency': 1 bit,
    'ideal_4fsk_kernel': 32 bits,

    # Part 2: Implicit dedicated slot (follows beacon)
    'slot_timing': {
        'start': beacon_end + 0ms,        # Immediately after beacon
        'duration': 2000ms,                # 2 second window
        'purpose': 'target_kernels_to_me'
    }
}

# Beacon timeline:
0.0-1.3s: My beacon transmits
1.3-3.3s: DEDICATED SLOT - I listen for:
          ├─ Target kernels (stations wanting to talk to me)
          ├─ QSO requests
          ├─ Net control messages (if I'm member)
          └─ Model decodes overlapping transmissions (10-15 possible)
3.3-5.0s: General activity (anyone can transmit)
```

**Stations transmitting to me use my slot:**
```python
# K0BB wants to send kernel to W2DEF
w2def_beacon = heard_beacons['W2DEF']

# Calculate transmission timing
transmit_time = w2def_beacon.end_time + random_offset(0, 2000)  # Within 2s slot

# Transmit with mutation
transmit_4fsk(
    my_target_kernel,
    pattern=mutate(w2def_beacon.ideal_kernel.base_pattern, my_random),
    start_time=transmit_time,
    duration=5.1s  # 64-bit kernel
)

# W2DEF decodes during their dedicated slot
# Higher priority decode (knows to expect target kernels)
```

## Multi-User Overlap Example

**5 stations send kernels to W2DEF simultaneously:**

```
W2DEF's dedicated slot (1.3-3.3s after W2DEF beacon):

K0BB transmits: Pattern 5-mutated-A, starts 1.350s
W1ABC transmits: Pattern 5-mutated-B, starts 1.423s  (73ms later)
K5XYZ transmits: Pattern 5-mutated-C, starts 1.891s  (541ms after K0BB)
N7MNO transmits: Pattern 5-mutated-D, starts 2.103s
VE3QWE transmits: Pattern 5-mutated-E, starts 2.567s

All five overlap in time!

W2DEF's model receives mixed signal:
├─ Onset detection: 5 distinct amplitude steps at [0, 73, 541, 753, 1217]ms
├─ Envelope analysis: Beating patterns from 5 overlapping tones
├─ Clock drift: 5 unique drift signatures
└─ Pattern correlation: 5 mutations of base Pattern 5

Model separates: 3-4 kernels decoded (60-80% success)
Lost: 1-2 kernels (retry next beacon cycle)

Total time: 2 seconds (vs 5 × 5s = 25s if sequential!)
Efficiency: 10× improvement via overlap tolerance
```

## Shannon Limit Compliance

**4-FSK channel bandwidth and capacity:**

```
Bandwidth: ~2000 Hz (span from 378 to 2253 Hz, beacon frequencies)
SNR: 0 dB (typical for control channel)
Shannon capacity: 2000 Hz × log₂(1 + 1) = 2000 Hz × 1 = 2000 bps

Current usage:
├─ 50 beacons/min: 50 × 1.3s × 12.5 bps / 60s = 13.5 bps avg
├─ 10 kernel exchanges/min: 10 × 5s × 12.5 bps / 60s = 10.4 bps avg
├─ 20 anti-kernels/min: 20 × 5s × 12.5 bps / 60s = 20.8 bps avg
└─ Total: 44.7 bps average

Efficiency: 44.7 / 2000 = 2.2% of Shannon

With 10 simultaneous: 10 × 12.5 bps = 125 bps peak
Peak efficiency: 125 / 2000 = 6.25% of Shannon ✓
```

**Intentionally inefficient** (6% Shannon) for robustness:
- Wide frequency span (poor spectral efficiency)
- Long symbols (low symbol rate)
- Conservative modulation (4-FSK, not complex QAM)
- Enables -22 dB operation (excellent weak-signal performance)

**Trade-off accepted**: Control channel prioritizes robustness over efficiency.

## Training for Overlapping Signals

**Model trained on asynchronous overlapping 4-FSK:**

```python
def train_4fsk_overlap_separation():
    """Train model to separate overlapping 4-FSK transmissions"""

    for scenario in training_data:
        # Generate 5-15 overlapping 4-FSK signals
        num_users = random.randint(5, 15)

        signals = []
        for user_id in range(num_users):
            # Each user:
            # - Different pattern mutation
            # - Random microsecond start time
            # - Unique clock drift
            signal = generate_4fsk(
                pattern=mutate_pattern(base=random.choice(64), seed=random),
                start_offset_us=random.randint(0, 2000000),  # 0-2s in microseconds
                clock_drift_ppm=random.uniform(-50, 50)
            )
            signals.append(signal)

        # Mix all signals (simulate radio reception)
        mixed = sum_with_sample_alignment(signals)  # 48 kHz aligned

        # Add noise
        mixed_with_noise = mixed + generate_noise(snr=-5)  # Challenging SNR

        # Train model to separate
        separated = model.separate_4fsk(mixed_with_noise)

        # Loss: How many users correctly decoded?
        decode_accuracy = compare(separated, ground_truth_signals)

        # Acceptable: 50-70% decode rate for 10+ overlapping
        # Perfect: 90%+ for 5 overlapping
        loss = 1.0 - decode_accuracy
        optimizer.step(loss)
```

**Model learns**:
- Microsecond onset timing differences
- Envelope beating patterns (constructive/destructive interference)
- Clock drift signatures (unique per radio)
- Pattern correlation (base + mutations)

## Dedicated Slots for Target Kernels

**Beacon structure with target slot:**

```
Beacon cycle (5 seconds total):

0.0-1.3s: Beacon transmission
          ├─ Callsign hash (16 bits)
          ├─ Emergency flag (1 bit)
          └─ Ideal 4-FSK kernel (32 bits)

1.3-3.3s: DEDICATED TARGET SLOT (2 seconds)
          ├─ I listen for messages TO ME
          ├─ Priority: Target kernels, QSO requests, directed messages
          ├─ Model expects: Mutations of my base pattern
          ├─ 10-15 stations can transmit simultaneously
          └─ Decode rate: 50-70% (overlapping, best-effort)

3.3-5.0s: General activity slot
          ├─ Anti-kernel broadcasts
          ├─ General coordination
          └─ Lower priority decode
```

**Stations use target slots:**
```python
# When to transmit kernel to W2DEF:
w2def_beacon_heard_at = 10.5s
dedicated_slot = (10.5 + 1.3, 10.5 + 3.3)  # 11.8s to 13.8s

# Transmit within this window
my_transmit_time = 11.8 + random(0, 2.0)  # Random within slot

transmit_4fsk(
    my_kernel_for_w2def,
    pattern=mutate(w2def.ideal_kernel.base_pattern, my_random),
    start=my_transmit_time
)
```

## Anti-Kernel Best-Effort Protocol

**Anti-kernels allow loss** (advisory, not critical):

```python
# Interference detected - broadcast anti-kernel
def broadcast_anti_kernel(interferers):
    """Best-effort anti-kernel broadcast (loss acceptable)"""

    anti_kernel_msg = {
        'from': my_hash,
        'type': 'ANTI_KERNEL',
        'interferers': [  # Top 3 interferers
            (hash_K5XYZ, level=0.4),
            (hash_N7ABC, level=0.25),
            (hash_W1MNO, level=0.15)
        ],
        'my_anti_kernel': my_kernel  # How to avoid interfering with me
    }
    # Total: ~80 bits

    # Transmit on 4-FSK (no dedicated slot, overlaps OK)
    transmit_4fsk_broadcast(
        anti_kernel_msg,
        pattern=my_assigned_pattern,
        start_time=now() + random(0, 1000),  # Random timing
        no_ack=True,          # Don't expect confirmation
        no_retry=True         # Don't retry if lost
    )

# Interferers:
# - Some hear (50-70%), adapt
# - Some don't hear, keep interfering
# - Retry next minute (eventually converges)
```

**Acceptable loss because:**
- Anti-kernels are optimization hints (not required for operation)
- Retry is automatic (next broadcast)
- Interference is tolerable (just reduces efficiency, doesn't break protocol)
- Statistical delivery over time (most get through within 2-3 minutes)

## Three-Round Kernel Protocol

**Complete kernel lifecycle with anti-kernel feedback:**

```markdown
**Round 1: Initial transmission (message patterns)**
Station A transmits message → Uses current kernel (or default)

**Round 2: Anti-kernel feedback (4-FSK, best-effort, overlapping)**
Stations B, C, D broadcast anti-kernels → "A is interfering"
(Transmitted simultaneously, 4-FSK, 50-70% delivery)

**Round 3: Adapted kernel (4-FSK, in A's next beacon or dedicated update)**
Station A broadcasts adapted kernel → Incorporates heard anti-kernels
Stations update: "A's new kernel reduces interference to B and C"

**Ongoing: Optimized communication**
Network uses A's adapted kernel (less interference)
Kernel converges over 2-3 cycles (minutes)
```

## Capacity Analysis

**With pattern mutations and microsecond timing:**

```
Virtual 4-FSK capacity:
├─ 64 base patterns
├─ ~10 mutations per base (limited by tolerance)
├─ ~100 microsecond timing slots per 2s window
└─ Total: 64 × 10 × 100 = 64,000 virtual slots!

Realistic (accounting for interference):
├─ 10-15 simultaneous transmissions decodable
├─ Model separation: -10 to -15 dB (vs -30 dB pure orthogonal)
├─ Decode success: 50-70% with 10 overlapping
└─ Good enough for control channel (not data)

Effective 4-FSK capacity: ~150 concurrent transmissions per minute
(10 simultaneous × beacon cycle)
```

## Use Case Summary

| Use Case | Delivery | Overlaps | Success Rate | Protocol |
|----------|----------|----------|--------------|----------|
| Beacons | Best-effort | Yes (48 beacon patterns) | 90%+ | No ACK, retry next minute |
| Target kernels | Dedicated slot | Yes (10-15) | 50-70% | No ACK, retry next beacon |
| Anti-kernels | Best-effort | Yes (unlimited) | 50-70% | No ACK, statistical delivery |
| Emergency | Guaranteed | Minimal (4-FSK, reserved) | 95%+ | 4-tone detect, includes grid |

**4-FSK intentionally accepts loss** on non-critical traffic (anti-kernels, some target kernels) to maximize throughput and minimize coordination.

## See Also

- **[Signal Specification](signal_specification.md)** - 4-FSK physical layer details
- **[Net Operations](net_operations.md)** - How nets use 4-FSK for coordination
- **[Model/Signal Expert](../model/experts.md)** - How model separates overlapping signals
