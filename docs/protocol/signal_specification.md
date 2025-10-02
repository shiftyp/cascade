# CASCADE Signal Specification

This document defines the physical layer signal characteristics that all CASCADE implementations must follow for interoperability.

## Overview

CASCADE uses fixed orthogonal patterns with adaptive modulation to achieve near-Shannon efficiency across heterogeneous hardware deployments. The protocol layer defines 64 mathematically orthogonal time-frequency patterns, while the model layer adaptively modulates within these patterns based on channel conditions.

## Base Signal Parameters

### Symbol Timing

**Base symbol duration**: 50ms (20 symbols/second)

**Rationale**:
- Exceeds HF multipath delay spread (1-5ms) by 10× margin
- Allows filter settling time (5-10ms)
- Provides margin for Doppler spread (±2 Hz = 10% of 20 Hz symbol rate)
- Compatible with standard amateur radio sound cards (48 kHz sample rate)
- Sufficient samples per symbol: 48000 × 0.05 = 2,400 samples

**Model adaptation**: The model may stretch or compress symbols within ±20% (40-60ms) based on channel conditions, but 50ms is the nominal baseline.

### Pattern Structure

**Symbols per pattern**: 32 symbols
**Pattern duration**: 32 × 50ms = 1.6 seconds

**Pattern definition**:
Each of the 64 patterns is defined as a sequence of tone indices over time:

```python
pattern_structure = {
    'pattern_id': int,           # 0-63
    'sequence': [int] * 32,      # 32 symbols, each 0-7 (tone index)
    'orthogonality': float       # <-30 dB vs all other patterns
}

# Example:
pattern_0 = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, ...]  # 32 tone indices
pattern_1 = [7, 6, 5, 4, 3, 2, 1, 0, 7, 6, ...]  # Different sequence
```

**Orthogonality requirement**: <-30 dB cross-correlation between all pattern pairs ensures multi-user separation.

**Pattern generation**: Walsh-Hadamard sequences or similar mathematical construction (deterministic, not learned).

### Modulation

**Base modulation**: 8-QAM constellation (8 points in IQ plane)

**Constellation points (ideal, high SNR)**:
```python
# 8-QAM constellation (3 bits per symbol)
constellation_8qam = [
    complex( 1.0,  1.0),  # 000
    complex( 1.0, -1.0),  # 001
    complex(-1.0,  1.0),  # 010
    complex(-1.0, -1.0),  # 011
    complex( 0.0,  1.5),  # 100
    complex( 0.0, -1.5),  # 101
    complex( 1.5,  0.0),  # 110
    complex(-1.5,  0.0),  # 111
]
```

**Model-driven adaptation**: The model continuously adapts constellation point positions based on:
- Channel SNR
- Interference from other users
- Multipath and fading conditions
- Hardware capabilities

**Constellation collapse hierarchy**:
- **High SNR (>10 dB)**: Full 8-QAM (3 bits/symbol)
- **Medium SNR (0-10 dB)**: Collapse toward QPSK (2 bits/symbol)
- **Low SNR (-10-0 dB)**: Collapse toward BPSK (1 bit/symbol)
- **Very low SNR (<-10 dB)**: Collapse to single tone, heavy repetition

The collapse is **continuous** (gradual point movement in IQ space), not discrete mode switching.

### Frequency Allocation

**Channel bandwidth**: 2.5 kHz per CASCADE channel
**Tone structure**: 8 tones within the 2.5 kHz bandwidth
**Tone spacing**: ~312 Hz (2500 Hz / 8 tones)

**Multi-user access**: All users transmit simultaneously within the same 2.5 kHz, separated by orthogonal patterns (CDMA-like).

```python
# All 50 users share the same 2.5 kHz:
User 1: Pattern 5 across 2.5 kHz
User 2: Pattern 12 across 2.5 kHz  # Overlapping!
...
User 50: Pattern 58 across 2.5 kHz

# Separated by pattern orthogonality, not frequency
```

## Beacon Protocol (Network Discovery)

CASCADE uses **asynchronous beacons** transmitted on interstitial frequencies between message tones. The model (Signal Expert) decodes beacons along with messages - no protocol-layer decoding required.

### Beacon Signal Parameters

**Frequency allocation** (interstitial channels):
```
Message tones:     0,  312,  625,  937, 1250, 1562, 1875, 2187 Hz (8 tones)

Interstitial tones (6 total):
├─ Normal beacons: 78, 234, 1718, 1953 Hz (4 outer tones, 4-FSK)
└─ Emergency:      468, 1093 Hz (2 inner tones, BPSK)

Visual layout:
0    78   156  234  312       468      625      781      937     1093    1250
M    NB   (gap) NB   M        EM       M       (gap)     M       EM      M   ...

M = Message (8 tones, 8-QAM, 50ms symbols)
NB = Normal Beacon (4 tones, 4-FSK, 160ms symbols)
EM = Emergency Beacon (2 tones, BPSK, 500ms symbols)

Separation:
- Message to normal beacon: 78 Hz minimum
- Message to emergency: 156 Hz minimum
- Normal to emergency: 234 Hz minimum
Interference: <-25 dB (adequate separation with different symbol rates)
```

**Normal Beacon Specification:**
```python
NORMAL_BEACON_SPEC = {
    # Symbol parameters
    'symbol_duration': 160,        # ms (FT8-style)
    'symbol_rate': 6.25,           # symbols/second
    'modulation': '4-FSK',         # 2 bits/symbol
    'tones': [78, 234, 1718, 1953], # 4 outer interstitial tones
    'tone_spacing': 156,           # Hz between beacon tones
    'bandwidth': 25,               # Hz per tone

    # Beacon content and FEC
    'payload': 16,                 # bits (callsign hash only)
    'fec': 'LDPC rate 1/2',        # Forward error correction
    'coded_bits': 32,              # bits (16 data + 16 parity)
    'throughput': 6.25,            # bps effective (with FEC)
    'duration': 2.56,              # seconds (16 symbols for 32 coded bits)

    # Transmission strategy
    'repetitions': 3,              # 3× per minute
    'timing': 'random',            # Async (no slots)
    'patterns': 'any',             # Use any of 64 patterns
    'power': 'normal',             # User's typical power

    # Performance
    'min_snr': -22,                # dB (with 3× integration)
    'range_100w': 24000,           # km (worldwide DX)
}
```

**Emergency Beacon Specification:**
```python
EMERGENCY_BEACON_SPEC = {
    # Symbol parameters (more robust than normal)
    'symbol_duration': 500,        # ms (3× longer for better SNR)
    'symbol_rate': 2,              # symbols/second
    'modulation': 'BPSK',          # 1 bit/symbol (maximum robustness)
    'tones': [468, 1093],          # 2 inner interstitial tones (RESERVED)
    'tone_spacing': 625,           # Hz between emergency tones
    'bandwidth': 10,               # Hz per tone (very narrow)

    # Beacon content and FEC
    'payload': 16,                 # bits (callsign hash only, emergency implicit from frequency)
    'fec': 'LDPC rate 1/3',        # Heavy FEC for maximum robustness
    'coded_bits': 48,              # bits (16 data + 32 parity)
    'throughput': 1.33,            # bps effective (with FEC)
    'duration': 12,                # seconds (24 symbols for 48 coded bits)

    # Transmission strategy
    'repetitions': 6,              # 6× per minute (2× normal beacon rate)
    'timing': 'every_10_seconds',  # Regular (not random - predictable)
    'patterns': 'any',             # Use any of 64 patterns
    'power': 'MAXIMUM',            # Full legal power (1.5 kW if available)

    # Performance
    'min_snr': -28,                # dB (BPSK + long symbols + repetition)
    'range_100w': 40000,           # km (global coverage)
    'priority': 'CRITICAL',        # Always decoded first

    # Spectrum reservation
    'reserved_frequencies': True,  # 468, 1093 Hz not used by messages/normal beacons
}
```

**Spectrum layout with emergency:**
```
Frequency (Hz):
0    78   156  234  312       468*      625      781      937     1093*    1250
M    NB   gap  NB   M         EM        M        gap      M       EM       M

M = Message (8 tones, 8-QAM)
NB = Normal beacon (4 tones, 4-FSK)
EM = Emergency (2 tones, BPSK) ← RESERVED, always monitored

* = Frequencies 468 and 1093 Hz are RESERVED for emergency only
```

### Detection Guarantee

**All stations (including RPi) monitor emergency frequencies:**

```python
def continuous_emergency_monitor():
    """Simple emergency detector (runs in parallel, low overhead)"""

    while True:
        # Check ONLY emergency frequencies (cheap!)
        signal_468 = bandpass_filter(received, center=468, bw=10)
        signal_1093 = bandpass_filter(received, center=1093, bw=10)

        # Simple BPSK detector (not full model inference)
        emergency_detected = detect_bpsk_tone(signal_468) or detect_bpsk_tone(signal_1093)

        if emergency_detected:
            # Alert user immediately
            # Full decode via model (priority override)
            emergency_beacon = model.decode_emergency(signal_468 + signal_1093)
            SOUND_ALARM()
            display_emergency(emergency_beacon.callsign)
```

**Overhead**: ~0.5ms per check (simple tone detection, not full model)
**Always runs** regardless of capacity limits

### Training with Reserved Emergency Frequencies

```python
def train_emergency_beacon_detection():
    """Train with guaranteed emergency channel"""

    for scenario in training_data:
        # Generate mixed scenario
        messages = generate_messages(50)  # High traffic

        # Add emergency beacon on RESERVED frequencies
        if random.random() < 0.1:  # 10% of scenarios have emergency
            emergency_beacon = generate_emergency_beacon(
                frequencies=[468, 1093],  # Reserved!
                symbol_duration=500,
                modulation='BPSK'
            )
            messages.append(emergency_beacon)

        mixed = sum(messages) + noise

        # Train model
        decoded = model.decode(mixed)

        # Critical: Emergency beacon MUST be in output
        if has_emergency_beacon(ground_truth) and not has_emergency_beacon(decoded):
            # Catastrophic failure
            loss = 10000.0  # Force model to fix this
        else:
            # Normal loss
            loss = standard_loss(decoded, ground_truth)

            # Still weight emergency higher
            if has_emergency_beacon(decoded):
                emergency_accuracy = check_emergency_correct(decoded, ground_truth)
                loss += 10.0 * (1.0 - emergency_accuracy)  # 10× weight

        optimizer.step(loss)
```

### Benefits Summary

**With 2 reserved emergency tones ([468, 1093 Hz]):**

✅ **Guaranteed detection**: Even RPi at full capacity monitors these frequencies
✅ **Simple detector**: Tone detection (0.5ms), not full model inference
✅ **Maximum robustness**: BPSK, 500ms symbols, -28 dB capable
✅ **Global range**: 40,000 km at 100W (reaches anywhere)
✅ **6× repetition**: Every 10 seconds (vs 20 seconds for normal beacons)
✅ **Training priority**: 10× loss weight + catastrophic miss penalty
✅ **Still have 4 tones** for normal beacons (12.5 bps maintained)

**Beacon transmission schedule:**
```python
def beacon_strategy(my_callsign, emergency_mode=False):
    """Dual beacon protocol: normal and emergency"""

    while operating:
        if emergency_mode:
            # EMERGENCY BEACONS (6× per minute, every 10 seconds)
            for i in range(6):
                transmit_emergency_beacon(
                    callsign_hash=hash(my_callsign, 16),  # 16 bits
                    frequencies=[468, 1093],               # RESERVED emergency tones
                    symbol_duration=500,                   # ms (robust)
                    modulation='BPSK',                     # Maximum robustness
                    power='MAXIMUM',                       # Full legal power
                    duration=4                             # seconds
                )
                sleep(10)  # Every 10 seconds

        else:
            # NORMAL BEACONS (3× per minute, random timing)
            for i in range(3):
                # Random timing (0-60 seconds)
                delay = random.uniform(0, 60)
                sleep(delay)

                # Random pattern from assigned pool
                pattern = random.choice(my_assigned_patterns)

                # Transmit on normal beacon frequencies (outer 4 tones)
                transmit_beacon(
                    callsign_hash=hash(my_callsign, 16),  # 16 bits
                    frequencies=[78, 234, 1718, 1953],    # Normal beacon tones
                    pattern=pattern,
                    symbol_duration=160,                  # ms
                    modulation='4-FSK',
                    duration=1.28                         # seconds
                )

        # Repeat every minute
        sleep_until_next_minute()
```

### Model Decodes Beacons (Not Protocol)

**Signal Expert handles both beacon types:**

```python
class SignalExpert:
    def separate_all_signals(self, mixed_signal):
        """
        Separate messages, normal beacons, and emergency beacons

        Model learns to recognize by frequency and symbol characteristics:
        - Messages: 50ms symbols, 8-QAM, [0, 312, 625, 937, 1250, 1562, 1875, 2187]
        - Normal beacons: 160ms symbols, 4-FSK, [78, 234, 1718, 1953]
        - Emergency beacons: 500ms symbols, BPSK, [468, 1093] ← Reserved frequencies
        """

        # Single inference separates everything
        separated = self.forward(mixed_signal)

        return [
            {'type': 'message', 'pattern': 5, 'data': bytes},
            {'type': 'beacon', 'callsign_hash': 0xA3F2, 'emergency': False, 'snr': -12},
            {'type': 'beacon', 'callsign_hash': 0x7F3D, 'emergency': True, 'snr': -20},  # ← On 468/1093 Hz
            {'type': 'message', 'pattern': 12, 'data': bytes},
        ]

# Emergency status inferred from frequency:
# - Detected on [468, 1093] → emergency=True
# - Detected on [78, 234, 1718, 1953] → emergency=False
```

**Protocol layer routing:**
```python
# Protocol receives decoded items from model
decoded_items = model.decode(received_signal)

for item in decoded_items:
    if item['type'] == 'beacon':
        # Emergency status inferred from frequency (model detected on [468, 1093])
        if item['emergency']:
            # EMERGENCY BEACON DETECTED
            SOUND_ALARM()
            emergency_cache[item['callsign_hash']] = {
                'activated': now(),
                'snr': item['snr'],
                'priority': 'CRITICAL'
            }
        else:
            # Normal beacon
            beacon_cache[item['callsign_hash']] = {
                'last_seen': now(),
                'pattern': item['pattern'],
                'snr': item['snr']
            }

    elif item['type'] == 'message':
        # Route message
        route_to_handler(item)
```

## ACK Protocol for Beacons

Beacon ACKs are transmitted on **message patterns** (not beacon frequencies) to keep interstitial channels clear for continuous beacon transmission.

### Normal Beacon ACK

**After receiving normal beacon:**

```python
# Station B heard Station A's normal beacon at measured SNR
beacon_received = {
    'callsign_hash': hash_A,
    'frequency': [78, 234, 1718, 1953],  # Normal beacon tones
    'measured_snr': +8                    # dB
}

# Generate ACK (transmitted on message patterns, NOT beacon frequencies)
normal_beacon_ack = {
    'beacon_hash': hash_A,        # 16 bits (which beacon we're ACKing)
    'my_call': hash_B,            # 24 bits (who I am)
    'snr_report': +8              # 4 bits (how well I heard)
}
# Total: 44 bits

# ACK transmission (SNR-adaptive, on MESSAGE patterns):
if measured_snr > 0:
    # Strong signal - fast ACK on message frequencies
    transmit_ack(
        payload=44 bits,
        frequencies=[0, 312, 625, 937, 1250, 1562, 1875, 2187],  # Message tones!
        modulation='8-QAM',
        symbol_duration=50,  # ms (message symbol rate)
        duration=0.09        # seconds
    )
elif measured_snr > -10:
    # Fair signal - medium ACK
    transmit_ack(
        payload=44 bits,
        frequencies=[0, 312, 625, 937, 1250, 1562, 1875, 2187],
        modulation='4-FSK',
        symbol_duration=160,  # ms
        duration=3.5          # seconds
    )
else:
    # Weak signal - slow ACK
    transmit_ack(
        payload=44 bits,
        frequencies=[0, 312, 625, 937, 1250, 1562, 1875, 2187],
        modulation='BPSK',
        symbol_duration=500,  # ms
        duration=22           # seconds
    )

# QSO established: Beacon + ACK = valid contact
```

### Emergency Beacon ACK (Enhanced)

**After receiving emergency beacon:**

```python
# Station B heard emergency beacon on [468, 1093] Hz
emergency_beacon_received = {
    'callsign_hash': hash_A,
    'frequency': [468, 1093],        # RESERVED emergency tones
    'measured_snr': -20,             # dB (weak emergency)
    'emergency': True                # Inferred from frequency
}

# Generate emergency ACK (more information than normal ACK)
emergency_ack = {
    'type': 'EMERGENCY_ACK',
    'beacon_hash': hash_A,           # 16 bits (which emergency beacon)
    'my_call': hash_B,               # 24 bits (who I am)
    'my_grid': grid_4char,           # 12 bits (rough location for coordination)
    'can_relay': True,               # 1 bit (can I relay messages?)
    'snr_report': -20                # 7 bits (fine-grained for emergency: -28 to +4 dB)
}
# Total: 60 bits

# CRITICAL: ACK on MESSAGE patterns (NOT emergency frequencies!)
# Keeps [468, 1093] Hz clear for continuous emergency beacon transmission
transmit_ack(
    payload=60 bits,
    frequencies=[0, 312, 625, 937, 1250, 1562, 1875, 2187],  # Message tones
    modulation='QPSK',               # Conservative but faster than BPSK
    symbol_duration=160,             # ms (emergency ACKs use standard timing)
    pattern=random_available,        # Any message pattern
    duration=2.4                     # seconds (60 bits @ 25 bps for QPSK)
)

# Multiple stations can ACK emergency beacon:
# - All use message patterns (lots of capacity)
# - No congestion on emergency frequencies
# - Emergency station monitors message patterns for ACKs
```

**Why ACKs use message patterns:**

✅ **Keeps emergency channel clear**: [468, 1093] Hz only for outbound emergency beacons
✅ **Higher capacity**: Message patterns support 50+ concurrent ACKs
✅ **Faster ACKs**: Can use faster modulation (QPSK/8-QAM vs emergency's BPSK)
✅ **No interference**: Emergency beacons continue uninterrupted

### ACK Reception by Beacon Sender

**Station A (sent beacon) monitors for ACKs:**

```python
# After sending normal beacon on [78, 234, 1718, 1953]:
# Monitor message patterns for ACKs (not beacon frequencies)

ack_window = 5  # seconds after beacon transmission
acks_received = []

for t in range(ack_window):
    # Decode message patterns (normal CASCADE message decode)
    decoded = model.decode(received_on_message_patterns)

    for item in decoded:
        if item['type'] == 'beacon_ack' and item['beacon_hash'] == my_hash:
            acks_received.append(item)

# Process ACKs:
# - Update beacon cache with responders
# - Note SNR reports
# - Establish kernel exchange if SNR > -10 dB
```

**Emergency beacon sender:**
```python
# After sending emergency beacon on [468, 1093]:
# Continue emergency beacons every 10 seconds
# WHILE ALSO monitoring message patterns for ACKs

emergency_acks = []

# Background process:
while emergency_mode:
    # Decode message patterns for emergency ACKs
    decoded = model.decode(message_patterns)

    for item in decoded:
        if item['type'] == 'emergency_ack':
            emergency_acks.append(item)
            # Log: Station B (grid FN42) heard me, can relay
            # Use for coordination

    # Continue emergency beacons (doesn't stop for ACKs)
    if time_for_next_beacon():
        transmit_emergency_beacon([468, 1093])
```

### Training with Interstitial Beacons

```python
def train_with_beacons():
    """Train Signal Expert to handle messages + beacons"""

    for batch in training_data:
        # Generate mixed scenario
        num_messages = random.randint(5, 50)
        num_beacons = random.randint(1, 10)  # Async beacons

        # Create signals
        message_signals = generate_messages(
            count=num_messages,
            tones=[0, 312, 625, 937, 1250, 1562, 1875, 2187],
            symbol_duration=50
        )

        beacon_signals = generate_beacons(
            count=num_beacons,
            tones=[156, 468, 781, 1093],  # Interstitial!
            symbol_duration=160
        )

        # Mix together (overlap in time and frequency)
        mixed = message_signals + beacon_signals + noise

        # Train model to separate
        decoded = signal_expert(mixed)

        # Loss: correctly identify all messages AND beacons
        loss = separation_loss(decoded, ground_truth_messages + ground_truth_beacons)
```

## Pattern Storage (Unchanged)

```python
# Minimal storage (tone sequence only)
pattern = {
    'id': int,                    # 0-63
    'sequence': np.uint8[32],     # 32 bytes (tone indices 0-7)
}

# Per pattern: ~256 bytes
# Total 64 patterns: ~16 KB
```

## Interoperability Requirements

**All CASCADE implementations must:**

1. **Use identical patterns**: The 64 orthogonal sequences are fixed and identical across all deployments
2. **Support variable modulation**: Decode 8-QAM, QPSK, BPSK constellations (model adapts continuously)
3. **Handle mixed modulations**: User A's 8-QAM must coexist with User B's BPSK (different SNR conditions)
4. **Maintain timing**: 50ms ±10ms symbol timing tolerance
5. **Preserve pattern orthogonality**: Cross-correlation must remain <-30 dB even as constellations adapt

**Critical**: Patterns are **protocol-defined** (fixed), constellations are **model-optimized** (variable). The fixed patterns ensure interoperability while adaptive constellations approach Shannon efficiency.

## Signal Processing Requirements

**Minimum receiver capabilities**:
- Sample rate: 48 kHz (standard amateur radio sound card)
- Bit depth: 16-bit (standard)
- Frequency stability: ±50 Hz maximum (GPS-locked preferred for >20 users)
- Processing: Pattern correlation + 8-QAM demodulation per 50ms

**Sound card compatibility**:
- Buffer size: 128-2048 samples (supports 50ms symbols)
- Latency: 10-20ms typical (acceptable for 50ms symbols)
- Interface: USB, built-in, or SignaLink/Digirig

## Throughput Calculations

**Single user, high SNR (+15 dB):**
```
Symbol rate: 20 symbols/sec
8-QAM: 3 bits/symbol
Tones: 8
Raw rate: 20 × 3 × 8 = 480 bps
With rate 0.5 FEC: 240 bps effective
```

**Multi-user (50 users, all high SNR):**
```
50 users × 480 bps = 24,000 bps raw aggregate
Shannon (2.5 kHz @ +15 dB): 12,575 bps
Required FEC rate: 0.52
Effective: 12,000 bps (95% Shannon efficiency)
```

**Hardware-limited scenarios** (e.g., Raspberry Pi 4 decoding 15 users):
```
15 users × 480 bps = 7,200 bps aggregate to this station
Other 35 users not decoded (insufficient hardware capacity)
Shannon efficiency for this receiver: 27% (limited by hardware, not protocol)
```

## See Also

- **[Hardware Requirements](../deployment/hardware_requirements.md)** - Deployment tiers and capabilities
- **[Protocol Overview](README.md)** - Protocol layer responsibilities
- **[Model Adaptation](../model/README.md)** - How model optimizes within protocol constraints
- **[Pattern Details](../model/patterns.md)** - 64 orthogonal pattern specifications
