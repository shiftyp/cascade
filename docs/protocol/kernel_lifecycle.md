# CASCADE Kernel Lifecycle Protocol

Kernels are **receiver-optimized hints** that evolve through a unified exchange protocol incorporating message context, retry coordination, and antikernel interference avoidance.

---

## Table of Contents

1. [Kernel Semantics](#kernel-semantics-rx-optimization)
2. [Unified KERNEL_EXCHANGE Message](#unified-kernel_exchange-message)
3. [Three-Round Exchange Protocol](#three-round-exchange-protocol)
4. [Antikernel Pileup Handling](#antikernel-pileup-handling)
5. [Retry via Kernel Exchange](#retry-via-kernel-exchange)
6. [Kernel Refresh Mechanisms](#kernel-refresh-mechanisms)
7. [Overhead Analysis](#overhead-analysis)

---

## Kernel Semantics: RX-Optimization

### What Kernels Represent

**Kernels are decoder hints, not encoder hints**:

```python
# Each station generates a kernel for THEIR OWN receiver
my_rx_kernel = model.generate_kernel(
    my_decoder_state=my_neural_state,
    my_hardware='rpi4',
    my_noise_floor=-95,  # dBm
    my_capabilities={
        'max_users': 15,           # Can decode up to 15 simultaneous
        'max_constellation': 'QPSK', # 8-QAM too complex for my hardware
        'preferred_fec': 0.7        # Need strong FEC
    }
)

# I broadcast my_rx_kernel to the network
# Others use my_rx_kernel when transmitting TO ME
# Kernel tells them: "Here's how to encode so I can decode easily"
```

**Direction of use**:
```python
kernel_usage = {
    'generator': 'Station A (receiver)',
    'kernel_represents': "A's decoder preferences/capabilities",
    'broadcasted_to': 'Network (all stations)',
    'used_by': 'Any station transmitting TO Station A',
    'purpose': 'Optimize encoding for A's receiver',
    'optimization': 'RX-side (not TX-side)'
}
```

### Why RX-Optimization Makes Sense

**Receiver knows their own limitations**:

```python
# Station A: Raspberry Pi 4, noisy urban location, selective fading
a_rx_kernel = {
    'hardware_tier': 'rpi4',
    'noise_floor_dbm': -95,        # Noisy
    'max_simultaneous_users': 15,  # Limited CPU
    'max_patterns_simultaneous': 2,  # Can decode 2 patterns at once
    'preferred_fec_rate': 0.8,     # Heavy FEC needed
    'preferred_constellation': 'QPSK',  # Can't handle 8-QAM
    'available_tones': [0-3],  # All 4 tones usable (run-length encoded)
    'available_tone_count': 4,     # All 4 tones usable
}

# Station B: x86 desktop, quiet rural location, excellent propagation
b_rx_kernel = {
    'hardware_tier': 'x86',
    'noise_floor_dbm': -110,       # Quiet
    'max_simultaneous_users': 50,  # Powerful CPU
    'max_patterns_simultaneous': 4,  # Can decode 4 patterns at once
    'preferred_fec_rate': 0.5,     # Light FEC sufficient
    'preferred_constellation': '16QAM',  # Can handle complex
    'available_tones': [0-3],      # All 4 tones usable
    'available_tone_count': 4,     # Perfect propagation
}

# When B transmits to A:
# - B uses a_rx_kernel as hints
# - Encodes with QPSK, FEC 0.8
# - Uses only 2 patterns (A can decode 2 max)
# - Avoids tones 35-39 (A has QRM there)
# - Selects from A's 65 available tones only
# - A can decode despite limitations
# - Throughput: 2 patterns × 40 bps = 80 bps

# When A transmits to B:
# - A uses b_rx_kernel as hints
# - Encodes with 16-APSK, FEC 0.5
# - Uses 4 patterns (B can decode 4)
# - Can use all 78 tones (B has excellent propagation)
# - B decodes easily with powerful hardware
# - Throughput: 4 patterns × 80 bps = 320 bps
# - Higher throughput A→B than B→A (asymmetric, natural)
```

**Asymmetric links are natural**:
- A→B might use 8-QAM (B has good hardware)
- B→A might use QPSK (A has limited hardware)
- Both directions work optimally for receiver capabilities

---

## Available Tone Subset Encoding

### Kernel Encodes Which Tones Receiver Can Decode

**Critical innovation**: Each receiver measures and announces which of 78 discrete reference tones are usable at their location:

```python
def measure_and_encode_available_tones():
    """
    Receiver measures SNR and interference at each discrete tone
    Encodes availability into kernel (40 bits of 64-bit kernel)
    """

    available_tones = []

    # Measure each of 78 discrete reference tones
    for tone_idx in range(78):
        freq_hz = REFERENCE_TONES[tone_idx]

        # Measure SNR at this exact frequency
        snr_db = measure_snr_at_frequency(freq_hz)

        # Check for local interference
        qrm = detect_local_qrm(freq_hz)

        # Tone is available if decodable
        if snr_db > -10 and not qrm:
            available_tones.append(tone_idx)

    # Examples of availability patterns:
    # Excellent: [0-77] (all 78 tones) → 1 range
    # Selective fading: [0-34, 40-69] (60 tones, gap at 35-39) → 2 ranges
    # Heavy QRM: [5-12, 25-35, 50-69] (34 tones) → 3 ranges
    # Extreme: [10, 25, 40, 55] (4 tones only) → 4 single-tone ranges

    # Encode using run-length encoding
    encoded_40bit = run_length_encode_tones(available_tones)

    return encoded_40bit  # Fits in kernel


def run_length_encode_tones(tone_indices):
    """
    Encode available tones as ranges (40 bits)

    Format:
    - 4 bits: Number of ranges (0-15)
    - 36 bits: Up to 4 ranges (9 bits each)
      - Each range: 7-bit start index + 2-bit length_code
        - length_code 00: length=1 (single tone)
        - length_code 01: length in next 7 bits
        - length_code 10: length=remaining tones
        - length_code 11: reserved
    """

    ranges = find_contiguous_ranges(tone_indices)
    # e.g., [0-34, 40-69] → [(0, 35), (40, 30)]

    num_ranges = min(len(ranges), 4)  # Max 4 ranges in 40 bits
    encoded = num_ranges << 36  # First 4 bits

    for i, (start, length) in enumerate(ranges[:4]):
        if length == 1:
            # Single tone
            range_bits = (start << 2) | 0b00
        elif length <= 127:
            # Explicit length
            range_bits = (start << 2) | 0b01
            range_bits = (range_bits << 7) | length
        else:
            # Remaining tones
            range_bits = (start << 2) | 0b10

        encoded |= (range_bits << (i * 9))

    return encoded  # 40 bits

# Examples:
# All tones [0-69]: num_ranges=1, (start=0, length=70)
#   → 0x01 | (0<<2|0b01)<<7|(70) = 0x...
#
# Selective [0-34, 40-69]: num_ranges=2, (0,35), (40,30)
#   → 0x02 | range1 | range2 = 0x...
```

---

## Unified KERNEL_EXCHANGE Message

### Message Format

**Single message type handles all kernel-related operations**:

```python
class KERNEL_EXCHANGE:
    """Unified message for all kernel operations"""

    # Core fields (always present)
    from_station: hash          # 32 bits - who is sending
    my_rx_kernel: bytes         # 64 bits - MY receiver's kernel (includes available tones)
    for_message_id: hash        # 64 bits - which message this relates to
    timestamp: uint32           # 32 bits

    # Optional fields (context-dependent)
    to_station: hash            # 32 bits - specific recipient (omit for broadcast)
    anti_kernel: bytes          # 64 bits - MY antikernel (if experiencing interference)
    adapted_for: list[hash]     # N×16 bits - who I adapted for
    retry_flag: bool            # 1 bit - is this a retry request?
    retry_count: uint8          # 8 bits - attempt number
    retry_reason: enum          # 8 bits - why original failed
    ready: bool                 # 1 bit - ready to receive message
    interference_level: float   # 16 bits - 0.0-1.0 (if anti_kernel present)
    message_priority: enum      # 8 bits - EMERGENCY/HIGH/NORMAL/LOW
```

### Size by Scenario

| Scenario | Fields Used | Size (bytes) | 4-FSK Time |
|----------|-------------|--------------|------------|
| **Initial contact** | from + my_rx_kernel + for_message_id + timestamp + ready | 29 | 4.6 sec |
| **Antikernel feedback** | Above + anti_kernel + interference_level | 37 | 5.9 sec |
| **Kernel adaptation** | from + my_rx_kernel + for_message_id + adapted_for | 33 | 5.3 sec |
| **Retry request** | Above + retry_flag + retry_count + retry_reason | 32 | 5.1 sec |
| **Proactive refresh** | from + my_rx_kernel + for_message_id + ready | 29 | 4.6 sec |

### Field Usage by Scenario

| Scenario | my_rx_kernel | for_message_id | anti_kernel | retry_flag | adapted_for |
|----------|--------------|----------------|-------------|------------|-------------|
| Initial contact | ✅ | ✅ msg_001 | ❌ | ❌ | ❌ |
| Antikernel feedback | ✅ | ✅ msg_002 | ✅ | ❌ | ❌ |
| Kernel adaptation | ✅ | ✅ msg_002 | ❌ | ❌ | ✅ |
| Retry after failure | ✅ | ✅ msg_001 | ❌ | ✅ | ❌ |
| Proactive refresh | ✅ | ✅ ongoing | ❌ | ❌ | ❌ |

---

## Three-Round Exchange Protocol

### Round 1: Initial Exchange (Message Preparation)

**Station A wants to send message, broadcasts RX kernel**:

```python
# A sends: "Here's MY RX kernel, I want to send msg_12345"
a_kernel_exchange = {
    'from': hash_A,
    'my_rx_kernel': kernel_A,           # Hints for decoding transmissions TO A
    'for_message_id': 'msg_12345',      # Message context
    'message_priority': 'NORMAL',
    'message_size_bytes': 128,
    'timestamp': now()
}

transmit_4fsk(a_kernel_exchange)  # ~5 seconds

# Network receives:
# - Stations cache kernel_A: "When I transmit TO A, use kernel_A as hints"
# - Station B (recipient) prepares to receive msg_12345
```

### Round 2: Response with Antikernel (If Needed)

**Station B responds with THEIR RX kernel + optional antikernel**:

```python
# B sends: "Here's MY RX kernel, here's my antikernel if you interfere with me"
b_kernel_exchange = {
    'from': hash_B,
    'my_rx_kernel': kernel_B,           # Hints for decoding transmissions TO B
    'for_message_id': 'msg_12345',      # Same message

    # Optional: If B detects A might interfere
    'anti_kernel': b_anti_kernel,       # What interferes with B's RX
    'interference_level': 0.35,         # 35% interference detected
    'affected_patterns': [12, 15],      # Which of B's patterns affected

    'ready': True,                      # Ready to receive msg_12345
    'timestamp': now()
}

transmit_4fsk(b_kernel_exchange)  # ~6 seconds (includes antikernel)

# A receives:
# - Caches kernel_B: "When I transmit TO B, use kernel_B"
# - Sees B's antikernel: "B's RX is experiencing interference"
# - Needs to adapt A's RX kernel to avoid interfering with B
```

### Round 3: Adaptation (If Antikernel Received)

**Station A adapts THEIR RX kernel to avoid interfering with B**:

```python
# A sends: "Here's my ADAPTED RX kernel, adjusted to avoid your interference"
a_adapted_exchange = {
    'from': hash_A,
    'my_rx_kernel': kernel_A_adapted,   # Updated A's RX kernel
    'for_message_id': 'msg_12345',
    'adapted_for': [hash_B],            # Adapted to avoid B's interference
    'adaptation_type': 'frequency_shift',
    'ready': True,
    'timestamp': now()
}

transmit_4fsk(a_adapted_exchange)  # ~5 seconds

# Adaptation details:
# - A's kernel now includes frequency offset to avoid B's patterns
# - When others transmit TO A, they use adapted kernel
# - A's transmissions use different frequency (less interference to B)

# Network effect:
# - Transmissions TO A shifted +50 Hz
# - Reduces interference TO B's reception
# - A's RX kernel = A's TX behavior (since others use A's kernel when TX to A)
```

### After Exchange: Message Transmission

```python
# All kernels exchanged, antikernels incorporated
# A transmits msg_12345 on message patterns

a_transmits = {
    'message_id': 'msg_12345',
    'payload': b'Hello W2DEF',
    'using_kernel': kernel_B,  # B's RX kernel as hints
    'transmission_time': 2_seconds
}

# B decodes using A's adapted kernel context
# Success rate higher (A adapted to avoid B's interference)
```

---

## Antikernel Pileup Handling

### The Pileup Problem

**One interferer, multiple victims**:

```python
# Station A transmits, causes interference to 5 stations
interfered_stations = ['B', 'C', 'D', 'E', 'F']

# All 5 want to broadcast antikernels simultaneously
# On same 4-FSK channel
# Collision risk: High
```

### Pileup Solution: Hybrid Strategy

**Combine pattern separation + priority delay + jitter**:

```python
class AntikernelPileupHandler:
    """Handle multiple simultaneous antikernel broadcasts"""

    def transmit_antikernel(self, interferer, interference_level):
        """Optimized antikernel transmission with pileup handling"""

        # 1. Deterministic pattern (based on hashes)
        my_pattern = hash(my_station + interferer) % 64
        # Different stations get different patterns
        # Pattern orthogonality provides separation

        # 2. Priority delay (based on interference severity)
        # More interference = shorter delay = transmit first
        max_delay_ms = 2000  # 2 seconds max
        priority_delay = max_delay_ms * (1.0 - interference_level)
        # Examples:
        #   100% interference: 0ms delay
        #   50% interference: 1000ms delay
        #   10% interference: 1800ms delay

        # 3. Random jitter (avoid exact timing collisions)
        jitter = random.randint(0, 200)  # 0-200ms

        # 4. Total delay
        total_delay = priority_delay + jitter
        time.sleep(total_delay / 1000)

        # 5. Transmit on 4-FSK with assigned pattern
        transmit_4fsk(
            antikernel_message,
            pattern=my_pattern,
            allow_overlap=True  # Multiple stations may transmit
        )
```

### Expected Performance

**Success rates by collision count**:

| Simultaneous Antikernels | Pattern Separation | + Priority Delay | + Jitter | Combined |
|--------------------------|--------------------|-----------------|---------:|----------|
| 2 stations | 95% | 98% | 99% | ~99% |
| 3 stations | 85% | 92% | 95% | ~95% |
| 5 stations | 70% | 80% | 85% | ~90% |
| 10 stations | 45% | 60% | 70% | ~75% |

**Convergence with losses**:

```python
convergence_simulation = {
    # 5 stations experiencing interference
    'round_1': {
        'transmitted': 5,
        'received_by_interferer': 4.5,  # 90% success (one lost)
        'interferer_adapts_for': 4      # Adapts for 4 stations
    },

    'round_2': {
        'transmitted': 1,  # Only the missed station retransmits
        'received': 1,     # No collision (only one transmitting)
        'interferer_adapts_for': 1  # Adapts for remaining station
    },

    'total_rounds': 2,
    'total_time': 8 + 8,  # = 16 seconds
    'final_coverage': 5,   # All 5 stations accommodated
    'verdict': 'Acceptable - 90% first-round success, 100% after 2 rounds'
}
```

### Pileup Timeline Example

```python
# A causes different interference levels to 5 stations
interference_map = {
    'B': 0.60,  # 60% interference (severe)
    'C': 0.40,  # 40% interference
    'D': 0.30,  # 30% interference
    'E': 0.15,  # 15% interference
    'F': 0.10   # 10% interference
}

# Antikernel transmission timeline
timeline = {
    # Station B (60% interference)
    't=0.0-0.2': 'B: priority_delay = 800ms, jitter = 120ms, starts at 0.92s',
    't=0.92-6.92': 'B transmits antikernel (pattern 23)',

    # Station C (40% interference)
    't=0.0-0.2': 'C: priority_delay = 1200ms, jitter = 87ms, starts at 1.29s',
    't=1.29-7.29': 'C transmits antikernel (pattern 47)',  # Overlaps B

    # Station D (30% interference)
    't=0.0-0.2': 'D: priority_delay = 1400ms, jitter = 156ms, starts at 1.56s',
    't=1.56-7.56': 'D transmits antikernel (pattern 8)',   # Overlaps B, C

    # Station E (15% interference)
    't=0.0-0.2': 'E: priority_delay = 1700ms, jitter = 34ms, starts at 1.73s',
    't=1.73-7.73': 'E transmits antikernel (pattern 61)',  # Overlaps B, C, D

    # Station F (10% interference)
    't=0.0-0.2': 'F: priority_delay = 1800ms, jitter = 193ms, starts at 2.0s',
    't=2.0-8.0': 'F transmits antikernel (pattern 5)',     # Overlaps C, D, E
}

# A receives (uses model multi-user separation):
received = {
    'B': 'Clear (transmitted first)',
    'C': 'Decoded (pattern 47 separated from B's pattern 23)',
    'D': 'Decoded (pattern 8 separated)',
    'E': 'Lost (too many overlaps)',  # 4 simultaneous
    'F': 'Decoded (pattern 5 separated)'
}

# Round 1 result: 4 out of 5 (80%) - acceptable
# E retransmits in Round 2 (no collision)
```

---

## Retry via Kernel Exchange

### Retry as Kernel Refresh

**Key insight**: Retry after multiple failures suggests kernel issues. Combine retry request with kernel refresh:

```python
# After 3 failed attempts on message patterns
# A sends kernel exchange with retry flag

retry_kernel_exchange = {
    'from': hash_A,
    'my_rx_kernel': generate_fresh_kernel(),  # Fresh kernel for MY RX
    'for_message_id': 'msg_12345',            # Which message to retry

    # Retry metadata
    'retry_flag': True,
    'retry_count': 3,                         # 3 previous failures
    'retry_reason': 'multiple_pattern_failures',  # or 'validation_failure', 'timeout'

    'timestamp': now()
}

transmit_4fsk(retry_kernel_exchange)  # ~5 seconds

# B receives:
# 1. Updates A's RX kernel (use when transmitting TO A)
# 2. Sees retry request for msg_12345
# 3. Retransmits msg_12345 using A's FRESH RX kernel
# 4. Fresh kernel might fix the decode issue
```

### Adaptive Retry Strategy

**Fast retries first, kernel refresh fallback**:

```python
class AdaptiveRetryStrategy:
    """Message patterns first, 4-FSK kernel exchange after failures"""

    def __init__(self):
        self.consecutive_failures = {}  # station -> count
        self.fsk_fallback_threshold = 3

    def handle_message_failure(self, station, message_id, failure_type):
        """Choose retry mechanism"""

        # Track failures
        self.consecutive_failures[station] = \
            self.consecutive_failures.get(station, 0) + 1

        failures = self.consecutive_failures[station]

        if failures < self.fsk_fallback_threshold:
            # Fast retry on message patterns (1 second)
            retry_request = {
                'type': 'RETRY_REQUEST',
                'message_id': message_id,
                'retry_count': failures
            }
            transmit_on_patterns(retry_request)
            return 'fast_retry', 1.0  # seconds

        else:
            # 3+ failures - use kernel exchange (robust)
            kernel_exchange = {
                'from': my_hash,
                'my_rx_kernel': generate_fresh_kernel(),
                'for_message_id': message_id,
                'retry_flag': True,
                'retry_count': failures,
                'retry_reason': failure_type
            }
            transmit_4fsk(kernel_exchange)
            return 'kernel_refresh_retry', 5.1  # seconds

    def reset_on_success(self, station):
        """Reset failure count on successful transmission"""
        self.consecutive_failures[station] = 0
```

**Retry flow**:
```
Attempt 1: Message patterns (fast) - 1 second [FAIL]
Attempt 2: Message patterns (fast) - 1 second [FAIL]
Attempt 3: Message patterns (fast) - 1 second [FAIL]
Attempt 4: 4-FSK kernel exchange - 5 seconds [SUCCESS]
           Partner retransmits with fresh kernel

Total: 8 seconds for 4 attempts
```

---

## Kernel Refresh Mechanisms

### Primary: ACK-Piggybacked Refresh (Zero Overhead)

**Every message ACK includes refreshed RX kernel**:

```python
# W2DEF sends message ACK to K0BB
message_ack = {
    'message_id': msg_id,
    'status': 'RECEIVED',
    'snr': +10,

    # Always include fresh RX kernel!
    'refreshed_rx_kernel': generate_fresh_kernel(my_current_state),
    'kernel_age': 0  # Just generated
}

# Transmitted on message patterns (fast, ~0.1s)
# K0BB receives: W2DEF's updated RX kernel
# Zero overhead (would send ACK anyway)
# K0BB uses new kernel when next transmitting TO W2DEF
```

### Secondary: Proactive Kernel Exchange (During Idle)

**If no recent messages, proactively update**:

```python
# Monitor MY issued kernels
for station in my_kernel_cache:
    kernel_age = now() - my_kernel_cache[station].timestamp

    # Approaching expiration?
    if kernel_age > estimated_valid_seconds * 0.8:
        # Proactively send updated RX kernel on 4-FSK
        kernel_update = {
            'from': my_hash,
            'my_rx_kernel': generate_fresh_kernel(),
            'for_message_id': 'ongoing',  # No specific message
            'reason': 'proactive_expiration_management'
        }

        transmit_4fsk(kernel_update)  # ~5s, during idle
```

### Tertiary: Explicit Request (Fallback)

**If kernel expired and no update received**:

```python
# About to transmit to station, check their RX kernel
partner_rx_kernel = kernel_cache[partner_station]

if is_expired(partner_rx_kernel):
    # Request fresh kernel via 4-FSK
    kernel_request = {
        'from': my_hash,
        'to': partner_station,
        'my_rx_kernel': generate_kernel(),  # Include mine too
        'for_message_id': upcoming_message_id,
        'type': 'REQUEST'
    }

    transmit_4fsk(kernel_request)  # ~5s

    # Wait for response
    response = listen_4fsk(timeout=10)

    if response:
        kernel_cache[partner_station] = response.my_rx_kernel
        # Proceed with message
    else:
        # No response - transmit on 4-FSK (slower but works)
        transmit_4fsk(message)
```

---

## Overhead Analysis

### Per-Exchange Overhead

**Message_id field cost**:

```python
message_id_overhead = {
    'field_size': 8_bytes,  # 64-bit message ID
    'transmission_time': 8 * 8 / 50,  # = 1.28 seconds at 50 bps

    'comparison': {
        'without_message_id': '21 bytes = 3.36 seconds',
        'with_message_id': '29 bytes = 4.64 seconds',
        'difference': '+1.28 seconds per exchange'
    }
}
```

### Typical QSO Overhead

**20-message QSO with unified protocol**:

```python
typical_qso_overhead = {
    # Initial kernel exchange
    'initial': {
        'a_to_b': 4.6_seconds,  # A's RX kernel
        'b_to_a': 4.6_seconds,  # B's RX kernel
        'subtotal': 9.2_seconds
    },

    # Message exchanges (fast patterns)
    'messages': {
        'count': 20,
        'time_per': 2_seconds,
        'subtotal': 40_seconds
    },

    # ACKs with kernel refresh (fast patterns)
    'acks': {
        'count': 20,
        'time_per': 0.1_seconds,
        'subtotal': 2_seconds
    },

    # Antikernel cycle (1 per QSO typical)
    'antikernel': {
        'b_antikernel': 5.9_seconds,  # Includes interference report
        'a_adaptation': 5.3_seconds,   # A adapts RX kernel
        'subtotal': 11.2_seconds
    },

    # Retry (1 per QSO typical)
    'retry': {
        'fast_attempts': 3_seconds,   # 3 × 1 second
        'kernel_exchange': 5.1_seconds,
        'subtotal': 8.1_seconds
    },

    # Totals
    'total_time': 70.5_seconds,
    'kernel_exchange_time': 25.4_seconds,
    'kernel_overhead_percent': 36.0,

    # Message_id specific overhead
    'message_id_exchanges': 5,  # Initial (2) + antikernel (2) + retry (1)
    'message_id_overhead': 5 * 1.28,  # = 6.4 seconds
    'message_id_percent': 9.1  # 9.1% of QSO for message_id context
}
```

### Overhead Justification

**Benefits of mandatory message_id**:

Including message_id in all KERNEL_EXCHANGE messages provides:
- **Protocol unification**: One message type instead of 5 separate types
- **Clear context**: Every kernel tied to specific message
- **Simplified retry**: Retry = kernel exchange with retry_flag
- **Better telemetry**: Easy to track message → kernel → outcome correlation
- **Reduced ambiguity**: No "which message is this kernel for?" confusion
- **Natural batching**: Kernel refresh + retry in single transmission

**Costs**:
- Time: +6.4 seconds per typical QSO (9.1% overhead)
- Bandwidth: +8 bytes per exchange

**Verdict**: Include message_id in all KERNEL_EXCHANGE messages. The 9% overhead is justified by major protocol simplification and improved telemetry clarity. Alternative approach (optional message_id) saves ~3% overhead but loses context benefits.

---

## Kernel Convergence

**Kernels improve over multiple rounds**:

```
Minute 0: Station A broadcasts with default RX kernel
          → A's transmissions (using others' RX kernels) cause 35% interference to B, 20% to C

Minute 1: B and C broadcast antikernels (their RX experiencing interference)
          → A receives both via pattern separation (~90% success with pileup handling)

Minute 2: A broadcasts adapted RX kernel v1
          → When others transmit TO A, they use adapted kernel
          → A's RX characteristics now avoid interfering with B, C
          → Interference to B: 35% → 20%, C: 20% → 10%

Minute 3: B broadcasts antikernel (still 20% interference)
          → A receives (single transmission, 100% success)

Minute 4: A broadcasts adapted RX kernel v2
          → Further optimization
          → Interference to B: 20% → 10%, C: 10% → 5%

Minute 5: No new antikernels (interference <10% = acceptable)
          → RX kernels converged

Result: Network self-optimizes over 5 minutes
        Interference reduced from 35% to 10% (71% reduction)
```

---

## Net Operations with RX Kernels

### Net Topology

**Typical net structure**:
- **Net Control Station (NCS)**: Central coordinator, typically powerful hardware
- **Member stations**: Participants, varied hardware capabilities
- **Relay stations**: Bridge members who can't reach NCS directly
- **All stations**: Broadcast their RX kernels during net startup

**Example net**:
- NCS: x86 desktop, can decode 50 simultaneous users
- Members A, B, C: Direct reach to NCS
- Members D, E: Need relay to reach NCS
- Relay R1: Coral TPU, bridges D/E to NCS
- Relay R2: Backup relay for E

### Net Startup: Kernel Collection

**All stations broadcast their RX kernels**:

During net startup, each station broadcasts a KERNEL_EXCHANGE message containing their RX preferences. These broadcasts happen in parallel using pattern separation.

```python
# NCS broadcasts (controller RX kernel)
ncs_kernel = {
    'from': hash('NCS'),
    'my_rx_kernel': kernel_NCS,  # Powerful x86, can handle 8-QAM, 50 users
    'for_message_id': 'net_session_001',
    'net_role': 'controller'
}

# Member A broadcasts (RPi4 RX kernel)
a_kernel = {
    'from': hash('A'),
    'my_rx_kernel': kernel_A,  # Limited hardware, needs QPSK, 15 users max
    'for_message_id': 'net_session_001',
    'net_role': 'member'
}

# Relay R1 broadcasts (Coral RX kernel)
r1_kernel = {
    'from': hash('R1'),
    'my_rx_kernel': kernel_R1,  # Medium hardware, reliable QPSK, 30 users
    'for_message_id': 'net_session_001',
    'net_role': 'relay',
    'relay_capacity': 10
}
```

**Parallel broadcast with pileup handling**:
- 13 stations broadcast simultaneously on 4-FSK
- Pattern separation: Each uses hash-derived pattern (0-63)
- Timing spread: Priority jitter over ~3 seconds
- Model separates: 10-12 out of 13 in first round (90%+ success)
- Round 2: Remaining 1-3 stations re-broadcast (no collision)
- **Total time**: 15-20 seconds for complete net kernel collection

**Result**: All stations cache all RX kernels (13 × 8 bytes = 104 bytes per station). Now everyone knows how to transmit to everyone else optimally.

### Check-In Flow with Relay

**Member D checking in (can't reach NCS directly)**:

Member D transmits check-in on message patterns. Relay R1 hears it, NCS does not.

```python
# D transmits check-in
d_checkin = {
    'from': hash('D'),
    'to': 'broadcast',
    'message': 'D checking in, grid FN42',
    'using_rx_kernel': kernel_broadcast  # Conservative for broadcast
}
```

R1 hears D's check-in (D is within R1's range). R1 decodes successfully using R1's own RX capabilities.

```python
# R1 relays to NCS
r1_relay = {
    'from': hash('R1'),
    'to': hash('NCS'),
    'original_from': hash('D'),
    'message': 'D checking in, grid FN42',
    'message_id': 'checkin_D_001',  # SAME message ID
    'relay_depth': 1,
    'using_rx_kernel': kernel_NCS  # Use NCS's RX kernel!
}
```

R1 re-encodes using NCS's RX kernel as hints. Since NCS has powerful hardware (x86), R1 can use 8-QAM and light FEC for higher throughput on the R1→NCS link.

NCS receives and decodes, now knows D is in the net and reachable via R1.

**NCS acknowledges back to D**:

```python
# NCS sends ack to D
ncs_ack = {
    'from': hash('NCS'),
    'to': hash('D'),
    'message': 'Roger D, you're #5',
    'message_id': 'ack_D_001',
    'relay_via': hash('R1'),  # Request R1 relay
    'using_rx_kernel': kernel_R1  # First hop: NCS → R1
}
```

NCS transmits using R1's RX kernel. R1 hears and decodes (addressed to D, relay_via = R1).

```python
# R1 relays to D
r1_to_d = {
    'from': hash('R1'),
    'to': hash('D'),
    'original_from': hash('NCS'),
    'message': 'Roger D, you're #5',
    'message_id': 'ack_D_001',  # SAME ID
    'relay_depth': 1,
    'using_rx_kernel': kernel_D  # Use D's RX kernel!
}
```

R1 re-encodes using D's RX kernel. Since D has limited hardware (RPi4), R1 uses QPSK and heavy FEC. D receives and decodes successfully.

### Multi-Hop Kernel Optimization

**Each hop uses destination's RX kernel**:

Message from D → R1 → NCS illustrates pairwise optimization:

**Hop 0 (D → R1)**:
- D encodes using kernel_R1 (R1's RX preferences)
- R1's kernel requests QPSK, FEC 0.7 (reliability for relay)
- Transmission time: ~2.5 seconds

**Hop 1 (R1 → NCS)**:
- R1 re-encodes using kernel_NCS (NCS's RX preferences)
- NCS's kernel accepts 8-QAM, FEC 0.5 (powerful hardware)
- Transmission time: ~1.8 seconds
- Higher throughput than D→R1 hop (NCS has better RX)

**Total time**: 4.3 seconds for 2-hop relay vs 2 seconds direct (acceptable overhead for extending range)

### Net Controller Optimization

**NCS uses cached RX kernels to optimize traffic**:

The controller analyzes all member RX kernels to understand capabilities and optimize net operations.

From member A's RX kernel, NCS learns:
- A has RPi4 (limited to 15 simultaneous users)
- A needs QPSK (can't decode 8-QAM reliably)
- A prefers heavy FEC (0.8)
- A has moderate SNR to NCS (-8 dB based on check-in)

From relay R1's RX kernel, NCS learns:
- R1 has Coral TPU (can handle 30 users)
- R1 offers relay capacity for 10 stations
- R1 emphasizes reliability (FEC 0.7, QPSK)
- R1 has good SNR to NCS (+5 dB)

**NCS assigns relay roles**:
- Members D, E: Use R1 as primary relay
- Member E: Use R2 as backup relay
- Direct members A, B, C: No relay needed

**NCS optimizes transmissions**:
- To powerful members: Use 8-QAM, light FEC (fast)
- To limited members: Use QPSK, heavy FEC (reliable)
- Broadcast announcements: Use most conservative kernel (reach everyone)

### Relay Station Kernel Strategy

**Relay R1 optimizes for reliability**:

Relay stations emphasize reliable reception since relay errors cascade (failed relay = failed end-to-end delivery).

R1's RX kernel requests:
- QPSK modulation (not 8-QAM) for reliability
- Heavy FEC (0.7) to minimize decode errors
- Narrow bandwidth (150 Hz) to avoid interference
- High priority for relay traffic

When members transmit TO R1, they use these conservative parameters, ensuring relay rarely fails.

### Antikernel in Net Context

**Member A interferes with member B**:

Both A and B are transmitting on message patterns. A's transmissions cause 30% interference to B's reception.

B broadcasts antikernel on 4-FSK with pileup handling (pattern separation + priority delay). The antikernel includes B's updated RX kernel and reports interference.

A receives B's antikernel and adapts A's own RX kernel. The adaptation shifts A's RX frequency by +50 Hz, which means stations transmitting TO A will shift +50 Hz (they use A's kernel). This frequency shift reduces interference to B's reception.

**Net-wide benefit**: NCS hears the antikernel exchange and updates routing/optimization to account for A-B interference. May route A's traffic through relay to reduce on-air overlap with B.

### Overhead in Net Operations

**15-minute net with 10 members + 2 relays + controller**:

**Startup overhead**:
- Kernel exchange: 15-20 seconds (one-time, parallel broadcasts)
- Check-ins: 36 seconds (10 × 3 seconds, some relayed)
- Total startup: ~55 seconds

**Ongoing overhead**:
- Kernel refreshes: 0 seconds (piggybacked on ACKs)
- Relayed messages: +2 seconds per relay (second hop)
- Antikernel (rare): ~11 seconds if needed

**Total 15-minute net**:
- Startup: 55 seconds
- Messages: 50 seconds (15 direct + 5 relayed)
- Closing: 15 seconds
- Total: ~120 seconds active time in 900-second window
- **Kernel overhead**: 20 seconds / 900 seconds = 2.2%

The kernel exchange overhead is minimal (one-time startup cost), and ongoing operations benefit from optimized per-hop encoding.

---

## Telemetry and Training

### What Telemetry Captures

Kernel exchanges generate telemetry that improves model performance over time.

**TX telemetry** (1040-D neural state):
- Pattern expert (512-D): MY RX kernel generation
- Spectrum expert (512-D): MY RX frequency preferences
- Station fingerprint (16-D): MY equipment characteristics
- Used to improve kernel generation quality

**RX telemetry** (3581-D neural state):
- All five experts active during message reception
- Records which kernel was used (partner's RX kernel from cache)
- Captures decode success and kernel effectiveness
- Used to validate kernel helped encoding

**Kernel lifecycle telemetry**:
- Round 1: TX broadcasts own RX kernel
- Round 2: RX responds with own RX kernel + antikernel if interference
- Round 3: TX adapts own RX kernel based on received antikernels
- Message transmission: TX uses RX's kernel, RX validates effectiveness

**Cross-station correlation**:

Telemetry from both sides of transmission provides ground truth labels. TX telemetry includes estimated SNR and generated RX kernel, while RX telemetry shows measured SNR and whether kernel helped. Correlation via message_id enables supervised learning for kernel generation optimization.

**Training improvements from telemetry**:
- Kernel generation quality: Learn which configurations work for given conditions
- Kernel lifetime prediction: Optimize estimated_valid_seconds field
- Antikernel strategy: Learn when/how to broadcast antikernels
- Kernel adaptation: Improve incorporation of antikernel feedback
- Convergence dynamics: Learn typical rounds-to-convergence (2-3)

See [telemetry_research.md](../../telemetry_research.md) for comprehensive telemetry strategies, cross-station correlation, model retraining decisions, and real-time adaptation techniques.

---

## See Also

- **[Message Validation](message_validation.md)** - CRC32 + xxHash32 dual validation
- **[Message Format](message_format.md)** - Binary wire format specification
- **[Net Operations](net_operations.md)** - Detailed net protocol and coordination
- **[Protocol Overview](README.md)** - Multi-stage protocol flow
- **[Continuous Improvement](../training/continuous_improvement.md)** - Model updates and federated learning
- **[telemetry_research.md](../../telemetry_research.md)** - Kernel telemetry and training

---

*Last updated: 2025-10-02*
