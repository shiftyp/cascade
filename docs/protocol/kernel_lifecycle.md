# CASCADE Kernel Lifecycle (V2)

**Purpose:** How kernels are generated, transmitted, and used for coordination
**Architecture:** Single RX kernel per beacon, TX kernel in RTS/headers

---

## Overview

CASCADE V2 uses **two kernel types**:
1. **RX Kernel**: How to reach this station (in beacons)
2. **TX Kernel**: Current transmission state (in RTS/headers, for anti-collision)

**Kernel provides pattern ID** - eliminates need for blind detection.

---

## RX Kernel Lifecycle

### Generation

**Decoder observes channel and generates RX kernel:**

```python
def generate_rx_kernel(observed_iq_samples):
    """Decoder generates RX kernel from channel observations"""

    # Measure which patterns decode well
    pattern_quality = {}
    for pattern_id in range(8):
        success_rate = measure_decode_success(pattern_id)
        snr = measure_snr(pattern_id)
        pattern_quality[pattern_id] = (success_rate, snr)

    # Select best pattern
    best_pattern = max(pattern_quality, key=lambda p: pattern_quality[p][0])

    # Measure best frequency pairs
    freq_quality = measure_frequency_quality()
    best_freq = select_best_frequency_pair(freq_quality)

    # Determine supported modulation from SNR
    avg_snr = estimate_snr()
    if avg_snr > 20:
        modulation = '16-APSK'
    elif avg_snr > 10:
        modulation = '8-PSK'
    elif avg_snr > 0:
        modulation = 'QPSK'
    else:
        modulation = 'BPSK'

    # NN generates embedding from observed IQ
    embedding = neural_network.generate_embedding(observed_iq_samples)

    return {
        'pattern_id': best_pattern,
        'frequency_pair': best_freq,
        'modulation': modulation,
        'polar_rate': select_polar_rate(avg_snr),
        'embedding': embedding  # 24 bytes, 48 dims × 4-bit
    }
```

### Transmission

**When to transmit RX kernel (in beacon):**
- Calling CQ (every 30-60s until response)
- Active QSO (updates every 60s if conditions change)
- Net participation (per net protocol)

**When NOT to transmit:**
- Idle/listening only
- QSO complete
- Propagation dead (no point)

**Beacon structure:**
```
Payload: RX kernel (28 bytes)
Pattern: 512 symbols @ BPSK
Duration: 2.56s @ 200 sym/s
Frequency: Station picks available pair
```

### Usage by Others

**Other station wants to transmit TO this station:**

```python
# Read RX kernel from beacon
rx_kernel = decode_beacon(station_callsign)

# Use recommended parameters
transmission = {
    'pattern_id': rx_kernel.pattern_id,      # Use pattern 3
    'frequency_pair': rx_kernel.frequency_pair,  # On freq 25
    'modulation': rx_kernel.modulation,      # Use QPSK
    'polar_rate': rx_kernel.polar_rate         # Use rate 3/4
}

# Check not blocked by TX kernels
if not blocked_by_tx_kernel(transmission):
    send_rts(transmission)
```

### Update Triggers

**RX kernel regenerated when:**
- SNR changes >3 dB
- Pattern success rate drops >10%
- Best frequency shifts
- Every 60s minimum (even if stable)

**Update sent in next beacon**

---

## TX Kernel Lifecycle

### Generation

**Protocol generates TX kernel when starting transmission:**

```python
def generate_tx_kernel(my_transmission):
    """Protocol generates TX kernel for anti-collision"""

    return {
        'pattern_id': my_transmission.pattern_id,
        'frequency_pair': my_transmission.frequency_pair,
        'modulation': my_transmission.modulation,
        'polar_rate': my_transmission.polar_rate,
        'embedding': current_nn_state()  # From encoder
    }
```

### Transmission

**TX kernel transmitted in:**

**1. RTS (Request-to-Send):**
```
Payload: TX kernel (28B) + metadata (3B)
Pattern: 512s @ BPSK
Duration: 2.05s
Purpose: Announce transmission intent, request channel
```

**2. Message Header (conditional):**
```
Included when:
- Conditions changed since RTS
- Pattern/frequency switched
- Modulation adapted
- >120s since last kernel

Adds: +2.05s to message (512s @ BPSK overhead)
```

**3. Cleared when idle:**
```
No TX kernel = Not transmitting
Allows others to use that pattern/frequency
```

### Usage by Network

**Other stations check TX kernels before transmitting:**

```python
# Collect all active TX kernels
active_channels = []
for station in network:
    if station.tx_kernel and station.tx_kernel.active:
        active_channels.append((
            station.tx_kernel.pattern_id,
            station.tx_kernel.frequency_pair
        ))

# Before transmitting, verify channel free
my_channel = (my_pattern, my_freq)
if my_channel in active_channels:
    # Collision! Find alternative
    my_pattern = find_free_pattern(my_freq)
    # Or different frequency
    my_freq = find_free_frequency(my_pattern)
```

---

## Complete Kernel Flow Example

### Initial Contact

**Station A (calling CQ):**
```
T=0s: Beacon with RX kernel
      {pattern: 3, freq: 25, mod: QPSK, polar: 3/4, embedding: [...]}
      512s @ BPSK = 2.05s

T=60s: Next beacon (kernel may update if conditions changed)
```

**Station B (responding):**
```
T=5s: Decode A's beacon, extract RX kernel

T=6s: Send RTS with TX kernel
      {pattern: 3, freq: 25, mod: QPSK, polar: 3/4, embedding: [...]}
      + destination: A, request: message
      512s @ BPSK = 2.05s

T=9s: Receive CTS from A
      Session granted
      128s @ BPSK = 0.51s

T=10s: Transmit message (pattern 3, freq 25, QPSK)
       1024s @ QPSK = 4.10s
       TX kernel active during transmission

T=14s: Receive ACK from A
       128s @ BPSK = 0.51s

T=15s: Clear TX kernel (go idle)
```

**Other stations (C, D, E) during this:**
```
T=6-15s: See B's TX kernel
         Avoid pattern 3, freq 25
         Use different combinations
         No collisions!
```

### Collision Scenario with QSY

**Station B and C both trying to contact A:**
```
T=0s:  Station A beacons RX kernel
       {pattern: 3, freq: 25, mod: QPSK, polar: 3/4, embedding: [...]}

T=5s:  Station C sends RTS to A
       {pattern: 3, freq: 25} - uses A's RX kernel
       512s @ BPSK = 2.05s

T=6s:  Station B sends RTS to A (doesn't hear C)
       {pattern: 3, freq: 25} - same as C!
       COLLISION IMMINENT

T=7s:  Station A receives both RTS
       Decodes B's RTS clearly (arrived later, stronger)
       Protocol extracts TX kernel from B's RTS: {pattern: 3, freq: 25}
       Compares against 8 closest context signals
       Finds C's TX kernel in context: {pattern: 3, freq: 25}
       CONFLICT DETECTED - same pattern + frequency triple

T=8s:  Station A sends QSY to B
       {rx_kernel: latest, reason: collision}
       256s @ BPSK = 1.28s
       Protocol tells B: "Your TX kernel conflicts with another station,
                          use my RX kernel instead or pick alternative"

T=9s:  Station B receives QSY
       Extracts RX kernel from QSY
       Waits 500ms (collision avoidance backoff)

T=10s: Station B retransmits RTS with alternative
       {pattern: 4, freq: 25} - different pattern, same freq triple
       512s @ BPSK = 2.05s

T=12s: Station A sends CTS to B
       {session_id: XYZ, status: OK}
       128s @ BPSK = 0.51s
       No conflict now!

T=13s: Station B transmits message
       {pattern: 4, freq: 25, QPSK}
       No collision with C's transmission

T=7.5s: (Meanwhile) Station A sends CTS to C
        {session_id: ABC, status: OK}
        Different session, pattern 3, freq: 25

T=8s:   Station C transmits message
        {pattern: 3, freq: 25, QPSK}
        Overlaps with B's QSY reception but different pattern!
```

**Result:**
- Both B and C complete successfully
- QSY avoided collision by separating patterns
- ~1.3s overhead for collision resolution
- Better than exponential backoff (would be 0.5-2s + retry)

---

## Kernel Coordination

### Distributed Collision Avoidance

**Each station maintains:**
```python
kernel_cache = {
    'W1ABC': {
        'rx_kernel': {...},  # From beacon
        'tx_kernel': {...},  # From RTS/message
        'last_seen': timestamp
    },
    'W2DEF': {
        'rx_kernel': {...},
        'tx_kernel': None,  # Idle
        'last_seen': timestamp
    },
    ...
}
```

**Before transmitting:**
1. Get target's RX kernel (pattern, frequency preference)
2. Check all TX kernels (avoid busy channels)
3. Select free channel near target's preference
4. Send RTS with own TX kernel
5. Wait for CTS

**Network self-organizes:**
- Busy channels avoided
- Traffic spreads across 536 logical channels (8 patterns × 67 freqs)
- No central coordinator
- Robust to station joining/leaving

### Kernel Expiration

**RX kernel expires:**
- After 120s without beacon refresh
- Remove from cache
- Station considered offline

**TX kernel expires:**
- After transmission complete
- Station clears it (goes idle)
- Immediate (no timeout needed)

---

## Kernel Update Strategy

### When to Include Kernel in Message Header

```python
def should_include_tx_kernel(last_kernel, current_conditions):
    """Decide if message should include updated TX kernel"""

    # First message in QSO: always include
    if first_message_in_qso:
        return True, "Initial TX state"

    # No previous kernel: include
    if last_kernel is None:
        return True, "No previous kernel"

    # Pattern changed
    if current_pattern != last_kernel.pattern_id:
        return True, f"Pattern changed {last_kernel.pattern_id} → {current_pattern}"

    # Frequency changed
    if current_freq != last_kernel.frequency_pair:
        return True, f"Frequency changed"

    # Modulation adapted
    if current_mod != last_kernel.modulation:
        return True, f"Modulation adapted to {current_mod}"

    # Polar rate changed
    if current_polar != last_kernel.polar_rate:
        return True, f"Polar rate changed"

    # Long time since last update
    if time_since_last_kernel > 120:
        return True, "Periodic refresh (120s)"

    # Conditions stable: no kernel needed
    return False, "Stable conditions"
```

**Cost of including kernel:**
- +28 bytes payload
- Use 512s pattern @ BPSK
- +2.05s to message
- Only when needed (not every message)

---

## Beacon Timing

### Randomized Transmission

**Avoid synchronized collisions:**
```python
# Random interval: 30-60s
beacon_interval = 30 + random.uniform(0, 30)

next_beacon_time = last_beacon + beacon_interval
```

**With 40 active users:**
- 40 beacons/min = 0.67 beacons/sec
- @ 2.56s each = ~60% total beacon airtime
- Distributed across 67 freq pairs = <1% per pair

**Collision rate:** <5% (random timing + large frequency/pattern space)

---

## Archived Documentation

**V1 (4-kernel structure):** See `kernel_lifecycle_v1_archived.md`

V1 used 3 RX kernels + 1 TX kernel per beacon (112 bytes total).
V2 uses 1 RX kernel per beacon, TX kernel only when transmitting.
