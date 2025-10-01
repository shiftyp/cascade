# Protocol Layer - Discrete Decisions

The protocol layer handles all discrete decisions in CASCADE. These are choices that cannot be optimized through gradient descent and require explicit rules.

## Responsibilities

### WHO - Identity and Access
- Callsign management
- Pattern pool assignment (which users get which patterns)
- User authentication state

### WHETHER - Binary Decisions
- Relay approval (forward or not)
- Trust transitions (TOTP verified or not)
- Emergency override activation

### WHAT - Discrete Classifications
- Message priority levels (EMERGENCY/HIGH/NORMAL/LOW)
- Hash exchange content (which stations to share)
- ACK information (success/failure, SNR bucket)

## Key Components

### Message Format
Simple 5-field structure:
```python
{
    'from': 'W1ABC',      # Sender callsign
    'to': 'W2DEF',        # Destination
    'id': 12345,          # Message ID
    'priority': 'NORMAL', # Priority level
    'data': 'Hello'       # Content
}
```

### Pattern Pool Assignment
- All 64 patterns: Dynamically assigned to users
- Emergency messages: Highest power/priority (not reserved patterns)
- Beacons: Use interstitial frequencies (not dedicated patterns)
- 8-16 patterns per active user
- Rotation every 100 transmissions
- Pattern reuse: 2-3× across spatially/temporally separated users (up to 150 virtual slots)

### Hash Exchange
- Callsign-based hashes (no salting)
- SNR-scaled exchange frequency
- 15-minute memory window
- Enables distributed mesh discovery

## Multi-Stage Protocol Flow

CASCADE operates in three progressive stages, adapting based on link quality:

### Stage 1: FT8-Style Discovery (Universal, -22 dB capable)

**Initial contact without prior kernels:**

```markdown
**Beacon** (1.28 seconds, asynchronous):
- Content: 16-bit callsign hash only
- Modulation: 4-FSK on interstitial frequencies [156, 468, 781, 1093 Hz]
- Symbol duration: 160ms (FT8-style)
- Repetitions: 3× per minute (random timing/patterns)
- Min SNR: -22 dB
- Purpose: Network discovery, regulatory identification

**ACK** (adaptive duration):
- Content: beacon_hash(16) + my_call(24) + snr(4) = 44 bits
- Modulation: Adapts to measured SNR
  - High SNR (>0 dB): 50ms symbols, 8-QAM, 0.09s
  - Fair SNR (-10 to 0): 160ms symbols, 4-FSK, 3.5s
  - Weak SNR (<-10 dB): 500ms symbols, BPSK, 22s
- Purpose: Confirm contact, report SNR, complete QSO

**QSO Complete** after beacon + ACK exchange ✓
- Valid FT8-equivalent contact for logbook
- No kernels needed
- Works globally at -22 dB
```

**When to stay in Stage 1:**
- SNR < -10 dB (weak DX)
- Occasional check-ins
- Legacy compatibility mode
- Emergency fallback

### Stage 2: Kernel Negotiation (Optional, SNR > -10 dB)

**Upgrade to kernel-optimized mode:**

```python
# After Stage 1 QSO, if SNR supports it:
if measured_snr > -10:
    # Request kernel exchange
    kernel_request = {
        'to': station,
        'my_kernel': select_kernel_size(measured_snr),  # 16/64/256 bits
        'request_their_kernel': True
    }

    # Transmitted on message pattern (faster than beacons)
    # Duration: 0.2-0.5s depending on SNR

# Receive kernel response:
kernel_response = {
    'my_kernel': their_kernel,
    'grid_square': grid,  # Optional, included in 256-bit extended kernel
    'capabilities': hardware_tier
}
```

**Kernel size selection by SNR:**

| SNR Range | Kernel Size | Includes | TX Time @ SNR |
|-----------|-------------|----------|---------------|
| < -18 dB | 0 bits | SNR only | N/A (no kernel) |
| -18 to -10 | 16 bits | Modulation, hardware tier | ~8s |
| -10 to 0 | 64 bits | + Capacity, interference map | ~0.4s |
| 0 to +10 | 64 bits | Full standard kernel | ~0.2s |
| > +10 dB | 256 bits | + Grid, network topology, timing | ~0.05s |

### Stage 3: High-Speed Messaging (Kernel-Optimized)

**Using exchanged kernels:**

```python
# Now have target + anti-kernels from multiple ACKs
transmission = model.encode(
    message_data,
    kernel_context={
        'target': target_station_kernel,     # Optimize FOR them
        'anti': aggregate(bystander_kernels) # Optimize AGAINST interference
    }
)

# Adaptive modulation based on kernels:
# - Strong target link: 8-QAM, 480 bps
# - Weak target link: BPSK, 20 bps
# - Anti-kernels: Adjust frequency/timing to reduce interference

# Message duration: 1.6s (short) to 6.4s (long with repetition)
```

**When to use Stage 3:**
- SNR > -10 dB (supports kernel overhead)
- Frequent communication (kernel exchange amortized)
- Contest/net operations
- Multi-message exchanges

### Protocol Stage Decision Logic

```python
def select_protocol_stage(station, link_quality):
    """Determine which stage to use"""

    snr = link_quality.measured_snr
    have_kernel = station in kernel_cache
    message_frequency = estimate_qso_rate(station)

    if snr < -10 or not have_kernel:
        return {
            'stage': 1,
            'mode': 'ft8_equivalent',
            'expected_rate': '12.5 bps (beacon) or adaptive ACK',
            'qso_time': '5-25 seconds',
            'use_when': 'DX, weak signals, initial contact'
        }

    elif snr < 0:
        return {
            'stage': 2,
            'mode': 'kernel_16bit',
            'expected_rate': '200-1000 bps',
            'qso_time': '2-5 seconds',
            'use_when': 'Regional, occasional messages'
        }

    elif message_frequency < 1/minute:
        # Slow messaging - kernel overhead not worth it
        return {
            'stage': 1,
            'mode': 'ft8_fallback',
            'expected_rate': '12.5 bps',
            'reason': 'Infrequent - kernel overhead > benefit'
        }

    else:
        return {
            'stage': 3,
            'mode': 'full_kernel',
            'expected_rate': '1000-11000 bps',
            'qso_time': '1.6-3.2 seconds',
            'use_when': 'Local, frequent messages, contests'
        }
```

### Kernel Hint Routing
- Pairwise hint exchange (Stage 2+)
- Receiver generates, transmitter uses
- 10-minute expiration
- Improves weak link performance

### Trust State Machine
```
UNTRUSTED → TOTP_TRUSTED → HMAC_ALLOWED
```
Transitions based on verification and link quality.

## Multi-Resolution Kernel System

Kernel complexity scales with link quality - better links exchange richer information:

### Kernel Tiers

**Extended Kernel (256 bits)** - Excellent links (SNR > +10 dB):
```python
extended_kernel = {
    # Core adaptation (64 bits)
    'modulation_pref': 8,          # bits (fine-grained preferences)
    'hardware_tier': 8,            # bits (detailed capabilities)
    'capacity_users': 8,           # bits (exact user count)
    'interference_coarse': 16,     # bits (basic interference map)
    'timing_offset': 8,            # bits (clock sync)
    'snr_floor': 8,                # bits (noise floor)
    'power_request': 8,            # bits (TX power recommendation)

    # Extended information (192 bits)
    'grid_square': 16,             # bits (6-char Maidenhead)
    'interference_detailed': 64,   # bits (per 50 Hz bin across 2.5 kHz)
    'temporal_preferences': 24,    # bits (time-of-day activity)
    'network_topology': 32,        # bits (connected stations, relay capability)
    'propagation_obs': 32,         # bits (current band conditions observed)
    'multi_pattern_hints': 24      # bits (pattern combination preferences)
}
# Total: 256 bits = 32 bytes
# TX time at 5,000 bps: 0.05 seconds
```

**Standard Kernel (64 bits)** - Good links (0 to +10 dB):
```python
standard_kernel = {
    'modulation_pref': 3,          # bits (8 levels)
    'hardware_tier': 2,            # bits (4 tiers)
    'capacity_users': 5,           # bits (0-31 users)
    'snr_floor': 5,                # bits (32 levels: -24 to +8 dB)
    'interference_map': 8,         # bits (coarse 8-zone map)
    'frequency_pref': 8,           # bits (256 frequency bins)
    'timing_offset': 4,            # bits (16 timing levels)
    'noise_floor': 5,              # bits (32 noise levels)
    'power_request': 4,            # bits (16 power levels)
    'features': 4,                 # bits (capability flags)
    'reserved': 16                 # bits (future use)
}
# Total: 64 bits = 8 bytes
# TX time at 320 bps: 0.2 seconds
```

**Compressed Kernel (16 bits)** - Fair links (-10 to 0 dB):
```python
compressed_kernel = {
    'modulation': 2,               # bits (4 levels: 8QAM/QPSK/BPSK/emergency)
    'hardware': 2,                 # bits (4 tiers)
    'snr_floor': 4,                # bits (16 levels: -24 to +8 dB, 2dB steps)
    'freq_zone': 3,                # bits (8 zones in 2.5 kHz)
    'capacity': 5                  # bits (0-31 user capacity)
}
# Total: 16 bits = 2 bytes
# TX time at 100 bps: 0.16 seconds
```

**No Kernel (3 bits)** - Weak links (<-18 dB):
```python
minimal_ack = {
    'snr_report': 3                # bits (8 levels: -24 to +15 dB, coarse)
}
# Total: 3 bits
# TX time at 2.5 bps: 1.2 seconds
# Transmitter uses conservative fallback (BPSK, assume RPi hardware)
```

### Progressive Kernel Refinement

**Links naturally upgrade kernels as SNR improves:**

```markdown
**First contact** (beacon @ -12 dB):
→ ACK with 16-bit compressed kernel (Stage 1→2 transition)

**Second message** (improved to -8 dB):
→ ACK with 64-bit standard kernel (better optimization)

**Third message** (now +5 dB, stable):
→ ACK with 256-bit extended kernel (includes grid, full detail)

**Subsequent messages**:
→ Kernel refresh every 10-20 messages (conditions may change)
→ Model uses latest kernel for optimal adaptation
```

### ACK System (Adaptive)
- SNR-dependent ACK modulation (50ms to 500ms symbols)
- Kernel size adapts to link quality (0 to 256 bits)
- Between-frame transmission
- Pattern success feedback
- Optional anti-kernel reporting (interference complaints)

## Interface with Model

The protocol provides constraints to the model:
- Assigned pattern pool
- Priority weight
- Time constraints
- Target destination

The model returns optimizations within these constraints.

## Heterogeneous Hardware Networks

CASCADE networks naturally accommodate mixed hardware capabilities without requiring central coordination or explicit hardware discovery.

### Natural Self-Organization

**Stations with different hardware decode different user subsets:**

```
50 users transmitting on channel

Station A (RPi only): Decodes 12 users (strongest signals)
Station B (Coral TPU): Decodes 55 users (nearly everyone)
Station C (Desktop): Decodes 32 users (most signals)
Station D (GPU): Decodes 68 users (everyone + weak signals)
```

**Network properties emerge automatically:**
- Strong transmissions reach all stations (100% connectivity)
- Medium transmissions reach capable stations (60-80% connectivity)
- Weak transmissions reach only powerful receivers (20-40% connectivity)

**This is Shannon-optimal**: Limited hardware naturally prioritizes strong signals, maximizing total network capacity.

### Multi-Kernel Coordination

All stations that successfully decode send ACKs with kernel hints. The protocol layer categorizes these:

**Target kernel** (intended recipient):
```
Message: "W1ABC to W2DEF"
W2DEF decodes → sends ACK with kernel
Protocol: "This is target kernel" (for W2DEF)
Model: Optimize transmission to maximize W2DEF's decode
```

**Anti-kernels** (interfered bystanders):
```
K5XYZ also decoded → sends ACK: "You're interfering with my QSO"
Protocol: "This is anti-kernel" (K5XYZ is bystander, not target)
Model: Adjust transmission to reduce K5XYZ's interference
```

**Neutral kernels** (no issues):
```
N7MNO decoded → sends ACK: "Heard you fine"
Protocol: "Neutral kernel" (no action needed)
Model: No constraints from this station
```

**Model input** (identity-blind):
```python
# Protocol passes to model (NO CALLSIGNS):
kernel_context = {
    'target': decompress(W2DEF_kernel),      # Optimize FOR
    'anti': aggregate([K5XYZ_kernel, ...]),  # Optimize AGAINST
    'neutral': aggregate([N7MNO_kernel])     # Informational
}

# Model adapts constellation/timing to satisfy constraints
adapted_signal = model.encode(message, kernel_context)
```

**Emergent behaviors:**
- Weak receivers automatically get simpler modulation (they request it via kernels)
- Interfered stations get relief (transmitters adapt to reduce interference)
- Strong receivers get complex modulation (they can handle it)
- No explicit negotiation needed (kernel hints convey everything)

### Emergency Priority

High-power emergency transmissions naturally reach all hardware tiers:

```
Emergency station: 100W transmission
SNR at receivers: +20 to +30 dB

Even RPi-only stations (10-user limit) decode emergency messages
→ 100% network penetration
→ No special hardware required for emergency participation
```

### Hardware Upgrade Incentives

Better hardware provides better experience:
- **RPi only**: Hears strong signals, participates in nets
- **RPi + Coral ($60 upgrade)**: Hears nearly everyone, full CASCADE experience
- **Desktop/GPU**: Maximum capacity for contest/club operations

Graceful degradation ensures entry-level hardware remains useful while creating natural upgrade path.

## See Also

- **[Model Layer](../model/README.md)** - Continuous optimization that works within protocol constraints
- **[Interface Documentation](../interface/README.md)** - Detailed protocol/model boundary definitions
- **[Priority Handling](priority_handling.md)** - Emergency and priority message processing
- **[Beacons](beacons.md)** - Network discovery and capabilities exchange
- **[Link Adaptation](link_adaptation.md)** - Pairwise communication optimization
- **[Hardware Requirements](../deployment/hardware_requirements.md)** - Deployment tiers and performance