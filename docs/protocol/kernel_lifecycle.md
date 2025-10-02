# CASCADE Kernel Lifecycle Protocol

Kernels evolve through a three-round feedback protocol, incorporating target optimization and anti-kernel interference avoidance. This document describes the complete kernel exchange, refresh, and adaptation mechanisms.

## Kernel Lifecycle Overview

```markdown
**Phase 1: Bootstrap** (4-FSK, no kernel)
├─ Beacon exchange (discovery)
├─ Initial ACK (FT8-style contact)
└─ First kernel exchange (64-bit standard kernel)

**Phase 2: Optimization** (multi-round adaptation)
├─ Round 1: Message transmission (uses current kernel)
├─ Round 2: Anti-kernel feedback (interference reports, best-effort)
├─ Round 3: Adapted kernel broadcast (incorporates anti-kernels)
└─ Repeat until convergence (2-3 cycles)

**Phase 3: Maintenance** (ongoing refresh)
├─ ACK-piggybacked kernel updates (zero overhead)
├─ Proactive expiration management
└─ Explicit refresh if needed (rare)
```

## Three-Round Kernel Protocol

### Round 1: Initial Transmission

**Station A transmits message using current kernel:**

```python
# Transmission
message_transmission = model.encode(
    message_data=b"Hello W2DEF",
    kernel_context={
        'target': w2def_kernel,       # Current kernel for W2DEF
        'anti': aggregate([...]),     # Known anti-kernels
    }
)

transmit_on_message_patterns(message_transmission)
# Duration: ~3s (model-determined)
```

### Round 2: Anti-Kernel Feedback (Best-Effort)

**Stations experiencing interference broadcast anti-kernels:**

```python
# Multiple stations detect interference from Station A
# Broadcast anti-kernels on 4-FSK (overlapping, lossy)

# Station B (experiencing 40% interference from A):
b_anti_kernel_broadcast = {
    'from': hash_B,
    'type': 'ANTI_KERNEL',
    'interferer': hash_A,
    'interference_level': 0.4,
    'my_anti_kernel': generate_anti_kernel(my_state),
    'affected_patterns': [12, 15]  # Which patterns A interferes on
}

transmit_4fsk_broadcast(
    b_anti_kernel_broadcast,
    pattern=mutate(my_pattern, random_seed),
    timing=random_offset(0, 1000),  # ms jitter
    no_ack=True,                    # Best-effort, no confirmation
    allow_overlap=True               # Other stations also broadcasting
)

# Stations C, D, E also broadcast (simultaneously)
# 4-FSK channel: 5 overlapping transmissions
# Model separates: 3-4 decoded (60-80% success)
# 1-2 lost (acceptable - will retry next cycle)
```

### Round 3: Adapted Kernel Broadcast

**Station A incorporates heard anti-kernels, broadcasts adapted kernel:**

```python
# Station A received anti-kernels from B and C (D's was lost)
heard_anti_kernels = [B_anti_kernel, C_anti_kernel]

# Generate adapted kernel
adapted_kernel = model.generate_kernel(
    my_decoder_state=my_state,
    anti_kernels=heard_anti_kernels,
    optimization={
        'my_decode_quality': weight=1.0,          # Maintain my performance
        'reduce_interference_to_B': weight=0.4,   # Adapt for B (40% interference)
        'reduce_interference_to_C': weight=0.25   # Adapt for C (25% interference)
    }
)

# Broadcast adapted kernel on 4-FSK
adapted_kernel_broadcast = {
    'from': hash_A,
    'type': 'ADAPTED_KERNEL',
    'kernel': adapted_kernel,
    'adapted_for': [hash_B, hash_C],   # Who I adapted for
    'adaptation_changes': {
        'frequency_shifted': +50,       # Hz (moved away from B's patterns)
        'power_reduced': -3,            # dB
        'timing_adjusted': True
    },
    'timestamp': now()
}

transmit_4fsk(adapted_kernel_broadcast)  # 5s for 64-bit kernel on 4-FSK
```

**Network updates caches:**
```python
# Stations B and C see:
# "Station A adapted to reduce interference with us"
# Update A's kernel → Use adapted version when encoding to A

# Other stations:
# "Station A has new kernel"
# Update cache → Use adapted kernel (also benefits from reduced interference)
```

## Kernel Self-Expiration

**Kernels include model-predicted validity:**

```python
kernel_with_expiration = {
    # Standard fields
    'version': 4 bits,
    'hardware': 2 bits,
    // ...

    # NEW: Expiration prediction
    'estimated_valid_seconds': 6 bits,   # 0-63 × 10sec = 0-630 seconds
    'confidence': 2 bits                 # 0=low, 1=med, 2=high, 3=very_high
}

# Model learns to predict kernel lifetime during training
def predict_kernel_validity(channel_state):
    """Model predicts how long kernel will remain valid"""

    # Analyze channel characteristics:
    if channel_stable(channel_state):
        # Stable propagation → long kernel life
        return {'valid_seconds': 600, 'confidence': 'very_high'}

    elif channel_slowly_varying(channel_state):
        # Gradual changes → medium kernel life
        return {'valid_seconds': 300, 'confidence': 'high'}

    elif channel_fading(channel_state):
        # Fast fading → short kernel life
        return {'valid_seconds': 60, 'confidence': 'medium'}

    else:
        # Chaotic/unknown → very short
        return {'valid_seconds': 30, 'confidence': 'low'}
```

## Kernel Refresh Mechanisms

### Primary: ACK-Piggybacked Refresh (Zero Overhead)

**Every message ACK includes refreshed kernel:**

```python
# W2DEF sends message ACK to K0BB
message_ack = {
    'message_id': msg_id,
    'status': 'RECEIVED',
    'snr': +10,

    # Always include fresh kernel!
    'refreshed_kernel': generate_fresh_kernel(my_current_state),
    'kernel_age': 0  # Just generated
}

# Transmitted on message patterns (fast, 0.1s)
# K0BB receives: W2DEF's updated kernel
# Zero overhead (would send ACK anyway)

# Kernel stays fresh automatically during active QSO
```

### Secondary: Proactive Receiver-Initiated (During Idle)

**If no recent messages, receiver proactively updates:**

```python
# W2DEF monitors kernels it issued
for station, kernel_issued in my_issued_kernels.items():
    kernel_age = now() - kernel_issued.timestamp

    # Approaching expiration? (80% of predicted lifetime)
    if kernel_age > kernel_issued.estimated_valid * 0.8:
        # Proactively send updated kernel on 4-FSK
        kernel_update = {
            'to': station,
            'type': 'KERNEL_REFRESH',
            'new_kernel': generate_fresh_kernel(),
            'reason': 'proactive_expiration_management'
        }

        transmit_4fsk(kernel_update)  # 5s, during idle time

# Station receives update, uses it (no ACK needed)
# Seamless refresh
```

### Tertiary: Transmitter-Requested (Fallback)

**If kernel expired and no update received:**

```python
# K0BB about to transmit, checks kernel
kernel = kernel_cache['W2DEF']
if is_expired(kernel):
    # Request fresh kernel on 4-FSK
    kernel_request = {
        'to': hash_W2DEF,
        'type': 'KERNEL_REQUEST',
        'my_fresh_kernel': generate_kernel()  # Include mine too
    }

    transmit_4fsk(kernel_request)  # 5s

    # Wait for response (4-FSK)
    response = listen_4fsk(timeout=10)  # Up to 10s

    if response:
        kernel_cache['W2DEF'] = response.kernel
        # Proceed with message
    else:
        # No response - fall back to 4-FSK for message
        transmit_4fsk(message)  # Slower but works
```

**Frequency:**
- Most refreshes: Via ACKs (zero overhead)
- Idle periods: Proactive (occasional, ~1/5min)
- Fallback: Explicit request (rare, <1% of time)

## Kernel Convergence

**Kernels improve over multiple rounds:**

```
Minute 0: Station A broadcasts with default kernel
          → Causes 35% interference to B, 20% to C

Minute 1: B and C broadcast anti-kernels
          → A hears both (via 4-FSK best-effort)

Minute 2: A broadcasts adapted kernel v1
          → Frequency shifted, power reduced
          → Interference to B: 35% → 20%, C: 20% → 10%

Minute 3: B broadcasts anti-kernel (still 20% interference)
          → A hears

Minute 4: A broadcasts adapted kernel v2
          → Further optimization
          → Interference to B: 20% → 10%, C: 10% → 5%

Minute 5: No new anti-kernels (interference acceptable)
          → Kernel converged

Result: Network self-optimizes over 5 minutes
        Interference reduced from 35% to 10% (74% improvement)
```

## Adapted Kernel Format

**Shows anti-kernel incorporation:**

```python
adapted_kernel_64bit = {
    'version': 4 bits,
    'my_preferences': 40 bits,  # Standard kernel fields

    # Adaptation metadata (20 bits)
    'adapted_from_count': 2 bits,        # 0-3 anti-kernels incorporated
    'anti_kernel_hashes': 12 bits,       # 3 × 4-bit short hashes
    'adaptation_type': 3 bits,           # Freq shift / power / timing / pattern
    'adaptation_magnitude': 3 bits       # How much changed (0-7 scale)
}
# Total: 64 bits

# Stations seeing adapted kernel know:
# - How many anti-kernels A incorporated
# - Who A adapted for (short hashes match their kernels)
# - What kind of adaptation (helps predict behavior)
```

## Kernel Expiration Training

**Model learns to predict kernel lifetime:**

```python
def train_expiration_prediction():
    """Train model to predict when kernels become stale"""

    for scenario in training:
        # Generate kernel at t=0
        kernel_t0 = model.generate_kernel(channel_state_t0)

        # Simulate time passing with channel evolution
        for time_delta in [60, 120, 300, 600]:  # 1, 2, 5, 10 minutes
            channel_evolved = evolve_channel(channel_state_t0, time_delta)

            # Does old kernel still work?
            test_msg = encode_with_kernel(data, kernel_t0)
            decode_result = decode_with_conditions(test_msg, channel_evolved)

            # Label: 1 if still works, 0 if failed
            still_valid = decode_success(decode_result)

        # Find actual expiration time
        actual_lifetime = max(t for t in [60,120,300,600] if validity[t] == 1)

        # Train predictor
        predicted_lifetime = model.predict_expiration(channel_state_t0)
        loss = abs(predicted_lifetime - actual_lifetime) / actual_lifetime

        optimizer.step(loss)
```

**Model learns**:
- Stable channels → long kernels (10 min)
- Fading channels → short kernels (1-2 min)
- Encodes prediction in kernel's expiration field

## See Also

- **[Adaptive 4-FSK](adaptive_4fsk.md)** - 4-FSK channel capacity and separation
- **[Protocol Overview](README.md)** - Multi-stage protocol flow
- **[Version Compatibility](../interface/version_compatibility.md)** - Kernel versioning
