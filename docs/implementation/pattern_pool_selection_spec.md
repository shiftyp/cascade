# CASCADE Pattern Pool Selection Algorithm Specification

**Purpose:** Specification for selecting appropriate pattern pool based on HF propagation
**Status:** Ready for implementation
**Used by:** Pattern Complexity Expert, protocol layer pattern assignment

---

## Overview

CASCADE's 128 patterns are organized into hierarchical pools matched to HF propagation characteristics. The model measures multipath delay spread and selects the appropriate pool.

**Key insight:** Most HF operation uses typical DX pool (IDs 80-207, 128 patterns). Exceptional conditions use smaller specialized pools.

---

## Pattern Pool Organization

### Complete Pool Specification

```python
PATTERN_POOLS = {
    # BEACON PATTERNS (0-63)
    'beacon_emergency': {
        'range': range(0, 16),
        'count': 16,
        'iq_complexity': 'minimal (BPSK line)',
        'lambda_range': (0.0, 0.1),
        'use_when': 'Emergency beacon negotiation',
        'propagation': 'Any (maximum robustness)',
    },

    'beacon_normal': {
        'range': range(16, 64),
        'count': 48,
        'iq_complexity': 'simple (circles/ellipses)',
        'lambda_range': (0.2, 0.3),
        'use_when': 'Normal kernel exchange, beacons',
        'propagation': 'Typical HF',
        'anti_kernel_resilient': True,  # Large pool handles blocking
    },

    # MESSAGE PATTERNS (64-255)
    'message_emergency': {
        'range': range(64, 80),
        'count': 16,
        'iq_complexity': 'minimal (BPSK)',
        'lambda_range': (0.0, 0.2),
        'use_when': 'Emergency traffic, disturbed conditions',
        'multipath_range': (10, 50),  # ms (severe)
        'snr_range': (-28, 0),  # dB
    },

    'message_typical_dx': {
        'range': range(80, 208),
        'count': 128,  # LARGEST POOL
        'iq_complexity': 'simple to moderate (circles/ellipses)',
        'lambda_range': (0.3, 0.5),
        'use_when': 'Multi-hop DX (MOST COMMON on HF)',
        'multipath_range': (3, 8),  # ms
        'snr_range': (-5, 15),  # dB
        'percentage_of_hf_operation': 70,  # 70% of HF contacts use this pool
    },

    'message_good_prop': {
        'range': range(208, 240),
        'count': 32,
        'iq_complexity': 'moderate to complex (ellipses/Lissajous)',
        'lambda_range': (0.5, 0.7),
        'use_when': 'Single-hop F2, good conditions',
        'multipath_range': (1, 3),  # ms
        'snr_range': (5, 15),  # dB
        'percentage_of_hf_operation': 20,  # 20% of contacts
    },

    'message_nvis': {
        'range': range(240, 256),
        'count': 16,
        'iq_complexity': 'complex (Lissajous)',
        'lambda_range': (0.7, 0.9),
        'use_when': '80m/40m NVIS, exceptional propagation',
        'multipath_range': (0.5, 1),  # ms (clean single-hop)
        'snr_range': (10, 20),  # dB (typically strong on NVIS)
        'percentage_of_hf_operation': 10,  # Rare
    },
}
```

---

## Pool Selection Algorithm

### Primary: Multipath-Based Selection

```python
def select_pattern_pool_from_propagation(multipath_delay_ms, snr_db, channel_type):
    """
    Select pattern pool based on measured propagation

    Args:
        multipath_delay_ms: Measured delay spread (0.5-50 ms)
        snr_db: Signal-to-noise ratio
        channel_type: 'message' or 'beacon'

    Returns:
        (pool_name, pattern_range)
    """

    if channel_type == 'beacon':
        # Beacon channel selection (simple)
        if snr_db < -10:
            return ('beacon_emergency', range(0, 16))
        else:
            return ('beacon_normal', range(16, 64))

    # Message channel selection (propagation-based)

    if multipath_delay_ms < 1.0:
        # Exceptional propagation (NVIS or very short path)
        if snr_db > 10:
            return ('message_nvis', range(240, 256))
        else:
            return ('message_good_prop', range(208, 240))

    elif multipath_delay_ms < 3.0:
        # Good propagation (single-hop F2)
        return ('message_good_prop', range(208, 240))

    elif multipath_delay_ms < 10.0:
        # Typical DX (2-3 hop multipath) - MOST COMMON
        return ('message_typical_dx', range(80, 208))

    else:
        # Severe multipath (long path, disturbed)
        if snr_db < -5:
            return ('message_emergency', range(64, 80))
        else:
            return ('message_typical_dx', range(80, 144))  # Lower half of typical


def measure_multipath_delay_spread():
    """
    Measure multipath from received beacon
    """

    # Analyze IQ smearing from beacon reception
    beacon_iq = receive_beacon_on_4fsk()

    # Power delay profile
    pdp = compute_power_delay_profile(beacon_iq)

    # Find delay spread (time containing 90% of energy)
    delay_spread_ms = compute_90percent_delay(pdp)

    return delay_spread_ms
```

### Pattern Assignment Within Pool

```python
def assign_patterns_from_pool(user_id, pool_name, pool_range):
    """
    Assign 8 patterns to user from selected pool

    Args:
        user_id: User identifier
        pool_name: 'typical_dx', etc.
        pool_range: range(80, 208) etc.

    Returns:
        List of 8 pattern IDs
    """

    pool_size = len(pool_range)
    patterns_per_user = 8

    # Hash user_id into pool
    base_idx = (hash(user_id) * 8) % pool_size

    # Assign 8 consecutive patterns from pool
    assigned = []
    for i in range(8):
        pattern_idx = (base_idx + i) % pool_size
        pattern_id = pool_range[pattern_idx]
        assigned.append(pattern_id)

    return assigned

# Example:
# User "W2DEF" on typical DX (pool 80-207):
#   hash("W2DEF") = 12345
#   base = (12345 * 8) % 128 = 40
#   Assigned: [120, 121, 122, 123, 124, 125, 126, 127]
#   (These are from typical DX pool, λ ≈ 0.42-0.44)
```

---

## Pool Transition Strategy

### Hysteresis to Prevent Oscillation

```python
def pool_selection_with_hysteresis(current_pool, measured_multipath):
    """
    Add hysteresis to prevent rapid pool switching
    """

    # Define pool boundaries with hysteresis
    thresholds = {
        'nvis_to_good': (0.8, 1.2),      # ms (2σ window)
        'good_to_typical': (2.5, 3.5),   # ms
        'typical_to_emergency': (9, 11), # ms
    }

    # Current pool determines thresholds
    if current_pool == 'message_nvis':
        if measured_multipath > thresholds['nvis_to_good'][1]:
            return 'message_good_prop'  # Degrade
        else:
            return 'message_nvis'  # Stay

    elif current_pool == 'message_good_prop':
        if measured_multipath < thresholds['nvis_to_good'][0]:
            return 'message_nvis'  # Upgrade
        elif measured_multipath > thresholds['good_to_typical'][1]:
            return 'message_typical_dx'  # Degrade
        else:
            return 'message_good_prop'  # Stay

    elif current_pool == 'message_typical_dx':
        if measured_multipath < thresholds['good_to_typical'][0]:
            return 'message_good_prop'  # Upgrade
        elif measured_multipath > thresholds['typical_to_emergency'][1]:
            return 'message_emergency'  # Degrade
        else:
            return 'message_typical_dx'  # Stay (most stable)

    elif current_pool == 'message_emergency':
        if measured_multipath < thresholds['typical_to_emergency'][0]:
            return 'message_typical_dx'  # Upgrade
        else:
            return 'message_emergency'  # Stay

    # Hysteresis creates 2σ stable bands
    # Prevents oscillation during marginal conditions
```

---

## Integration with Model

### Model Inputs

```python
def model_pattern_pool_selection(features):
    """
    Neural network selects pattern pool
    """

    # Extract propagation features
    multipath_estimate = propagation_estimation_head(features)  # ms
    snr_estimate = snr_estimation_head(features)  # dB

    # Pool selection (6-way classification)
    pool_logits = pool_selection_head(features)  # [6 values]
    pool_probs = softmax(pool_logits)

    # Select pool
    pool_idx = argmax(pool_probs)
    pool_names = ['beacon_emergency', 'beacon_normal',
                  'message_emergency', 'message_typical_dx',
                  'message_good_prop', 'message_nvis']

    selected_pool = pool_names[pool_idx]

    return {
        'pool': selected_pool,
        'multipath_ms': multipath_estimate,
        'snr_db': snr_estimate,
        'confidence': pool_probs[pool_idx],
    }
```

---

## Training Strategy

### Pool-Specific Training

```python
def train_pattern_pool_expert():
    """
    Train model to select appropriate pools
    """

    for batch in training_data:
        # Batch has ground-truth propagation labels
        true_multipath_ms = batch.metadata['multipath_delay']
        true_snr = batch.metadata['snr']

        # Determine correct pool
        if true_multipath_ms < 1:
            target_pool = 'message_nvis'
        elif true_multipath_ms < 3:
            target_pool = 'message_good_prop'
        elif true_multipath_ms < 10:
            target_pool = 'message_typical_dx'  # Most common
        else:
            target_pool = 'message_emergency'

        # Model prediction
        predicted = model.select_pool(batch.signal)

        # Loss
        pool_loss = cross_entropy(predicted['pool'], target_pool)
        multipath_loss = mse(predicted['multipath_ms'], true_multipath_ms)
        snr_loss = mse(predicted['snr_db'], true_snr)

        total_loss = pool_loss + 0.1*multipath_loss + 0.1*snr_loss

        optimize(total_loss)

    # Model learns to classify propagation modes
    # And assign appropriate pattern pools
```

---

## See Also

- **[Pattern Architecture](../model/pattern_architecture.md)** - Complete pool organization
- **[Pattern Complexity Expert](../model/experts.md#pattern-complexity-expert-network)** - Neural network that selects pools
- **[Adaptive Tone Grid](../protocol/adaptive_tone_grid.md)** - Propagation characteristics by band

---

*Ready for implementation*
