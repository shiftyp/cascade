# CASCADE Pattern Usage (V2)

**Purpose:** How 8 patterns are used in beacons and messages
**Architecture:** No pools - all 8 patterns universally usable
**Selection:** Kernel-coordinated dynamic assignment

---

## Overview

CASCADE V2 uses **8 patterns** that can be used for any purpose:
- Beacons
- Messages
- RTS/CTS control
- Emergency traffic

**No pattern pools or hierarchy** - all 8 are equivalent.

---

## Pattern Assignment

### Kernel-Coordinated Selection

**Pattern ID provided in kernel:**
```python
rx_kernel = {
    'pattern_id': 3,      # Use pattern 3 to reach me
    'frequency_pair': 25,  # On tone pair 25
    'modulation': 'QPSK',
    ...
}
```

**Receiver determines pattern:**
- Decoder observes which patterns it decodes well
- Selects pattern with best orthogonality on current frequency
- Includes in RX kernel
- Transmitters use that pattern when messaging this station

**No static assignment** - patterns dynamically allocated based on:
- Current channel conditions
- Interference from other users
- TX kernel anti-collision map

### Pattern Reuse

**Same pattern used across:**
- 67 different frequency pairs (primary isolation)
- Different time slots (asynchronous)
- Different stations (kernel coordinates to avoid collisions)

**Example:**
```
Pattern 3 simultaneously used by:
- Station A: Freq pair 10 (500-520 Hz)
- Station B: Freq pair 35 (1500-1520 Hz) } No interference
- Station C: Freq pair 60 (2600-2620 Hz) } (different frequencies)
```

---

## Pattern Selection Examples

### Beacon

```python
# Station picks any available pattern
available_patterns = [0,1,2,3,4,5,6,7]

# Check TX kernels to avoid busy channels
for pattern_id in available_patterns:
    for freq_pair in preferred_frequencies:
        if not in_use(pattern_id, freq_pair):
            return (pattern_id, freq_pair)

# Transmit beacon with selected pattern
```

### Message Transmission

```python
# Use pattern from target's RX kernel
pattern_id = target_rx_kernel.pattern_id
freq_pair = target_rx_kernel.frequency_pair

# Verify not in use (check TX kernels)
if in_use(pattern_id, freq_pair):
    # Find nearby alternative
    pattern_id = find_free_pattern_near(pattern_id, freq_pair)

# Transmit message
```

---

## All Patterns Equal

**Unlike V1 (specialized pools), V2 patterns are:**
- ✅ All same orthogonality quality
- ✅ All usable for beacons or messages
- ✅ All usable at any SNR (modulation adapts separately)
- ✅ No emergency vs normal distinction

**Benefits:**
- Simpler protocol (no pool selection logic)
- Better utilization (no reserved patterns)
- Easier optimization (all 8 optimized together)
- Cleaner architecture

---

## Capacity

**8 patterns × 67 frequency pairs = 536 logical channels**

**Supports:**
- 40-45 active users simultaneously
- Higher with temporal reuse (asynchronous transmissions)

---

## Archived Documentation

**V1 (128-pattern pools):** See `pattern_pool_selection_spec_v1_archived.md`

V1 used hierarchical pools (emergency/beacon/message) with 128 patterns.
V2 uses 8 universal patterns with kernel-coordinated dynamic assignment.
