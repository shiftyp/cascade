# CASCADE QSO Coordination Protocol

Pairwise QSO operations use temporary pattern reservation and time-division to reduce interference and improve decode reliability.

## QSO Mode Overview

**When two stations establish active QSO:**
- Temporarily reserve pattern pair (2 of 128 patterns available)
- Use time-division (turn-taking on half-duplex radios)
- Priority decode for QSO partner
- Pattern release after QSO complete
- Reduces interference from/to other network traffic

## QSO Establishment

**Via kernel exchange:**

```python
# After initial contact (beacon + ACK with kernels)
def establish_qso_mode(partner_station):
    """Negotiate dedicated QSO patterns"""

    # Request QSO mode via message
    qso_request = {
        'to': partner_station,
        'type': 'QSO_MODE_REQUEST',
        'my_proposed_pattern': 5,      # I'll use Pattern 5
        'your_pattern': 7,             # You use Pattern 7
        'duration_minutes': 10,        # Reserve for 10 minutes
        'my_kernel_with_qso': my_kernel_updated
    }

    # Partner accepts
    qso_accept = {
        'type': 'QSO_MODE_ACCEPT',
        'confirmed_patterns': (5, 7),  # Agreed pattern pair
        'my_kernel_with_qso': partner_kernel_updated
    }

# Both kernels now have qso_active=1, qso_partner_pattern set
# Network knows: Patterns 5 and 7 reserved for this QSO
```

## Time-Division Protocol

**Turn-taking on half-duplex radios:**

```python
QSO_TIMING = {
    'turn_length': 5,              # seconds max per turn
    'turn_indicator': 'K',         # "K" or "OVER" signals end of turn
    'implicit_timeout': 10          # seconds of silence = turn forfeit
}

# QSO flow:
def qso_exchange():
    # Turn 1: K0BB transmits
    k0bb_turn = {
        'pattern': 5,                    # K0BB's QSO pattern
        'duration': 3.2,                 # seconds (model-determined)
        'end_marker': 'K'                # Signals "your turn"
    }

    transmit(k0bb_message, pattern=5)
    # K0BB: TX on Pattern 5
    # W2DEF: RX on Pattern 5 (priority decode, expects partner)

    # Turn 2: W2DEF responds
    w2def_turn = {
        'pattern': 7,                    # W2DEF's QSO pattern
        'duration': 4.8,                 # seconds
        'end_marker': 'K'
    }

    transmit(w2def_message, pattern=7)
    # W2DEF: TX on Pattern 7
    # K0BB: RX on Pattern 7 (priority decode)

# Repeat until QSO complete
```

**Priority decode during QSO:**
```python
def decode_during_qso(signal, qso_partner_pattern, my_capacity=50):
    """Partner gets decode priority"""

    # Always decode partner first (reserved capacity)
    partner_signal = extract_pattern(signal, qso_partner_pattern)
    partner_decoded = model.decode_priority(partner_signal)  # Guaranteed

    # Decode others with remaining capacity
    remaining_capacity = my_capacity - 1  # 1 slot reserved for partner
    other_patterns = get_all_patterns_except(qso_partner_pattern)
    others_decoded = model.decode(other_patterns, capacity=remaining_capacity)

    return [partner_decoded] + others_decoded  # Partner always first
```

## QSO Partner as Anti-Kernel Target

**During active QSO, partner becomes highest-priority anti-kernel:**

```python
# K0BB encoding message to someone else (not W2DEF)
# But in active QSO with W2DEF

message_to_k5xyz = model.encode(
    data="Message for K5XYZ",
    kernel_context={
        'target': k5xyz_kernel,
        'anti': aggregate([
            w2def_kernel,  # QSO partner - HIGH anti-kernel weight!
            other_anti_kernels...
        ]),
        'anti_weights': [
            0.8,  # W2DEF (QSO partner, don't interfere)
            0.2,  # Others (normal anti-kernel weight)
            ...
        ]
    }
)

# Model adapts:
# - Avoid interfering with W2DEF's patterns
# - Shift frequency/timing to minimize overlap
# - Reduce power if needed
```

**Result**: Active QSO partners automatically reduce interference with each other!

## Pattern Release

**QSO ends, patterns returned to pool:**

```python
def end_qso(partner_station):
    """Release reserved patterns"""

    # Send QSO end message
    qso_end = {
        'type': 'QSO_END',
        'to': partner_station,
        'duration_was': qso_elapsed_time,
        'my_kernel_updated': generate_fresh_kernel(qso_active=False)
    }

    transmit(qso_end)

    # Update own kernel
    my_kernel.qso_active = False
    my_kernel.qso_partner_pattern = None

    # Patterns 5 and 7 now available for other users
    release_patterns([5, 7])

# Auto-release on timeout (10 minutes default, or 5 minutes silence)
```

## QSO Coordination Benefits

**Compared to free-for-all mode:**

| Metric | Free Mode | QSO Mode | Improvement |
|--------|-----------|----------|-------------|
| Partner decode reliability | 80% | 98% | +18% (priority + reduced interference) |
| Decode latency | Variable | Predictable | Know when partner transmits |
| Interference to/from partner | Normal | Minimized | Anti-kernel priority |
| Hardware efficiency | Decode all 50 | Decode partner + best-effort others | CPU reserved for partner |

**Trade-off**: 2 patterns reserved (3% of capacity) for significant QSO quality improvement

## Multi-QSO Scenarios

**Multiple simultaneous QSOs:**

```
Network with 50 users:
├─ 10 active QSOs (20 users, 20 patterns reserved)
├─ 30 casual users (use remaining 44 patterns)
└─ Capacity: 44 patterns × 2× reuse = 88 virtual slots (adequate)

QSO pairs don't interfere with each other:
├─ K0BB ↔ W2DEF: Patterns 5,7
├─ K5XYZ ↔ N7ABC: Patterns 12,15
└─ Pattern separation (-30 dB) ensures clean decode
```

## See Also

- **[Net Operations](net_operations.md)** - Group QSO coordination
- **[Kernel Lifecycle](kernel_lifecycle.md)** - Kernel exchange and refresh
- **[Signal Specification](signal_specification.md)** - Pattern details
