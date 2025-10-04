# CASCADE Net Operations Protocol

CASCADE supports coordinated net operations with relay-based topology, pre-encoding slot allocation, and configurable net profiles optimized for different use cases.

## Net Profiles

Net controllers configure message size and timing constraints based on net purpose:

### DX Net Profile

**Optimized for**: Rapid check-ins, maximum stations, minimal transmission time

```python
DX_NET_PROFILE = {
    'max_message_bytes': 32,       # Callsign + grid + brief report
    'max_slot_seconds': 2.0,       # Brief transmissions (model determines actual time)
    'typical_message': "K0BB FN42 599",  # 12 bytes
    'relay_required': True,        # Most DX needs relay
    'check_in_style': 'rapid'      # One after another, no gaps
}

# Protocol doesn't specify modulation (that's model's job!)
# Model adapts: weak link might use BPSK (slower), strong link 8-QAM (faster)
# Protocol just enforces: max 2 seconds and max 32 bytes
```

**Example DX net operation:**
```
# Protocol view (doesn't know modulation, just sees durations):

0.0-0.5s: W1NET "VK2ZOI" (model encodes, outputs 0.5s signal)
0.5-2.1s: VK2ZOI via K0BB relay "VK2 599" (model outputs 1.6s signal - weak link)
2.1-2.6s: W1NET "ZS6ABC" (model outputs 0.5s)
2.6-4.2s: ZS6ABC via N7 relay "ZS6 579" (model outputs 1.6s - weak link)
...

# Model chose modulation (protocol doesn't care):
# - W1NET: 8-QAM (strong signal, fast encoding)
# - VK2ZOI via relay: BPSK (weak link, slower encoding)
# - Protocol just allocates time based on model's output duration

20 stations check in: ~40 seconds total
Rate: 30 check-ins/minute possible
```

### Ragchew Net Profile

**Optimized for**: Extended conversations, detailed exchanges

```python
RAGCHEW_NET_PROFILE = {
    'max_message_bytes': 256,      # Full paragraph
    'max_slot_seconds': 12.0,      # Extended transmissions allowed
    'typical_message': 150,        # bytes (multiple sentences)
    'relay_optional': True,        # Local ragchew usually doesn't need relay
    'check_in_style': 'conversational'  # Natural flow, pauses
}

# Model determines actual encoding time (protocol just sets limits)
# Same 256 bytes might be:
# - 4.8s on strong link (8-QAM, 3 patterns)
# - 12.8s on weak link (BPSK, 8 patterns)
# Protocol allows up to 12s, model optimizes within that
```

**Example ragchew net:**
```
0-7s: K0BB "Well the weather here has been quite nice, we had some
           storms last week but cleared up nicely..." (256 bytes, 4 patterns)
7-8s: [Natural pause]
8-14s: W2DEF "That's great to hear! We've been having similar weather..."
            (256 bytes, 4 patterns)
...

More relaxed timing, extended messages
5-8 transmissions per minute
```

### Emergency Net Profile

**Optimized for**: Coordinated response, reliable relay, clear communication

```python
EMERGENCY_NET_PROFILE = {
    'max_message_bytes': 128,      # Concise but detailed
    'max_slot_seconds': 8.0,       # Allows weak links (model adapts)
    'typical_message': 96,         # bytes (2-3 sentences)
    'relay_required': True,        # Ensure all stations reached
    'check_in_style': 'structured', # Strict order, net control coordinates
    'priority': 'HIGH',            # Override other traffic
    'confirmation_required': True  # All transmissions ACKed
}

# Protocol sets constraints, model optimizes encoding
# 128 bytes encoded by model might be:
# - 3.2s (strong link, efficient encoding)
# - 8.0s (weak link, robust encoding with repetition)
# Protocol allows up to 8s to accommodate weak links
```

**Emergency net operation:**
```
W1NET (controller): "K0BB, status report"
K0BB: "All clear, 3 operators standing by" (96 bytes, 1.6s)
W1NET: "Copy K0BB. W2DEF, your traffic?"
W2DEF: "Need medical supplies relay to county EOC" (128 bytes, 3.2s)
W1NET: "Copy W2DEF, K5XYZ will relay to EOC. Next check VK2..."
...

Structured, confirmed, everyone hears
10-15 transmissions per minute
```

### Contest Net Profile

**Optimized for**: Maximum throughput, rapid exchanges

```python
CONTEST_NET_PROFILE = {
    'max_message_bytes': 64,       # Brief exchanges only
    'max_slot_seconds': 1.0,       # Fast turnaround (assumes good signals)
    'typical_message': 32,         # bytes ("599 001 K")
    'relay_optional': False,       # Direct only (contest, local, strong signals)
    'check_in_style': 'rapid_fire',# Minimal gaps
    'auto_numbering': True         # Sequential contest numbers
}

# Assumes strong local signals (contest operations)
# Model likely uses 8-QAM (fast)
# 64 bytes typically encodes to 0.2-0.5s
# Protocol allows up to 1.0s (safety margin)
```

**Contest net operation:**
```
0.0-0.3s: W1NET "K0BB"
0.3-0.5s: K0BB "599 001"  (32 bytes, 0.2s @ 8-QAM!)
0.5-0.8s: W1NET "W2DEF"
0.8-1.0s: W2DEF "599 002"
1.0-1.3s: W1NET "K5XYZ"
1.3-1.5s: K5XYZ "599 003"
...

20 exchanges in ~10 seconds
120 QSOs per minute possible!
```

## Net Formation Protocol

### Controller Initiates Net

```python
def form_net(net_profile, purpose):
    """Net controller starts net"""

    # Broadcast net formation on emergency or message patterns
    net_announcement = {
        'type': 'NET_FORMING',
        'controller': my_callsign,
        'net_id': generate_net_id(),      # Unique ID for this net
        'profile': net_profile,           # DX/Ragchew/Emergency/Contest
        'purpose': purpose,               # "10m DX", "Emergency coordination", etc.
        'region': my_grid[:2],            # Geographic focus
        'max_size_bytes': net_profile.max_message_bytes,
        'max_slot_seconds': net_profile.max_slot_seconds
    }

    transmit_broadcast(net_announcement, power='HIGH')

    # Wait for check-ins (60 seconds)
    check_ins = collect_check_ins(timeout=60)

    # Analyze topology, select relays
    relays = select_relay_stations(check_ins)

    # Broadcast net roster and relay assignments
    net_roster = {
        'net_id': net_id,
        'controller': my_callsign,
        'relays': relay_list,
        'members': member_list,
        'patterns_assigned': pattern_assignments
    }

    return net_roster
```

### Member Check-In

```python
def check_in_to_net(net_announcement):
    """Join announced net"""

    # Send check-in with capabilities
    check_in = {
        'net_id': net_announcement.net_id,
        'my_call': my_callsign,
        'my_grid': my_grid,
        'my_kernel': my_kernel,          # Includes hardware tier, capacity
        'can_relay': my_hardware_tier in ['coral', 'desktop', 'gpu'],
        'snr_to_controller': measure_snr(net_announcement),
        'snr_to_peers': measure_peer_snr(other_check_ins)  # Who else I hear
    }

    transmit_to_controller(check_in)
```

## Relay Selection Algorithm

**Controller analyzes check-ins to choose relays:**

```python
def select_relay_stations(check_ins):
    """Choose optimal relay stations based on coverage and capability"""

    relay_candidates = []

    for station in check_ins:
        # Relay criteria:
        relay_score = 0

        # 1. Strong link to controller (critical)
        if station.snr_to_controller > 5:
            relay_score += 100
        elif station.snr_to_controller > 0:
            relay_score += 50
        else:
            continue  # Can't be relay if weak to controller

        # 2. Coverage of other members
        peers_heard = sum(1 for p in station.snr_to_peers if p > -10)
        coverage_percent = peers_heard / len(check_ins)
        relay_score += coverage_percent * 100

        # 3. Hardware capability
        if station.hardware == 'gpu':
            relay_score += 50
        elif station.hardware in ['coral', 'desktop']:
            relay_score += 30

        # 4. Geographic diversity
        if station.grid[:2] != controller_grid[:2]:
            relay_score += 20  # Prefer geographic spread

        relay_candidates.append({
            'call': station.call,
            'score': relay_score,
            'coverage': peers_heard,
            'hardware': station.hardware
        })

    # Select top 3-5 relays
    relays = sorted(relay_candidates, key=lambda r: r.score, reverse=True)[:5]

    return relays
```

## Pre-Encoding Slot Allocation

**Member composes message, system pre-encodes for exact slot:**

```python
def request_transmission_slot(message_text, net_id):
    """Pre-encode message and request exact-duration slot"""

    # Step 1: Get message bytes
    message_bytes = message_text.encode('utf-8')

    # Check net message size limit
    net_profile = get_net_profile(net_id)
    if len(message_bytes) > net_profile.max_message_bytes:
        alert(f"Message too long! Max {net_profile.max_message_bytes} bytes for this net.")
        return

    # Step 2: Model encodes message NOW (protocol doesn't know HOW, just gets duration)
    encoded_signal = model.encode(
        message_bytes,
        kernel_context={
            'target': net_broadcast_kernel,
            'anti': current_interference
        }
    )

    # Model returns: opaque encoded signal + metadata
    # Protocol doesn't see modulation scheme, just duration!
    exact_duration = encoded_signal.duration_seconds  # e.g., 4.8s (from model)

    # Model might have used:
    # - 8-QAM, 3 patterns = 4.8s (strong link)
    # - BPSK, 8 patterns = 12.8s (weak link)
    # Protocol doesn't know or care! Just sees duration.

    # Step 3: Check against net time limit
    if exact_duration > net_profile.max_slot_seconds:
        alert(f"Encoded message is {exact_duration}s, exceeds net limit of
               {net_profile.max_slot_seconds}s. Try shortening message.")
        return

    # Step 4: Request slot with exact time from model
    slot_request = {
        'from': my_callsign,
        'net_id': net_id,
        'exact_duration_seconds': exact_duration,  # From model encoding
        'message_size_bytes': len(message_bytes),  # For info only
        'encoded_ready': True,
        'message_id': generate_message_id(),
        'via_relay': my_assigned_relay if needs_relay else None
    }

    # Send to net control
    send_slot_request(slot_request, net_id)

    # Step 5: Buffer opaque encoded signal
    tx_buffer[message_id] = {
        'encoded_signal': encoded_signal,  # Opaque to protocol
        'valid_until': now() + 60,         # Re-encode if stale
        'awaiting_slot': True
    }
```

### Net Control Slot Assignment

**Controller packs slots efficiently using exact durations:**

```python
def assign_net_slots(pending_requests, net_profile):
    """Assign variable-duration slots based on pre-encoded messages"""

    # Sort requests
    sorted_reqs = sorted(pending_requests, key=lambda r: (
        r.priority,              # Emergency first
        -r.exact_duration        # Then shortest (keeps net moving)
    ))

    current_time = now() + 2  # Start in 2 seconds
    slot_schedule = []

    for req in sorted_reqs:
        # Check slot fits within net limits
        if req.exact_duration > net_profile.max_slot_seconds:
            # Message too long for net profile
            reject_slot(req, reason=f"Exceeds {net_profile.max_slot_seconds}s limit")
            continue

        # Assign exact-fit slot
        slot = {
            'speaker': req.from_callsign,
            'start_time': current_time,
            'duration': req.exact_duration,  # EXACT (pre-encoded)
            'pattern': assign_temp_pattern(),  # Temporary for this slot
            'relay': req.via_relay,
            'message_hash': req.message_hash  # Verify correct message transmitted
        }

        slot_schedule.append(slot)

        # Next slot starts immediately (no gap!)
        current_time += req.exact_duration

    # Broadcast schedule to net
    broadcast_slot_schedule(slot_schedule)

    return slot_schedule
```

**Net schedule broadcast:**

```
Net Control Transmits Schedule:

Next 5 transmissions:
├─ K0BB: 0.2s (Pattern 12, direct)
├─ W2DEF: 1.3s (Pattern 15, direct)
├─ VK2ZOI: 6.4s (Pattern 18, via K0BB relay)
├─ K5XYZ: 0.8s (Pattern 21, direct)
└─ N7ABC: 2.1s (Pattern 24, direct)

Total cycle: 10.8 seconds (efficient!)

All stations know when to expect each speaker
Decode priority set automatically
```

## Net Operation Examples

### DX Net (32-byte limit, rapid)

```
Pacific DX Net - 15 stations

Slot assignments (pre-encoded, exact):
0.0-0.3s: W6NET "K0BB"
0.3-0.5s: K0BB "FN42 589" (32 bytes, BPSK via relay, 0.2s)
0.5-0.8s: W6NET "VK2ZOI"
0.8-2.4s: VK2ZOI "QF22 559" (via W6 relay, 1.6s)
2.4-2.7s: W6NET "JA1XYZ"
2.7-4.3s: JA1XYZ "PM95 569" (via W6, 1.6s)
...

15 DX stations: ~25 seconds total
36 check-ins per minute achievable
```

### Ragchew Net (256-byte limit, conversational)

```
Saturday Morning Ragchew - 8 stations

Flexible slots (pre-encoded):
0-8s: K0BB "Well folks, went to the hamfest last weekend and picked up
           a new antenna tuner. Works great on 40m..." (256 bytes, 4 patterns, 6.4s)
8-9s: [Natural pause]
9-16s: W2DEF "That sounds nice! I've been looking for a good tuner myself.
            What brand did you get?" (256 bytes, 6.4s)
16-18s: [Pause]
18-22s: K0BB "It's an MFJ-998, works really well..." (128 bytes, 3.2s)
...

Casual pace, extended messages
Natural conversation flow
```

### Emergency Net (128-byte limit, structured)

```
ARES Emergency Net - 12 stations, 3 relays

Structured check-ins (pre-encoded, confirmed):
0-1s: W1EOC "All stations, sound off. K0BB status"
1-4s: K0BB "FN42 operational, 3 operators, emergency power active"
          (128 bytes, 2 patterns, 3.2s)
4-5s: W1EOC "Copy K0BB. W2DEF status"
5-8s: W2DEF via N7 relay "FN31 operational, hospital liaison established"
          (128 bytes, relayed, 3.2s)
...

All transmissions confirmed
12 stations: ~2 minutes full check-in
Coordinated, reliable
```

### Contest Net (64-byte limit, maximum speed)

```
Contest Coordination Net - 6 stations

Rapid exchanges (pre-encoded):
0.0-0.3s: W1CC "K0BB mult?"
0.3-0.5s: K0BB "VK4 new" (32 bytes, 8-QAM, 0.2s!)
0.5-0.8s: W1CC "W2DEF QRM?"
0.8-1.0s: W2DEF "Clear here" (32 bytes, 0.2s)
1.0-1.3s: W1CC "K5XYZ rate?"
1.3-1.5s: K5XYZ "145/hr" (32 bytes, 0.2s)

6 exchanges in 1.5 seconds
240 exchanges per minute possible!
```

## Relay-Based Topology

### Automatic Relay Assignment

**Net members assigned to relays based on SNR:**

```python
def assign_members_to_relays(members, relays):
    """Each member gets primary and backup relay"""

    assignments = {}

    for member in members:
        # Find best relay for this member
        relay_scores = []

        for relay in relays:
            # Score based on member→relay SNR
            snr = member.snr_to_peers.get(relay.call, -30)
            if snr > -5:  # Viable relay
                score = snr + relay.capability_score
                relay_scores.append((relay.call, score))

        # Primary relay (best SNR)
        if relay_scores:
            relay_scores.sort(key=lambda x: x[1], reverse=True)
            assignments[member.call] = {
                'primary_relay': relay_scores[0][0],
                'backup_relay': relay_scores[1][0] if len(relay_scores) > 1 else None
            }
        else:
            # No relay needed (member has direct link to controller)
            assignments[member.call] = {
                'primary_relay': None,
                'direct': True
            }

    return assignments
```

### Hierarchical Decode Priority

**Net members decode in priority order:**

```python
def net_decode_priority(signal, net_state, my_capacity=15):
    """Priority decode for net operation (half-duplex)"""

    priorities = []

    # Priority 1: Net controller (always decode)
    priorities.append(('controller', net_state.controller_pattern, weight=100))

    # Priority 2: Current speaker (from slot schedule)
    if net_state.current_slot:
        priorities.append(('current_speaker', net_state.current_slot.pattern, weight=90))

    # Priority 3: Relays (especially my relay)
    for relay in net_state.relays:
        weight = 80 if relay == my_assigned_relay else 60
        priorities.append(('relay', relay.pattern, weight))

    # Priority 4: Other members (best-effort)
    for member in net_state.members:
        priorities.append(('member', member.pattern, weight=20))

    # Decode up to capacity, prioritized
    decoded = []
    for (type, pattern, weight) in sorted(priorities, key=lambda x: x[2], reverse=True):
        if len(decoded) >= my_capacity:
            break  # Capacity exhausted

        signal_on_pattern = extract_pattern(signal, pattern)
        if signal_detected(signal_on_pattern):
            decoded_item = model.decode_pattern(signal_on_pattern)
            decoded.append(decoded_item)

    return decoded  # Controller and current speaker always included (if present)
```

## Pre-Encoding Workflow

**Complete message transmission flow:**

```markdown
**User types message in UI:**
"Traffic for W2DEF medical update all clear"

**UI pre-encodes immediately:**
├─ Message: 41 bytes
├─ Encoding: QPSK (medium SNR to relay)
├─ Patterns: 1 pattern
├─ Exact time: 1.6 seconds
└─ Status: Ready to transmit

**User clicks "Request Slot":**
→ Request sent to net control (via relay if needed)
→ Message buffered (encoded, awaiting slot assignment)

**Net control assigns slot:**
→ "VK2ZOI: Slot at +15 seconds, Pattern 42, 1.6s duration"
→ Schedule broadcast to all net members

**At t=+15 seconds:**
→ VK2ZOI transmits pre-encoded message on Pattern 42
→ Exactly 1.6 seconds (matches slot)
→ All net members expect this (priority decode Pattern 42)
→ If relayed: K0BB (relay) rebroadcasts to full net

**Net control confirms:**
→ "Copy VK2ZOI via K0BB"
→ Next slot begins immediately
```

## Dynamic Pattern Reuse

**Patterns temporarily assigned per slot (not permanent):**

```python
# 20-person net, but only 3-4 talking in any 10-second window
NET_PATTERN_POOL = {
    'net_id': net_id,
    'reserved_patterns': [5, 12, 15, 18, 21, 24, 27, 30],  # 8 patterns
    'controller_permanent': 5,     # Controller always Pattern 5
    'member_temporary': [12, 15, 18, 21, 24, 27, 30]  # Shared among members
}

# During operation:
def assign_pattern_for_slot(speaker):
    # Find available pattern (not in use by current/recent speakers)
    recent_patterns = get_patterns_used_last_5_seconds()
    available = [p for p in member_temporary if p not in recent_patterns]

    # Assign first available
    return available[0] if available else wait_for_pattern()

# Result: 20-person net uses only ~8 patterns (not 20!)
# Remaining 56 patterns for other traffic
# Net impact on network: Minimal!
```

## Net Profiles Summary

| Profile | Max Msg | Max Slot | Typical Msg | Check-ins/min | Use Case |
|---------|---------|----------|-------------|---------------|----------|
| **DX** | 32 bytes | 1.0s | 12 bytes | 30+ | Rapid DX check-ins |
| **Ragchew** | 256 bytes | 10.0s | 150 bytes | 5-8 | Extended conversations |
| **Emergency** | 128 bytes | 5.0s | 96 bytes | 10-15 | Coordinated response |
| **Contest** | 64 bytes | 2.0s | 32 bytes | 120+ | Maximum throughput |

**Controller sets profile** when forming net, all members comply with size/timing limits.

## See Also

- **[Message Size Limits](README.md#message-format-and-size-limits)** - Protocol-wide limits
- **[Emergency Protocol](emergency_validation.md)** - Emergency message handling
- **[Signal Specification](signal_specification.md)** - Physical layer details
