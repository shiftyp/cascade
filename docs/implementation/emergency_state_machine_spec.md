# CASCADE Emergency Protocol State Machine Specification

**Purpose:** Formal state machine specification for emergency protocol implementation
**Status:** Ready for implementation
**Reference:** [Emergency Relay Network](../protocol/emergency_relay_network.md)

---

## Overview

CASCADE emergency protocol is a 6-phase state machine that coordinates:
- Emergency declaration (1550 Hz alert)
- Network clearing (beacon channel reservation)
- Ad-hoc relay network formation (prokernels)
- Worldwide emergency traffic relay

**Supports:** Up to 4 simultaneous emergencies via pattern separation

---

## State Machine Definition

### States

```python
EmergencyStates = {
    'NORMAL': 0,                    # No emergency active
    'EMERGENCY_ALERT': 1,           # Phase 1: Transmitting/hearing alert (1550 Hz)
    'NETWORK_CLEARING': 2,          # Phase 2: Stations clearing beacon channel
    'NEGOTIATION': 3,               # Phase 3: Emergency details on 4-FSK
    'PROKERNEL_COLLECTION': 4,      # Phase 4: Collecting relay capabilities
    'FINAL_KERNEL': 5,              # Phase 5: Relay network map
    'EMERGENCY_TRAFFIC': 6,         # Phase 6: Actual emergency data flow
}
```

### Transitions

```
NORMAL ──[Emergency Detected on 1550 Hz]──> EMERGENCY_ALERT
   ↓
EMERGENCY_ALERT ──[Alert Complete (24s)]──> NETWORK_CLEARING
   ↓
NETWORK_CLEARING ──[Cleared (10s)]──> NEGOTIATION
   ↓
NEGOTIATION ──[Negotiation Complete (38s)]──> PROKERNEL_COLLECTION
   ↓
PROKERNEL_COLLECTION ──[Prokernels Received (~48s)]──> FINAL_KERNEL
   ↓
FINAL_KERNEL ──[Final Sent (40s)]──> EMERGENCY_TRAFFIC
   ↓
EMERGENCY_TRAFFIC ──[Emergency Resolved]──> NORMAL
```

---

## State Handlers

### State: NORMAL

```python
def state_normal():
    """
    Normal operation - monitor for emergencies
    """

    while state == NORMAL:
        # Monitor emergency alert tone (1550 Hz) continuously
        signal_1550 = bandpass_filter(received, center=1550, bw=10)

        if detect_emergency_alert(signal_1550):
            # Emergency detected!
            emergency_data = decode_emergency_alert(signal_1550)

            # Transition
            state = EMERGENCY_ALERT
            emergency_context = {
                'callsign': emergency_data['callsign'],
                'grid': emergency_data['grid'],
                'type': emergency_data['type'],
                'priority': emergency_data['priority'],
                'heard_at': time.now(),
            }

            # Trigger actions
            SOUND_ALARM()
            display_emergency(emergency_context)

            # If multiple emergencies, assign ID
            emergency_id = assign_emergency_slot(emergency_context)
            # Returns: 0-3 (which of 4 emergency slots)

        sleep(0.1)  # Check every 100ms
```

### State: EMERGENCY_ALERT

```python
def state_emergency_alert(role):
    """
    role: 'originator' or 'listener'
    """

    if role == 'originator':
        # I am declaring emergency
        alert = {
            'frequency': 1550,  # Hz
            'modulation': 'BPSK',
            'callsign': my_callsign,  # Full 29-bit encoding
            'grid': my_grid_4char,  # 12 bits
            'type': emergency_type,  # 4 bits
            'priority': 3,  # CRITICAL
        }

        transmit(alert, duration=24)  # seconds

        # Transition after transmission
        state = NETWORK_CLEARING
        wait_for_clearing_responses(10)  # seconds

    else:  # listener
        # Heard emergency from someone else
        # Wait for alert to complete
        wait(24 - time_since_heard)

        # Transition
        state = NETWORK_CLEARING
        execute_clearing_protocol()
```

### State: NETWORK_CLEARING

```python
def state_network_clearing():
    """
    All stations participate in clearing
    """

    # Stop all normal beacon transmissions
    stop_normal_beacons()

    # Transmit CLEARING on 1550 Hz
    clearing = {
        'frequency': 1550,
        'modulation': 'BPSK',
        'message': 'CLEARING',
        'my_hash': hash_16bit(my_callsign),
        'duration': 10,  # seconds total for network
        'stagger': random.uniform(0, 3)  # My transmit delay
    }

    sleep(clearing['stagger'])
    transmit(clearing, duration=2)  # My clearing transmission

    # Reserve beacon channel
    beacon_channel_mode = 'EMERGENCY_ONLY'

    # Transition
    state = NEGOTIATION
```

### State: NEGOTIATION

```python
def state_negotiation(role):
    """
    Emergency station sends details on 4-FSK
    Others listen
    """

    if role == 'originator':
        # Determine my emergency ID (0-3)
        my_emergency_id = get_my_emergency_id()  # Based on declaration order

        # My beacon patterns
        beacon_patterns = [
            my_emergency_id * 4,
            my_emergency_id * 4 + 1,
            my_emergency_id * 4 + 2,
            my_emergency_id * 4 + 3
        ]  # e.g., [0,1,2,3] for emergency 1

        # Transmit negotiation on 4-FSK using Pattern 0
        negotiation = {
            'pattern': beacon_patterns[0],  # Use first beacon pattern
            'tones': [1490, 1520, 1580, 1610],
            'callsign_full': my_callsign,  # 29 bits
            'grid_6char': my_grid_precise,  # 18 bits
            'lat_lon': my_coordinates,  # 41 bits
            'emergency_details': details,  # Variable
            'kernel': my_emergency_kernel,  # 64 bits
        }

        transmit(negotiation, duration=38)  # seconds

        state = PROKERNEL_COLLECTION

    else:  # listener
        # Decode negotiation (model handles pattern separation)
        negotiation_data = model.decode_4fsk()

        # Store emergency details
        emergency_contexts[emergency_id] = negotiation_data

        # Decide if I can help
        can_relay = check_relay_capability()

        if can_relay:
            state = PROKERNEL_COLLECTION
            prepare_prokernel_response(negotiation_data)
        else:
            # Listen only, don't respond
            state = EMERGENCY_TRAFFIC
```

### State: PROKERNEL_COLLECTION

```python
def state_prokernel_collection(role):
    """
    Stations respond with capabilities
    """

    if role == 'responder':
        # Calculate stagger delay (distance-based)
        distance_km = calculate_distance(my_grid, emergency_grid)
        stagger_delay = (distance_km / 100) + random.uniform(0, 5)

        sleep(stagger_delay)

        # Determine my emergency pattern allocation
        emergency_id = which_emergency_responding_to()
        my_beacon_pattern = emergency_id * 4 + 1  # Second pattern in emergency's allocation

        # Transmit prokernel
        prokernel = {
            'pattern': my_beacon_pattern,  # Beacon pattern
            'tones': [1490, 1520, 1580, 1610],
            'emergency_call': emergency_callsign,
            'responder_call': my_callsign,  # Full 29 bits
            'my_kernel': my_kernel_64bit,
            'capabilities': {
                'can_contact_911': has_phone,
                'can_relay': True,
                'power_watts': my_power,
            },
            'heard_stations': list_of_heard,
        }

        transmit(prokernel, duration=48)  # seconds

        state = FINAL_KERNEL  # Wait for final

    else:  # originator
        # Collect prokernels for ~48 seconds
        prokernels_received = []

        for t in range(48):
            # Model decodes 4-FSK (pattern-separated)
            decoded = model.decode_4fsk()
            if decoded and decoded['type'] == 'PROKERNEL':
                prokernels_received.append(decoded)

            sleep(1)

        # Form relay network
        relay_network = form_adhoc_network(prokernels_received)

        state = FINAL_KERNEL
```

### State: FINAL_KERNEL

```python
def state_final_kernel(role):
    """
    Emergency station sends relay map
    """

    if role == 'originator':
        # Build network map
        network_map = {
            'tier1': identify_tier1(prokernels),  # Direct, strong
            'tier2': identify_tier2(prokernels),  # Regional
            'routing': {
                '911_contact': find_911_capable(),
                'hospital': find_hospital_contact(),
                'regional_hubs': find_regional(),
            }
        }

        # Transmit final kernel
        final = {
            'pattern': my_beacon_patterns[0],
            'tones': [1490, 1520, 1580, 1610],
            'network_map': network_map,
            'routing_instructions': routing,
        }

        transmit(final, duration=40)

        state = EMERGENCY_TRAFFIC

    else:  # listener
        # Receive final kernel
        final_data = model.decode_4fsk()

        # Store relay map
        my_role = find_my_role_in_network(final_data)
        # e.g., "tier1_911_contact" or "tier2_regional_relay"

        state = EMERGENCY_TRAFFIC
```

### State: EMERGENCY_TRAFFIC

```python
def state_emergency_traffic(role):
    """
    Ongoing emergency communications
    """

    if role == 'originator':
        # Transmit emergency updates
        while emergency_active:
            update = {
                'patterns': my_message_patterns,  # [64-67] for emergency 1
                'tones': ALL_MESSAGE_TONES,  # 4 tones
                'content': get_emergency_update(),
            }

            transmit(update)
            sleep(300)  # Every 5 minutes

    elif role == 'tier1':
        # Direct relay responsibilities
        while emergency_active:
            # Receive from originator
            emergency_data = receive_emergency_traffic()

            # Execute role (call 911, contact hospital, etc.)
            execute_tier1_role(emergency_data)

            # Relay to tier 2
            relay_to_tier2(emergency_data)

            sleep(60)

    elif role == 'tier2':
        # Regional relay
        ...

    # Transition out when resolved
    if emergency_resolved:
        transmit_emergency_cleared()
        state = NORMAL
```

---

## Multi-Emergency State Tracking

```python
class MultiEmergencyStateMachine:
    """
    Track up to 4 simultaneous emergencies
    """

    def __init__(self):
        self.emergencies = {
            0: {'state': 'INACTIVE', 'context': None},
            1: {'state': 'INACTIVE', 'context': None},
            2: {'state': 'INACTIVE', 'context': None},
            3: {'state': 'INACTIVE', 'context': None},
        }

    def process_emergency_alert(self, alert_data):
        """
        New emergency detected - assign slot
        """

        # Find free slot
        for slot_id in range(4):
            if self.emergencies[slot_id]['state'] == 'INACTIVE':
                # Assign to this slot
                self.emergencies[slot_id] = {
                    'state': 'EMERGENCY_ALERT',
                    'context': alert_data,
                    'beacon_patterns': [slot_id*4, slot_id*4+1, slot_id*4+2, slot_id*4+3],
                    'message_patterns': [64+slot_id*4, 65+slot_id*4, 66+slot_id*4, 67+slot_id*4],
                    'my_role': determine_role(alert_data),
                }
                return slot_id

        # All 4 slots full!
        return None  # Must wait for slot to free

    def tick(self):
        """
        Process all active emergencies
        """

        for slot_id in range(4):
            if self.emergencies[slot_id]['state'] != 'INACTIVE':
                # Process this emergency's state
                process_emergency_state(
                    slot_id,
                    self.emergencies[slot_id]
                )
```

---

## Implementation Checklist

- [ ] State machine enum definitions
- [ ] State transition handlers
- [ ] Emergency slot allocation (4 simultaneous)
- [ ] 1550 Hz alert tone detector
- [ ] 4-FSK pattern-separated decoder
- [ ] Prokernel collection and network formation
- [ ] Tier assignment algorithm
- [ ] Relay routing logic
- [ ] Emergency resolved detection

---

*Ready for implementation*
*Estimated implementation time: 2-3 weeks*
