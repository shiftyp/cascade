# CASCADE Examples

This directory contains example scenarios demonstrating how CASCADE's protocol and model layers work together.

## Basic QSO Example

A simple contact between two stations showing ACK-based learning:

```python
# Station W1ABC initiates contact
def basic_qso():
    # Protocol creates message
    message = {
        'from': 'W1ABC',
        'to': 'W2DEF',
        'id': 1001,
        'priority': 'NORMAL',
        'data': 'CQ from grid FN42'
    }

    # Protocol determines constraints
    constraints = ModelConstraints(
        assigned_patterns=[5, 12, 19, 27, 34],
        priority=0.5,  # NORMAL = 0.5
        max_time_seconds=5.0,
        target_callsign='W2DEF',
        kernel_hint=kernel_cache.get('W2DEF')
    )

    # Model optimizes encoding
    encoding = model.optimize_encoding(message, constraints)
    # Returns: patterns=[5, 19], redundancy=1.8, collapse=1 (16 clusters)

    # Stream fragments
    for fragment in model.fragment_generator(message, encoding):
        transmit(fragment)

    # W2DEF receives and sends ACK
    # Protocol measures SNR
    measured_snr = measure_snr(received_signal)  # Returns 2 (0 dB bucket)

    ack = {
        'type': 'ACK',
        'for_id': 1001,
        'snr': measured_snr,
        'patterns_decoded': [5, 19],
        'kernel_generated': 0xABCDEF123456
    }

    # W1ABC processes ACK
    # Update link quality
    model.update_link_quality('W2DEF', ack['snr'])

    # Store kernel hint for next transmission
    kernel_cache['W2DEF'] = ack['kernel_generated']
```

## Emergency Relay Example

Multi-hop relay with manual approval and hash-based routing:

```python
def emergency_relay():
    # W1ABC needs emergency help
    emergency = {
        'from': 'W1ABC',
        'to': 'EMRG',
        'id': 911,
        'priority': 'EMERGENCY',
        'data': {
            'destination': 'W3GHI',
            'message': 'Medical emergency at grid FN42',
            'gps': [42.3601, -71.0589]
        }
    }

    # Protocol assigns emergency patterns
    constraints = ModelConstraints(
        assigned_patterns=[0, 1, 2, 3],  # Emergency reserved
        priority=1.0,  # EMERGENCY = 1.0
        max_time_seconds=10.0,  # Extended time
        target_callsign='ALL',  # Broadcast
        kernel_hint=None
    )

    # Model uses maximum robustness
    encoding = model.optimize_encoding(emergency, constraints)
    # Returns: patterns=[0,1,2,3], redundancy=3.0, collapse=3 (binary)

    # W2DEF receives emergency
    if hash('W3GHI') in known_stations:
        # Have path to destination
        print("EMERGENCY RELAY REQUEST")
        print(f"From: W1ABC")
        print(f"To: W3GHI")
        print(f"Message: {emergency['data']['message']}")

        if operator_approves():
            # Create relay message
            relay = {
                'from': 'W2DEF',
                'to': 'W3GHI',
                'id': generate_id(),
                'priority': 'EMERGENCY',
                'data': emergency['data'],
                'relayed_from': 'W1ABC'
            }

            # Use learned link quality to W3GHI
            relay_constraints = ModelConstraints(
                assigned_patterns=get_patterns('W2DEF'),
                priority=1.0,
                max_time_seconds=10.0,
                target_callsign='W3GHI',
                kernel_hint=kernel_cache.get('W3GHI')
            )

            transmit_message(relay, relay_constraints)
```

## Mesh Formation Example

Distributed topology discovery through SNR-scaled hash exchange:

```python
def mesh_formation():
    # Track known stations
    mesh_topology = {}

    def process_transmission(message, measured_snr):
        # Update topology
        callsign_hash = hash(message['from'])
        mesh_topology[callsign_hash] = {
            'callsign': message['from'],
            'last_heard': time.time(),
            'snr': measured_snr
        }

        # Clean old entries (15 minute window)
        for h, info in list(mesh_topology.items()):
            if time.time() - info['last_heard'] > 900:
                del mesh_topology[h]

        # Determine if we should exchange hashes
        messages_sent = get_message_count(message['from'])

        if measured_snr > 10 and messages_sent % 10 == 0:
            # Strong link, frequent exchange
            include_hashes = True
        elif measured_snr > 0 and messages_sent % 25 == 0:
            # Moderate link, occasional exchange
            include_hashes = True
        elif measured_snr > -10 and messages_sent % 50 == 0:
            # Weak link, rare exchange
            include_hashes = True
        else:
            include_hashes = False

        if include_hashes:
            # Add hash table to response
            response = {
                'from': 'W2DEF',
                'to': message['from'],
                'id': generate_id(),
                'priority': 'NORMAL',
                'data': 'Roger your message',
                'mesh_hashes': list(mesh_topology.keys())[:20]  # Limit size
            }
        else:
            response = {
                'from': 'W2DEF',
                'to': message['from'],
                'id': generate_id(),
                'priority': 'NORMAL',
                'data': 'Roger your message'
            }

        return response
```

## Adaptive Fragmentation Example

Model decides fragment duration based on conditions:

```python
def adaptive_fragmentation():
    # Large file transfer
    file_data = read_file('document.pdf')  # 1MB file

    message = {
        'from': 'W1ABC',
        'to': 'W2DEF',
        'id': generate_id(),
        'priority': 'LOW',
        'data': file_data
    }

    # Check link quality history
    link_snr = get_link_snr('W2DEF')

    constraints = ModelConstraints(
        assigned_patterns=get_patterns('W1ABC'),
        priority=0.25,  # LOW
        max_time_seconds=60.0,  # 1 minute total
        target_callsign='W2DEF',
        kernel_hint=kernel_cache.get('W2DEF')
    )

    # Model adapts fragmentation to SNR
    if link_snr > 10:
        # Good link: large fragments (5 seconds)
        expected_fragments = 12
    elif link_snr > 0:
        # Moderate: medium fragments (2 seconds)
        expected_fragments = 30
    else:
        # Poor: small fragments (0.5 seconds)
        expected_fragments = 120

    # Stream fragments with ACK windows
    fragments_sent = 0
    for fragment in model.fragment_generator(message, constraints):
        transmit(fragment)
        fragments_sent += 1

        # Model indicates ACK window after every 10 fragments
        if fragments_sent % 10 == 0:
            wait_for_ack(0.5)  # 500ms window
```

## Kernel Hint Learning Example

Bidirectional optimization improving weak links:

```python
def kernel_hint_learning():
    # W1ABC struggles to decode W2DEF

    def receive_with_difficulty():
        # Receive weak signal
        signal = receive()
        initial_decode = model.decode(signal, kernel_hint=None)

        if initial_decode.success_rate < 0.5:
            # Generate optimized kernel hint
            kernel_hint = model.generate_kernel_hint(
                signal,
                initial_decode
            )

            # Try decode again with generated hint
            improved_decode = model.decode(signal, kernel_hint)

            # Include hint in ACK for W2DEF to use
            ack = {
                'type': 'ACK',
                'for_id': initial_decode.message_id,
                'snr': -5,  # Poor SNR
                'success': improved_decode.success,
                'kernel_generated': kernel_hint
            }

            return ack, improved_decode

        return None, initial_decode

    # W2DEF receives hint and uses for next transmission
    def transmit_with_hint(destination, kernel_hint):
        constraints = ModelConstraints(
            assigned_patterns=get_patterns('W2DEF'),
            priority=0.5,
            max_time_seconds=5.0,
            target_callsign=destination,
            kernel_hint=kernel_hint  # Use received hint
        )

        # Model uses hint for better encoding
        encoding = model.optimize_encoding(message, constraints)
        # Kernel hint helps model choose optimal parameters
        # for W1ABC's specific conditions
```

## Pattern Pool Rotation Example

Fair access through pattern rotation:

```python
def pattern_rotation():
    # Track pattern assignments
    pattern_pools = {}
    transmission_counts = defaultdict(int)

    def assign_patterns(callsign):
        if callsign not in pattern_pools:
            # New station: assign initial pool
            available = set(range(4, 64))  # 0-3 reserved
            for existing in pattern_pools.values():
                available -= set(existing)

            # Assign 8 patterns initially
            assigned = list(available)[:8]
            pattern_pools[callsign] = assigned
            transmission_counts[callsign] = 0

        # Check if rotation needed
        if transmission_counts[callsign] >= 100:
            # Rotate patterns for fairness
            old_patterns = pattern_pools[callsign]

            # Release old patterns
            available = set(old_patterns)

            # Get different patterns
            all_patterns = set(range(4, 64))
            used = set()
            for other_patterns in pattern_pools.values():
                used.update(other_patterns)

            new_available = all_patterns - used
            new_patterns = list(new_available)[:len(old_patterns)]

            pattern_pools[callsign] = new_patterns
            transmission_counts[callsign] = 0

            print(f"Rotated {callsign} patterns: {old_patterns} -> {new_patterns}")

        return pattern_pools[callsign]
```

## Trust State Transition Example

Authentication adaptation based on link quality:

```python
def trust_transitions():
    trust_states = {}
    trust_timers = {}

    def update_trust(callsign, event_type, snr=None):
        current_state = trust_states.get(callsign, 'UNTRUSTED')

        if event_type == 'TOTP_VERIFIED':
            # Successful TOTP verification
            trust_states[callsign] = 'TOTP_TRUSTED'
            trust_timers[callsign] = time.time()
            print(f"{callsign}: UNTRUSTED -> TOTP_TRUSTED")

        elif event_type == 'SNR_UPDATE' and current_state == 'TOTP_TRUSTED':
            # Check if eligible for HMAC
            if snr > 10:
                # High SNR
                if callsign not in trust_timers:
                    trust_timers[callsign] = time.time()
                elif time.time() - trust_timers[callsign] > 300:  # 5 minutes
                    trust_states[callsign] = 'HMAC_ALLOWED'
                    print(f"{callsign}: TOTP_TRUSTED -> HMAC_ALLOWED")
            else:
                # Reset timer if SNR drops
                trust_timers[callsign] = time.time()

        elif event_type == 'SNR_UPDATE' and current_state == 'HMAC_ALLOWED':
            # Check if should revert
            if snr < 5:
                trust_states[callsign] = 'TOTP_TRUSTED'
                print(f"{callsign}: HMAC_ALLOWED -> TOTP_TRUSTED (SNR degraded)")

        elif event_type == 'TIMEOUT':
            # No contact for 30 minutes
            if callsign in trust_timers:
                if time.time() - trust_timers[callsign] > 1800:
                    trust_states[callsign] = 'UNTRUSTED'
                    del trust_timers[callsign]
                    print(f"{callsign}: -> UNTRUSTED (timeout)")

        return trust_states.get(callsign, 'UNTRUSTED')
```

These examples demonstrate the clean separation between protocol (discrete decisions) and model (continuous optimization) while showing how they work together to achieve optimal communication.