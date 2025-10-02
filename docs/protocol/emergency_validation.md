# Emergency Beacon Validation Protocol

While CASCADE regulations allow anyone to transmit emergency beacons, the network uses crowd-sourced validation to prevent false alarm cascades while ensuring legitimate emergencies propagate reliably.

## Design Constraints

**Regulatory requirement**: Anyone can transmit emergency (including unlicensed operators in true emergency)

**Network requirement**: Prevent single malicious actor from causing network-wide false alarm

**Solution**: Multi-station confirmation before coordinated relay, while individual stations can still relay immediately (regulatory compliance).

## Validation Layers

### Layer 1: Transmission Validation (Immediate, Automatic)

**All stations perform basic sanity checks:**

```python
def validate_emergency_transmission(emergency_beacon):
    """Immediate checks on received emergency beacon"""

    # Check 1: Proper frequency (must be on [468, 1093] Hz)
    if emergency_beacon.frequencies != [468, 1093]:
        return {'valid': False, 'reason': 'Wrong frequency'}

    # Check 2: Proper modulation (must be BPSK, 500ms symbols)
    if emergency_beacon.modulation != 'BPSK' or emergency_beacon.symbol_duration != 500:
        return {'valid': False, 'reason': 'Invalid modulation'}

    # Check 3: Reasonable callsign hash (basic format check)
    if not is_valid_hash_format(emergency_beacon.callsign_hash):
        return {'valid': False, 'reason': 'Malformed callsign'}

    # Check 4: Not rate-limited
    if count_emergencies_from(emergency_beacon.callsign_hash, last_hour) > 5:
        return {'valid': False, 'reason': 'Rate limit exceeded', 'log': 'SUSPICIOUS'}

    # Passed basic validation
    return {'valid': True, 'confidence': 'LOW'}  # Still needs confirmation
```

**Action if Layer 1 fails**: Log suspicious activity, don't relay

### Layer 2: Multi-Station Confirmation (30-Second Window)

**Wait for independent confirmations from other stations:**

```python
def collect_confirmations(emergency_beacon, wait_seconds=30):
    """Collect confirmations from other stations"""

    confirmations = []
    start_time = now()

    while (now() - start_time) < wait_seconds:
        # Listen for confirmation ACKs on message patterns
        acks = receive_emergency_acks()

        for ack in acks:
            if ack.emergency_beacon_hash == emergency_beacon.hash:
                confirmations.append({
                    'from_callsign': ack.my_call,
                    'from_grid': ack.my_grid,
                    'snr_reported': ack.snr_report,
                    'can_relay': ack.can_relay,
                    'timestamp': now()
                })

    return confirmations
```

**Validation criteria:**

```python
def validate_via_confirmations(confirmations):
    """Check if emergency is confirmed by network"""

    # Minimum 3 confirmations
    if len(confirmations) < 3:
        return {'status': 'UNCONFIRMED', 'wait': True}

    # Geographic diversity (at least 2 different grid squares)
    unique_grids = set(c['from_grid'] for c in confirmations)
    if len(unique_grids) < 2:
        return {'status': 'SUSPICIOUS', 'reason': 'All confirmations from same location'}

    # SNR consistency (all reporters heard emergency at reasonable SNR)
    snr_reports = [c['snr_reported'] for c in confirmations]
    if max(snr_reports) - min(snr_reports) > 20:  # dB
        return {'status': 'SUSPICIOUS', 'reason': 'Inconsistent SNR reports'}

    # Timing consistency (all confirmations within 30 seconds)
    time_spread = max(c['timestamp'] for c in confirmations) - min(c['timestamp'] for c in confirmations)
    if time_spread > 30:
        return {'status': 'SUSPICIOUS', 'reason': 'Time-spread confirmations'}

    # Validated!
    return {
        'status': 'CONFIRMED',
        'confidence': 'HIGH',
        'confirming_stations': len(confirmations),
        'geographic_diversity': len(unique_grids)
    }
```

### Layer 3: Reputation Tracking (Informational, Not Enforced)

**Track historical behavior** (optional, helps users decide):

```python
class EmergencyReputationTracker:
    """Track emergency transmission history (not enforced, informational only)"""

    def __init__(self):
        self.history = {}

    def record_emergency(self, callsign, confirmed):
        """Record emergency and outcome"""

        if callsign not in self.history:
            self.history[callsign] = {
                'total_emergencies': 0,
                'confirmed': 0,
                'false_alarms': 0,
                'last_emergency': None
            }

        self.history[callsign]['total_emergencies'] += 1

        if confirmed:
            self.history[callsign]['confirmed'] += 1
        else:
            self.history[callsign]['false_alarms'] += 1

        self.history[callsign]['last_emergency'] = now()

    def get_reputation(self, callsign):
        """Get informational reputation score"""

        if callsign not in self.history:
            return {'reputation': 'UNKNOWN', 'score': 0.5}

        h = self.history[callsign]

        if h['total_emergencies'] == 0:
            return {'reputation': 'NO_HISTORY', 'score': 0.5}

        # Calculate score
        confirmation_rate = h['confirmed'] / h['total_emergencies']

        return {
            'reputation': 'TRUSTED' if confirmation_rate > 0.8 else 'QUESTIONABLE',
            'score': confirmation_rate,
            'total': h['total_emergencies'],
            'confirmed': h['confirmed'],
            'display': f"{h['confirmed']}/{h['total_emergencies']} confirmed"
        }
```

**UI Display:**

```
Emergency Beacon Received from K0BB

Validation Status: ⏳ Waiting for confirmations (1/3 received)

Station Reputation: ✓ Trusted (4/5 past emergencies confirmed)

[ Relay Immediately ]  [ Wait for Confirmations ]  [ Ignore ]

Note: You may relay any emergency per FCC Part 97
      Waiting for confirmations helps prevent false alarm cascades
```

## Relay Decision Matrix

**Stations decide relay strategy based on validation:**

| Confirmations | Reputation | Recommended Action | Why |
|---------------|------------|-------------------|-----|
| 0 (immediate) | Unknown | Relay if comfortable | Individual judgment, regulatory compliant |
| 1-2 | Unknown | Wait 30s for more | Prudent, prevents single-source false alarm |
| 3+ | Any | Relay immediately | Network consensus, high confidence |
| 0 | Poor (<50%) | Wait or skip | Past false alarms suggest caution |
| 3+ | Poor | Relay anyway | Network consensus overrides reputation |

**Key**: Individual stations always decide (no automatic relay), but validation provides information for informed decision.

## Crowd-Sourced Geographic Triangulation

**Confirmation ACKs include grid squares**, enabling triangulation:

```python
def triangulate_emergency_location(confirmations):
    """Estimate emergency location from confirmations"""

    # Extract reporter locations and SNR
    reports = [
        {'grid': c['from_grid'], 'snr': c['snr_reported']}
        for c in confirmations
    ]

    # Simple triangulation (more reports = better accuracy)
    if len(reports) >= 3:
        # Calculate center of mass weighted by SNR
        estimated_location = weighted_centroid(
            [grid_to_coords(r['grid']) for r in reports],
            weights=[snr_to_weight(r['snr']) for r in reports]
        )

        # Uncertainty estimate
        spread = max_distance(reports)

        return {
            'estimated_grid': coords_to_grid(estimated_location),
            'uncertainty_km': spread,
            'confidence': 'MEDIUM' if len(reports) < 5 else 'HIGH'
        }
    else:
        return {'estimated_grid': None, 'confidence': 'LOW'}
```

**Display to operators:**

```
Emergency Location Estimate (from 7 confirmations):
├─ Grid: FN42 (±50 km uncertainty)
├─ Confirmations from: FN41, FN42, FN43, FN32
└─ Geographic spread: 150 km (consistent)

Relay coordination: 3 stations in FN42 can provide direct assistance
                   4 stations relaying to wider network
```

## Anti-Jamming Through Limits

**Even without validation, limits prevent network collapse:**

**Single malicious actor:**
- Can send 5 emergencies/hour (rate limit)
- Each propagates 3 hops (155 relays total)
- Each expires after 5 minutes (TTL)
- Network impact: 1% overhead for 5 minutes
- **Self-limiting**: Can't sustain jamming

**Coordinated attack (10 malicious actors):**
- 10 × 5 = 50 emergencies/hour
- Each 155 relays = 7,750 relays/hour
- Spread over 50 patterns = 155 relays/hour/pattern
- Impact: 155 relays / 3600 seconds = 4.3% overhead per pattern
- **Still manageable**: 95% capacity remains

**Defense**: Protocol limits prevent emergency jamming from being effective attack vector.

## Regulatory Compliance

**FCC Part 97 compliance maintained:**
- Any station CAN transmit emergency (no permission needed)
- Any station CAN relay emergency (individual decision)
- CASCADE doesn't block emergencies (just provides validation info)
- Operators make final relay decision (informed by validation)

**Validation is advisory, not mandatory** - maintains regulatory compliance while reducing false alarm impact.

## See Also

- **[Emergency Traffic Limits](README.md#emergency-traffic-limits)** - Protocol-level abuse prevention
- **[Signal Specification](signal_specification.md)** - Emergency beacon technical spec
- **[Priority Handling](priority_handling.md)** - How emergency messages are prioritized
