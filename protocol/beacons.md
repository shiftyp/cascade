# Protocol-Level Beacon System

Beacons are protocol-level decisions for coordination and presence, optimized to not degrade efficiency at low SNR.

## Overview

Traditional packet radio systems often require regular beacons for network maintenance - think of AX.25 packet radio sending UI frames every few minutes, or WiFi access points beaconing every 100ms. These beacons consume significant bandwidth and can dominate the channel at low data rates.

CASCADE takes a radically different approach: beacons are rare, adaptive, and never degrade communication efficiency. At very low SNR where every bit counts, beacons may never be sent at all. The protocol makes intelligent decisions about when beaconing makes sense, while the content scales from ultra-minimal (6 bytes) to full station information (64 bytes) based on available bandwidth.

This design philosophy recognizes that at -25 dB SNR, sending a 6-byte beacon takes 48 seconds at 1 bps. That same time could transmit an actual emergency message. The protocol therefore treats beacons as optional coordination aids, not mandatory overhead.

## Beacon Decision Logic (Protocol)

The protocol decides when to send beacons based on conditions. This decision tree considers link SNR, time since last beacon, current activity level, and efficiency impact:

```python
class BeaconManager:
    """Protocol-level beacon decisions"""

    def should_send_beacon(self, link_snr, last_beacon_time, activity_level):
        """Protocol decides if beacon needed"""

        time_since_last = time.time() - last_beacon_time

        # SNR-based beacon frequency
        if link_snr < -10:
            # Very low SNR: Rarely or never
            beacon_interval = float('inf')  # No beacons
        elif link_snr < 0:
            # Low SNR: Very infrequent
            beacon_interval = 600  # 10 minutes
        elif link_snr < 10:
            # Medium SNR: Occasional
            beacon_interval = 300  # 5 minutes
        else:
            # High SNR: More frequent
            beacon_interval = 120  # 2 minutes

        # Activity-based adjustment
        if activity_level == 'emergency':
            beacon_interval /= 10  # 10× more frequent
        elif activity_level == 'idle':
            beacon_interval *= 2   # Half as frequent

        return time_since_last >= beacon_interval
```

## Adaptive Beacon Content

One of CASCADE's key innovations is scaling beacon content to match available bandwidth. Rather than always sending fixed-format beacons, the protocol adapts the information density based on link quality. This ensures beacons provide maximum value without wasting precious bandwidth.

At very low SNR, a beacon might only indicate that a station exists - just a compressed callsign. At high SNR, the same beacon can include full station capabilities, mesh topology information, and traffic availability. This scaling happens automatically based on measured link conditions.

Beacon content scales with available bandwidth:

```python
class AdaptiveBeacon:
    """Scale beacon content to SNR"""

    def create_beacon(self, snr, station_info):
        """Create appropriate beacon for conditions"""

        if snr < -15:
            # Ultra-minimal: Just presence
            return self.minimal_beacon(station_info)
        elif snr < -5:
            # Basic: Callsign + grid
            return self.basic_beacon(station_info)
        elif snr < 5:
            # Standard: Add status and capabilities
            return self.standard_beacon(station_info)
        else:
            # Full: Complete station info
            return self.full_beacon(station_info)

    def minimal_beacon(self, info):
        """6 bytes only"""
        return {
            'callsign': compress_callsign(info['callsign']),  # 6 bytes
            'type': 'MINIMAL'
        }

    def basic_beacon(self, info):
        """10 bytes"""
        return {
            'callsign': compress_callsign(info['callsign']),  # 6 bytes
            'grid': compress_grid(info['grid']),               # 4 bytes
            'type': 'BASIC'
        }

    def standard_beacon(self, info):
        """18 bytes"""
        return {
            'callsign': compress_callsign(info['callsign']),   # 6 bytes
            'grid': compress_grid(info['grid']),                # 4 bytes
            'status': info['status'],                           # 1 byte
            'capabilities': info['capabilities'],               # 2 bytes
            'snr_estimate': quantize_snr(info['snr']),         # 1 byte
            'pattern_mode': info['pattern_complexity'],         # 1 byte
            'timestamp': compress_time(time.time()),            # 3 bytes
            'type': 'STANDARD'
        }

    def full_beacon(self, info):
        """Up to 64 bytes"""
        return {
            'callsign': info['callsign'],                       # Full text
            'grid': info['grid'],
            'status': info['status'],
            'capabilities': info['capabilities'],
            'pattern_mode': info['pattern_complexity'],
            'frequency_usage': info['frequency_range'],
            'active_nets': info['nets'],
            'relay_available': info['relay'],
            'timestamp': time.time(),
            'type': 'FULL'
        }
```

## Efficiency Protection

CASCADE's most important beacon principle: never let beacons degrade communication efficiency, especially at low SNR. This section describes the mathematical framework for ensuring beacons remain helpful rather than harmful.

The efficiency check calculates the actual impact of sending a beacon versus using that same time for data. At -25 dB SNR, if the choice is between a 6-byte beacon or 6 bytes of emergency traffic, the emergency traffic wins. This prevents the common problem in packet radio where beacons consume most of the channel capacity in poor conditions.

The protocol enforces strict efficiency limits that become tighter as SNR decreases. At very low SNR, even a 1% efficiency loss from beaconing is unacceptable. This ensures the channel remains available for critical communications when conditions are marginal.

Beacons never degrade low-SNR efficiency:

```python
def beacon_efficiency_check(snr, beacon_size, message_queue):
    """Ensure beacon doesn't hurt efficiency"""

    # Calculate overhead
    beacon_time = beacon_size / get_data_rate(snr)
    useful_data_time = sum(m.size / get_data_rate(snr) for m in message_queue)
    total_time = beacon_time + useful_data_time

    efficiency_with_beacon = useful_data_time / total_time
    efficiency_without = 1.0  # No overhead

    # Only send beacon if efficiency loss is acceptable
    if snr < -10:
        max_efficiency_loss = 0.01  # 1% max at very low SNR
    elif snr < 0:
        max_efficiency_loss = 0.02  # 2% max at low SNR
    elif snr < 10:
        max_efficiency_loss = 0.05  # 5% max at medium SNR
    else:
        max_efficiency_loss = 0.10  # 10% max at high SNR

    efficiency_loss = efficiency_without - efficiency_with_beacon
    return efficiency_loss <= max_efficiency_loss
```

## Beacon Scheduling

Intelligent beacon scheduling prevents beacons from interfering with active communications. The protocol monitors channel activity and finds optimal windows for beacon transmission. This is similar to how polite speakers wait for a pause in conversation before interjecting.

During emergency traffic, beacons are completely suppressed - no network maintenance is worth interfering with life safety communications. The protocol also identifies quiet periods in the channel's natural rhythm, inserting beacons when they cause least disruption.

When patterns are selected for beaconing, the protocol chooses the least active ones to minimize collision probability. This creates a self-organizing system where beacons naturally avoid busy frequencies.

Protocol schedules beacons to minimize interference:

```python
class BeaconScheduler:
    """Smart beacon scheduling"""

    def schedule_beacon(self, current_transmissions):
        """Find optimal beacon slot"""

        # Never interrupt emergency traffic
        if any(t.priority == 'EMERGENCY' for t in current_transmissions):
            return None  # Don't beacon now

        # Look for quiet periods
        quiet_window = self.find_quiet_window(current_transmissions)

        if quiet_window:
            return {
                'time': quiet_window.start,
                'duration': quiet_window.duration,
                'patterns': self.least_used_patterns()
            }

        # If no quiet period, skip beacon
        return None

    def least_used_patterns(self):
        """Select patterns with least activity"""
        pattern_activity = measure_pattern_occupancy()
        # Use least active patterns for beacon
        return sorted(pattern_activity.items(), key=lambda x: x[1])[:2]
```

## Integration with Model

The separation between protocol and model is particularly clear with beacons. The protocol makes the discrete decision of WHETHER and WHEN to beacon based on efficiency calculations and activity monitoring. The model then optimizes HOW to transmit that beacon - which patterns to use, what encoding parameters, and how to adapt to current channel conditions.

This separation ensures beacons remain a protocol-level coordination tool while benefiting from the model's sophisticated channel adaptation. The protocol ensures beacons don't harm efficiency; the model ensures they get through reliably when sent.

The protocol decides WHEN to beacon, the model optimizes HOW:

```python
# Protocol decision
if beacon_manager.should_send_beacon(snr, last_beacon):
    # Protocol creates beacon content
    beacon = adaptive_beacon.create_beacon(snr, station_info)

    # Model optimizes transmission
    encoding = model.optimize_encoding(
        beacon,
        ModelConstraints(
            priority=0.1,  # Low priority
            patterns=protocol.beacon_patterns(),
            max_time=beacon_time_limit(snr)
        )
    )

    # Only transmit if efficiency preserved
    if beacon_efficiency_check(snr, len(beacon), message_queue):
        transmit(beacon, encoding)
```

## Beacon Types

CASCADE defines three beacon types, each serving different purposes and used in different conditions:

### Presence Beacon
The absolute minimum beacon, used when bandwidth is precious:
- **Size**: 6 bytes only
- **Content**: Compressed callsign
- **Purpose**: "I exist" signal
- **When Used**: Very low SNR, bandwidth constrained
- **Value**: Allows others to know station is reachable
- **Time at -20 dB**: ~24 seconds to transmit

### Capability Beacon
Moderate information density for operational coordination:
- **Size**: 10-18 bytes
- **Content**: Callsign, grid square, basic capabilities
- **Purpose**: Enable intelligent routing decisions
- **When Used**: Medium SNR, normal operations
- **Value**: Helps mesh optimize traffic flow
- **Includes**: Pattern complexity support, relay availability

### Coordination Beacon
Full information for advanced mesh operations:
- **Size**: Up to 64 bytes
- **Content**: Complete station profile and network state
- **Purpose**: Enable sophisticated multi-hop routing
- **When Used**: High SNR only
- **Value**: Full mesh topology awareness
- **Includes**: Active nets, frequency usage, relay rules

## Beacon Reception

When beacons are received, the protocol extracts maximum value from minimal information. Each beacon updates multiple system databases, building a picture of network topology and capabilities over time. This incremental learning approach means even rare beacons contribute to mesh intelligence.

The protocol maintains separate databases for link quality, station capabilities, and mesh topology. These databases decay over time, with older information weighted less in routing decisions. This creates a system that adapts to changing conditions without requiring constant beacon traffic.

Protocol processes received beacons:

```python
def process_beacon(beacon, measured_snr):
    """Update station database from beacon"""

    station_id = beacon['callsign']

    # Update link quality matrix
    link_quality_matrix.update(station_id, measured_snr)

    # Update capability database
    if beacon['type'] in ['STANDARD', 'FULL']:
        capability_db[station_id] = {
            'pattern_mode': beacon.get('pattern_mode'),
            'relay_available': beacon.get('relay_available'),
            'last_heard': time.time()
        }

    # Update mesh topology
    if 'active_nets' in beacon:
        mesh_topology.update(station_id, beacon['active_nets'])
```

## Benefits

The protocol-level beacon system provides crucial advantages:

1. **Protocol Control**: Deterministic beacon decisions
   - Beacons follow clear rules, not probabilistic models
   - Operators can predict and control beacon behavior
   - Debugging and troubleshooting remain straightforward

2. **Efficiency First**: Never degrades low-SNR performance
   - Mathematical guarantee that beacons won't harm weak signal work
   - Automatic suppression when channel capacity is limited
   - Priority always given to actual traffic over maintenance

3. **Adaptive Content**: Information scales with bandwidth
   - 6-byte minimums for existence proof
   - 64-byte maximums for full coordination
   - Smooth scaling between extremes based on link quality

4. **Smart Scheduling**: Avoids interference with traffic
   - Never interrupts emergency communications
   - Finds natural quiet periods in channel rhythm
   - Uses least-active patterns to minimize collisions

5. **Coordination**: Enables multi-station awareness when possible
   - Builds topology knowledge incrementally
   - Enables sophisticated routing when conditions permit
   - Degrades gracefully to point-to-point when necessary

## Design Philosophy

CASCADE's beacon system embodies the principle that protocol overhead should never compromise primary communication goals. In emergency situations at marginal SNR, every bit should carry emergency traffic, not network maintenance. Beacons exist to enhance communication when excess capacity exists, not as mandatory overhead that degrades service.

This philosophy stands in contrast to traditional packet systems where beacon overhead can consume 50% or more of channel capacity at low data rates. CASCADE ensures that beacons remain helpers, not hindrances, especially when communication margins are thin.