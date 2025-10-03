# Center-Band Beacon Reservation

CASCADE reserves 175 Hz in the center of the SSB passband (1475-1625 Hz) for beacon transmission, kernel exchange, and emergency coordination. This center-band placement ensures optimal HF propagation and separates coordination traffic from message data.

## Frequency Allocation

### 175 Hz Reservation at 1475-1625 Hz

```
Total CASCADE channel: 2500 Hz (300-2800 Hz)

Lower message tones: 300-1388 Hz (35 discrete tones, 32 Hz spacing)
│
│ 75 Hz guard band
│
Beacon reservation: 1475-1625 Hz (150 Hz)
├─ Emergency alert: 1550 Hz (exact center, single tone, BPSK)
├─ 4-FSK beacon tones: [1490, 1520, 1580, 1610] Hz (symmetric around 1550 Hz)
└─ Padding: 30 Hz clear zone around emergency
│
│ 75 Hz guard band
│
Upper message tones: 1700-2788 Hz (35 discrete tones, 32 Hz spacing)
```

### Rationale for Center-Band Placement

**HF Propagation advantages:**
1. **Optimal filter response**: Center of SSB passband (300-3000 Hz typical)
2. **Flat frequency response**: No filter rolloff (edges at 300 Hz and 2800 Hz have rolloff)
3. **Ionospheric stability**: Mid-frequencies less affected by ionospheric tilt
4. **Universal compatibility**: All amateur radios pass center frequencies cleanly
5. **Multipath resilience**: Center frequencies experience less dispersion on HF
6. **Equipment variance**: Works with both narrow (1.8 kHz) and wide (2.4 kHz) SSB filters

**Operational advantages:**
1. **Always monitored**: Emergency tone (1475 Hz) never blocked by message traffic
2. **Separate from data**: No collision with 70 message tones
3. **Symmetric**: Equal message bandwidth above and below (35 + 35 tones)
4. **Acoustically pleasant**: 1550 Hz is audible but not harsh (human monitoring)

## Beacon Band Components

### Emergency Alert Tone (1475 Hz)

```python
EMERGENCY_ALERT_TONE = {
    'frequency': 1475,  # Hz (single discrete tone)
    'modulation': 'BPSK',  # On/off keying (most robust)
    'symbol_duration': 500,  # ms (very slow for maximum range)
    'bandwidth': 2,  # Hz (1 / 0.5s)

    # Always monitored by all stations
    'monitoring': 'continuous',
    'detector_type': 'simple_bandpass',  # Not full model inference
    'cpu_overhead': '<1ms per check',

    # Emergency payload
    'payload_bits': 48,  # Callsign (29) + grid (12) + type (4) + priority (2) + flag (1)
    'duration': 24,  # seconds
    'min_snr': -28,  # dB
    'power': 'MAXIMUM',

    # Purpose
    'purpose': [
        'Immediate emergency alert',
        'Trigger network clearing',
        'Provide basic identification/location',
        'Legal callsign transmission',
    ]
}
```

### 4-FSK Beacon Tones (1490, 1520, 1580, 1610 Hz)

```python
BEACON_4FSK_TONES = {
    'frequencies': [1490, 1520, 1580, 1610],  # Hz (4 discrete tones)
    'spacing': 30,  # Hz between tones
    'total_span': 90,  # Hz (1510-1600)
    'modulation': '4-FSK',  # 2 bits per symbol

    # Normal operation (kernel exchange)
    'normal_beacon': {
        'symbol_duration': 100,  # ms
        'bandwidth': 10,  # Hz per tone
        'payload_bits': 85,  # Callsign (29) + kernel (64) + status (5)
        'symbols': 43,  # 85 bits ÷ 2 bits/symbol
        'duration': 4.3,  # seconds
        'fec_rate': 0.5,  # Rate 1/2 LDPC
        'min_snr': -22,  # dB
        'power': 'NORMAL',
        'repeat_rate': 'adaptive',  # Based on kernel changes
    },

    # Emergency operation
    'emergency_negotiation': {
        'symbol_duration': 200,  # ms (more robust)
        'bandwidth': 5,  # Hz per tone
        'payload_bits': 192,  # Full emergency details
        'symbols': 96,
        'duration': 19.2,  # seconds
        'fec_rate': 0.5,
        'min_snr': -25,  # dB
        'power': 'MAXIMUM',
    },

    # Purpose
    'purposes': [
        'Kernel exchange (64-bit receiver optimization)',
        'Network coordination',
        'Emergency negotiation (phase 3)',
        'Prokernel responses (phase 4)',
        'Final kernel with relay map (phase 5)',
    ]
}
```

## Beacon Protocol (Normal Operation)

### Sparse Adaptive Beaconing

```python
def normal_beacon_strategy():
    """
    Beacons transmitted only when needed (kernel changes)
    Not continuous (reduces overhead to ~2%)
    """

    # When to beacon:
    beacon_triggers = {
        'kernel_changed': True,      # Kernel updated (new capabilities)
        'new_station': True,          # Just joined network
        'periodic_refresh': 300,      # Every 5 minutes if quiet
        'request_response': True,     # Someone requested our kernel
        'emergency_cleared': True,    # Emergency mode ended
    }

    # Beacon transmission
    beacon = {
        'callsign_full': 'K2DEF',     # 29 bits (legal ID)
        'kernel_current': 0x1234,     # 64 bits (receiver capabilities)
        'status': {
            'hardware_tier': 'RPI_CORAL',  # 3 bits
            'available_tones': [0-69],  # 64 bits (run-length encoded)
            'max_patterns': 4,           # 2 bits
            'emergency_capable': True,   # 1 bit
        },
        'total_bits': 85
    }

    # Transmit on 4-FSK tones
    transmit_beacon(
        frequencies=[1490, 1520, 1580, 1610],
        modulation='4-FSK',
        symbol_duration=100,  # ms
        payload=beacon,
        duration=4.3,  # seconds
        power='NORMAL'
    )

    # Next beacon: Only when kernel changes or 5 min timeout
    # Not every minute like before
    # Reduces beacon overhead from 10% to ~2%
```

### Beacon Detection

```python
def detect_beacons():
    """
    Monitor 4-FSK tones for beacons
    Model decodes (protocol doesn't parse)
    """

    # Continuous monitoring of beacon band
    beacon_signal = bandpass_filter(
        received_audio,
        center=1555,  # Hz (center of 4-FSK span)
        bandwidth=100  # Hz
    )

    # Model decodes 4-FSK
    # Returns: beacon type (normal vs emergency phase)
    decoded = model.decode_4fsk(
        beacon_signal,
        tones=[1490, 1520, 1580, 1610]
    )

    if decoded:
        # Extract beacon type from content
        if 'emergency_flag' in decoded and decoded['emergency_flag']:
            # Emergency negotiation (phase 3, 4, or 5)
            handle_emergency_beacon(decoded)
        else:
            # Normal beacon (kernel exchange)
            update_kernel_cache(decoded['callsign'], decoded['kernel'])
```

## Emergency Protocol (Beacon Band Usage)

### Six-Phase Emergency Flow

```
Phase 1: Emergency Alert (1475 Hz only)
├─ Duration: ~24 seconds
├─ All stations monitor this single tone
└─ Triggers: Network clearing

Phase 2: Network Clearing (1475 Hz)
├─ Duration: ~10 seconds
├─ Stations transmit "CLEARING" on 1475 Hz
└─ 4-FSK tones now reserved for emergency

Phase 3: Emergency Negotiation (4-FSK: 1510-1600 Hz)
├─ Duration: ~38 seconds
├─ Emergency station sends full details on 4-FSK
├─ Full callsign (29 bits, legal compliance)
└─ Emergency kernel (64 bits)

Phase 4: Prokernel Responses (4-FSK: 1510-1600 Hz)
├─ Duration: ~50 seconds (staggered)
├─ All hearing stations respond on 4-FSK
├─ Full callsigns for all responders (legal)
└─ Announce capabilities (911 access, relay, power)

Phase 5: Final Kernel (4-FSK: 1510-1600 Hz)
├─ Duration: ~40 seconds
├─ Emergency station sends relay network map
└─ Routing instructions (who calls 911, etc.)

Phase 6: Emergency Traffic (Message tones: 300-1388, 1700-2788 Hz)
├─ Duration: Ongoing until resolved
├─ Uses patterns [0-3] on 70 message tones
├─ Beacon band returns to normal (available for new emergencies)
└─ Multi-pattern for throughput (4× on strong links)

Total setup time: ~160 seconds (~2.7 minutes)
Then: Continuous emergency traffic on message tones
```

## Bandwidth Utilization Analysis

### Normal Operation

```python
def normal_beacon_bandwidth_usage():
    """
    Calculate beacon channel usage in normal operation
    """

    # Total beacon bandwidth: 175 Hz
    # Total CASCADE bandwidth: 2500 Hz
    # Beacon percentage: 7% of total bandwidth

    # But time utilization is low:
    beacons_per_minute = {
        'network_100_stations': 20,  # ~1 beacon per station per 5 min
        'avg_beacon_duration': 4.3,  # seconds
        'total_beacon_time': 20 * 4.3,  # 86 seconds per minute
        'duty_cycle': 86 / 60,  # 1.43 (143%)
    }

    # This seems >100%! But beacons overlap (4-FSK decoded by model)
    # Model can separate ~5 simultaneous beacons

    effective_usage = {
        'time_utilization': 0.86 / 5,  # 17.2% (5 beacons parallel)
        'bandwidth_reserved': 0.07,  # 7%
        'effective_overhead': 0.07 * 0.172,  # 1.2%
    }

    return effective_overhead  # ~1.2% total overhead

# Normal beacons: Only ~1-2% overhead (very efficient!)
```

### Emergency Operation

```python
def emergency_beacon_bandwidth_usage():
    """
    Emergency uses beacon band heavily for ~160 seconds
    Then returns to normal
    """

    emergency_phases = {
        'phase1_alert': {
            'bandwidth': 20,  # Hz (single tone + spreading)
            'duration': 24,  # seconds
            'utilization': 1.0,  # 100%
        },
        'phase2_clearing': {
            'bandwidth': 20,  # Hz
            'duration': 10,
            'utilization': 1.0,
        },
        'phase3_negotiation': {
            'bandwidth': 90,  # Hz (4-FSK span)
            'duration': 38,
            'utilization': 1.0,
        },
        'phase4_prokernels': {
            'bandwidth': 90,  # Hz
            'duration': 50,
            'utilization': 1.0,  # Multiple staggered responses
        },
        'phase5_final': {
            'bandwidth': 90,  # Hz
            'duration': 40,
            'utilization': 1.0,
        },
    }

    total_emergency_setup = sum(p['duration'] for p in emergency_phases.values())
    # = 162 seconds

    # After setup: Beacon band mostly idle
    # Emergency traffic uses message tones (patterns 0-3)

    # Amortized over 1 hour with one emergency:
    emergency_overhead = (162 / 3600) * 0.07  # 0.3%

    return {
        'setup_duration': 162,  # seconds
        'hourly_overhead': 0.003,  # 0.3% (one emergency per hour)
        'impact': 'minimal',
    }
```

## Beacon Tone Specifications

### Reference Tone Grid Gaps

```
Message tone grid (32 Hz spacing):

Lower band last tone: 1388 Hz (tone index 34)
Gap: 1388 + 32 = 1420 Hz (would be tone 35)
     1420 + 32 = 1452 Hz (would be tone 36)

Beacon reservation starts: 1475 Hz
Emergency alert: 1550 Hz (exact center of 300-2800 Hz spectrum)
4-FSK tones: 1490, 1520, 1580, 1610 Hz (symmetric around 1550 Hz)
Beacon reservation ends: 1625 Hz

Gap: 1625 + 32 = 1657 Hz
     1657 + 32 = 1689 Hz
Upper band starts at: 1700 Hz (clean spacing)

Upper band first tone: 1701 Hz (tone index 35)

Note: Tone indices 35-40 (would be 1420-1580 Hz) are SKIPPED
      Reserved for beacon band
```

### Detailed Frequency Map

```
Frequency Layout:

300 ──┬── Lower message tones (35 discrete tones)
      │   [300, 332, 364, ..., 1324, 1356, 1388] Hz
      │   Tone indices: 0-34
1388 ──┘

      ┌── Guard band (87 Hz)
1475 ──┤
      │
1550 ──┼── EMERGENCY ALERT (single tone, BPSK, exact center)
      │   ├─ Always monitored
      │   ├─ Triggers network clearing
      │   └─ Min SNR: -28 dB
      │
1510 ──┼── 4-FSK BEACON TONE 1
1540 ──┼── 4-FSK BEACON TONE 2
1570 ──┼── 4-FSK BEACON TONE 3
1600 ──┼── 4-FSK BEACON TONE 4
      │   └─ Normal: kernel exchange
      │       Emergency: negotiation, prokernels, final kernel
1625 ──┘

      └── Guard band (75 Hz)
1700 ──┬
      │   Upper message tones (35 discrete tones)
      │   [1700, 1732, 1764, ..., 2724, 2756, 2788] Hz
      │   Tone indices: 35-69
2788 ──┘
```

## Normal Beacon Format

### Kernel Exchange Beacon

```python
NORMAL_BEACON = {
    'frequencies': [1490, 1520, 1580, 1610],  # Hz (4-FSK)
    'symbol_duration': 100,  # ms
    'modulation': '4-FSK',  # 2 bits per symbol
    'bandwidth_per_tone': 10,  # Hz

    # Payload structure
    'payload': {
        'callsign': 'K2DEF',           # 29 bits (full, legal)
        'kernel': {
            'hardware_tier': 2,         # 3 bits (RPi/Coral/Desktop/GPU)
            'max_patterns': 4,          # 2 bits (1-4 simultaneous)
            'max_constellation': 6,     # 3 bits (BPSK=1, QPSK=2, 8QAM=3, 16QAM=4, 64QAM=6, 256QAM=8)
            'available_tones_encoded': 0xABCD,  # 40 bits (run-length)
            'preferred_fec_rate': 5,    # 3 bits (0.3-0.9 in steps)
            'noise_floor_quantized': 12, # 5 bits (-110 to -85 dBm)
            'total_kernel_bits': 64
        },
        'status': {
            'battery_powered': False,   # 1 bit
            'emergency_capable': True,  # 1 bit
            'can_contact_911': True,    # 1 bit
            'internet_available': True, # 1 bit
            'reserved': 0,              # 1 bit
        },
        'total_bits': 29 + 64 + 5 = 98
    },

    # With rate 1/2 FEC:
    'coded_bits': 196,
    'symbols': 98,  # 196 ÷ 2
    'duration': 9.8,  # seconds

    # Transmission strategy
    'trigger': 'kernel_change',  # Not periodic!
    'max_rate': 'once_per_5min',
    'power': 'NORMAL',
}
```

## Emergency Beacon Formats

### Phase 3: Emergency Negotiation

```python
EMERGENCY_NEGOTIATION = {
    'frequencies': [1490, 1520, 1580, 1610],  # Hz (same 4-FSK tones)
    'symbol_duration': 200,  # ms (slower for robustness)
    'modulation': '4-FSK',

    'payload': {
        'message_type': 'EMERGENCY_NEGOTIATION',  # 4 bits
        'callsign_emergency': 'W1ABC',  # 29 bits
        'grid_6char': 'FN31pr',  # 18 bits (precise)
        'lat_precise': 41.234,  # 20 bits (0.001° resolution)
        'lon_precise': -73.456,  # 21 bits (0.001° resolution)
        'emergency_type': 'MEDICAL_HEART_ATTACK',  # 8 bits
        'severity': 'CRITICAL',  # 3 bits
        'patient_count': 1,  # 4 bits
        'kernel_emergency': 0xABCD,  # 64 bits
        'situation_summary': 'CHEST_PAIN_UNCONSCIOUS',  # 24 bits
        'relay_needed': True,  # 1 bit
        'total_bits': 196
    },

    'fec_rate': 0.5,
    'coded_bits': 392,
    'symbols': 196,
    'duration': 39.2,  # seconds
    'power': 'MAXIMUM',
}
```

### Phase 4: Prokernel Response

```python
PROKERNEL_RESPONSE = {
    'frequencies': [1490, 1520, 1580, 1610],  # Hz (4-FSK)
    'symbol_duration': 200,  # ms
    'modulation': '4-FSK',

    'payload': {
        'message_type': 'PROKERNEL',  # 4 bits
        'emergency_callsign': 'W1ABC',  # 29 bits (who's in emergency)
        'responder_callsign': 'K2DEF',  # 29 bits (legal ID)
        'responder_grid': 'FN42jk',  # 18 bits
        'my_kernel': 0x5678,  # 64 bits
        'snr_heard_alert': +12,  # 8 bits
        'snr_heard_negotiation': +8,  # 8 bits
        'capabilities': {
            'can_relay': True,  # 1 bit
            'can_contact_911': True,  # 1 bit
            'can_contact_hospital': False,  # 1 bit
            'power_watts': 1500,  # 10 bits
            'antenna_gain': 6,  # 5 bits
            'hardware_tier': 2,  # 3 bits
        },
        'heard_stations': [  # Other stations I can hear
            0x3A,  # 8-bit hashes (up to 5 stations)
            0x7F,
            0xB2,
        ],
        'distance_km': 150,  # 12 bits
        'total_bits': 243
    },

    'fec_rate': 0.5,
    'coded_bits': 486,
    'symbols': 243,
    'duration': 48.6,  # seconds
    'stagger_delay': 'distance_based',  # Closer responds first
    'power': 'NORMAL',  # Save power for relay
}
```

## Beacon Channel Coordination

### Collision Avoidance

```python
def beacon_collision_avoidance():
    """
    Multiple strategies to prevent beacon collisions
    """

    strategies = {
        'normal_beacons': {
            'timing': 'random_jitter',  # Random 0-60s delay
            'rate_limit': 'max_once_per_5min',
            'listen_first': True,  # Carrier sense
            'backoff': 'exponential',  # If collision detected
        },

        'emergency_clearing': {
            'timing': 'random_0_to_3s',  # Short jitter
            'brief': True,  # Only 10 seconds total
            'expected_collisions': 0.3,  # 30% (acceptable)
        },

        'prokernels': {
            'timing': 'distance_based_stagger',  # Closer first
            'jitter': 'random_0_to_5s',
            'duration': 48.6,  # Long (sequential OK)
            'max_responses': 20,  # Practical limit
        },
    }

    # Model separates overlapping beacons
    # 4-FSK allows ~3-5 simultaneous decodes
    # Collisions rare due to staggering
```

## Return to Normal Operation

### After Emergency Resolved

```python
def return_to_normal():
    """
    Emergency resolved, beacon band returns to normal
    """

    # Emergency station broadcasts: "EMERGENCY CLEARED"
    emergency_cleared = {
        'frequency': 1475,  # Hz (emergency alert tone)
        'modulation': 'BPSK',
        'message': 'CLEARED',  # 8 bits
        'callsign': 'W1ABC',  # 29 bits (who's clearing)
        'duration': 18.5,  # seconds
        'power': 'MAXIMUM',
    }

    transmit(emergency_cleared)

    # All stations resume normal beacon operation
    # 4-FSK tones: Return to kernel exchange
    # 1475 Hz: Return to monitoring for next emergency

    # Message tones: Continue normal operation
    # Patterns [0-3]: Released from emergency reservation
```

## Spectral Efficiency Impact

### Beacon Reservation Cost

```
Total bandwidth: 2500 Hz
Message tones: 2200 Hz (70 tones × ~31 Hz effective)
Beacon reservation: 175 Hz
Guard bands: 125 Hz

Utilization:
- Message tones: 88% of bandwidth
- Beacon band: 7% of bandwidth
- Guards: 5% of bandwidth

Shannon efficiency:
- Without beacons (theoretical): 2500 Hz available
- With beacons (actual): 2200 Hz available
- Cost: 12% bandwidth reduction
- Benefit: Coordination, emergency capability

Trade-off: Worth it! ✓
- Enables kernel exchange (improves Shannon efficiency by 40%+)
- Enables emergency (life-safety capability)
- Net positive even with 12% bandwidth cost
```

## See Also

- **[Emergency Relay Network](emergency_relay_network.md)** - Full emergency protocol details
- **[Adaptive Tone Grid](adaptive_tone_grid.md)** - 70 discrete reference tones
- **[Kernel Lifecycle](kernel_lifecycle.md)** - Kernel exchange protocol
- **[Signal Specification](signal_specification.md)** - Physical layer parameters
- **[4D Pattern Envelope](../model/4d_pattern_envelope.md)** - Pattern structure
