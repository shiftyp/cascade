# CASCADE Emergency Relay Network Protocol

CASCADE implements a self-organizing ad-hoc relay network for emergency communications. When an emergency is declared, the network automatically forms relay tiers based on signal strength and geographic distribution, enabling worldwide emergency traffic coordination without infrastructure.

## Emergency Protocol Overview

### Six-Phase Emergency Protocol

1. **Emergency Alert** (1550 Hz BPSK, ~24s): Single-tone alert with callsign and grid (exact center of spectrum)
2. **Network Clearing** (immediate): All stations stop beacon transmissions, transmit "CLEARING"
3. **Emergency Negotiation** (4-FSK, ~37s): Full details, location, kernel on beacon channel
4. **Prokernel Responses** (4-FSK, staggered): Stations announce relay capabilities
5. **Final Kernel** (4-FSK, ~26s): Emergency station sends relay network map
6. **Emergency Traffic** (message tones): Actual emergency data relayed via ad-hoc net

---

## Multiple Simultaneous Emergencies (4D Pattern Separation)

CASCADE can handle **up to 4 simultaneous emergencies** using 4D pattern orthogonality, without requiring additional frequency allocation.

### Emergency Pattern Allocation

**8 emergency patterns (0-7) divided among emergencies:**

```python
EMERGENCY_PATTERN_ALLOCATION = {
    'total_emergency_patterns': 32,  # 16 beacon + 16 message reserved
    'max_simultaneous': 4,  # Up to 4 concurrent emergencies

    'allocation': {
        'emergency_1': {
            'beacon_patterns': [0, 1, 2, 3],  # 4 beacon patterns (4-FSK)
            'message_patterns': [64, 65, 66, 67],  # 4 message patterns (70 tones)
            'alert_tone': 1550,  # Hz (shared with all)
            'alert_offset': 0,  # seconds (first to alert)
            'freq_preference': range(0, 25),  # Lower frequency bias (message tones)
            'example': 'Hurricane Maria (Puerto Rico)',
            'benefit': '4× beacon throughput for negotiation'
        },

        'emergency_2': {
            'beacon_patterns': [4, 5, 6, 7],  # 4 beacon patterns
            'message_patterns': [68, 69, 70, 71],  # 4 message patterns
            'alert_tone': 1550,  # Hz (same tone)
            'alert_offset': 15,  # seconds (staggered)
            'freq_preference': range(20, 50),  # Mid-frequency bias
            'example': 'Haiti Earthquake'
        },

        'emergency_3': {
            'beacon_patterns': [8, 9, 10, 11],
            'message_patterns': [72, 73, 74, 75],
            'alert_tone': 1550,  # Hz
            'alert_offset': 30,  # seconds
            'freq_preference': range(45, 70),  # Upper frequency bias
            'example': 'Maritime MAYDAY'
        },

        'emergency_4': {
            'beacon_patterns': [12, 13, 14, 15],
            'message_patterns': [76, 77, 78, 79],
            'alert_tone': 1550,  # Hz
            'alert_offset': 45,  # seconds
            'freq_preference': range(0, 70),  # Can use any (fills gaps)
            'example': 'McMurdo Infrastructure'
        },
    },

    'normal_operation': {
        'beacon_patterns_available': 48,  # IDs 16-63 (anti-kernel resilient)
        'message_patterns_available': 176,  # IDs 80-255
        'typical_dx_pool': 128,  # IDs 80-207 (most HF operation)
    }
}
```

### How Simultaneous Emergencies Work

#### Phase 1: Time-Staggered Alerts

```python
# Emergency 1 (Hurricane Maria) starts at T=0:
alert_1 = transmit_emergency_alert(
    frequency=1550,  # Hz
    callsign='KP4XXX',
    grid='FK68',
    type='HURRICANE_CAT5',
    start_time=0,
    duration=24  # seconds
)

# Emergency 2 (Haiti Earthquake) starts at T=15s:
# Overlaps with Emergency 1 by 9 seconds!
alert_2 = transmit_emergency_alert(
    frequency=1550,  # Hz (SAME tone)
    callsign='HH2XXX',
    grid='FK48',
    type='EARTHQUAKE_7.8',
    start_time=15,  # Staggered
    duration=24
)

# Stations hear BOTH alerts (overlapped on same frequency)
# Model separates by:
# - Different bit patterns (callsigns differ)
# - Time offset (15s stagger helps)
# - Successive decoding (decode strong one first, cancel, decode second)
```

#### Phase 3-5: Pattern-Separated Negotiation

**Key innovation:** Negotiations use different patterns on 4-FSK tones simultaneously!

```python
# All 4 emergencies negotiate in parallel:

# Emergency 1: Pattern 0 on 4-FSK
negotiate_1 = {
    'pattern': 0,  # Emergency 1's pattern
    'tones': [1490, 1520, 1580, 1610],  # 4-FSK
    'callsign': 'KP4XXX',
    'details': 'Hurricane Maria...',
    'duration': 38  # seconds
}

# Emergency 2: Pattern 2 on SAME 4-FSK tones (simultaneously!)
negotiate_2 = {
    'pattern': 2,  # Emergency 2's pattern (orthogonal to pattern 0)
    'tones': [1490, 1520, 1580, 1610],  # Same frequencies!
    'callsign': 'HH2XXX',
    'details': 'Haiti earthquake...',
    'duration': 38  # Parallel with E1
}

# Emergency 3 & 4: Patterns 4 and 6 (also parallel)

# Pattern orthogonality allows all 4 to transmit simultaneously
# on the SAME 4-FSK tones
# Model separates via pattern correlation
# This is the power of 4D separation!
```

#### Phase 6: Fully Parallel Emergency Traffic

```python
# All 4 emergencies transmit simultaneously on message tones:

# Emergency 1 (Hurricane):
traffic_1 = {
    'patterns': [0, 1],
    'tones': ALL_MESSAGE_TONES,  # All 70 tones
    'freq_bias': 'lower',  # Prefer 0-25 (reduces collision)
    'throughput': '2 patterns × 80 bps = 160 bps'
}

# Emergency 2 (Earthquake):
traffic_2 = {
    'patterns': [2, 3],
    'tones': ALL_MESSAGE_TONES,  # All 70 tones
    'freq_bias': 'mid',  # Prefer 20-50 (different from E1)
    'throughput': '2 patterns × 80 bps = 160 bps'
}

# Emergency 3 & 4: Similar

# Separation:
# - Different patterns (128-pattern orthogonality)
# - Frequency preferences reduce collision (smart routing)
# - 4D correlation separates overlaps
# - All 4 coexist in 2.5 kHz bandwidth!
```

### Timeline Example: 2 Simultaneous Emergencies

```
Time    Emergency 1 (Hurricane)           Emergency 2 (Earthquake)
────────────────────────────────────────────────────────────────────
0s      Alert on 1550 Hz (KP4XXX)         (Not started yet)
15s     Alert continuing...                Alert on 1550 Hz (HH2XXX)
        (9s overlap on same tone!)         (Overlaps with E1!)
24s     Alert complete → clearing          Alert continuing...
34s     Negotiation (Pattern 0, 4-FSK)     Alert complete → clearing
39s     Negotiation continuing...          Clearing complete
44s     Negotiation continuing...          Negotiation (Pattern 2, 4-FSK)
                                           (SIMULTANEOUS on same 4-FSK!)
72s     Negotiation complete               Negotiation continuing...
        Prokernels (pattern 0)             Negotiation continuing...
82s     Prokernels continuing...           Negotiation complete
                                           Prokernels (pattern 2)
120s    Prokernels complete                (Both collecting prokernels)
        Final kernel (pattern 0)           Prokernels continuing...
130s    Final complete                     Prokernels complete
        Traffic starts (patterns 0,1)      Final kernel (pattern 2)
160s    Traffic ongoing...                 Final complete
                                           Traffic starts (patterns 2,3)

Both traffic flows simultaneously on message tones (patterns 0,1 vs 2,3)
Separated by 4D orthogonality
280 user capacity shared across both emergencies + normal traffic
```

### Capacity Allocation

```python
def emergency_capacity_sharing():
    """
    How 280 user capacity divides with multiple emergencies
    """

    scenarios = {
        'single_emergency': {
            'emergency_users': 40,  # 20 stations × 2 patterns avg
            'normal_traffic': 240,  # Continues during emergency
            'total': 280,
        },

        'two_emergencies': {
            'emergency_1_users': 30,  # 15 stations × 2 patterns
            'emergency_2_users': 30,  # 15 stations × 2 patterns
            'normal_traffic': 220,  # Slightly reduced
            'total': 280,
        },

        'four_emergencies': {
            'each_emergency_users': 20,  # 10 stations × 2 patterns each
            'total_emergency': 80,
            'normal_traffic': 200,  # Still 200 users for normal!
            'total': 280,
        },
    }

    # Key: Even with 4 emergencies, normal traffic continues!
    # 4D separation allows graceful capacity sharing
```

### Real-World Scenario: Cascading Disasters

```
September 2017 (actual events):

Week 1:
- Hurricane Irma (Florida) - Emergency 1
  Uses patterns [0,1], freq preference lower tones

Week 2:
- Hurricane Maria (Puerto Rico) - Emergency 2
  Uses patterns [2,3], freq preference mid tones
  Both emergencies active simultaneously!

Week 3:
- Mexico 7.1 Earthquake - Emergency 3 (if CASCADE existed)
  Uses patterns [4,5], freq preference upper tones
  Three emergencies parallel!

Week 4:
- All three continuing recovery operations
  + Normal traffic (contests, DX, nets)
  = 4D separation handles all simultaneously
```

---

## Phase 1: Emergency Alert (Single Tone)

### Time-Staggered Alert Protocol

**Multiple emergencies use same 1550 Hz tone with 15-second stagger:**

```python
def multi_emergency_alert_protocol():
    """
    When multiple emergencies declared within short time
    """

    # Station detects multiple emergencies needed
    emergencies_pending = [
        {'call': 'KP4XXX', 'type': 'HURRICANE'},
        {'call': 'HH2XXX', 'type': 'EARTHQUAKE'},
    ]

    # Automatic stagger assignment
    for i, emerg in enumerate(emergencies_pending):
        emerg['alert_start'] = i * 15  # seconds
        emerg['patterns'] = [i*2, i*2+1]  # Pattern allocation

        # Transmit alert
        transmit_emergency_alert(
            frequency=1550,  # All use same tone
            callsign=emerg['call'],
            start_time=emerg['alert_start'],
            duration=24
        )

    # Network hears: Multiple emergencies
    # 15s stagger prevents total collision (9s overlap acceptable)
    # Model can decode overlapped section if signals strong enough
    # If not: Each emergency gets majority of alert time clear
```

---

## Phase 1: Emergency Alert (Single Tone)

### Emergency Alert Tone Specification

```python
EMERGENCY_ALERT = {
    # Single dedicated tone (always monitored)
    'frequency': 1550,  # Hz (exact center of 300-2800 Hz spectrum)
    'modulation': 'BPSK',  # Most robust
    'symbol_duration': 500,  # ms (very slow for maximum range)
    'bandwidth': 2,  # Hz (1/0.5s symbol duration)

    # Payload (full callsign for legal compliance)
    'payload': {
        'emergency_flag': 1,       # 1 bit (always 1)
        'callsign_full': 'W1ABC',  # 29 bits (encoded efficiently)
        'grid_4char': 'FN31',      # 12 bits
        'emergency_type': 5,       # 4 bits (MEDICAL, FIRE, WEATHER, etc.)
        'priority': 3,             # 2 bits (CRITICAL=3, HIGH=2, MEDIUM=1)
        'total_bits': 48
    },

    # Transmission characteristics
    'duration': 24,  # seconds (48 bits × 0.5s per bit)
    'power': 'MAXIMUM',  # Full legal power (1.5 kW if available)
    'min_snr': -28,  # dB (extremely robust)
    'range_100w': 40000,  # km (global coverage potential)
    'repeat': 'continuous',  # Until network clears
}
```

### Callsign Encoding (29 bits)

```python
def encode_callsign(callsign):
    """
    Efficiently encode amateur radio callsign

    Format: [prefix][number][suffix]
    Examples: W1ABC, K2DEF, VE3XYZ, G4MNO, JA1XYZ

    Encoding:
    - Prefix (1-2 chars): 10 bits (supports 1024 prefixes)
    - Number (1 digit): 4 bits (0-9)
    - Suffix (1-3 chars): 15 bits (supports 32768 combinations)
    Total: 29 bits
    """

    # Parse callsign
    prefix = extract_prefix(callsign)  # "W", "VE", "G", etc.
    number = extract_number(callsign)  # 0-9
    suffix = extract_suffix(callsign)  # "ABC", "XYZ", etc.

    # Encode prefix (base-26 or base-676)
    if len(prefix) == 1:
        prefix_value = ord(prefix) - ord('A')  # 0-25
    else:  # 2 chars
        prefix_value = (ord(prefix[0]) - ord('A')) * 26 + \
                      (ord(prefix[1]) - ord('A')) + 26  # 26-701

    # Encode suffix (base-26³ for 1-3 chars)
    suffix_value = 0
    for i, char in enumerate(reversed(suffix)):
        suffix_value += (ord(char) - ord('A')) * (26 ** i)

    # Pack into 29 bits
    encoded = (prefix_value << 19) | (number << 15) | suffix_value

    return encoded  # 29 bits

# Examples:
# W1ABC: prefix=22(W), num=1, suffix=ABC(28) → 0b...10110_0001_000000000011100
# VE3XYZ: prefix=547, num=3, suffix=XYZ → 29 bits
# G4MNO: prefix=6(G), num=4, suffix=MNO → 29 bits
```

### Emergency Types

```python
EMERGENCY_TYPES = {
    0: 'GENERAL',      # Unspecified emergency
    1: 'MEDICAL',      # Medical emergency, injury, illness
    2: 'FIRE',         # Fire, explosion
    3: 'WEATHER',      # Hurricane, tornado, flooding
    4: 'SEARCH_RESCUE', # Missing person, SAR
    5: 'INFRASTRUCTURE', # Power, water, communications down
    6: 'CIVIL',        # Civil unrest, evacuation
    7: 'MARITIME',     # Vessel in distress
    8: 'AVIATION',     # Aircraft emergency
    9: 'HAZMAT',       # Hazardous materials
    10: 'EARTHQUAKE',  # Seismic event
    11: 'TSUNAMI',     # Tsunami warning/impact
    12: 'NUCLEAR',     # Nuclear incident
    13: 'TEST',        # Practice drill (low priority)
    14: 'CANCEL',      # Cancel previous emergency
    15: 'UPDATE',      # Update to ongoing emergency
}
```

### Emergency Alert Detection

```python
def continuous_emergency_monitor():
    """
    All stations monitor 1550 Hz continuously
    Simple detector, <1ms overhead
    """

    while True:
        # Bandpass filter at emergency frequency
        emergency_signal = bandpass_filter(
            received_audio,
            center=1550,  # Hz (exact spectrum center)
            bandwidth=10  # Hz (wide enough for drift)
        )

        # Simple BPSK tone detector
        power = np.abs(emergency_signal) ** 2
        threshold = noise_floor * 100  # -20 dB SNR threshold

        if np.mean(power) > threshold:
            # Potential emergency - decode
            decoded_bits = decode_bpsk(emergency_signal, duration=24)

            if decoded_bits[0] == 1:  # Emergency flag
                # EMERGENCY DETECTED
                emergency_data = parse_emergency_alert(decoded_bits)

                # Immediate actions
                SOUND_ALARM()
                display_emergency(emergency_data)
                execute_phase2_clearing()

        sleep(0.1)  # Check every 100ms
```

---

## Phase 2: Network Clearing

### Clearing Protocol

When stations hear emergency alert on 1550 Hz, they immediately transmit "CLEARING":

```python
def execute_network_clearing():
    """
    Station heard emergency - participate in network clearing
    """

    # STEP 1: Stop all beacon transmissions
    stop_normal_beacons()
    stop_kernel_exchanges()

    # STEP 2: Transmit CLEARING signal on emergency tone
    clearing_transmission = {
        'frequency': 1550,  # Hz (same emergency tone)
        'modulation': 'BPSK',
        'payload': {
            'message_type': 'CLEARING',    # 4 bits
            'my_callsign_hash': hash_self, # 16 bits (hash OK here, not legal ID)
            'total_bits': 20
        },
        'duration': 10,  # seconds
        'power': 'NORMAL',
        'stagger_delay': random.uniform(0, 3)  # Avoid collision
    }

    # Random delay before transmitting
    sleep(clearing_transmission['stagger_delay'])
    transmit(clearing_transmission)

    # STEP 3: Enter emergency mode
    emergency_mode = {
        'beacon_channel': 'RESERVED_FOR_EMERGENCY',
        'message_tones': 'CONTINUE_NORMAL',  # Don't stop regular traffic!
        'monitoring': '4FSK_NEGOTIATION',
        'ready_to_relay': True
    }

    set_mode(emergency_mode)
```

### Beacon Channel Status

```
NORMAL OPERATION:
├─ 1550 Hz: Silent (monitored for emergency)
└─ [1490, 1520, 1580, 1610] Hz: Normal beacons, kernel exchange

EMERGENCY DECLARED (Phase 1-2):
├─ 1550 Hz: Emergency alert + CLEARING responses
└─ 1510-1600 Hz: Paused (waiting for negotiation)

EMERGENCY NEGOTIATION (Phase 3-5):
├─ 1475 Hz: Silent (alert complete)
└─ 1510-1600 Hz: Emergency negotiation, prokernels, final kernel

EMERGENCY TRAFFIC (Phase 6):
├─ 1550 Hz: Silent (available for new emergencies)
├─ 1510-1600 Hz: Reserved (emergency can use if needed)
└─ Message tones (70): Emergency traffic on patterns [0-3]
```

---

## Phase 3: Emergency Negotiation

### Negotiation on 4-FSK Beacon Tones

```python
EMERGENCY_NEGOTIATION = {
    # Use 4-FSK beacon tones (now cleared)
    'frequencies': [1490, 1520, 1580, 1610],  # Hz
    'modulation': '4-FSK',  # 2 bits per symbol
    'symbol_duration': 200,  # ms
    'bandwidth': 5,  # Hz per tone

    # Detailed emergency payload
    'payload': {
        'callsign_full': 'W1ABC',         # 29 bits (legal compliance)
        'grid_6char': 'FN31pr',           # 18 bits (precise location)
        'latitude': 41.234,                # 20 bits (0.001° precision)
        'longitude': -73.456,              # 21 bits (0.001° precision)
        'emergency_type_detailed': 'MEDICAL_HEART_ATTACK',  # 8 bits
        'severity': 'CRITICAL',            # 3 bits (CRITICAL/HIGH/MEDIUM/LOW)
        'patient_count': 1,                # 4 bits (1-15 patients)
        'kernel_emergency': 0xABCD,        # 64 bits (emergency kernel)
        'relay_request': True,             # 1 bit
        'message_preview': 'CHEST_PAIN',   # 24 bits (keywords)
        'total_bits': 192
    },

    # FEC for robustness
    'fec_rate': 0.5,      # Rate 1/2 LDPC
    'coded_bits': 384,    # 192 × 2
    'symbols': 192,       # 384 bits ÷ 2 bits/symbol
    'duration': 38.4,     # seconds
    'power': 'MAXIMUM',
}
```

### Why Full Callsign is Critical

- **Legal compliance**: FCC requires full callsign identification
- **911 coordination**: Dispatchers need real callsign for contact
- **Ham radio lookup**: Callsign databases provide contact info, address
- **Verification**: Prevents false emergencies (hash would be anonymous)
- **International**: DX stations recognize callsign prefix/country

---

## Phase 4: Prokernel Responses

### Stations Announce Relay Capabilities

```python
PROKERNEL_RESPONSE = {
    # Same 4-FSK beacon tones
    'frequencies': [1510, 1540, 1570, 1600],
    'modulation': '4-FSK',
    'symbol_duration': 200,  # ms

    # Relay station identifies itself
    'payload': {
        'emergency_callsign': 'W1ABC',     # 29 bits (who's in emergency)
        'responder_callsign': 'K2DEF',     # 29 bits (full callsign - legal)
        'responder_grid_6char': 'FN42jk',  # 18 bits (my location)
        'my_kernel_emergency': 0x5678,     # 64 bits (my decoder kernel)
        'snr_heard_alert': +12,            # 8 bits (how well I heard)
        'snr_heard_negotiation': +8,       # 8 bits

        # Relay capabilities
        'can_relay_radio': True,           # 1 bit (can relay RF)
        'can_contact_911': True,           # 1 bit (have phone/internet)
        'can_contact_hospital': False,     # 1 bit
        'power_watts': 1500,               # 10 bits (1-1500W)
        'antenna_gain_dbi': 6,             # 5 bits (0-31 dBi)
        'hardware_tier': 'RPI_CORAL',      # 3 bits (RPi/Coral/Desktop/GPU)

        # Network topology
        'heard_emergency': True,           # 1 bit
        'heard_stations': [                # 40 bits (up to 5 stations)
            'N3GHI',  # 8 bits (hash)
            'W4JKL',  # 8 bits
            'K5MNO',  # 8 bits
        ],
        'distance_to_emergency_km': 150,   # 12 bits (0-4095 km)

        'total_bits': 237
    },

    'fec_rate': 0.5,
    'coded_bits': 474,
    'symbols': 237,
    'duration': 47.4,  # seconds
    'stagger_delay': random.uniform(0, 10),  # Avoid collision
}
```

### Staggered Response to Avoid Collisions

```python
def respond_with_prokernel(emergency_alert, negotiation):
    """
    Respond to emergency with capabilities
    Stagger based on distance (closer responds first)
    """

    # Calculate distance to emergency
    my_grid = 'FN42jk'
    emergency_grid = emergency_alert['grid']
    distance_km = calculate_distance(my_grid, emergency_grid)

    # Stagger delay: farther stations wait longer
    # This prioritizes nearby stations (better for local 911 coordination)
    base_delay = distance_km / 100  # 1 second per 100 km
    random_jitter = random.uniform(0, 5)  # Avoid exact collisions
    total_delay = base_delay + random_jitter

    # Wait, then transmit
    sleep(total_delay)
    transmit_prokernel(PROKERNEL_RESPONSE)
```

---

## Phase 5: Final Kernel (Relay Network Map)

### Emergency Station Sends Network Map

After collecting prokernels, emergency station broadcasts relay network structure:

```python
FINAL_KERNEL_WITH_NETWORK_MAP = {
    'frequencies': [1510, 1540, 1570, 1600],
    'modulation': '4-FSK',
    'symbol_duration': 200,  # ms

    'payload': {
        'emergency_callsign': 'W1ABC',  # 29 bits
        'final_kernel': 0xFINAL,        # 64 bits (optimized for relay net)

        # Relay network tiers (organized by signal strength/distance)
        'tier1_stations': [  # Direct to emergency
            {'call': 'K2DEF', 'role': 'CALL_911'},      # 29 + 3 bits
            {'call': 'N3GHI', 'role': 'HOSPITAL'},      # 29 + 3 bits
        ],

        'tier2_stations': [  # Via tier1 relay
            {'call': 'W4JKL', 'via': 'N3GHI'},          # 29 + 8 bits
            {'call': 'K3STU', 'via': 'K2DEF'},          # 29 + 8 bits
        ],

        'tier3_stations': [  # Long distance DX
            {'call': 'K5MNO', 'via': 'W4JKL'},          # 29 + 8 bits
            {'call': 'W6VWX', 'via': 'K5MNO'},          # 29 + 8 bits
            {'call': 'VK2ABC', 'via': 'W6VWX'},         # 29 + 8 bits (Australia!)
        ],

        # Routing instructions
        'primary_911': 'K2DEF',        # 8 bits (hash of tier1 call)
        'backup_911': 'N3GHI',         # 8 bits
        'hospital_contact': 'N3GHI',   # 8 bits
        'worldwide_awareness': 'tier3', # All tier3 relay globally

        # Message priorities
        'immediate_relay': ['SITUATION_UPDATE', 'MEDEVAC_ETA'],
        'periodic_relay': ['STATUS_CHECK'],

        'total_bits': ~400  # Variable based on network size
    },

    'duration': 40,  # seconds (approx)
    'power': 'MAXIMUM',
}
```

---

## Phase 6: Emergency Traffic Relay

### Emergency Message Structure

```python
EMERGENCY_MESSAGE = {
    # Use message tones (NOT beacon channel)
    'tones': LOWER_TONES + UPPER_TONES,  # All 70 reference tones
    'patterns': [64, 65, 66, 67],  # Emergency 1 message patterns (4 total)

    # Can use all 4 message patterns simultaneously
    'patterns_simultaneous': 4,  # 4× throughput for urgent traffic
    # Plus Emergency 1 has beacon patterns [0,1,2,3] for negotiation

    # Conservative modulation for reliability
    'modulation': 'QPSK',  # 2 bits/symbol
    'fec_rate': 0.33,      # Heavy FEC (rate 1/3)
    'symbol_duration': 50,  # ms (standard)

    # Actual emergency content (realistic large-scale disaster scenario)
    # Emergency 1 uses patterns [64-67] for message traffic
    # Can use 4 patterns simultaneously for 4× throughput if needed
    'payload': """
        PRIORITY EMERGENCY - HURRICANE CATEGORY 5 LANDFALL

        Location: San Juan, Puerto Rico (FK68)
        Emergency contact: KP4XXX
        All phone/internet systems DOWN

        Situation:
        - Hurricane Maria Cat 5 direct hit San Juan 0600Z
        - Municipal Hospital 300 patients, generators failing
        - Estimated 12 hours generator fuel remaining
        - Multiple buildings collapsed, unknown casualties
        - Water treatment plants offline, flooding widespread
        - Airport damaged but runway operational for C-130

        Critical needs (URGENT):
        - Insulin (500+ diabetic patients)
        - Antibiotics (wound infections spreading)
        - Diesel fuel (20,000 gallons for hospital generators)
        - Water purification equipment
        - Search and rescue teams (multiple collapse sites)

        Relay coordination:
        - W2XXX/N2XXX (NY): Contact FEMA Region 2, coordinate airlift
        - K4XXX (FL): Coast Guard liaison, maritime rescue coordination
        - W1XXX (MA): Contact pharmaceutical suppliers, insulin emergency
        - K5XXX (TX): Alert military bases, heavy equipment transport
        - Worldwide: Request assistance any stations with .gov/.mil contacts

        HF is ONLY communication - all infrastructure destroyed
        Updates every 30 minutes this frequency
        Coordinating relief flights on 14.265 MHz

        KP4XXX San Juan Emergency Net Control
    """,

    'duration': 60,  # seconds (detailed message)
    'power': 'MAXIMUM',
    'repeat_interval': 300,  # Repeat every 5 minutes
}
```

### Ad-Hoc Relay Network in Action

```
Geographic distribution (Hurricane Maria example - 2017 actual scenario):

KP4XXX (Emergency) ────────────┐
FK68 (San Juan, Puerto Rico)   │ All infrastructure destroyed
HF is ONLY communication       │ Phone/internet DOWN
                              │
                              ├─→ K4XXX (Tier 1, 1600km)
                              │   EL87 (Miami, Florida)
                              │   Role: Coast Guard liaison, maritime rescue
                              │   │
                              │   ├─→ K4YYY (Tier 2, via K4XXX)
                              │   │   EM66 (Tampa - Gulf Coast coordination)
                              │   │
                              │   └─→ N4ZZZ (Tier 2, via K4XXX)
                              │       EM84 (Atlanta - SE regional hub)
                              │
                              ├─→ W2XXX (Tier 1, 2600km)
                              │   FN30 (New York)
                              │   Role: FEMA Region 2 contact, federal coordination
                              │   │
                              │   ├─→ W1YYY (Tier 2, via W2XXX)
                              │   │   FN42 (Boston - medical supply coordination)
                              │   │
                              │   └─→ K2ZZZ (Tier 2, via W2XXX)
                              │       FN20 (New Jersey - logistics hub)
                              │
                              ├─→ K5XXX (Tier 1, 3200km)
                              │   EM10 (Houston, Texas)
                              │   Role: Military base coordination, heavy equipment
                              │   │
                              │   └─→ K5YYY (Tier 2, via K5XXX)
                              │       EL29 (San Antonio - Air Force transport)
                              │
                              └─→ W6XXX (Tier 1, 5400km)
                                  DM13 (Los Angeles, California)
                                  Role: Pacific coordination, additional resources
                                  │
                                  └─→ KH6YYY (Tier 2, via W6XXX)
                                      BL10 (Hawaii - staging point for Pacific aid)


Relay flow (realistic disaster response):
1. KP4XXX transmits emergency on patterns [64-67] (4 message patterns, all 70 tones)
   - Multi-pattern transmission for maximum redundancy
   - Weak signal (-15 to -5 dB at mainland due to distance)
   - CASCADE's -25 dB capability critical for reaching mainland

2. Tier 1 stations receive directly (1600-5400km range)
   - K4XXX (FL): Immediately contacts Coast Guard, coordinates maritime rescue
   - W2XXX (NY): Calls FEMA Region 2, triggers federal disaster response
   - K5XXX (TX): Alerts military bases (Lackland AFB, Fort Hood)
   - W6XXX (CA): Coordinates Pacific resources, backup staging

3. Tier 1 stations relay to Tier 2 (regional hubs)
   - N4ZZZ (Atlanta): Coordinates SE region hospitals, supplies
   - W1YYY (Boston): Emergency insulin procurement (pharmaceutical companies)
   - K5YYY (San Antonio): Air Force C-130 transport coordination
   - KH6YYY (Hawaii): Pacific staging for additional resources

4. Within 1 hour of emergency declaration:
   - FEMA activated, resources mobilizing
   - Coast Guard rescue operations underway
   - C-130 flights arranged with insulin, fuel, water equipment
   - International aid coordinated (if needed via Caribbean/South American relays)

5. Ongoing updates every 30 minutes via HF
   - Hospital status, fuel remaining, patient conditions
   - Relief flight ETAs, supply delivery confirmations
   - Evolving needs as situation develops

6. This is why HF emergency capability matters:
   - Reached mainland from Puerto Rico despite weak signals
   - Coordinated multi-agency response (FEMA, Coast Guard, Military)
   - Organized resource delivery (insulin, fuel, equipment)
   - All while phone/internet completely DOWN
   - Real example: 2017 Hurricane Maria, HF was only comm for 2-3 weeks
```

---

## Organic Network Formation

### Self-Organization Algorithm

```python
class AdHocRelayNetwork:
    """
    Network forms automatically based on:
    - Signal strength (who heard emergency)
    - Geographic distribution (grid squares)
    - Capabilities (911 access, hospital, power)
    - Existing connections (who hears whom)
    """

    def __init__(self):
        self.emergency_station = None
        self.tiers = {}
        self.routing = {}

    def form_network(self, emergency_alert, prokernels):
        """
        Organize responding stations into relay tiers
        Purely organic - based on actual signal reports
        """

        self.emergency_station = emergency_alert['callsign']

        # Tier 1: Strong signals to emergency (<500km, SNR>0)
        tier1 = [
            pk for pk in prokernels
            if pk['snr_heard_alert'] > 0 and
               pk['distance_km'] < 500
        ]

        # Sort by capabilities
        tier1_sorted = sorted(tier1, key=lambda x: (
            x['can_contact_911'] * 10 +      # Prioritize 911 access
            x['can_contact_hospital'] * 5 +
            x['snr_heard_alert']
        ), reverse=True)

        self.tiers['tier1'] = tier1_sorted

        # Tier 2: Heard by tier1 stations (500-2000km)
        tier2 = {}
        for t1 in tier1_sorted:
            # Who did this tier1 station hear?
            heard_by_t1 = [
                pk for pk in prokernels
                if pk['callsign'] in t1['heard_stations'] and
                   pk not in tier1
            ]
            tier2[t1['callsign']] = heard_by_t1

        self.tiers['tier2'] = tier2

        # Tier 3: Long distance (>2000km)
        tier3 = [
            pk for pk in prokernels
            if pk['distance_km'] > 2000
        ]

        # Sort by distance (farthest first for global coverage)
        tier3_sorted = sorted(tier3,
                             key=lambda x: x['distance_km'],
                             reverse=True)

        self.tiers['tier3'] = tier3_sorted

        return self.tiers

    def assign_roles(self):
        """
        Assign specific roles based on capabilities
        """

        # Find station with 911 access
        for station in self.tiers['tier1']:
            if station['can_contact_911']:
                self.routing['primary_911'] = station['callsign']
                break

        # Find station with hospital access
        for station in self.tiers['tier1']:
            if station['can_contact_hospital']:
                self.routing['hospital'] = station['callsign']
                break

        # Assign regional relay hubs
        regions = ['northeast', 'southeast', 'midwest', 'west', 'pacific']
        for region in regions:
            # Find highest-power station in region
            regional_stations = [
                s for s in self.tiers['tier2']
                if self.in_region(s['grid'], region)
            ]
            if regional_stations:
                best = max(regional_stations, key=lambda x: x['power_watts'])
                self.routing[f'{region}_hub'] = best['callsign']

        return self.routing
```

---

## Emergency Traffic Patterns

### Message Relay Sequence

```python
def emergency_traffic_relay():
    """
    Emergency traffic flows through ad-hoc network
    """

    # Timeline:

    # T+0s: Emergency alert (1475 Hz)
    # T+24s: Alert complete
    # T+24-34s: Network clearing
    # T+34s: Emergency negotiation (4-FSK)
    # T+72s: Negotiation complete
    # T+72-120s: Prokernels (staggered responses)
    # T+120s: Final kernel (relay map)
    # T+160s: Emergency traffic begins

    while emergency_active:
        # Emergency station sends updates
        update = {
            'timestamp': now(),
            'patient_status': get_patient_status(),
            'location_update': current_location(),  # If mobile
            'requests': ['MEDEVAC_ETA', 'HOSPITAL_READY'],
        }

        # Transmit on message patterns with multi-pattern
        transmit_emergency_message(update, patterns=[64,65,66,67])  # Emergency 1 message patterns

        # Tier 1 stations relay immediately
        tier1_relay(update, destinations=['911', 'hospital'])

        # Tier 2/3 relay for geographic coverage
        tier2_relay(update, regional=True)
        tier3_relay(update, worldwide=True)

        # Wait for next update
        sleep(300)  # Every 5 minutes
```

### Relay Coordination

```python
# K2DEF (Tier 1 - 911 contact):
def tier1_911_relay():
    """Station with 911 access coordinates with emergency services"""

    # Heard emergency from W1ABC
    emergency_data = decode_emergency()

    # IMMEDIATE: Call 911 via phone/internet
    call_911({
        'emergency_type': 'Medical - Heart Attack',
        'location': 'Grid square FN31pr (41.234°N, 73.456°W)',
        'patient': 'Adult male, ~65 years, chest pain',
        'contact': 'Amateur radio operator W1ABC',
        'access': 'Route 7 near mile marker 45',
        'status': 'Conscious, battery power for 2 hours',
    })

    # Relay 911 response back to emergency station
    dispatcher_response = {
        'medevac_dispatched': True,
        'eta_minutes': 15,
        'helicopter_callsign': 'LIFEFLIGHT_2',
        'landing_zone': 'Route 7 MM45 parking area',
        'instructions': 'Stay with vehicle, turn on hazards',
    }

    # Transmit response to W1ABC
    transmit_to_emergency(dispatcher_response, target='W1ABC')


# N3GHI (Tier 1 - Hospital contact):
def tier1_hospital_relay():
    """Station contacts hospital directly"""

    # Contact Springfield Hospital ER via radio/phone
    hospital_alert = {
        'incoming_patient': 'Cardiac emergency',
        'eta': '15 minutes via medevac',
        'patient_info': 'Male ~65, chest pain, conscious',
        'request': 'Cardiologist on standby',
        'contact': 'Relay via amateur radio W1ABC/N3GHI',
    }

    contact_hospital(hospital_alert)

    # Relay hospital response
    hospital_response = {
        'er_ready': True,
        'cardiologist_available': True,
        'helipad_clear': True,
        'instructions': 'Direct to ER, trauma team standing by',
    }

    transmit_to_emergency(hospital_response, target='W1ABC')


# W4JKL (Tier 2 - Regional relay):
def tier2_regional_relay():
    """Relay to southern region for coverage"""

    # Relay emergency to southern states
    # Uses message patterns (not beacon channel)
    relay_message({
        'original': 'W1ABC emergency in Massachusetts',
        'type': 'Heart attack, medevac requested',
        'status': '911 contacted, helicopter ETA 15 min',
        'relay_instructions': 'Southern stations maintain awareness',
        'via': 'N3GHI → W4JKL relay chain',
    })


# VK2ABC (Tier 3 - Worldwide relay):
def tier3_worldwide_relay():
    """Station in Australia relays to Pacific/Asia"""

    # Relay emergency awareness to VK/ZL/JA stations
    worldwide_relay({
        'emergency_origin': 'W1ABC (USA, Massachusetts)',
        'type': 'Medical emergency - heart attack',
        'status': 'Under control, 911 responding, medevac dispatched',
        'relay_chain': 'W1ABC → N3GHI → W4JKL → K5MNO → W6VWX → VK2ABC',
        'purpose': 'Awareness, demonstrate global relay capability',
        'hops': 6,
        'distance_km': 16000,  # Massachusetts to Australia!
    })
```

---

## Network Resilience

### Automatic Rerouting on Failure

```python
def handle_relay_failure():
    """
    If tier1 station fails, network automatically adapts
    """

    # K2DEF (tier1, 911 contact) loses power
    # Network detects: No ACK from K2DEF for 60 seconds

    # Automatic failover:
    # N3GHI (tier1, hospital contact) takes over 911 role
    # OR: Tier2 station (W2XYZ) promoted to tier1

    # Emergency station broadcasts updated routing:
    routing_update = {
        'failed_station': 'K2DEF',
        'new_primary_911': 'N3GHI',
        'promoted_tier1': 'W2XYZ',
        'relay_map_version': 2,  # Incremented
    }

    transmit_routing_update(routing_update)

    # Network adapts organically
    # No central coordination needed
    # Surviving stations reconfigure automatically
```

### Multi-Path Redundancy

```python
# Emergency message reaches destination via multiple paths:

Path 1: W1ABC → K2DEF (direct, tier1)
Path 2: W1ABC → N3GHI (direct, tier1)
Path 3: W1ABC → N3GHI → W4JKL (tier2)
Path 4: W1ABC → N3GHI → K3STU (tier2)

# If tier1 fails, tier2/tier3 provide backup
# Message still gets through
# Network is fault-tolerant by design
```

---

## Additional Realistic Emergency Scenarios

CASCADE's emergency relay network handles diverse large-scale emergencies where HF becomes primary or only communication:

### Scenario 1: Earthquake - Infrastructure Collapse

```
PRIORITY EMERGENCY - MAJOR EARTHQUAKE

Location: Port-au-Prince, Haiti (FK48)
Contact: HH2XXX
All communications infrastructure DESTROYED

Situation:
- 7.8 magnitude earthquake 0410Z
- Estimate 1000+ persons trapped in collapsed buildings
- Presidential Palace, multiple hospitals severely damaged
- Airport control tower damaged, runway cracked but usable
- No running water, power grid completely failed
- Aftershocks continuing (6.0+ magnitude)

Critical needs:
- Search and rescue teams (urban collapse specialists)
- Heavy equipment (excavators, concrete cutters)
- Medical teams (trauma surgeons, field hospitals)
- Water purification (municipal system destroyed)
- Communications equipment (all repeaters down)

Relay instructions:
- W4XXX (FL): Contact USAID, international rescue coordination
- K2XXX (NY): United Nations liaison, disaster response teams
- Worldwide: Any stations with government/military contacts
- Maritime: Ships in Caribbean, medical vessels

HF coordination on 7.185 MHz and 14.205 MHz
Airport open for C-130 heavy transport
Updates hourly as situation develops

HH2XXX Port-au-Prince Emergency Coordinator
```

### Scenario 2: Maritime Distress - Vessel Sinking

```
MAYDAY MAYDAY MAYDAY

Vessel: Fishing vessel Pacific Star
Position: 47°15'N 125°30'W (300 miles offshore Washington)
Contact: Maritime mobile WDX4523

Emergency:
- Taking water, engine room flooding
- Pump failure, water level rising
- 6 persons aboard, donning survival suits
- 35-foot seas, 50 knot winds
- Drifting toward rocks, ETA impact 3 hours
- EPIRB activated, strobe lights deployed
- Water temperature 48°F (survival time <1 hour if in water)

Immediate assistance required:
- Coast Guard rescue helicopter (range limit, weather)
- Commercial vessels in area for pickup
- Canadian Coast Guard (closer to position)

Relay coordination:
- W7XXX (Seattle): Contact USCG District 13, helicopter dispatch
- VE7XXX (Vancouver): Alert Canadian Coast Guard, vessels in area
- Maritime stations: Any ships within 100 miles of position
- Weather: Current conditions critical for helicopter ops

MAYDAY continues on 2182 kHz voice, HF digital backup
Position updates every 15 minutes
Vessel has 12 hours battery power for HF

WDX4523 Pacific Star
```

### Scenario 3: Remote Infrastructure - Power Failure

```
PRIORITY EMERGENCY - CRITICAL INFRASTRUCTURE FAILURE

Location: McMurdo Station, Antarctica (RB50)
Contact: KC4USV
Limited time-critical situation

Situation:
- Primary generator failed 1200Z
- Backup generator online but fuel leak detected
- Estimated 48-72 hours backup power remaining
- Temperature -42°C (-44°F), wind chill -65°C
- 148 personnel, winter-over crew
- Next resupply flight not scheduled for 3 weeks
- Cannot evacuate (weather, darkness, no flight capability)

Critical needs:
- Generator parts (specific model, limited availability)
- Airlift from Christchurch NZ (8+ hour flight, weather-dependent)
- Alternative: Parts from South Pole station (if compatible)
- Fuel leak repair equipment
- Cold weather survival gear (if power fails completely)

Coordination required:
- W2XXX (NY): Contact National Science Foundation HQ
- ZL1XXX (NZ): Coordinate with Christchurch support base
- K1XXX (Boston): Antarctic logistics contractor (Raytheon)
- Military: US Air Force Antarctic support (LC-130 capability)

This is NOT life-threatening yet but time-critical
If backup fails: Life-safety emergency in 48 hours
Need parts/personnel on next available flight (weather window)
Runway operational, winds 25 knots, visibility 5 miles

Updates every 4 hours on 14.255 MHz
KC4USV McMurdo Emergency Coordinator
```

### Why These Scenarios Demonstrate CASCADE Value

**Common characteristics:**
1. **Infrastructure destroyed** - HF is only communication
2. **Weak signals** - Long distance, poor propagation
3. **Multi-agency coordination** - FEMA, Coast Guard, Military, International
4. **Time-critical** - Minutes/hours matter
5. **Complex logistics** - Multiple resources, locations, agencies

**CASCADE advantages for these emergencies:**
- **-25 dB sensitivity**: Puerto Rico → Mainland despite weak signals
- **280+ users**: Coordinate large emergency nets (50+ stations)
- **Multi-pattern**: 4× throughput on strong links (mainland stations)
- **Automatic relay**: Self-organizing tier structure
- **Priority system**: Life-safety messages preempt normal traffic
- **Efficiency**: More information in limited bandwidth vs FT8/voice

**Historical precedent:**
- 2017 Hurricane Maria: HF only communication Puerto Rico for weeks
- 2010 Haiti Earthquake: HF coordinated international response
- 2011 Japan Tsunami: HF carried emergency traffic when phones overloaded
- Ongoing maritime emergencies: HF reaches vessels beyond VHF range

CASCADE's emergency relay network is designed for EXACTLY these scenarios where lives depend on HF communication reaching across continents despite infrastructure failure.

---

### Emergency Protocol Overhead

```python
def emergency_protocol_timing():
    """
    Total time from alert to traffic flowing
    """

    timeline = {
        'phase1_alert': 24,            # seconds (BPSK on 1475 Hz)
        'phase2_clearing': 10,         # seconds (network response)
        'phase3_negotiation': 38,      # seconds (4-FSK details)
        'phase4_prokernels': 48,       # seconds (average, staggered)
        'phase5_final_kernel': 40,     # seconds (relay map)
        'total_setup': 160,            # seconds (~2.7 minutes)

        'phase6_traffic_starts': 160,  # seconds after initial alert
    }

    # After setup, emergency traffic flows continuously
    # Updates every 5 minutes

    return timeline

# Compare to traditional emergency nets:
# - Phone 911: ~30 seconds (if phone works!)
# - HF voice net: 5-10 minutes to establish
# - FT8 emergency: 15+ minutes (slow protocol)
# - CASCADE: ~2.7 minutes to full relay network

# CASCADE advantage: Global relay network formed automatically
```

### Beacon Channel Utilization

```
Normal operation:
- Beacon bandwidth: 175 Hz
- Beacon traffic: ~5% utilization (sparse beaconing)
- Beacons per minute: 30-60 across all stations

Emergency operation:
- Beacon bandwidth: 175 Hz (same)
- Emergency phases 1-5: 100% utilization for ~160 seconds
- Then: Reserved but mostly idle (emergency uses message tones)
- Overhead: Acceptable for life-safety communications
```

---

## Legal Compliance

### Full Callsign Requirements

```python
FCC_COMPLIANCE = {
    'part_97': 'Amateur Radio Service Rules',
    'section_119': 'Station identification',

    'requirements': {
        'frequency': 'Every 10 minutes during transmission',
        'emergency': 'Full callsign required (not hash)',
        'relay': 'Must identify both originator and relay station',
        'international': 'Callsign indicates country/license class',
    },

    'cascade_compliance': {
        'emergency_alert': 'Full callsign (29 bits)',
        'negotiation': 'Full callsign (29 bits)',
        'prokernels': 'Full callsigns for all participants',
        'relay_messages': 'Includes originator and relay callsigns',
        'periodic_id': 'Every emergency update includes callsign',
    }
}

# Cost of compliance:
# 29 bits vs 16 bits hash = +13 bits per callsign
# Emergency negotiation: +13 bits
# Each prokernel: +26 bits (emergency + responder)
# Total overhead: ~100-150 bits across all phases
# Duration impact: ~10-15 seconds
# Acceptable for legal/safety requirements ✓
```

---

## See Also

- **[Beacon Reservation](beacon_reservation.md)** - 175 Hz center-band specification
- **[Priority Handling](priority_handling.md)** - Emergency message prioritization
- **[Net Operations](net_operations.md)** - Directed net protocols
- **[Kernel Lifecycle](kernel_lifecycle.md)** - Prokernel and final kernel exchange
- **[Signal Specification](signal_specification.md)** - Physical layer details
