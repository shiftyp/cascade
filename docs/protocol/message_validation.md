# CASCADE Message Validation Protocol

## Overview

CASCADE uses **dual-layer validation** to prevent neural network hallucinations while maintaining low overhead:

1. **CRC32**: Error detection that the NN can learn (training signal)
2. **xxHash32**: Validity checking that the NN cannot forge (hallucination protection)

This approach enables the model to learn error patterns from CRC while xxHash provides cryptographic-quality validation against false positives.

---

## The Hallucination Problem

### What is NN Hallucination?

At low SNR, the neural network may produce plausible-looking output that passes CRC validation but is incorrect:

```python
# Low SNR scenario (SNR = -20dB)
received_signal = true_signal + heavy_noise

# NN decodes
decoded = model.decode(received_signal)
# Output: "Hello W2DEF" (plausible callsign, valid syntax)

# CRC validation
predicted_crc = model.predict_crc(decoded.payload)
# CRC matches! (NN learned to predict valid CRCs)

# Problem: Message is actually wrong
# True message was: "Hello W1ABC"
# But it passes validation!
```

**Why this happens**:
- NN trains on payload + CRC pairs
- Learns CRC calculation patterns
- At low SNR, can "guess" plausible payload + matching CRC
- Especially problematic with short messages (limited entropy)

### Single-Layer Validation Vulnerability

```python
# Training exposes CRC patterns to NN
training_sample = {
    'input': noisy_iq_signal,
    'true_payload': b'Hello W2DEF',
    'true_crc': 0x8A3B5C2D
}

# After 100K training samples, NN learns:
# - "Hello W2DEF" → CRC 0x8A3B5C2D
# - "QRZ K0BB" → CRC 0x1F8E3A9C
# - ...
# - General CRC32 polynomial structure

# At low SNR, NN can hallucinate:
hallucinated = {
    'payload': 'Plausible text',  # Looks valid
    'crc': 0x????               # NN predicts matching CRC
}

# Validation passes, but message is wrong!
```

---

## Dual-Layer Validation Architecture

### Layer 1: CRC32 (NN-Learned)

**Purpose**: Fast error detection, provides training signal for NN

```python
def compute_crc32(data):
    """Standard CRC32 (IEEE 802.3 polynomial)"""
    return zlib.crc32(data) & 0xffffffff  # 32-bit result
```

**Properties**:
- **Size**: 4 bytes (32 bits)
- **Computation**: ~0.002ms for 128 bytes
- **Error detection**: Detects 99.9998% of errors
- **NN learnable**: Yes (polynomial math, NN can approximate)

**NN training includes CRC**:
```python
# Loss function includes CRC prediction
loss = payload_loss + crc_loss

# NN learns to predict both payload and CRC
# This is GOOD - helps NN understand error patterns
```

### Layer 2: xxHash32 (NN-Proof)

**Purpose**: Validity checking that NN cannot forge

```python
import xxhash

def compute_xxhash32(data):
    """xxHash32 - non-cryptographic but NN-proof"""
    return xxhash.xxh32(data).intdigest()  # 32-bit result
```

**Properties**:
- **Size**: 4 bytes (32 bits)
- **Computation**: ~0.005ms for 128 bytes
- **Collision resistance**: 2^32 (sufficient for validity)
- **NN learnable**: No (complex mixing function beyond NN capability)

**Why NN cannot learn xxHash**:

```c
// xxHash32 algorithm (simplified)
uint32_t xxhash32(const void* input, size_t len) {
    uint32_t h32 = PRIME32_5;  // Seed: 374761393

    // Process 16-byte chunks with complex mixing
    for (size_t i = 0; i < len; i += 16) {
        h32 += read32(input, i) * PRIME32_2;     // 2246822519
        h32 = ROTL32(h32, 13);                    // Bit rotation
        h32 *= PRIME32_1;                         // 2654435761
        // ... repeat for each 4-byte word
    }

    // Final avalanche mixing (critical for quality)
    h32 ^= h32 >> 15;
    h32 *= PRIME32_2;
    h32 ^= h32 >> 13;
    h32 *= PRIME32_3;  // 3266489917
    h32 ^= h32 >> 16;

    return h32;
}

// NN would need to learn:
// ✗ Modular multiplication with large primes
// ✗ Bit rotations (ROTL32)
// ✗ XOR operations
// ✗ Avalanche mixing (5 stages)
// ✗ All with perfect bit-level accuracy
//
// This is beyond NN capability without memorizing all possible inputs
```

**xxHash properties that prevent learning**:
1. **Non-linear mixing**: Multiple multiplications by large primes
2. **Bit-level operations**: Rotations and XORs that NNs struggle with
3. **Avalanche effect**: 1-bit input change → ~16-bit output change
4. **Large state space**: 2^32 possible outputs
5. **No discernible patterns**: Designed to appear random

---

## Message Structure

### Wire Format Overview

CASCADE messages use a fixed binary format. See [Message Format](message_format.md) for complete specification.

**Binary layout**:
```
[Header: 19 bytes] [Payload: variable UTF-8] [Validation: 8 bytes]
```

**Header fields** (19 bytes total):
- from_hash (4 bytes): Sender identifier
- to_hash (4 bytes): Destination identifier
- message_id (8 bytes): Unique message ID
- priority (1 byte): EMERGENCY/HIGH/NORMAL/LOW
- payload_length (2 bytes): Payload size in bytes

**Validation fields** (8 bytes total):
- crc32 (4 bytes): Error detection layer
- xxhash32 (4 bytes): Hallucination prevention layer

### Validation Hash Computation

**CRC32 computed over header + payload**:
```python
def compute_crc32_for_message(header_bytes, payload_bytes):
    """CRC32 over complete message data"""

    message_data = header_bytes + payload_bytes
    crc = zlib.crc32(message_data) & 0xFFFFFFFF

    return crc
```

**xxHash32 computed over header + payload + CRC32**:
```python
def compute_xxhash32_for_message(header_bytes, payload_bytes, crc32):
    """xxHash32 over message data + CRC32"""

    validation_input = header_bytes + payload_bytes + crc32.to_bytes(4, 'little')
    xxh = xxhash.xxh32(validation_input)

    return xxh.intdigest()
```

**Complete serialization**:
```python
def create_message(from_hash, to_hash, message_id, priority, payload_text):
    """Create message with dual validation"""

    # Encode payload as UTF-8
    payload_bytes = payload_text.encode('utf-8')

    # Pack header
    header = struct.pack(
        '<IIQBH',
        from_hash,
        to_hash,
        message_id,
        priority,
        len(payload_bytes)
    )

    # Compute CRC32 over header + payload
    crc = zlib.crc32(header + payload_bytes) & 0xFFFFFFFF

    # Compute xxHash32 over header + payload + CRC
    validation_input = header + payload_bytes + struct.pack('<I', crc)
    xxh = xxhash.xxh32(validation_input).intdigest()

    # Assemble complete message
    return header + payload_bytes + struct.pack('<II', crc, xxh)
```

---

## Validation Protocol

### Two-Stage Validation

```python
class MessageValidator:
    """Dual-layer validation with telemetry tracking"""

    def validate(self, message):
        """Validate message with both layers"""

        # Stage 1: CRC32 validation (fast, NN-learned)
        crc_valid = verify_crc32(message.payload, message.crc32)

        if not crc_valid:
            # Early rejection - don't check xxHash
            return ValidationResult(
                valid=False,
                stage='crc',
                error_type='corruption_or_nn_error',
                telemetry_flag='crc_failure'
            )

        # Stage 2: xxHash32 validation (NN-proof)
        expected_hash = create_validation_hash(message.payload, message.crc32)
        hash_valid = (message.xxhash32 == expected_hash)

        if not hash_valid:
            # CRC passed but xxHash failed
            # NN hallucinated plausible payload + matching CRC!
            return ValidationResult(
                valid=False,
                stage='xxhash',
                error_type='nn_hallucination',
                telemetry_flag='potential_hallucination',
                snr=message.measured_snr,
                confidence=message.decode_confidence
            )

        # Both passed - message is valid
        return ValidationResult(
            valid=True,
            stage='complete',
            error_type=None,
            telemetry_flag='validated'
        )
```

### Validation Outcomes

| CRC32 | xxHash32 | Interpretation | Telemetry Flag | Action |
|-------|----------|----------------|----------------|--------|
| ❌ Fail | (skipped) | Corruption or NN wrong | `crc_failure` | Reject |
| ✅ Pass | ❌ Fail | **NN hallucination** | `hallucination_detected` | Reject + log |
| ✅ Pass | ✅ Pass | Valid message | `validated` | Accept |

---

## Hallucination Detection and Telemetry

### Telemetry Structure

```python
validation_telemetry = {
    # Standard telemetry fields
    'neural_state': {...},  # 3581-D
    'metadata': {...},

    # Validation tracking
    'validation': {
        # Validation outcomes
        'crc_valid': bool,
        'xxhash_valid': bool,
        'hallucination_detected': bool,  # crc_valid AND not xxhash_valid

        # Context
        'measured_snr_db': float,
        'decode_confidence': float,  # Model's own confidence score
        'payload_length': int,
        'bit_error_estimate': float,

        # For hallucinations
        'predicted_payload': bytes,   # What NN thought it decoded
        'predicted_crc': uint32,      # CRC NN predicted
        'predicted_xxhash': uint32,   # xxHash NN predicted (will be wrong)
        'expected_xxhash': uint32     # Correct xxHash
    }
}
```

### Hallucination Analysis from Telemetry

**Train hallucination predictor**:

```python
def analyze_hallucination_patterns(telemetry_dataset):
    """Find conditions that lead to hallucinations"""

    # Filter hallucination events
    hallucinations = [
        t for t in telemetry_dataset
        if t.validation.crc_valid and not t.validation.xxhash_valid
    ]

    # Analyze patterns
    analysis = {
        'count': len(hallucinations),
        'rate': len(hallucinations) / len(telemetry_dataset),

        # SNR correlation
        'avg_snr': np.mean([h.measured_snr_db for h in hallucinations]),
        'snr_threshold': np.percentile([h.measured_snr_db for h in hallucinations], 90),
        # Result: 90% of hallucinations occur at SNR < -18dB

        # Confidence correlation
        'avg_confidence': np.mean([h.decode_confidence for h in hallucinations]),
        'confidence_threshold': np.percentile([h.decode_confidence for h in hallucinations], 90),
        # Result: 90% of hallucinations have confidence < 0.4

        # Combined risk factors
        'high_risk_conditions': 'SNR < -18dB AND confidence < 0.4',
        'high_risk_rate': 0.45  # 45% hallucination rate in high-risk zone
    }

    return analysis
```

**Build hallucination predictor**:

```python
class HallucinationPredictor:
    """Predict hallucination risk from neural state"""

    def __init__(self):
        # Small classifier (50K params)
        self.predictor = train_on_telemetry(hallucination_events)

    def predict_hallucination_risk(self, neural_state, snr, confidence):
        """Estimate probability this decode is hallucinated"""

        features = {
            'conductor_weights': neural_state.conductor_weights,  # 5-D
            'noise_expert_activation': neural_state.noise_expert[-10:],  # Last 10 dims
            'measured_snr': snr,
            'decode_confidence': confidence
        }

        risk_score = self.predictor(features)
        # 0.0 = very unlikely hallucination
        # 1.0 = very likely hallucination

        return risk_score

    def should_warn_user(self, risk_score):
        """Decide if user should be warned"""

        if risk_score > 0.8:
            return True, 'HIGH RISK - verify content'
        elif risk_score > 0.5:
            return True, 'MODERATE RISK - check important details'
        else:
            return False, None
```

**User interface integration**:

```python
# During decode
decode_result = model.decode(iq_samples)

# Check validation
validation = validator.validate(decode_result)

if not validation.valid:
    if validation.error_type == 'nn_hallucination':
        # Detected hallucination via xxHash mismatch
        display_to_user("⚠️  Decode failed validation - rejecting message")
        log_telemetry(validation)
    else:
        # Normal CRC error
        display_to_user("CRC error - request retransmission")

elif validation.valid:
    # Additional check: Predict hallucination risk
    risk = predictor.predict_hallucination_risk(
        decode_result.neural_state,
        decode_result.snr,
        decode_result.confidence
    )

    warn, message = predictor.should_warn_user(risk)
    if warn:
        display_to_user(f"⚠️  {message}")

    # Display decoded message
    display_to_user(decode_result.payload)
```

---

## Overhead Analysis

### Per-Message Overhead

```python
overhead_breakdown = {
    # Validation bytes
    'crc32': 4_bytes,
    'xxhash32': 4_bytes,
    'total_validation': 8_bytes,

    # Example payloads
    'short_message_64b': {
        'payload': 64_bytes,
        'validation': 8_bytes,
        'total': 72_bytes,
        'overhead_percent': 12.5
    },

    'typical_message_128b': {
        'payload': 128_bytes,
        'validation': 8_bytes,
        'total': 136_bytes,
        'overhead_percent': 6.25
    },

    'long_message_256b': {
        'payload': 256_bytes,
        'validation': 8_bytes,
        'total': 264_bytes,
        'overhead_percent': 3.1
    }
}
```

**Transmission time impact**:

```python
# CASCADE effective rate: ~100 bps (with FEC)
transmission_time_impact = {
    'validation_bits': 64,  # 8 bytes
    'added_time': 64 / 100,  # = 0.64 seconds

    'typical_qso': {
        'message_count': 20,
        'total_validation_overhead': 20 * 0.64,  # = 12.8 seconds
        'qso_duration': 300,  # 5 minutes
        'overhead_percent': 12.8 / 300  # = 4.3%
    }
}

# 4.3% time overhead is acceptable for hallucination protection
```

### Computational Overhead

```python
computation_breakdown = {
    # Sender (creating message)
    'crc32_compute': 0.002_ms,
    'xxhash32_compute': 0.005_ms,
    'total_sender': 0.007_ms,

    # Receiver (validating message)
    'crc32_verify': 0.002_ms,
    'xxhash32_verify': 0.005_ms,
    'total_receiver': 0.007_ms,

    # Negligible compared to:
    'nn_decode_time': 10_ms,  # 1400x longer than validation
    'validation_percentage': 0.07  # 0.07% of decode time
}

# Validation is essentially free (0.07% overhead)
```

---

## xxHash32 Algorithm Details

### Algorithm Overview

xxHash32 uses a simple but effective mixing strategy:

```python
class XXHash32:
    """xxHash32 reference implementation"""

    # Prime constants (carefully chosen for mixing quality)
    PRIME32_1 = 2654435761
    PRIME32_2 = 2246822519
    PRIME32_3 = 3266489917
    PRIME32_4 = 668265263
    PRIME32_5 = 374761393

    def hash(self, data, seed=0):
        """Compute xxHash32"""

        h32 = seed + self.PRIME32_5

        # Process 16-byte chunks
        for i in range(0, len(data) - 16, 16):
            chunk = data[i:i+16]
            h32 = self._process_chunk(chunk, h32)

        # Process remaining bytes
        remaining = data[-(len(data) % 16):]
        h32 = self._process_remaining(remaining, h32)

        # Final avalanche mixing
        h32 ^= len(data)
        h32 = self._avalanche(h32)

        return h32 & 0xffffffff

    def _process_chunk(self, chunk, h32):
        """Process 16-byte chunk with complex mixing"""
        for j in range(0, 16, 4):
            val = int.from_bytes(chunk[j:j+4], 'little')
            h32 += val * self.PRIME32_2
            h32 = self._rotl32(h32, 13)
            h32 *= self.PRIME32_1
            h32 &= 0xffffffff
        return h32

    def _avalanche(self, h32):
        """Final mixing for avalanche effect"""
        h32 ^= h32 >> 15
        h32 *= self.PRIME32_2
        h32 ^= h32 >> 13
        h32 *= self.PRIME32_3
        h32 ^= h32 >> 16
        return h32 & 0xffffffff

    @staticmethod
    def _rotl32(value, shift):
        """32-bit left rotation"""
        return ((value << shift) | (value >> (32 - shift))) & 0xffffffff
```

**Why this prevents NN learning**:
- Requires bit-perfect arithmetic (NNs are approximate)
- Multiple prime number multiplications
- Bit rotations (NNs struggle with bit operations)
- Modulo 2^32 after each step
- Avalanche mixing destroys linear patterns

---

## Implementation

### Python Reference Implementation

```python
import xxhash
import zlib

class CascadeMessageValidator:
    """Dual-layer message validation"""

    def create_message(self, payload: bytes) -> dict:
        """Create message with dual validation"""

        # Layer 1: CRC32 (NN learns)
        crc = zlib.crc32(payload) & 0xffffffff

        # Layer 2: xxHash32 (NN cannot forge)
        validation_input = payload + crc.to_bytes(4, byteorder='little')
        xxh = xxhash.xxh32(validation_input)
        validation_hash = xxh.intdigest()

        return {
            'payload': payload,
            'crc32': crc,
            'xxhash32': validation_hash
        }

    def validate(self, message: dict) -> tuple[bool, str, dict]:
        """
        Validate message with dual layers

        Returns:
            (valid, error_type, telemetry_data)
        """

        # Stage 1: CRC32 check
        computed_crc = zlib.crc32(message['payload']) & 0xffffffff
        crc_valid = (computed_crc == message['crc32'])

        if not crc_valid:
            return False, 'crc_failure', {
                'crc_valid': False,
                'xxhash_valid': None,  # Not checked
                'hallucination_detected': False
            }

        # Stage 2: xxHash32 check
        validation_input = message['payload'] + message['crc32'].to_bytes(4, byteorder='little')
        expected_hash = xxhash.xxh32(validation_input).intdigest()
        hash_valid = (message['xxhash32'] == expected_hash)

        if not hash_valid:
            # CRC passed but xxHash failed = hallucination
            return False, 'hallucination', {
                'crc_valid': True,
                'xxhash_valid': False,
                'hallucination_detected': True
            }

        # Both passed
        return True, 'validated', {
            'crc_valid': True,
            'xxhash_valid': True,
            'hallucination_detected': False
        }
```

### C Implementation (for embedded systems)

```c
#include <stdint.h>
#include "xxhash.h"  // From official xxHash library

// CRC32 table (standard IEEE 802.3)
static const uint32_t crc32_table[256] = { /* ... */ };

uint32_t cascade_crc32(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i]) & 0xFF];
    }
    return ~crc;
}

typedef struct {
    uint8_t *payload;
    size_t payload_len;
    uint32_t crc32;
    uint32_t xxhash32;
} cascade_message_t;

int cascade_validate_message(const cascade_message_t *msg) {
    // Stage 1: CRC32
    uint32_t computed_crc = cascade_crc32(msg->payload, msg->payload_len);
    if (computed_crc != msg->crc32) {
        return -1;  // CRC failure
    }

    // Stage 2: xxHash32
    // Concatenate payload + CRC for validation input
    uint8_t validation_input[msg->payload_len + 4];
    memcpy(validation_input, msg->payload, msg->payload_len);
    memcpy(validation_input + msg->payload_len, &msg->crc32, 4);

    uint32_t computed_hash = XXH32(validation_input, msg->payload_len + 4, 0);
    if (computed_hash != msg->xxhash32) {
        return -2;  // Hallucination detected
    }

    return 0;  // Valid
}
```

---

## Training Considerations

### NN Training with Dual Validation

```python
def train_decoder_with_dual_validation():
    """Train NN to predict payload and CRC, but not xxHash"""

    for batch in training_data:
        # Ground truth
        true_payload = batch.payload
        true_crc = compute_crc32(true_payload)
        true_xxhash = compute_xxhash32(true_payload + true_crc)

        # NN prediction
        predicted = model.decode(batch.noisy_iq)

        # Loss function - ONLY payload and CRC
        loss = (
            payload_loss(predicted.payload, true_payload) +
            crc_loss(predicted.crc, true_crc)
            # NOTE: NO xxHash in loss function!
        )

        optimizer.step(loss)

    # Result:
    # ✅ NN learns to decode payloads accurately
    # ✅ NN learns to predict CRC (helps understand errors)
    # ❌ NN never sees xxHash in loss (cannot learn to forge)
```

**Why xxHash is not in loss function**:
- NN sees xxHash values in training data
- But never gets gradient signal for xxHash prediction
- Cannot learn to optimize for xxHash
- Even if it memorized training xxHashes, cannot generate new valid ones

### Preventing Overfitting to xxHash

```python
def ensure_xxhash_remains_unlearnable():
    """Validation that NN hasn't learned xxHash"""

    # Test: Can NN predict xxHash for novel payloads?
    test_payloads = generate_novel_payloads(n=1000)  # Never seen in training

    hallucination_attempts = 0
    for payload in test_payloads:
        # Let NN try to hallucinate
        nn_payload = model.hallucinate_at_low_snr(noise_only_input)
        nn_crc = model.predict_crc(nn_payload)
        nn_xxhash = compute_xxhash32(nn_payload + nn_crc)

        # Check if NN accidentally got xxHash right
        true_xxhash = compute_xxhash32(payload + compute_crc32(payload))

        if nn_xxhash == true_xxhash:
            hallucination_attempts += 1

    false_positive_rate = hallucination_attempts / 1000
    expected_rate = 1 / (2**32)  # Random chance: 1 in 4 billion

    assert false_positive_rate < 0.001  # Should be near-zero
    # If false_positive_rate is high, NN somehow learned xxHash (shouldn't happen)
```

---

## Comparison to Alternatives

### xxHash32 vs Other Approaches

| Approach | Overhead | Computation | NN-Proof | Key Mgmt | FCC ID |
|----------|----------|-------------|----------|----------|--------|
| **CRC32 only** | 4 bytes | 0.002ms | ❌ No | None | No |
| **CRC32 + xxHash32** | 8 bytes | 0.007ms | ✅ Yes | None | No |
| **CRC32 + SHA256** | 36 bytes | 0.052ms | ✅✅ Strong | None | No |
| **CRC32 + HMAC64** | 12 bytes | 0.102ms | ✅✅✅ Crypto | 32 bytes/station | Yes |
| **Reed-Solomon only** | 32 bytes | 1.0ms | ⚠️ Moderate | None | No |

**Decision matrix**:

```python
use_cases = {
    'minimal_overhead_priority': {
        'recommendation': 'CRC32 + xxHash32',
        'overhead': '8 bytes (6.25%)',
        'hallucination_protection': 'Excellent',
        'fcc_compliance': 'Separate periodic announcement'
    },

    'fcc_identification_priority': {
        'recommendation': 'CRC32 + HMAC64',
        'overhead': '12 bytes (9.4%)',
        'hallucination_protection': 'Excellent',
        'fcc_compliance': 'Every message cryptographically signed'
    },

    'simplicity_priority': {
        'recommendation': 'Reed-Solomon only',
        'overhead': '32 bytes (25%)',
        'hallucination_protection': 'Good',
        'fcc_compliance': 'Separate periodic announcement'
    }
}

# For CASCADE: CRC32 + xxHash32
# - Low overhead (6.25%)
# - Fast (0.007ms)
# - No key management
# - Excellent hallucination protection
# - FCC ID via periodic 4-FSK announcements
```

---

## FCC Identification Strategy

### Periodic Callsign Announcement

**With xxHash32 (no per-message identification)**:

```python
class FCCIdentification:
    """FCC compliant identification without per-message overhead"""

    def __init__(self, callsign):
        self.callsign = callsign
        self.last_identification = 0
        self.identification_interval = 600  # 10 minutes (FCC requirement)

    def should_identify(self):
        """Check if identification needed"""
        elapsed = time.time() - self.last_identification
        return elapsed >= self.identification_interval

    def transmit_identification(self):
        """Send callsign on 4-FSK (robust)"""

        # Use 4-FSK for maximum robustness
        identification_message = {
            'type': 'IDENTIFICATION',
            'callsign': self.callsign,  # Plain text (FCC requirement)
            'timestamp': int(time.time()),
            'grid_square': self.grid_square  # Optional
        }

        # Transmit on 4-FSK channel
        # ~6-8 seconds for full callsign
        transmit_4fsk(identification_message)

        self.last_identification = time.time()

    def identification_schedule(self, qso_duration_minutes):
        """Schedule identifications during QSO"""

        num_identifications = int(qso_duration_minutes / 10) + 1

        return {
            'minute_0': 'Initial identification (includes callsign)',
            'minute_10': 'Re-identification',
            'minute_20': 'Re-identification',
            # ... every 10 minutes
            'total_overhead': f'{num_identifications * 7} seconds',
            'overhead_percent': f'{num_identifications * 7 / (qso_duration_minutes * 60) * 100:.1f}%'
        }
```

**Example 20-minute QSO**:
```python
qso_timeline = {
    # Minute 0: Initial contact
    '0:00-0:08': '4-FSK identification (K0BB)',
    '0:08-20:00': 'Fast messages with xxHash validation (no callsign)',

    # Minute 10: FCC re-identification
    '10:00-10:07': '4-FSK identification (K0BB)',
    '10:07-20:00': 'Continue fast messages',

    # Total overhead
    'identification_time': 15,  # seconds
    'message_time': 1185,      # seconds
    'overhead': 1.25           # percent
}

# Very low overhead (1.25%) for FCC compliance
```

---

## Telemetry-Driven Improvements

### Hallucination Rate Tracking

**Monitor hallucination frequency**:

```python
class HallucinationMonitor:
    """Track hallucination rates across conditions"""

    def analyze_hallucination_rates(self, telemetry_stream):
        """Real-time hallucination rate analysis"""

        results = {
            'by_snr': defaultdict(list),
            'by_confidence': defaultdict(list),
            'by_payload_length': defaultdict(list),
            'by_band': defaultdict(list)
        }

        for sample in telemetry_stream:
            if sample.validation.hallucination_detected:
                # Bin by SNR
                snr_bin = int(sample.measured_snr_db / 5) * 5  # 5dB bins
                results['by_snr'][snr_bin].append(1)

                # Bin by confidence
                conf_bin = int(sample.decode_confidence * 10) / 10
                results['by_confidence'][conf_bin].append(1)

                # Bin by payload length
                len_bin = int(sample.payload_length / 32) * 32
                results['by_payload_length'][len_bin].append(1)

                # Bin by band
                results['by_band'][sample.band].append(1)

        # Compute rates
        hallucination_rates = {
            'snr_threshold': self._find_threshold(results['by_snr']),
            'confidence_threshold': self._find_threshold(results['by_confidence']),
            'high_risk_length': max(results['by_payload_length'].keys()),
            'vulnerable_bands': sorted(results['by_band'].keys(),
                                      key=lambda b: len(results['by_band'][b]),
                                      reverse=True)
        }

        return hallucination_rates
```

### Model Improvements from Hallucination Telemetry

**Use hallucination events to improve model**:

```python
def retrain_with_hallucination_awareness(base_model, hallucination_telemetry):
    """Improve model using hallucination events"""

    for event in hallucination_telemetry:
        # This was a hallucination event
        # - NN predicted: payload_A + crc_A
        # - CRC matched (NN learned CRC)
        # - xxHash failed (payload_A was wrong)

        # Add to training with high weight
        # Teach NN to be more conservative at low SNR
        training_sample = {
            'input_iq': event.input_signal,
            'nn_prediction': event.predicted_payload,
            'true_payload': None,  # Unknown (was hallucinated)
            'special_flag': 'known_hallucination',
            'snr': event.measured_snr_db,
            'confidence': event.decode_confidence
        }

        # Train to lower confidence in these conditions
        # Or train to output "uncertain" marker
        loss = train_to_be_more_uncertain_at_low_snr(
            model_state=event.neural_state,
            hallucination_occurred=True
        )

        optimizer.step(loss)
```

---

## Security Considerations

### xxHash is NOT Cryptographic

**Important limitations**:

```python
security_properties = {
    'collision_resistance': '2^32 (weak for crypto)',
    'preimage_resistance': 'None (can find input for given hash)',
    'second_preimage': 'None (can find collision)',
    'intentional_attacks': 'Vulnerable',

    'use_case': {
        'checksum': '✅ Excellent',
        'hash_table': '✅ Excellent',
        'validity_checking': '✅ Excellent (against NN)',
        'data_deduplication': '✅ Good',
        'digital_signatures': '❌ DO NOT USE',
        'password_hashing': '❌ DO NOT USE',
        'cryptographic_auth': '❌ DO NOT USE'
    }
}
```

**Why xxHash works for CASCADE**:
- **Threat model**: NN hallucination (unintentional), not malicious attacks
- **NN cannot**: Learn arbitrary hash functions from observations
- **Attacker could**: Intentionally craft collisions (but we don't have attackers)
- **Goal**: Prevent false positives, not prevent forgery

**If you needed security** (authentication, anti-tampering):
```python
# Use HMAC instead
use_hmac_if = {
    'sender_authentication': 'Need to verify who sent message',
    'anti_tampering': 'Prevent malicious message modification',
    'replay_protection': 'Prevent message replay attacks',
    'fcc_identification': 'Cryptographic callsign proof'
}

# Use xxHash if:
use_xxhash_if = {
    'hallucination_protection': 'Prevent NN false positives',
    'data_integrity': 'Ensure message not corrupted',
    'low_overhead': 'Minimize bandwidth usage',
    'no_key_management': 'Avoid key distribution complexity'
}

# CASCADE uses xxHash: hallucination protection without crypto overhead
```

---

## Summary

**CASCADE message validation uses CRC32 + xxHash32**:

1. **CRC32** (4 bytes):
   - Fast error detection
   - NN learns CRC patterns (training signal)
   - 99.9998% error detection rate

2. **xxHash32** (4 bytes):
   - Validity checking NN cannot forge
   - Prevents hallucinations at low SNR
   - 2^32 collision resistance (sufficient for validity)

3. **Total overhead**: 8 bytes (6.25% for 128-byte payload)

4. **Computational cost**: 0.007ms (0.07% of NN decode time)

5. **Benefits**:
   - NN learns from CRC (improves training)
   - xxHash prevents hallucinations (improves reliability)
   - No key management (simpler deployment)
   - Minimal overhead (bandwidth efficient)

6. **FCC compliance**: Separate periodic callsign announcements on 4-FSK (every 10 minutes, ~7 seconds)

---

## Retry Strategy

### Adaptive Retry Mechanism

**Use fast retries first, fallback to robust kernel exchange**:

```python
class AdaptiveRetry:
    """Fast message patterns → Robust 4-FSK kernel exchange"""

    def __init__(self):
        self.consecutive_failures = {}
        self.fsk_fallback_threshold = 3

    def handle_validation_failure(self, station, message_id, validation_result):
        """Choose retry mechanism based on failure pattern"""

        failures = self.consecutive_failures.get(station, 0) + 1
        self.consecutive_failures[station] = failures

        if failures < self.fsk_fallback_threshold:
            # Retry 1-2: Fast retry on message patterns
            return self.fast_retry(message_id, failures)

        else:
            # Retry 3+: Kernel exchange on 4-FSK (robust + resets state)
            return self.kernel_exchange_retry(station, message_id, failures)

    def fast_retry(self, message_id, attempt):
        """Fast retry on message patterns"""
        retry_request = {
            'type': 'RETRY_REQUEST',
            'message_id': message_id,
            'retry_count': attempt
        }
        transmit_on_patterns(retry_request)  # ~1 second
        return 'fast_retry'

    def kernel_exchange_retry(self, station, message_id, attempt):
        """Robust retry via kernel exchange on 4-FSK"""

        # See kernel_lifecycle.md for unified KERNEL_EXCHANGE format
        kernel_exchange = {
            'from': my_hash,
            'my_rx_kernel': generate_fresh_kernel(),
            'for_message_id': message_id,
            'retry_flag': True,
            'retry_count': attempt,
            'retry_reason': 'validation_failure'
        }

        transmit_4fsk(kernel_exchange)  # ~5 seconds
        return 'kernel_refresh_retry'
```

**Retry flow example**:

```
Message validation fails (xxHash mismatch = hallucination)
  ↓
Attempt 1: Retry request on message patterns - 1 sec [FAIL - CRC error]
  ↓
Attempt 2: Retry request on message patterns - 1 sec [FAIL - xxHash mismatch]
  ↓
Attempt 3: Retry request on message patterns - 1 sec [FAIL - timeout]
  ↓
Attempt 4: KERNEL_EXCHANGE on 4-FSK with retry_flag - 5 sec
           Partner receives: fresh RX kernel + retry request
           Partner retransmits using fresh kernel
           [SUCCESS]

Total: 8 seconds, 4 attempts
```

### Why Kernel Exchange for Retry

**After 3 failures, kernel might be the problem**:

```python
reasons_for_kernel_refresh_retry = {
    'stale_kernel': 'Kernel optimized for old conditions',
    'decode_hints_wrong': 'Kernel hints not matching current propagation',
    'interference_changed': 'New interference requires adaptation',
    'hardware_state_drift': 'Receiver characteristics changed (temperature, etc.)',

    'benefit': 'Fresh kernel might fix underlying issue',
    'cost': '4 extra seconds (vs fast retry)',
    'justification': 'After 3 failures, something is fundamentally wrong'
}
```

---

*Last updated: 2025-10-02*

*Related documents*:
- [Kernel Lifecycle](kernel_lifecycle.md) - Unified KERNEL_EXCHANGE protocol with retry
- [Signal Specification](signal_specification.md) - Physical layer parameters
- [Emergency Validation](emergency_validation.md) - Emergency message validation
- [telemetry_research.md](../../telemetry_research.md) - Validation telemetry analysis
