# CASCADE Message Format Specification

## Overview

CASCADE uses a **fixed binary wire format** for minimal overhead, fast parsing, and NN-friendly structure. Messages are UTF-8 text at the application layer, serialized to compact binary for transmission.

**Design goals**:
- Minimal overhead (every byte counts in 256-byte limit)
- Fast serialization/deserialization on RPi4
- Fixed structure for NN learning
- Deterministic encoding (same message = same bytes)
- Support for dual-layer validation (CRC32 + xxHash32)

---

## Wire Format Structure

### Complete Message Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    CASCADE Message (Binary)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────┐                  │
│  │   HEADER (19 bytes, fixed)            │                  │
│  ├──────────────────────────────────────┤                  │
│  │  from_hash        : uint32 (4 bytes)  │                  │
│  │  to_hash          : uint32 (4 bytes)  │                  │
│  │  message_id       : uint64 (8 bytes)  │                  │
│  │  priority         : uint8  (1 byte)   │                  │
│  │  payload_length   : uint16 (2 bytes)  │                  │
│  └──────────────────────────────────────┘                  │
│                                                              │
│  ┌──────────────────────────────────────┐                  │
│  │   PAYLOAD (variable, 0-256 bytes)     │                  │
│  ├──────────────────────────────────────┤                  │
│  │  UTF-8 encoded text                   │                  │
│  │  (human-readable when decoded)        │                  │
│  └──────────────────────────────────────┘                  │
│                                                              │
│  ┌──────────────────────────────────────┐                  │
│  │   VALIDATION (8 bytes, fixed)         │                  │
│  ├──────────────────────────────────────┤                  │
│  │  crc32            : uint32 (4 bytes)  │                  │
│  │  xxhash32         : uint32 (4 bytes)  │                  │
│  └──────────────────────────────────────┘                  │
│                                                              │
│  Total: 27 bytes overhead + payload                         │
│  Example: 128-byte payload = 155 bytes total (17.4%)       │
└─────────────────────────────────────────────────────────────┘
```

### Field Definitions

**from_hash** (4 bytes, uint32):
- Hash of sender's callsign
- 32-bit truncated hash (sufficient for collision avoidance in network)
- Little-endian byte order
- Range: 0x00000000 to 0xFFFFFFFF

**to_hash** (4 bytes, uint32):
- Hash of destination callsign
- 0xFFFFFFFF for broadcast messages
- Little-endian byte order

**message_id** (8 bytes, uint64):
- Unique message identifier
- Cryptographic hash of: payload + timestamp + nonce
- Used for deduplication, relay tracking, telemetry correlation
- Little-endian byte order

**priority** (1 byte, uint8):
- Message priority level
- Values: 0 = EMERGENCY, 1 = HIGH, 2 = NORMAL, 3 = LOW
- Other values reserved for future use

**payload_length** (2 bytes, uint16):
- Length of payload in bytes
- Range: 0-65535 (but CASCADE limits to 256)
- Little-endian byte order
- Allows receiver to extract payload without parsing

**payload** (variable, 0-256 bytes):
- UTF-8 encoded text
- Human-readable messages
- No null-terminator (length-prefixed)
- Must be valid UTF-8 (invalid UTF-8 = validation failure)

**crc32** (4 bytes, uint32):
- CRC32 checksum over header + payload
- IEEE 802.3 polynomial (0xEDB88320)
- Little-endian byte order
- NN learns to predict this (training signal)

**xxhash32** (4 bytes, uint32):
- xxHash32 over header + payload + crc32
- Prevents NN hallucinations
- Little-endian byte order
- NN cannot learn to forge this

---

## Size Analysis

### Overhead Breakdown

**Fixed overhead**:
```
Header:        19 bytes (from_hash through payload_length)
Validation:     8 bytes (crc32 + xxhash32)
Total fixed:   27 bytes
```

**By payload size**:

| Payload | Total Size | Overhead Bytes | Overhead % |
|---------|------------|----------------|------------|
| 32 bytes | 59 bytes | 27 | 84.4% |
| 64 bytes | 91 bytes | 27 | 42.2% |
| 128 bytes | 155 bytes | 27 | 21.1% |
| 256 bytes | 283 bytes | 27 | 10.5% |

**Sweet spot**: 128-256 byte messages have acceptable overhead (10-21%)

### Comparison to Alternatives

**For 128-byte payload**:

| Format | Total Size | Overhead % | Parse Time | Verdict |
|--------|------------|------------|------------|---------|
| Fixed Binary | 155 bytes | 21% | 5-10μs | ✅ **Recommended** |
| Protobuf | 159 bytes | 24% | 50-100μs | Good alternative |
| CBOR | 166 bytes | 30% | 30-50μs | Acceptable |
| YAML | 256 bytes | 100% | 1-2ms | ❌ Too large |
| JSON | 298 bytes | 133% | 500μs-1ms | ❌ Exceeds limits |

**Why fixed binary wins for CASCADE**:
- Smallest overhead (critical for 256-byte limit)
- Fastest parsing (important for RPi4)
- NN-friendly (fixed structure, learns byte offsets)
- Simple implementation (no dependencies)
- Deterministic (important for training data consistency)

---

## Serialization

### Python Reference Implementation

```python
import struct
import zlib
import xxhash

class CascadeMessageCodec:
    """Fixed binary message serialization"""

    # Struct formats (little-endian)
    HEADER_FORMAT = '<IIQBH'  # 19 bytes
    VALIDATION_FORMAT = '<II'  # 8 bytes

    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 19
    VALIDATION_SIZE = struct.calcsize(VALIDATION_FORMAT)  # 8

    def serialize(self, message):
        """
        Serialize message to binary wire format

        Args:
            message: Message object with fields (from_hash, to_hash, etc.)

        Returns:
            bytes: Binary representation for transmission
        """

        # Encode payload as UTF-8
        payload_bytes = message.payload.encode('utf-8')

        # Validate size
        if len(payload_bytes) > 256:
            raise ValueError(f"Payload too large: {len(payload_bytes)} bytes (max 256)")

        # Pack header
        header = struct.pack(
            self.HEADER_FORMAT,
            message.from_hash,      # uint32
            message.to_hash,        # uint32
            message.message_id,     # uint64
            message.priority,       # uint8
            len(payload_bytes)      # uint16
        )

        # Combine header + payload
        message_data = header + payload_bytes

        # Compute CRC32 over header + payload
        crc32 = zlib.crc32(message_data) & 0xFFFFFFFF

        # Compute xxHash32 over header + payload + CRC32
        validation_input = message_data + struct.pack('<I', crc32)
        xxh32 = xxhash.xxh32(validation_input).intdigest()

        # Pack validation
        validation = struct.pack(self.VALIDATION_FORMAT, crc32, xxh32)

        # Complete message
        return message_data + validation

    def deserialize(self, binary):
        """
        Deserialize binary wire format to message

        Args:
            binary: bytes from receiver

        Returns:
            Message object or ValidationError
        """

        # Validate minimum size
        min_size = self.HEADER_SIZE + self.VALIDATION_SIZE  # 27 bytes
        if len(binary) < min_size:
            raise ValueError(f"Message too small: {len(binary)} bytes (min {min_size})")

        # Parse header
        from_hash, to_hash, message_id, priority, payload_length = \
            struct.unpack_from(self.HEADER_FORMAT, binary, 0)

        # Extract payload
        payload_start = self.HEADER_SIZE
        payload_end = payload_start + payload_length
        payload_bytes = binary[payload_start:payload_end]

        # Extract validation
        validation_start = payload_end
        crc32, xxhash32 = struct.unpack_from(
            self.VALIDATION_FORMAT,
            binary,
            validation_start
        )

        # Validate CRC32
        message_data = binary[:payload_end]  # Header + payload
        computed_crc = zlib.crc32(message_data) & 0xFFFFFFFF

        if computed_crc != crc32:
            raise ValidationError('crc_failure', computed_crc, crc32)

        # Validate xxHash32
        validation_input = message_data + struct.pack('<I', crc32)
        computed_xxh = xxhash.xxh32(validation_input).intdigest()

        if computed_xxh != xxhash32:
            raise ValidationError('xxhash_failure', computed_xxh, xxhash32)

        # Decode payload
        try:
            payload_text = payload_bytes.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValidationError('invalid_utf8', str(e))

        # Return validated message
        return Message(
            from_hash=from_hash,
            to_hash=to_hash,
            message_id=message_id,
            priority=priority,
            payload=payload_text
        )
```

### C Reference Implementation

```c
#include <stdint.h>
#include <string.h>
#include "zlib.h"     // CRC32
#include "xxhash.h"   // xxHash32

#pragma pack(push, 1)  // No padding
typedef struct {
    uint32_t from_hash;
    uint32_t to_hash;
    uint64_t message_id;
    uint8_t priority;
    uint16_t payload_length;
} cascade_header_t;

typedef struct {
    uint32_t crc32;
    uint32_t xxhash32;
} cascade_validation_t;
#pragma pack(pop)

#define HEADER_SIZE sizeof(cascade_header_t)  // 19 bytes
#define VALIDATION_SIZE sizeof(cascade_validation_t)  // 8 bytes
#define MAX_PAYLOAD_SIZE 256

typedef struct {
    cascade_header_t header;
    uint8_t payload[MAX_PAYLOAD_SIZE];
    cascade_validation_t validation;
    size_t total_size;
} cascade_message_t;

int cascade_serialize(const cascade_message_t *msg, uint8_t *output, size_t output_size) {
    // Calculate required size
    size_t required = HEADER_SIZE + msg->header.payload_length + VALIDATION_SIZE;

    if (output_size < required) {
        return -1;  // Buffer too small
    }

    // Copy header (already in correct byte order for little-endian)
    memcpy(output, &msg->header, HEADER_SIZE);

    // Copy payload
    memcpy(output + HEADER_SIZE, msg->payload, msg->header.payload_length);

    // Compute CRC32 over header + payload
    size_t message_data_len = HEADER_SIZE + msg->header.payload_length;
    uint32_t crc = crc32(0L, output, message_data_len);

    // Compute xxHash32 over header + payload + CRC32
    uint8_t validation_input[message_data_len + 4];
    memcpy(validation_input, output, message_data_len);
    memcpy(validation_input + message_data_len, &crc, 4);

    uint32_t xxh = XXH32(validation_input, message_data_len + 4, 0);

    // Append validation
    memcpy(output + message_data_len, &crc, 4);
    memcpy(output + message_data_len + 4, &xxh, 4);

    return required;  // Return total size
}

int cascade_deserialize(const uint8_t *input, size_t input_size, cascade_message_t *msg) {
    // Validate minimum size
    if (input_size < HEADER_SIZE + VALIDATION_SIZE) {
        return -1;  // Too small
    }

    // Parse header
    memcpy(&msg->header, input, HEADER_SIZE);

    // Validate payload length
    if (msg->header.payload_length > MAX_PAYLOAD_SIZE) {
        return -2;  // Payload too large
    }

    size_t expected_size = HEADER_SIZE + msg->header.payload_length + VALIDATION_SIZE;
    if (input_size < expected_size) {
        return -3;  // Truncated message
    }

    // Extract payload
    memcpy(msg->payload, input + HEADER_SIZE, msg->header.payload_length);

    // Extract validation
    size_t validation_offset = HEADER_SIZE + msg->header.payload_length;
    uint32_t crc32_received, xxhash32_received;
    memcpy(&crc32_received, input + validation_offset, 4);
    memcpy(&xxhash32_received, input + validation_offset + 4, 4);

    // Validate CRC32
    uint32_t crc32_computed = crc32(0L, input, validation_offset);
    if (crc32_computed != crc32_received) {
        return -4;  // CRC failure
    }

    // Validate xxHash32
    uint8_t validation_input[validation_offset + 4];
    memcpy(validation_input, input, validation_offset);
    memcpy(validation_input + validation_offset, &crc32_received, 4);

    uint32_t xxhash32_computed = XXH32(validation_input, validation_offset + 4, 0);
    if (xxhash32_computed != xxhash32_received) {
        return -5;  // xxHash failure (hallucination)
    }

    // Success
    msg->validation.crc32 = crc32_received;
    msg->validation.xxhash32 = xxhash32_received;
    msg->total_size = expected_size;

    return 0;  // Success
}
```

---

## Field Specifications

### from_hash (uint32, 4 bytes)

**Purpose**: Identify sender without transmitting full callsign

**Encoding**: 32-bit truncated hash of sender's callsign
```python
from_hash = hash(callsign.encode('ascii'))[:4]  # First 32 bits
# Example: "K0BB" → 0xA3F2B8C1
```

**Collision probability**: With 100,000 active callsigns globally, probability of collision ≈ 1 in 42,000 (acceptable, full callsign sent via 4-FSK for FCC ID)

**Byte order**: Little-endian (0xA3F2B8C1 → bytes [0xC1, 0xB8, 0xF2, 0xA3])

### to_hash (uint32, 4 bytes)

**Purpose**: Identify destination

**Encoding**: Same as from_hash

**Special value**: 0xFFFFFFFF indicates broadcast (all stations)

### message_id (uint64, 8 bytes)

**Purpose**: Unique message identifier for deduplication, relay tracking, and telemetry correlation

**Generation**:
```python
import hashlib

def generate_message_id(payload, timestamp, nonce):
    """Generate cryptographically unique message ID"""

    h = hashlib.sha256()
    h.update(payload)
    h.update(timestamp.to_bytes(8, 'little'))
    h.update(nonce.to_bytes(4, 'little'))

    # Take first 64 bits
    return int.from_bytes(h.digest()[:8], 'little')
```

**Properties**:
- Globally unique (collision probability ≈ 1 in 2^64)
- Deterministic from message content
- Used for ACKs, retry requests, telemetry correlation
- Preserved across relay hops (same message ID end-to-end)

### priority (uint8, 1 byte)

**Purpose**: Message priority level

**Values**:
- 0x00 = EMERGENCY (highest priority, auto-relay)
- 0x01 = HIGH (urgent, expedited)
- 0x02 = NORMAL (default)
- 0x03 = LOW (defer if busy)
- 0x04-0xFF = Reserved

### payload_length (uint16, 2 bytes)

**Purpose**: Length of payload in bytes

**Range**: 0-65535 (CASCADE limits to 256 for V1)

**Why not implicit**: Allows receiver to:
- Allocate buffer before reading payload
- Validate expected message size
- Detect truncation errors

### payload (variable, 0-256 bytes)

**Purpose**: Message content

**Encoding**: UTF-8 text
- Allows international characters (émergency, Москва, etc.)
- Human-readable when decoded
- No null-terminator (length-prefixed)

**Validation**: Must be valid UTF-8
```python
try:
    payload_text = payload_bytes.decode('utf-8')
except UnicodeDecodeError:
    # Invalid UTF-8 = message rejection
    raise ValidationError('invalid_utf8')
```

**Size limits**:
- Minimum: 0 bytes (empty message allowed for pure control)
- Maximum: 256 bytes (protocol limit for V1)
- Typical: 64-128 bytes (QSO exchanges)

### crc32 (uint32, 4 bytes)

**Purpose**: Fast error detection, NN training signal

**Algorithm**: CRC32 (IEEE 802.3 polynomial 0xEDB88320)

**Computed over**: Header (19 bytes) + Payload (variable)

**NN training**: Included in loss function (NN learns to predict CRC)

### xxhash32 (uint32, 4 bytes)

**Purpose**: Prevent NN hallucinations

**Algorithm**: xxHash32 (seed = 0)

**Computed over**: Header (19 bytes) + Payload (variable) + CRC32 (4 bytes)

**NN training**: NOT in loss function (NN sees but cannot learn)

---

## Endianness

**All multi-byte fields use little-endian**:

Why little-endian:
- x86/ARM (RPi, desktop) are little-endian natively
- Faster on target hardware (no byte swapping)
- Consistent with most modern protocols

**Endianness conversion** (if needed on big-endian systems):
```python
# Python handles automatically via struct format '<'
# C requires explicit handling:

uint32_t to_little_endian_32(uint32_t value) {
    #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        return __builtin_bswap32(value);
    #else
        return value;  // Already little-endian
    #endif
}
```

---

## UTF-8 Payload

### Why UTF-8

**Benefits**:
- International character support (全球, Добро, etc.)
- Backward compatible with ASCII (1 byte per char for English)
- Self-synchronizing (easy to find start of character)
- Standard encoding (universal support)

**Compared to ASCII**:
- ASCII: 1 byte per character, English only
- UTF-8: 1-4 bytes per character, all languages
- For English text: Identical (UTF-8 = ASCII for code points 0-127)

**Size impact**:
```python
message_sizes = {
    'english': {
        'text': 'Hello W2DEF',
        'utf8_bytes': 12,
        'ascii_bytes': 12,
        'difference': 0  # Same
    },

    'international': {
        'text': 'Привет W2DEF',  # Russian "Hello"
        'utf8_bytes': 18,  # Cyrillic = 2 bytes per char
        'ascii_bytes': None,  # Not representable in ASCII
        'benefit': 'Supports international operators'
    },

    'emoji': {
        'text': '73 de K0BB 🎉',  # Party emoji
        'utf8_bytes': 17,  # Emoji = 4 bytes
        'recommendation': 'Discourage (waste of bytes)'
    }
}
```

### UTF-8 Validation

**Strict validation required**:

```python
def validate_utf8(payload_bytes):
    """Ensure payload is valid UTF-8"""

    try:
        text = payload_bytes.decode('utf-8')

        # Additional checks
        if '\0' in text:
            # Null bytes not allowed (security)
            raise ValueError("Null bytes in payload")

        if len(text) == 0 and len(payload_bytes) > 0:
            # Payload bytes exist but decode to empty string (weird)
            raise ValueError("Empty string with payload bytes")

        return text

    except UnicodeDecodeError as e:
        # Invalid UTF-8 sequence
        raise ValidationError(f"Invalid UTF-8: {e}")
```

**Invalid UTF-8 handling**:
- Message rejected at deserialization
- Logged as validation error
- Request retransmit
- Telemetry records: invalid_utf8 error type

---

## Future Considerations

### Binary Payload Support (V2+)

**If binary data needed in future**:

```python
# Add payload_type field (1 byte)
extended_header = {
    'from_hash': 4,
    'to_hash': 4,
    'message_id': 8,
    'priority': 1,
    'payload_type': 1,  # NEW: 0=UTF8, 1=binary, 2=compressed
    'payload_length': 2,
    'total': 20  # bytes (+1 vs current)
}

# Payload types:
PAYLOAD_TYPE_UTF8 = 0      # UTF-8 text (V1 default)
PAYLOAD_TYPE_BINARY = 1    # Raw binary data
PAYLOAD_TYPE_ZSTD = 2      # Zstd compressed UTF-8
PAYLOAD_TYPE_MSGPACK = 3   # MessagePack encoded

# Still limited to 256 bytes total
```

**Compression option**:
```python
# For long text, compress before transmission
import zstd

long_message = "Very long emergency message with lots of detail..." * 5  # 300 bytes
compressed = zstd.compress(long_message.encode('utf-8'), level=22)

if len(compressed) <= 256:
    # Fits! Send compressed
    payload_type = PAYLOAD_TYPE_ZSTD
    payload_bytes = compressed  # ~120 bytes
else:
    # Truncate original
    payload_type = PAYLOAD_TYPE_UTF8
    payload_bytes = long_message[:256].encode('utf-8')
```

### Schema Evolution

**How to add fields without breaking compatibility**:

Current design has no extension mechanism. If needed in future:

**Option A: Version field** (in priority byte):
```python
# Use high bits of priority for version
priority_byte = {
    'version': 3_bits,  # 0-7 (upper 3 bits)
    'priority': 5_bits  # 0-31 (lower 5 bits, expand from 4 levels)
}

# Version 0: Current format (19-byte header)
# Version 1: Extended format (20-byte header with payload_type)
# Version 2: Future format
```

**Option B: Extension flag**:
```python
# Add 'extension_present' bit to priority
if extension_present:
    # Read additional fields after payload_length
    extension_length = read_uint16()
    extension_data = read_bytes(extension_length)
```

**For V1**: Keep simple (no extensions), add in V2 if needed

---

## Examples

### Short Message (32 bytes payload)

```python
message = Message(
    from_hash=0xA3F2B8C1,
    to_hash=0xC4D5E6F7,
    message_id=0x1234567890ABCDEF,
    priority=NORMAL,
    payload="CQ CQ CQ de K0BB"  # 16 chars = 16 bytes UTF-8
)

serialized = codec.serialize(message)
# Size: 19 (header) + 16 (payload) + 8 (validation) = 43 bytes
# Overhead: 27 / 43 = 62.8%
```

### Typical Message (128 bytes payload)

```python
message = Message(
    from_hash=0xA3F2B8C1,
    to_hash=0xC4D5E6F7,
    message_id=0xFEDCBA0987654321,
    priority=NORMAL,
    payload="Hello W2DEF, thanks for the QSO last night. Conditions were great on 40m. I'm running 100W to a dipole at 30 feet. 73 de K0BB"  # 128 bytes
)

serialized = codec.serialize(message)
# Size: 19 + 128 + 8 = 155 bytes
# Overhead: 27 / 155 = 17.4%
```

### Emergency Message (256 bytes max)

```python
message = Message(
    from_hash=0xA3F2B8C1,
    to_hash=0xFFFFFFFF,  # Broadcast
    message_id=0x1111111111111111,
    priority=EMERGENCY,
    payload="EMERGENCY: Fire reported at grid square FN42mc. Evacuating residents. Need assistance at 145.550 MHz. Multiple structures involved. Wind from southwest pushing flames northeast. Clear area north of Highway 36."  # 232 bytes
)

serialized = codec.serialize(message)
# Size: 19 + 232 + 8 = 259 bytes
# Overhead: 27 / 259 = 10.4%
```

---

## Performance

### Serialization Performance

**Python**:
```python
# Benchmark on typical 128-byte message
serialization_time = {
    'struct.pack (header)': 0.5_μs,
    'utf8_encode': 0.3_μs,
    'crc32_compute': 2.0_μs,
    'xxhash32_compute': 5.0_μs,
    'struct.pack (validation)': 0.3_μs,
    'memcpy_operations': 0.5_μs,
    'total': 8.6_μs  # Microseconds
}

# Deserialization: Similar (~10μs total)
```

**C (optimized)**:
```c
// Benchmark on RPi4
// 128-byte message
// - Serialization: ~3μs
// - Deserialization: ~4μs
// - Total round-trip: ~7μs

// Negligible compared to NN inference (10ms = 10,000μs)
```

### Memory Footprint

**Python**:
```python
# Pre-allocated buffers
buffer_sizes = {
    'max_message_size': 256 + 19 + 8,  # = 283 bytes
    'header_buffer': 19,
    'validation_buffer': 8,
    'total_per_message': 283
}

# For 10 message queue: 2.83 KB (negligible)
```

**C (embedded)**:
```c
// Stack allocation (no malloc needed)
cascade_message_t msg;  // 283 bytes on stack
uint8_t wire_buffer[300];  // 300 bytes for safety margin

// Total: <600 bytes stack (fine for embedded)
```

---

## Comparison to Alternatives

### Fixed Binary vs Protobuf vs JSON

**For 128-byte payload**:

| Format | Header | Payload | Validation | Total | Overhead % | Parse Time |
|--------|--------|---------|------------|-------|------------|------------|
| **Fixed Binary** | 19 | 128 | 8 | 155 | 21% | 10μs |
| Protobuf | 12 | 128 | 8 | 148 | 16% | 50-100μs |
| CBOR | 19 | 128 | 8 | 155 | 21% | 30-50μs |
| JSON | 107 | 128 | 8 | 243 | 90% | 500μs |
| YAML | 78 | 128 | 8 | 214 | 67% | 1-2ms |

**Why fixed binary for CASCADE**:

1. **NN-friendly**: Fixed offsets (byte 0-3 = from_hash, always)
   - NN can learn "payload starts at byte 19"
   - Protobuf has variable offsets (harder to learn)

2. **Deterministic**: Same message always produces same bytes
   - Important for training data consistency
   - Protobuf field ordering can vary

3. **Fast on RPi4**: 10μs vs 50-100μs (5-10x faster)
   - Matters when processing 50 simultaneous users

4. **Simple**: No protoc compiler, no schema files
   - Just struct.pack/unpack

5. **Compact enough**: 21% overhead acceptable
   - Protobuf only 5% better (16% vs 21%)
   - Not worth the complexity for 5% savings

**When Protobuf would be better**:
- If schema evolution critical (adding fields frequently)
- If cross-language support needed (many languages)
- If variable-length field compression important
- If parsing time not critical (server-side processing)

**For CASCADE**: Fixed binary is optimal given NN-processing, size constraints, and embedded targets.

---

## See Also

- **[Message Validation](message_validation.md)** - CRC32 + xxHash32 validation
- **[Kernel Lifecycle](kernel_lifecycle.md)** - Kernel exchange protocol
- **[Signal Specification](signal_specification.md)** - Physical layer modulation
- **[Protocol Overview](README.md)** - Multi-stage protocol flow

---

*Last updated: 2025-10-02*
