# CASCADE Kernel Encoding Specification

**Purpose:** Complete specification for 64-bit kernel encoding/decoding
**Status:** Ready for implementation
**Used in:** Beacons, ACKs, kernel exchange messages

---

## Overview

Kernels are 64-bit receiver optimization hints that encode:
- Hardware capabilities
- Available tone subset (critical for per-receiver adaptation)
- Preferred modulation/FEC
- Pattern pool recommendations

**Key constraint:** Must fit in 64 bits while encoding 4-tone availability (typically all 4 available).

---

## 64-Bit Kernel Structure

### Layout (64 bits total)

```python
KERNEL_64BIT_STRUCTURE = {
    # Metadata (8 bits)
    'version': 3,               # bits (8 versions: 0-7)
    'valid_seconds': 5,          # bits (0-31 × 30s = 0-930s validity estimate)

    # Hardware capabilities (12 bits)
    'hardware_tier': 3,          # bits (8 tiers: RPi, RPi+Coral, Desktop, GPU, etc.)
    'max_patterns_simultaneous': 3,  # bits (1-8 patterns)
    'max_constellation': 3,      # bits (BPSK=0, QPSK=1, 8QAM=2, 16QAM=3, 64QAM=4, etc.)
    'max_users_decodable': 3,    # bits (0-7 → 1-128 users, log scale)

    # Available tone encoding (40 bits) - CRITICAL
    'tone_availability': 40,     # bits (run-length encoding, up to 4 ranges)

    # Preferences (4 bits)
    'preferred_fec_rate': 2,     # bits (4 levels: 0.33, 0.5, 0.67, 0.75)
    'emergency_capable': 1,      # bit (can participate in emergency nets)
    'beacon_pattern_base': 1,    # bit (which beacon pattern pool: 0-15 or 16-63)
}

# Total: 3 + 5 + 12 + 40 + 4 = 64 bits ✓
```

---

## Available Tone Encoding (40 bits)

This is the most critical part - encoding which of 78 discrete reference tones are available at receiver.

### Run-Length Encoding Algorithm

```python
def encode_available_tones_40bit(available_tone_indices):
    """
    Encode available tones using run-length encoding

    Args:
        available_tone_indices: List of available tone indices
                               e.g., [0,1,2,3,5,6,7,...,68,69] (tone 4 missing)

    Returns:
        40-bit encoded representation
    """

    # Find contiguous ranges
    ranges = find_contiguous_ranges(available_tone_indices)
    # e.g., [0-3], [5-69] → [(0, 4), (5, 65)]

    # Encoding format (40 bits):
    # - 4 bits: Number of ranges (0-15, supports up to 15 ranges)
    # - 36 bits: Up to 4 ranges × 9 bits each

    num_ranges = min(len(ranges), 4)  # Max 4 ranges in 40 bits
    encoded = num_ranges  # First 4 bits

    for i, (start, length) in enumerate(ranges[:4]):
        # Each range: 9 bits
        # - 7 bits: start index (0-127, but we use 0-69)
        # - 2 bits: length encoding
        #   00: length = 1 (single tone)
        #   01: next 7 bits contain length
        #   10: "to end" (remaining tones)
        #   11: reserved

        if length == 1:
            # Single tone
            range_bits = (start << 2) | 0b00
            bit_count = 9

        elif length <= 127:
            # Explicit length (most common)
            range_bits = (start << 2) | 0b01
            range_bits = (range_bits << 7) | length
            bit_count = 9

        else:
            # Shouldn't happen with 4 tones, but handle anyway
            range_bits = (start << 2) | 0b10
            bit_count = 9

        # Pack into result
        encoded |= (range_bits << (4 + i * 9))

    return encoded  # 40 bits


def decode_available_tones_40bit(encoded_40bit):
    """
    Decode available tones from 40-bit encoding

    Returns:
        List of available tone indices
    """

    # Extract number of ranges
    num_ranges = encoded_40bit & 0xF

    available = []

    for i in range(num_ranges):
        # Extract range (9 bits)
        range_bits = (encoded_40bit >> (4 + i * 9)) & 0x1FF

        # Parse
        start = (range_bits >> 2) & 0x7F
        length_code = range_bits & 0x03

        if length_code == 0b00:
            # Single tone
            available.append(start)

        elif length_code == 0b01:
            # Explicit length
            length = (range_bits >> 2) & 0x7F  # Re-extract differently
            # Actually need to extract length from next 7 bits
            # This needs careful bit packing - implementation detail

            for j in range(length):
                available.append(start + j)

        elif length_code == 0b10:
            # To end (rare)
            for j in range(start, 70):
                available.append(j)

    return available
```

### Example Encodings

```python
# Example 1: All tones available [0-69]
all_tones = range(0, 78)
# Encoding: 1 range, (start=0, length=70)
# Bits: 0001 | (0<<2|01)<<7|(70) = ...
# Result: Compact (uses <20 bits of 40 available)

# Example 2: Selective fading [0-34, 40-69]
selective = list(range(0, 35)) + list(range(40, 70))
# Encoding: 2 ranges, (0, 35), (40, 30)
# Bits: 0010 | range1 | range2
# Result: Uses ~25 bits

# Example 3: Heavy QRM [5-12, 25-35, 50-69]
heavy_qrm = list(range(5, 13)) + list(range(25, 36)) + list(range(50, 70))
# Encoding: 3 ranges
# Result: Uses ~32 bits

# Example 4: Extreme sparse [10, 25, 40, 55]
extreme = [10, 25, 40, 55]
# Encoding: 4 single-tone ranges
# Result: Uses all 40 bits (worst case)
```

---

## Complete Kernel Encoding Implementation

```python
def encode_kernel_64bit(receiver_state):
    """
    Encode complete 64-bit kernel

    Args:
        receiver_state: {
            'hardware_tier': int (0-7),
            'max_patterns': int (1-8),
            'max_constellation': int (0-7),
            'max_users': int (1-128),
            'available_tones': List[int],
            'preferred_fec': float (0.33-0.75),
            'emergency_capable': bool,
            'beacon_pattern_base': int (0-1),
        }

    Returns:
        64-bit integer
    """

    kernel = 0

    # Version (3 bits, offset 0)
    version = 1
    kernel |= (version << 0)

    # Valid seconds (5 bits, offset 3)
    valid_slots = min(receiver_state['valid_estimate'] // 30, 31)
    kernel |= (valid_slots << 3)

    # Hardware tier (3 bits, offset 8)
    kernel |= (receiver_state['hardware_tier'] << 8)

    # Max patterns (3 bits, offset 11)
    max_patterns_encoded = receiver_state['max_patterns'] - 1  # 1-8 → 0-7
    kernel |= (max_patterns_encoded << 11)

    # Max constellation (3 bits, offset 14)
    kernel |= (receiver_state['max_constellation'] << 14)

    # Max users (3 bits, offset 17, log scale)
    max_users_log = int(np.log2(receiver_state['max_users']))  # 1→0, 128→7
    kernel |= (max_users_log << 17)

    # Preferred FEC (2 bits, offset 20)
    fec_map = {0.33: 0, 0.5: 1, 0.67: 2, 0.75: 3}
    fec_encoded = fec_map.get(receiver_state['preferred_fec'], 1)
    kernel |= (fec_encoded << 20)

    # Emergency capable (1 bit, offset 22)
    kernel |= (int(receiver_state['emergency_capable']) << 22)

    # Beacon pattern base (1 bit, offset 23)
    kernel |= (receiver_state['beacon_pattern_base'] << 23)

    # Available tones (40 bits, offset 24-63)
    tone_encoding = encode_available_tones_40bit(
        receiver_state['available_tones']
    )
    kernel |= (tone_encoding << 24)

    return kernel  # 64-bit integer


def decode_kernel_64bit(kernel_64bit):
    """
    Decode 64-bit kernel

    Returns:
        Dictionary with all kernel fields
    """

    decoded = {}

    # Extract fields (reverse of encoding)
    decoded['version'] = (kernel_64bit >> 0) & 0x7
    decoded['valid_seconds'] = ((kernel_64bit >> 3) & 0x1F) * 30
    decoded['hardware_tier'] = (kernel_64bit >> 8) & 0x7
    decoded['max_patterns'] = ((kernel_64bit >> 11) & 0x7) + 1
    decoded['max_constellation'] = (kernel_64bit >> 14) & 0x7
    decoded['max_users_log'] = (kernel_64bit >> 17) & 0x7
    decoded['max_users'] = 2 ** decoded['max_users_log']
    decoded['preferred_fec'] = [0.33, 0.5, 0.67, 0.75][(kernel_64bit >> 20) & 0x3]
    decoded['emergency_capable'] = bool((kernel_64bit >> 22) & 0x1)
    decoded['beacon_pattern_base'] = (kernel_64bit >> 23) & 0x1

    # Available tones (40 bits)
    tone_encoding = (kernel_64bit >> 24) & 0xFFFFFFFFFF  # 40 bits
    decoded['available_tones'] = decode_available_tones_40bit(tone_encoding)

    return decoded
```

---

## Kernel Usage in Transmission

```python
def transmit_using_kernel(data, target_kernel):
    """
    How transmitter uses decoded kernel
    """

    # Decode kernel
    kernel = decode_kernel_64bit(target_kernel)

    # Select pattern pool
    multipath = estimate_multipath()  # From beacon measurements
    if multipath < 1:
        pool_range = range(208, 240)  # Good prop
    elif multipath < 8:
        pool_range = range(80, 208)  # Typical DX (most common)
    else:
        pool_range = range(64, 80)  # Emergency

    # Filter by available tones
    available_tones = kernel['available_tones']
    # e.g., [0-34, 40-69] (tones 35-39 have QRM at receiver)

    # Select patterns from pool
    my_patterns = protocol.get_assigned_patterns()  # e.g., [88-95]
    patterns_in_pool = [p for p in my_patterns if p in pool_range]

    # Select up to max_patterns
    max_patterns = kernel['max_patterns']  # e.g., 4
    selected_patterns = patterns_in_pool[:max_patterns]

    # For each pattern, use only available tones
    for pattern in selected_patterns:
        for symbol in range(32):
            base_tone = pattern.freq_sequence[symbol]

            if base_tone in available_tones:
                use_tone = base_tone
            else:
                # Shift to nearest available (±3 tones)
                use_tone = find_nearest_available(base_tone, available_tones)

            transmit(
                frequency=REFERENCE_TONES[use_tone],
                iq=pattern.iq_trajectory[symbol],
                duration=50  # ms
            )
```

---

## See Also

- **[Kernel Lifecycle](../protocol/kernel_lifecycle.md)** - Complete kernel exchange protocol
- **[Pattern Architecture](../model/pattern_architecture.md)** - Pattern pool organization
- **[Adaptive Tone Grid](../protocol/adaptive_tone_grid.md)** - 78-tone specification

---

*Ready for implementation*
