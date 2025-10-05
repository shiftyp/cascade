"""CASCADE Binary Pattern File Format I/O

Binary Format Specification (v3 - with flip-orthogonality):
- Header (32 bytes):
  - Magic: b'CASC' (4 bytes)
  - Version: uint8 (1 byte) = 3
  - Pattern count: uint16 (2 bytes)
  - Reserved: 25 bytes for future use

- Per Pattern (304 bytes):
  - Pattern ID: uint8 (1 byte)
  - Freq sequence: 32 × uint8 (32 bytes)
  - IQ trajectory: 32 × complex64 (256 bytes = 32 × 8)
  - IQ complexity λ: float32 (4 bytes)
  - Flip-orthogonality stats (9 bytes):
    - Max flip correlation: float32 (4 bytes)
    - Avg flip correlation: float32 (4 bytes)
    - Adjacent channel safe: uint8 (1 byte, 0 or 1)
  - Checksum: uint16 CRC (2 bytes) - computed over pattern data
"""

from typing import List
import struct
import numpy as np
from .models import Pattern


def save_pattern_file(patterns: List[Pattern], filename: str) -> None:
    """Save patterns to CASCADE binary format (v3 with flip-orthogonality)

    Args:
        patterns: List of Pattern objects
        filename: Output file path
    """
    with open(filename, 'wb') as f:
        # Write header
        f.write(b'CASC')  # Magic bytes
        f.write(struct.pack('B', 3))  # Version 3 (with flip-orthogonality)
        f.write(struct.pack('<H', len(patterns)))  # Pattern count (little-endian uint16)
        f.write(b'\x00' * 25)  # Reserved

        # Write each pattern
        for pattern in patterns:
            pattern_data = _pack_pattern(pattern)
            f.write(pattern_data)


def load_pattern_file(filename: str) -> List[Pattern]:
    """Load patterns from CASCADE binary format

    Args:
        filename: Pattern file path

    Returns:
        List of Pattern objects

    Raises:
        ValueError: If magic bytes invalid or checksums fail
    """
    patterns = []

    with open(filename, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'CASC':
            raise ValueError(f"Invalid magic bytes: {magic}, expected b'CASC'")

        version = struct.unpack('B', f.read(1))[0]
        if version not in [2, 3]:
            raise ValueError(f"Unsupported version: {version}, expected 2 or 3")

        pattern_count = struct.unpack('<H', f.read(2))[0]
        f.read(25)  # Skip reserved bytes

        # Read each pattern
        for _ in range(pattern_count):
            if version == 2:
                # Old format without flip-orthogonality data
                pattern_data = f.read(295)
                if len(pattern_data) < 295:
                    raise ValueError("Incomplete pattern data in file")
                pattern = _unpack_pattern_v2(pattern_data)
            else:  # version == 3
                # New format with flip-orthogonality data
                pattern_data = f.read(304)
                if len(pattern_data) < 304:
                    raise ValueError("Incomplete pattern data in file")
                pattern = _unpack_pattern(pattern_data)

            patterns.append(pattern)

    return patterns


def _pack_pattern(pattern: Pattern) -> bytes:
    """Pack a Pattern into binary format v3 (with flip-orthogonality)

    Args:
        pattern: Pattern object

    Returns:
        304 bytes of packed pattern data
    """
    data = bytearray()

    # Pattern ID (1 byte)
    data.extend(struct.pack('B', pattern.pattern_id))

    # Freq sequence (32 bytes)
    data.extend(pattern.freq_sequence.tobytes())

    # IQ trajectory (256 bytes)
    # Complex64 = 2 × float32 per symbol × 32 symbols = 256 bytes
    data.extend(pattern.iq_trajectory.tobytes())

    # IQ complexity λ (4 bytes)
    data.extend(struct.pack('<f', pattern.iq_complexity_lambda))

    # Flip-orthogonality stats (9 bytes total)
    # Max flip correlation (4 bytes)
    max_flip = pattern.flip_orthogonality_stats.get('max_flip_correlation_db', -100.0)
    data.extend(struct.pack('<f', max_flip if max_flip is not None else -100.0))

    # Avg flip correlation (4 bytes)
    avg_flip = pattern.flip_orthogonality_stats.get('avg_flip_correlation_db', -100.0)
    data.extend(struct.pack('<f', avg_flip if avg_flip is not None else -100.0))

    # Adjacent channel safe (1 byte: 0 or 1)
    safe = pattern.flip_orthogonality_stats.get('adjacent_channel_safe', False)
    data.extend(struct.pack('B', 1 if safe else 0))

    # Compute checksum (CRC16 over all pattern data so far)
    checksum = _compute_crc16(bytes(data))
    data.extend(struct.pack('<H', checksum))

    assert len(data) == 304, f"Pattern data should be 304 bytes, got {len(data)}"
    return bytes(data)


def _unpack_pattern_v2(data: bytes) -> Pattern:
    """Unpack binary data into a Pattern object (v2 format, no flip data)

    Args:
        data: 295 bytes of pattern data

    Returns:
        Pattern object

    Raises:
        ValueError: If checksum fails
    """
    if len(data) != 295:
        raise ValueError(f"Pattern data must be 295 bytes, got {len(data)}")

    offset = 0

    # Pattern ID (1 byte)
    pattern_id = struct.unpack('B', data[offset:offset+1])[0]
    offset += 1

    # Freq sequence (32 bytes)
    freq_sequence = np.frombuffer(data[offset:offset+32], dtype='uint8')
    offset += 32

    # IQ trajectory (256 bytes)
    iq_trajectory = np.frombuffer(data[offset:offset+256], dtype='complex64')
    offset += 256

    # IQ complexity λ (4 bytes)
    iq_complexity_lambda = struct.unpack('<f', data[offset:offset+4])[0]
    offset += 4

    # Checksum (2 bytes)
    stored_checksum = struct.unpack('<H', data[offset:offset+2])[0]
    offset += 2

    # Verify checksum (computed over first 293 bytes)
    computed_checksum = _compute_crc16(data[:offset-2])
    if computed_checksum != stored_checksum:
        raise ValueError(f"Checksum mismatch: computed {computed_checksum:04x}, stored {stored_checksum:04x}")

    # Create Pattern object
    pattern = Pattern(
        pattern_id=pattern_id,
        freq_sequence=freq_sequence,
        iq_trajectory=iq_trajectory,
        iq_complexity_lambda=iq_complexity_lambda
    )

    return pattern


def _unpack_pattern(data: bytes) -> Pattern:
    """Unpack binary data into a Pattern object (v3 format with flip-orthogonality)

    Args:
        data: 304 bytes of pattern data

    Returns:
        Pattern object

    Raises:
        ValueError: If checksum fails
    """
    if len(data) != 304:
        raise ValueError(f"Pattern data must be 304 bytes, got {len(data)}")

    offset = 0

    # Pattern ID (1 byte)
    pattern_id = struct.unpack('B', data[offset:offset+1])[0]
    offset += 1

    # Freq sequence (32 bytes)
    freq_sequence = np.frombuffer(data[offset:offset+32], dtype='uint8')
    offset += 32

    # IQ trajectory (256 bytes)
    iq_trajectory = np.frombuffer(data[offset:offset+256], dtype='complex64')
    offset += 256

    # IQ complexity λ (4 bytes)
    iq_complexity_lambda = struct.unpack('<f', data[offset:offset+4])[0]
    offset += 4

    # Flip-orthogonality stats (9 bytes)
    # Max flip correlation (4 bytes)
    max_flip_corr = struct.unpack('<f', data[offset:offset+4])[0]
    offset += 4

    # Avg flip correlation (4 bytes)
    avg_flip_corr = struct.unpack('<f', data[offset:offset+4])[0]
    offset += 4

    # Adjacent channel safe (1 byte)
    adjacent_safe = struct.unpack('B', data[offset:offset+1])[0]
    offset += 1

    # Checksum (2 bytes)
    stored_checksum = struct.unpack('<H', data[offset:offset+2])[0]
    offset += 2

    # Verify checksum (computed over first 302 bytes)
    computed_checksum = _compute_crc16(data[:offset-2])
    if computed_checksum != stored_checksum:
        raise ValueError(f"Checksum mismatch: computed {computed_checksum:04x}, stored {stored_checksum:04x}")

    # Create Pattern object
    pattern = Pattern(
        pattern_id=pattern_id,
        freq_sequence=freq_sequence,
        iq_trajectory=iq_trajectory,
        iq_complexity_lambda=iq_complexity_lambda
    )

    # Populate flip-orthogonality stats
    pattern.flip_orthogonality_stats['max_flip_correlation_db'] = max_flip_corr if max_flip_corr != -100.0 else None
    pattern.flip_orthogonality_stats['avg_flip_correlation_db'] = avg_flip_corr if avg_flip_corr != -100.0 else None
    pattern.flip_orthogonality_stats['adjacent_channel_safe'] = bool(adjacent_safe)

    return pattern


def _compute_crc16(data: bytes) -> int:
    """Compute CRC16-CCITT checksum

    Args:
        data: Bytes to checksum

    Returns:
        16-bit CRC value
    """
    crc = 0xFFFF
    polynomial = 0x1021

    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ polynomial) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

    return crc
