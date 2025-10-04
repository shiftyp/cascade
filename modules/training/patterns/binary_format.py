"""CASCADE Binary Pattern File Format I/O

Binary Format Specification:
- Header (32 bytes):
  - Magic: b'CASC' (4 bytes)
  - Version: uint8 (1 byte) = 2
  - Pattern count: uint16 (2 bytes)
  - Reserved: 25 bytes for future use

- Per Pattern (295 bytes):
  - Pattern ID: uint8 (1 byte)
  - Freq sequence: 32 × uint8 (32 bytes)
  - IQ trajectory: 32 × complex64 (256 bytes = 32 × 8)
  - IQ complexity λ: float32 (4 bytes)
  - Checksum: uint16 CRC (2 bytes) - computed over pattern data
"""

from typing import List
import struct
import numpy as np
from .models import Pattern


def save_pattern_file(patterns: List[Pattern], filename: str) -> None:
    """Save patterns to CASCADE binary format

    Args:
        patterns: List of Pattern objects
        filename: Output file path
    """
    with open(filename, 'wb') as f:
        # Write header
        f.write(b'CASC')  # Magic bytes
        f.write(struct.pack('B', 2))  # Version
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
        if version != 2:
            raise ValueError(f"Unsupported version: {version}, expected 2")

        pattern_count = struct.unpack('<H', f.read(2))[0]
        f.read(25)  # Skip reserved bytes

        # Read each pattern
        for _ in range(pattern_count):
            pattern_data = f.read(295)
            if len(pattern_data) < 295:
                raise ValueError("Incomplete pattern data in file")

            pattern = _unpack_pattern(pattern_data)
            patterns.append(pattern)

    return patterns


def _pack_pattern(pattern: Pattern) -> bytes:
    """Pack a Pattern into binary format

    Args:
        pattern: Pattern object

    Returns:
        292 bytes of packed pattern data
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

    # Compute checksum (CRC16 over all pattern data so far)
    checksum = _compute_crc16(bytes(data))
    data.extend(struct.pack('<H', checksum))

    assert len(data) == 295, f"Pattern data should be 295 bytes, got {len(data)}"
    return bytes(data)


def _unpack_pattern(data: bytes) -> Pattern:
    """Unpack binary data into a Pattern object

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
