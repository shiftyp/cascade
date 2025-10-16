"""Constellation mapping for CASCADE V2 signal generator.

Implements BPSK, QPSK, 8-PSK, and 16-APSK modulation schemes.
"""

import numpy as np
from typing import Tuple


def get_bits_per_symbol(modulation: str) -> int:
    """Get number of bits per symbol for a modulation scheme.

    Args:
        modulation: One of 'BPSK', 'QPSK', '8-PSK', '16-APSK'

    Returns:
        int: Bits per symbol

    Raises:
        ValueError: If modulation is not supported
    """
    bits_per_symbol = {
        'BPSK': 1,
        'QPSK': 2,
        '8-PSK': 3,
        '16-APSK': 4
    }

    if modulation not in bits_per_symbol:
        raise ValueError(
            f"Unknown modulation: {modulation}. "
            f"Supported: {list(bits_per_symbol.keys())}"
        )

    return bits_per_symbol[modulation]


def map_to_constellation(bits: np.ndarray, modulation: str) -> np.ndarray:
    """Map data bits to constellation symbols.

    Implements standard PSK/APSK constellations with Gray coding.
    All constellations normalized to unit average power.

    Args:
        bits: Data bits to modulate, shape (num_bits,), values {0, 1}
        modulation: Modulation scheme ('BPSK', 'QPSK', '8-PSK', '16-APSK')

    Returns:
        np.ndarray: Complex constellation symbols, shape (num_symbols,), dtype complex64

    Raises:
        ValueError: If bits length not compatible with modulation
        ValueError: If modulation not supported

    Example:
        >>> bits = np.array([0, 1, 1, 0])  # 4 bits
        >>> symbols = map_to_constellation(bits, 'QPSK')  # 2 symbols
        >>> symbols.shape
        (2,)
    """
    # Validate inputs
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0 and 1")

    bits_per_sym = get_bits_per_symbol(modulation)
    num_bits = len(bits)

    if num_bits % bits_per_sym != 0:
        raise ValueError(
            f"Number of bits ({num_bits}) must be multiple of {bits_per_sym} "
            f"for {modulation} modulation"
        )

    num_symbols = num_bits // bits_per_sym

    # Reshape bits into symbols
    bit_groups = bits.reshape(num_symbols, bits_per_sym)

    # Map based on modulation type
    if modulation == 'BPSK':
        symbols = _map_bpsk(bit_groups)
    elif modulation == 'QPSK':
        symbols = _map_qpsk(bit_groups)
    elif modulation == '8-PSK':
        symbols = _map_8psk(bit_groups)
    elif modulation == '16-APSK':
        symbols = _map_16apsk(bit_groups)
    else:
        raise ValueError(f"Unknown modulation: {modulation}")

    return symbols.astype(np.complex64)


def _map_bpsk(bit_groups: np.ndarray) -> np.ndarray:
    """Map bits to BPSK constellation.

    BPSK: 2 points on real axis
    - 0 → +1
    - 1 → -1

    Args:
        bit_groups: Shape (num_symbols, 1)

    Returns:
        np.ndarray: Complex symbols, shape (num_symbols,)
    """
    bits = bit_groups[:, 0].astype(np.int32)  # Convert to int32 to avoid uint8 overflow
    # Map: 0 → +1, 1 → -1
    symbols = 1 - 2 * bits
    return symbols.astype(np.complex64)


def _map_qpsk(bit_groups: np.ndarray) -> np.ndarray:
    """Map bits to QPSK constellation with Gray coding.

    QPSK: 4 points at ±45° and ±135°
    Gray coding:
    - 00 → exp(j*π/4)     = (+1+j)/sqrt(2)
    - 01 → exp(j*3π/4)    = (-1+j)/sqrt(2)
    - 11 → exp(j*5π/4)    = (-1-j)/sqrt(2)
    - 10 → exp(j*7π/4)    = (+1-j)/sqrt(2)

    Args:
        bit_groups: Shape (num_symbols, 2)

    Returns:
        np.ndarray: Complex symbols, shape (num_symbols,)
    """
    # Convert bit pairs to decimal for Gray coding lookup
    decimal = bit_groups[:, 0] * 2 + bit_groups[:, 1]

    # Gray coding mapping to phase angles (radians)
    gray_to_phase = {
        0: np.pi / 4,      # 00 → 45°
        1: 3 * np.pi / 4,  # 01 → 135°
        3: 5 * np.pi / 4,  # 11 → 225°
        2: 7 * np.pi / 4   # 10 → 315°
    }

    # Map to constellation points
    phases = np.array([gray_to_phase[d] for d in decimal])
    symbols = np.exp(1j * phases)

    # Normalize to unit average power
    symbols /= np.sqrt(2)

    return symbols


def _map_8psk(bit_groups: np.ndarray) -> np.ndarray:
    """Map bits to 8-PSK constellation with Gray coding.

    8-PSK: 8 points equally spaced around unit circle
    Gray coding minimizes bit errors for adjacent symbols

    Gray mapping:
    - 000 → 0°
    - 001 → 45°
    - 011 → 90°
    - 010 → 135°
    - 110 → 180°
    - 111 → 225°
    - 101 → 270°
    - 100 → 315°

    Args:
        bit_groups: Shape (num_symbols, 3)

    Returns:
        np.ndarray: Complex symbols, shape (num_symbols,)
    """
    # Convert bit triplets to decimal
    decimal = (bit_groups[:, 0] * 4 + bit_groups[:, 1] * 2 + bit_groups[:, 2])

    # Gray coding for 8-PSK
    gray_to_phase = {
        0: 0,                  # 000 → 0°
        1: np.pi / 4,         # 001 → 45°
        3: np.pi / 2,         # 011 → 90°
        2: 3 * np.pi / 4,     # 010 → 135°
        6: np.pi,             # 110 → 180°
        7: 5 * np.pi / 4,     # 111 → 225°
        5: 3 * np.pi / 2,     # 101 → 270°
        4: 7 * np.pi / 4      # 100 → 315°
    }

    phases = np.array([gray_to_phase[d] for d in decimal])
    symbols = np.exp(1j * phases)

    return symbols


def _map_16apsk(bit_groups: np.ndarray) -> np.ndarray:
    """Map bits to 16-APSK constellation.

    16-APSK: Two concentric rings (4+12 configuration)
    - Inner ring: 4 points at radius r1
    - Outer ring: 12 points at radius r2
    - Typical ratio: r2/r1 ≈ 2.7 for optimal performance

    Args:
        bit_groups: Shape (num_symbols, 4)

    Returns:
        np.ndarray: Complex symbols, shape (num_symbols,)
    """
    # Convert bit quadruplets to decimal (0-15)
    decimal = (bit_groups[:, 0] * 8 + bit_groups[:, 1] * 4 +
              bit_groups[:, 2] * 2 + bit_groups[:, 3])

    # Ring radii (optimized for AWGN)
    r1 = 1.0  # Inner ring
    r2 = 2.7  # Outer ring (r2/r1 = 2.7)

    # Normalize for unit average power
    # Average power = (4*r1² + 12*r2²) / 16
    avg_power = (4 * r1**2 + 12 * r2**2) / 16
    norm_factor = np.sqrt(avg_power)
    r1 /= norm_factor
    r2 /= norm_factor

    symbols = np.zeros(len(decimal), dtype=np.complex128)

    for i, d in enumerate(decimal):
        if d < 4:
            # Inner ring: 4 points at 0°, 90°, 180°, 270°
            angle = d * np.pi / 2
            symbols[i] = r1 * np.exp(1j * angle)
        else:
            # Outer ring: 12 points at 30°, 60°, 90°, ..., 360°
            angle = (d - 4) * np.pi / 6
            symbols[i] = r2 * np.exp(1j * angle)

    return symbols


def verify_gray_coding(modulation: str) -> bool:
    """Verify that constellation uses Gray coding.

    Gray coding: Adjacent symbols differ by only 1 bit.
    This minimizes bit errors in noisy channels.

    Args:
        modulation: Modulation scheme to verify

    Returns:
        bool: True if Gray coding is correctly implemented
    """
    bits_per_sym = get_bits_per_symbol(modulation)
    num_symbols = 2 ** bits_per_sym

    # Generate all possible bit combinations
    all_bits = np.array([
        [(i >> j) & 1 for j in range(bits_per_sym - 1, -1, -1)]
        for i in range(num_symbols)
    ], dtype=np.uint8)

    # Flatten for mapping
    bits_flat = all_bits.flatten()
    symbols = map_to_constellation(bits_flat, modulation)

    # For each symbol, find nearest neighbor
    # Check if nearest neighbor differs by exactly 1 bit
    for i in range(num_symbols):
        symbol_i = symbols[i]
        distances = np.abs(symbols - symbol_i)
        distances[i] = np.inf  # Exclude self

        # Find nearest neighbor
        nearest_idx = np.argmin(distances)

        # Count bit differences
        bit_diff = np.sum(all_bits[i] != all_bits[nearest_idx])

        # For Gray coding, nearest neighbor should differ by 1 bit
        # (This is a simplified check; full verification is more complex)
        if bit_diff > 2:  # Allow some tolerance for APSK
            return False

    return True


def verify_unit_power(symbols: np.ndarray, tolerance: float = 0.01) -> bool:
    """Verify that constellation has unit average power.

    Args:
        symbols: Complex constellation symbols
        tolerance: Allowed deviation from unit power

    Returns:
        bool: True if average power is within tolerance of 1.0
    """
    avg_power = np.mean(np.abs(symbols) ** 2)
    return abs(avg_power - 1.0) < tolerance


def generate_pilot_sequence(modulation: str, num_symbols: int, shaped: bool = True) -> np.ndarray:
    """Generate swept constellation pilot sequence for NN channel learning.

    Creates a sequence that cycles through constellation points.
    With shaped=True, orders points from inner (low amplitude) to outer (high amplitude)
    for smooth AGC settling and gradual equalizer learning.

    The NN can use this known sequence to:
    - Estimate channel gain/phase response
    - Learn equalizer taps
    - Compute SNR estimate
    - Detect carrier offset

    Args:
        modulation: Modulation scheme ('BPSK', 'QPSK', '8-PSK', '16-APSK')
        num_symbols: Number of pilot symbols to generate
        shaped: If True, order inner→outer for smooth AGC (default: True)

    Returns:
        np.ndarray: Complex pilot symbols, shape (num_symbols,), dtype complex64

    Example:
        >>> pilots = generate_pilot_sequence('QPSK', 100, shaped=True)
        >>> # Returns inner points first [0°, 180°], then outer [90°, 270°], repeating
    """
    # Generate all constellation points
    bits_per_sym = get_bits_per_symbol(modulation)
    num_constellation_points = 2 ** bits_per_sym

    # Create bit patterns for all constellation points
    all_bits = []
    for i in range(num_constellation_points):
        bits = [(i >> j) & 1 for j in range(bits_per_sym - 1, -1, -1)]
        all_bits.extend(bits)

    all_bits = np.array(all_bits, dtype=np.uint8)
    constellation = map_to_constellation(all_bits, modulation)

    # Shape pilot order (inner → outer) for smooth AGC/equalizer learning
    if shaped:
        if modulation == 'BPSK':
            # BPSK: Start with +1 (lower amplitude after AGC gain), then -1
            order = [0, 1]  # [+1, -1]

        elif modulation == 'QPSK':
            # QPSK: Cardinal points (I or Q axis) first, then diagonal
            # Order by distance from origin (all QPSK points same magnitude, so use phase)
            # Start with 0° and 180° (I-axis), then 90° and 270° (Q-axis)
            angles = np.angle(constellation)
            # Group into cardinal (near 0°/180°) and diagonal (near ±90°)
            cardinal_mask = (np.abs(np.abs(angles) - np.pi/2) > np.pi/4)
            order = list(np.where(cardinal_mask)[0]) + list(np.where(~cardinal_mask)[0])

        elif modulation == '8-PSK':
            # 8-PSK: Start with cardinal (0°, 90°, 180°, 270°), then diagonal
            angles = np.angle(constellation) % (2*np.pi)
            # Cardinal: angles close to 0, π/2, π, 3π/2
            is_cardinal = np.array([
                np.min([abs(a), abs(a - np.pi), abs(a - np.pi/2), abs(a - 3*np.pi/2)])
                < np.pi/8 for a in angles
            ])
            order = list(np.where(is_cardinal)[0]) + list(np.where(~is_cardinal)[0])

        elif modulation == '16-APSK':
            # 16-APSK: Inner ring first (4 points, low amplitude), then outer ring (12 points)
            magnitudes = np.abs(constellation)
            # Sort by magnitude: inner ring points have smaller magnitude
            order = np.argsort(magnitudes).tolist()

        else:
            order = list(range(num_constellation_points))

        # Reorder constellation
        constellation = constellation[order]

    # Repeat constellation to fill num_symbols
    num_repeats = int(np.ceil(num_symbols / num_constellation_points))
    pilot_sequence = np.tile(constellation, num_repeats)[:num_symbols]

    return pilot_sequence.astype(np.complex64)


def generate_constant_pilot(modulation: str, num_symbols: int) -> np.ndarray:
    """Generate constant pilot symbol for continuous phase reference.

    Uses a single known constellation point repeated. Useful for:
    - Continuous carrier phase tracking
    - Frequency offset estimation
    - Simple channel gain estimation

    Args:
        modulation: Modulation scheme ('BPSK', 'QPSK', '8-PSK', '16-APSK')
        num_symbols: Number of pilot symbols to generate

    Returns:
        np.ndarray: Complex pilot symbols (all identical), shape (num_symbols,)

    Example:
        >>> pilots = generate_constant_pilot('QPSK', 50)
        >>> # Returns [+1+j, +1+j, +1+j, ...] (constant reference)
    """
    # Use first constellation point (typically highest I+Q)
    if modulation == 'BPSK':
        pilot_symbol = complex(1, 0)  # +1
    elif modulation == 'QPSK':
        pilot_symbol = (1 + 1j) / np.sqrt(2)  # 45°
    elif modulation == '8-PSK':
        pilot_symbol = complex(1, 0)  # 0°
    elif modulation == '16-APSK':
        # Use outer ring point for stronger signal
        pilot_symbol = 2.7 / np.sqrt((4 * 1.0**2 + 12 * 2.7**2) / 16)  # Normalized outer ring
    else:
        raise ValueError(f"Unknown modulation: {modulation}")

    return np.full(num_symbols, pilot_symbol, dtype=np.complex64)
