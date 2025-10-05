"""Zadoff-Chu Sequence Generation"""

import numpy as np


def generate_zadoff_chu_pattern(u: int, N: int = 31) -> np.ndarray:
    """Generate Zadoff-Chu sequence with root index u

    Zadoff-Chu sequences provide excellent auto-correlation and cross-correlation
    properties, making them ideal as base patterns for orthogonal signal design.

    Formula: phase = 2π × u × n(n+1) / (2N)
    The phase is then mapped to tone indices 0-1 (2-FSK architecture)

    Args:
        u: Root index (0 to N-1)
        N: Sequence length (if 31, adds guard symbol for 32 total)

    Returns:
        N-element (or 32 if N=31) uint8 array with tone indices 0-1 (2-FSK)
    """
    # Generate Zadoff-Chu sequence phase values
    n = np.arange(N)
    phase = 2 * np.pi * u * n * (n + 1) / (2 * N)

    # Map phase (0 to 2π) to tone indices (0 to 1) for 2-FSK
    # Divide phase space into 2 equal regions
    tone_indices = ((phase / (2 * np.pi)) * 2).astype('uint8') % 2

    # Special case: if N=31, add guard symbol for 32 total
    if N == 31:
        freq_sequence = np.zeros(32, dtype='uint8')
        freq_sequence[:N] = tone_indices
        freq_sequence[N:] = 0  # Guard symbol
    else:
        # Otherwise return exact length requested
        freq_sequence = tone_indices

    return freq_sequence


def generate_random_pattern(seed: int) -> np.ndarray:
    """Generate random pattern for non-Zadoff-Chu patterns

    Args:
        seed: Random seed for reproducibility

    Returns:
        32-element uint8 array with tone indices 0-1 (2-FSK)
    """
    rng = np.random.RandomState(seed)
    freq_sequence = rng.randint(0, 2, size=32, dtype='uint8')
    return freq_sequence
