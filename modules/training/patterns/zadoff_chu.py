"""Zadoff-Chu Sequence Generation"""

import numpy as np


def generate_zadoff_chu_pattern(u: int, N: int = 31) -> np.ndarray:
    """Generate Zadoff-Chu sequence with root index u

    Zadoff-Chu sequences provide excellent auto-correlation and cross-correlation
    properties, making them ideal as base patterns for orthogonal signal design.

    Formula: phase = 2π × u × n(n+1) / (2N)
    The phase is then mapped to tone indices 0-3

    Args:
        u: Root index (0 to N-1)
        N: Sequence length (default 31 for 32 symbols including guard)

    Returns:
        32-element uint8 array with tone indices 0-3
    """
    # Generate Zadoff-Chu sequence phase values
    n = np.arange(N)
    phase = 2 * np.pi * u * n * (n + 1) / (2 * N)

    # Map phase (0 to 2π) to tone indices (0 to 3)
    # Divide phase space into 4 equal regions
    tone_indices = ((phase / (2 * np.pi)) * 4).astype('uint8') % 4

    # Add guard symbol at end (tone 0)
    freq_sequence = np.zeros(32, dtype='uint8')
    freq_sequence[:N] = tone_indices
    freq_sequence[N:] = 0  # Guard symbol

    return freq_sequence


def generate_random_pattern(seed: int) -> np.ndarray:
    """Generate random pattern for non-Zadoff-Chu patterns

    Args:
        seed: Random seed for reproducibility

    Returns:
        32-element uint8 array with tone indices 0-3
    """
    rng = np.random.RandomState(seed)
    freq_sequence = rng.randint(0, 4, size=32, dtype='uint8')
    return freq_sequence
