"""4D Cross-Correlation Calculation (Time × Freq × IQ)"""

import numpy as np
from .models import Pattern


def compute_4d_correlation(pattern_i: Pattern, pattern_j: Pattern) -> float:
    """Compute 4D cross-correlation in Time × Freq × IQ space

    The correlation is computed across all three dimensions:
    - Time: 32 symbols
    - Frequency: Tone indices (0-3 from 78-tone grid)
    - IQ: Complex constellation points

    Patterns are orthogonal in frequency if they use different tones at the same
    time symbol. If they use the same tone, we check IQ orthogonality.

    Args:
        pattern_i: First pattern
        pattern_j: Second pattern

    Returns:
        Correlation in dB (must be < -37.5 dB for orthogonality)
    """
    correlation = 0.0

    # Loop over 32 time symbols
    for t in range(32):
        tone_i = pattern_i.freq_sequence[t]
        tone_j = pattern_j.freq_sequence[t]

        # If different tones at this time instant, they're orthogonal in frequency
        if tone_i != tone_j:
            continue  # No contribution to correlation

        # Same tone → check IQ orthogonality
        iq_i = pattern_i.iq_trajectory[t]
        iq_j = pattern_j.iq_trajectory[t]

        # Complex inner product (magnitude of dot product)
        correlation += np.abs(iq_i * np.conj(iq_j))

    # Normalize by number of symbols
    correlation_normalized = correlation / 32.0

    # Convert to dB
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    correlation_db = 20 * np.log10(correlation_normalized + epsilon)

    return float(correlation_db)


def compute_correlation_with_phase(
    pattern_i: Pattern,
    pattern_j: Pattern,
    phase_per_tone: np.ndarray,
    phase_per_symbol: np.ndarray = None
) -> float:
    """Compute correlation with phase distortion applied

    Models HF propagation effects:
    - Frequency-dependent phase rotation (different phase per tone)
    - Time-varying phase (optional, different phase per symbol)

    Args:
        pattern_i: First pattern
        pattern_j: Second pattern
        phase_per_tone: Phase rotation for each tone index (4 values, radians)
        phase_per_symbol: Optional per-symbol phase variation (32 values)

    Returns:
        Correlation in dB with phase distortion applied
    """
    correlation = 0.0

    for t in range(32):
        tone_i = pattern_i.freq_sequence[t]
        tone_j = pattern_j.freq_sequence[t]

        if tone_i != tone_j:
            continue  # Frequency orthogonal

        # Apply phase distortion
        phase_distortion = phase_per_tone[tone_i]
        if phase_per_symbol is not None:
            phase_distortion += phase_per_symbol[t]

        # Apply phase rotation to both patterns
        iq_i_distorted = pattern_i.iq_trajectory[t] * np.exp(1j * phase_distortion)
        iq_j_distorted = pattern_j.iq_trajectory[t] * np.exp(1j * phase_distortion)

        # Compute inner product
        correlation += np.abs(iq_i_distorted * np.conj(iq_j_distorted))

    correlation_normalized = correlation / 32.0
    epsilon = 1e-10
    correlation_db = 20 * np.log10(correlation_normalized + epsilon)

    return float(correlation_db)


def compute_robust_correlation(
    pattern_i: Pattern,
    pattern_j: Pattern,
    num_phase_trials: int = 100
) -> float:
    """Compute worst-case correlation under random phase distortion

    Tests correlation across multiple random phase scenarios to find
    worst-case (highest correlation).

    Args:
        pattern_i: First pattern
        pattern_j: Second pattern
        num_phase_trials: Number of random phase scenarios to test

    Returns:
        Worst-case correlation in dB (highest across all trials)
    """
    worst_case_corr = -float('inf')

    for trial in range(num_phase_trials):
        # Random phase per tone (models frequency-dependent distortion)
        phase_per_tone = np.random.uniform(-np.pi, np.pi, size=4)

        # Random phase per symbol (models time-varying channel)
        phase_per_symbol = np.random.uniform(-0.2, 0.2, size=32)

        corr_db = compute_correlation_with_phase(
            pattern_i,
            pattern_j,
            phase_per_tone,
            phase_per_symbol
        )

        worst_case_corr = max(worst_case_corr, corr_db)

    return worst_case_corr
