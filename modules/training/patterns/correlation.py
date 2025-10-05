"""4D Cross-Correlation Calculation (Time × Freq × IQ)

Includes flip-orthogonality support for adjacent-channel robustness.
"""

import numpy as np
from typing import Tuple, Dict
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


def compute_flip_correlation(pattern_i: Pattern, pattern_j: Pattern) -> float:
    """Compute correlation with pattern_j's frequency sequence inverted (FSK flip)

    This checks orthogonality when pattern_j experiences FSK inversion,
    which can happen due to phase inversions in multipath or adjacent-channel
    interference when patterns share tones.

    Args:
        pattern_i: First pattern (normal)
        pattern_j: Second pattern (will be FSK-inverted)

    Returns:
        Correlation in dB with pattern_j FSK-inverted
    """
    # Create a temporary pattern with inverted frequency sequence
    # We can use pattern_j's pre-computed inverse if available
    if hasattr(pattern_j, 'freq_sequence_inv'):
        freq_inv = pattern_j.freq_sequence_inv
    else:
        freq_inv = 1 - pattern_j.freq_sequence  # Invert 0↔1 for 2-FSK

    # Create temporary inverted pattern
    pattern_j_inv = Pattern(
        pattern_id=pattern_j.pattern_id,
        freq_sequence=freq_inv,
        iq_trajectory=pattern_j.iq_trajectory,
        iq_complexity_lambda=pattern_j.iq_complexity_lambda
    )

    # Compute correlation with inverted pattern
    return compute_4d_correlation(pattern_i, pattern_j_inv)


def compute_all_correlations(pattern_i: Pattern, pattern_j: Pattern) -> Dict[str, float]:
    """Compute all correlation metrics between two patterns

    Computes:
    - Normal correlation
    - Flip correlation (pattern_j inverted)
    - Reverse flip correlation (pattern_i inverted)
    - Mutual flip (both inverted)

    Args:
        pattern_i: First pattern
        pattern_j: Second pattern

    Returns:
        Dictionary with all correlation values in dB
    """
    results = {}

    # Normal correlation
    results['normal'] = compute_4d_correlation(pattern_i, pattern_j)

    # Pattern j flipped
    results['j_flipped'] = compute_flip_correlation(pattern_i, pattern_j)

    # Pattern i flipped
    results['i_flipped'] = compute_flip_correlation(pattern_j, pattern_i)

    # Both flipped (useful for symmetric validation)
    pattern_i_inv = Pattern(
        pattern_id=pattern_i.pattern_id,
        freq_sequence=1 - pattern_i.freq_sequence,
        iq_trajectory=pattern_i.iq_trajectory,
        iq_complexity_lambda=pattern_i.iq_complexity_lambda
    )
    pattern_j_inv = Pattern(
        pattern_id=pattern_j.pattern_id,
        freq_sequence=1 - pattern_j.freq_sequence,
        iq_trajectory=pattern_j.iq_trajectory,
        iq_complexity_lambda=pattern_j.iq_complexity_lambda
    )
    results['both_flipped'] = compute_4d_correlation(pattern_i_inv, pattern_j_inv)

    # Maximum (worst-case) correlation
    results['max_correlation'] = max(results.values())

    # Check if adjacent-channel safe (all correlations < -30 dB)
    results['adjacent_safe'] = all(corr < -30.0 for corr in results.values())

    return results


def check_adjacent_channel_safety(pattern_i: Pattern, pattern_j: Pattern,
                                   tone_pair_i: Tuple[int, int],
                                   tone_pair_j: Tuple[int, int]) -> bool:
    """Check if two patterns are safe to use on adjacent tone pairs

    Args:
        pattern_i: First pattern
        pattern_j: Second pattern
        tone_pair_i: Tone indices used by pattern_i (e.g., (34, 35))
        tone_pair_j: Tone indices used by pattern_j (e.g., (35, 36))

    Returns:
        True if patterns maintain sufficient orthogonality even when sharing a tone
    """
    # Check if tone pairs are adjacent (share a tone)
    shared_tones = set(tone_pair_i) & set(tone_pair_j)

    if not shared_tones:
        # No shared tones, automatically safe
        return True

    # If they share tones, check all correlations
    correlations = compute_all_correlations(pattern_i, pattern_j)

    # For adjacent channels with shared tones, require stricter orthogonality
    # Normal correlation should still be < -37.5 dB
    # Flip correlations should be < -30 dB
    return (correlations['normal'] < -37.5 and
            correlations['j_flipped'] < -30.0 and
            correlations['i_flipped'] < -30.0)
