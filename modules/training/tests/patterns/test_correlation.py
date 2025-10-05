"""Contract tests for 4D correlation calculation"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/workspaces/cascade')

from modules.training.patterns import compute_4d_correlation, Pattern


def test_correlation_returns_float_db():
    """T006: Verify returns float in dB"""
    # Create two simple test patterns
    pattern_i = Pattern(
        pattern_id=0,
        freq_sequence=np.zeros(32, dtype='uint8'),  # All tone 0
        iq_trajectory=np.ones(32, dtype='complex64'),  # All 1+0j
        iq_complexity_lambda=0.0
    )
    pattern_j = Pattern(
        pattern_id=1,
        freq_sequence=np.zeros(32, dtype='uint8'),  # All tone 0
        iq_trajectory=np.ones(32, dtype='complex64'),  # All 1+0j
        iq_complexity_lambda=0.0
    )

    correlation_db = compute_4d_correlation(pattern_i, pattern_j)

    # Verify returns float
    assert isinstance(correlation_db, (float, np.floating))
    # Verify in reasonable dB range
    assert -100 <= correlation_db <= 10


def test_identical_patterns_high_correlation():
    """T006: Test with identical patterns → expect ~0 dB"""
    # Create identical patterns (2-FSK)
    freq_seq = np.random.randint(0, 2, size=32, dtype='uint8')
    iq_traj = np.random.randn(32).astype('complex64')

    pattern_i = Pattern(
        pattern_id=0,
        freq_sequence=freq_seq.copy(),
        iq_trajectory=iq_traj.copy(),
        iq_complexity_lambda=0.0
    )
    pattern_j = Pattern(
        pattern_id=1,
        freq_sequence=freq_seq.copy(),
        iq_trajectory=iq_traj.copy(),
        iq_complexity_lambda=0.0
    )

    correlation_db = compute_4d_correlation(pattern_i, pattern_j)

    # Identical patterns should have high correlation (close to 0 dB)
    assert correlation_db > -3.0, f"Identical patterns should correlate highly, got {correlation_db} dB"


def test_orthogonal_patterns_low_correlation():
    """T006: Test with orthogonal patterns → expect <-30 dB"""
    # Create patterns with different tones (frequency-orthogonal)
    pattern_i = Pattern(
        pattern_id=0,
        freq_sequence=np.zeros(32, dtype='uint8'),  # All tone 0
        iq_trajectory=np.ones(32, dtype='complex64'),
        iq_complexity_lambda=0.0
    )
    pattern_j = Pattern(
        pattern_id=1,
        freq_sequence=np.ones(32, dtype='uint8'),  # All tone 1 (different!)
        iq_trajectory=np.ones(32, dtype='complex64'),
        iq_complexity_lambda=0.0
    )

    correlation_db = compute_4d_correlation(pattern_i, pattern_j)

    # Different tones should be orthogonal (<-30 dB)
    assert correlation_db < -30.0, f"Orthogonal patterns should have low correlation, got {correlation_db} dB"


def test_time_freq_iq_dimensions_checked():
    """T006: Verify Time × Freq × IQ dimensions are all checked"""
    # Pattern with mixed tones
    freq_seq_i = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 4, dtype='uint8')  # Mix of 0 and 1
    freq_seq_j = np.array([0, 0, 0, 0, 2, 2, 2, 2] * 4, dtype='uint8')  # Mix of 0 and 2

    # IQ trajectories
    iq_i = np.exp(1j * np.linspace(0, 2*np.pi, 32)).astype('complex64')
    iq_j = np.exp(1j * np.linspace(0, 2*np.pi, 32) + np.pi/2).astype('complex64')  # 90° phase shift

    pattern_i = Pattern(
        pattern_id=0,
        freq_sequence=freq_seq_i,
        iq_trajectory=iq_i,
        iq_complexity_lambda=0.0
    )
    pattern_j = Pattern(
        pattern_id=1,
        freq_sequence=freq_seq_j,
        iq_trajectory=iq_j,
        iq_complexity_lambda=0.0
    )

    correlation_db = compute_4d_correlation(pattern_i, pattern_j)

    # Should only correlate where tones match (first 4 symbols)
    # But IQ is 90° shifted, so should still be low
    assert isinstance(correlation_db, (float, np.floating))
    assert correlation_db < 0  # Some correlation from matching tones, but not perfect
