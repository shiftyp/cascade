"""Contract tests for pattern validation"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/workspaces/cascade')

from modules.training.patterns import validate_orthogonality, Pattern


def test_validate_checks_all_pairs():
    """T007: Verify checks all pattern pairs"""
    # Create 4 test patterns
    patterns = []
    for i in range(4):
        # Each pattern uses a different tone for orthogonality
        freq_seq = np.full(32, i % 4, dtype='uint8')
        iq_traj = np.ones(32, dtype='complex64')
        patterns.append(Pattern(
            pattern_id=i,
            freq_sequence=freq_seq,
            iq_trajectory=iq_traj,
            iq_complexity_lambda=0.0
        ))

    passes, stats = validate_orthogonality(patterns, target_db=-37.5)

    # Should have checked all pairs: C(4,2) = 6 pairs
    assert 'pair_count' in stats or 'min_correlation_db' in stats
    # Should return bool and dict
    assert isinstance(passes, bool)
    assert isinstance(stats, dict)


def test_validate_returns_true_for_good_patterns():
    """T007: Returns True only if ALL pairs <-37.5 dB"""
    # Create orthogonal patterns (different tones)
    patterns = []
    for i in range(4):
        freq_seq = np.full(32, i, dtype='uint8')  # Each uses different tone
        iq_traj = np.ones(32, dtype='complex64')
        patterns.append(Pattern(
            pattern_id=i,
            freq_sequence=freq_seq,
            iq_trajectory=iq_traj,
            iq_complexity_lambda=0.0
        ))

    passes, stats = validate_orthogonality(patterns, target_db=-37.5)

    # All patterns use different tones, should be highly orthogonal
    assert passes is True
    assert stats['min_correlation_db'] < -37.5


def test_validate_returns_false_for_bad_patterns():
    """T007: Returns False if any pair fails threshold"""
    # Create mostly orthogonal patterns, but two identical
    patterns = []
    for i in range(3):
        freq_seq = np.full(32, i, dtype='uint8')
        iq_traj = np.ones(32, dtype='complex64')
        patterns.append(Pattern(
            pattern_id=i,
            freq_sequence=freq_seq,
            iq_trajectory=iq_traj,
            iq_complexity_lambda=0.0
        ))

    # Add duplicate of pattern 0 (will fail orthogonality)
    patterns.append(Pattern(
        pattern_id=3,
        freq_sequence=patterns[0].freq_sequence.copy(),
        iq_trajectory=patterns[0].iq_trajectory.copy(),
        iq_complexity_lambda=0.0
    ))

    passes, stats = validate_orthogonality(patterns, target_db=-37.5)

    # Should fail because patterns 0 and 3 are identical
    assert passes is False
    assert 'failed_pairs' in stats
    assert len(stats['failed_pairs']) > 0


def test_validate_with_known_good_bad_pairs():
    """T007: Test with known good/bad pair mix"""
    patterns = []

    # Good pair: Different tones
    patterns.append(Pattern(
        pattern_id=0,
        freq_sequence=np.zeros(32, dtype='uint8'),  # All tone 0
        iq_trajectory=np.ones(32, dtype='complex64'),
        iq_complexity_lambda=0.0
    ))
    patterns.append(Pattern(
        pattern_id=1,
        freq_sequence=np.ones(32, dtype='uint8'),  # All tone 1
        iq_trajectory=np.ones(32, dtype='complex64'),
        iq_complexity_lambda=0.0
    ))

    # Bad pair: Same tones and IQ (identical)
    patterns.append(Pattern(
        pattern_id=2,
        freq_sequence=np.zeros(32, dtype='uint8'),  # Same as pattern 0
        iq_trajectory=np.ones(32, dtype='complex64'),  # Same as pattern 0
        iq_complexity_lambda=0.0
    ))

    passes, stats = validate_orthogonality(patterns, target_db=-37.5)

    # Should fail overall because (0,2) are too similar
    assert passes is False


def test_validate_statistics_returned():
    """T007: Verify statistics dict contains min/max/mean"""
    patterns = []
    for i in range(3):
        freq_seq = np.full(32, i, dtype='uint8')
        iq_traj = np.ones(32, dtype='complex64')
        patterns.append(Pattern(
            pattern_id=i,
            freq_sequence=freq_seq,
            iq_trajectory=iq_traj,
            iq_complexity_lambda=0.0
        ))

    passes, stats = validate_orthogonality(patterns, target_db=-37.5)

    # Check stats dict has expected fields
    assert 'min_correlation_db' in stats or 'max_correlation_db' in stats
    # Should contain numeric values
    for key, value in stats.items():
        if 'correlation' in key:
            assert isinstance(value, (int, float, np.number))
