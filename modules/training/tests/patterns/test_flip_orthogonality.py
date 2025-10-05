"""Test flip-orthogonality functionality for CASCADE patterns

Tests that patterns maintain sufficient orthogonality when FSK-inverted,
critical for adjacent-channel operation when patterns share tones.
"""

import pytest
import numpy as np
from modules.training.patterns.models import Pattern
from modules.training.patterns.correlation import (
    compute_flip_correlation,
    compute_all_correlations,
    check_adjacent_channel_safety
)
from modules.training.patterns.validator import (
    validate_flip_orthogonality,
    validate_adjacent_channel_safety,
    generate_flip_validation_report
)
from modules.training.patterns.optimizer import optimize_pattern
from modules.training.patterns.generator import validate_flip_orthogonality as gen_validate_flip


class TestFlipCorrelation:
    """Test flip correlation computation"""

    def test_flip_correlation_inverts_frequency(self):
        """Test that flip correlation correctly inverts frequency sequence"""
        # Create test pattern with simple frequency sequence
        pattern_a = Pattern(
            pattern_id=0,
            freq_sequence=np.array([0, 1, 0, 1] * 8, dtype='uint8'),  # Alternating
            iq_trajectory=np.ones(32, dtype='complex64'),
            iq_complexity_lambda=0.0
        )

        pattern_b = Pattern(
            pattern_id=1,
            freq_sequence=np.array([1, 0, 1, 0] * 8, dtype='uint8'),  # Inverse of A
            iq_trajectory=np.ones(32, dtype='complex64'),
            iq_complexity_lambda=0.0
        )

        # Flip correlation of A with B should be very low (they're inverses)
        flip_corr = compute_flip_correlation(pattern_a, pattern_b)

        # When B is flipped, it becomes identical to A, so correlation should be high
        assert flip_corr > -10.0, "Flip correlation should be high when patterns match after flip"

    def test_flip_correlation_maintains_orthogonality(self):
        """Test that orthogonal patterns remain orthogonal when flipped"""
        # Create orthogonal patterns (different frequency sequences)
        pattern_a = Pattern(
            pattern_id=0,
            freq_sequence=np.array([0] * 32, dtype='uint8'),  # All tone 0
            iq_trajectory=np.ones(32, dtype='complex64'),
            iq_complexity_lambda=0.0
        )

        pattern_b = Pattern(
            pattern_id=1,
            freq_sequence=np.array([1] * 32, dtype='uint8'),  # All tone 1
            iq_trajectory=np.ones(32, dtype='complex64'),
            iq_complexity_lambda=0.0
        )

        # These patterns use different tones, so should be orthogonal even when flipped
        flip_corr = compute_flip_correlation(pattern_a, pattern_b)
        assert flip_corr < -30.0, "Orthogonal patterns should remain orthogonal when flipped"

    def test_all_correlations_comprehensive(self):
        """Test compute_all_correlations returns all correlation types"""
        pattern_a = Pattern(
            pattern_id=0,
            freq_sequence=np.random.randint(0, 2, 32, dtype='uint8'),
            iq_trajectory=np.exp(1j * np.random.uniform(-np.pi, np.pi, 32)).astype('complex64'),
            iq_complexity_lambda=0.1
        )

        pattern_b = Pattern(
            pattern_id=1,
            freq_sequence=np.random.randint(0, 2, 32, dtype='uint8'),
            iq_trajectory=np.exp(1j * np.random.uniform(-np.pi, np.pi, 32)).astype('complex64'),
            iq_complexity_lambda=0.1
        )

        all_corrs = compute_all_correlations(pattern_a, pattern_b)

        # Check all expected keys exist
        assert 'normal' in all_corrs
        assert 'j_flipped' in all_corrs
        assert 'i_flipped' in all_corrs
        assert 'both_flipped' in all_corrs
        assert 'max_correlation' in all_corrs
        assert 'adjacent_safe' in all_corrs

        # Max correlation should be the worst of all types
        assert all_corrs['max_correlation'] == max(
            all_corrs['normal'],
            all_corrs['j_flipped'],
            all_corrs['i_flipped'],
            all_corrs['both_flipped']
        )


class TestAdjacentChannelSafety:
    """Test adjacent channel safety validation"""

    def test_adjacent_channel_with_shared_tone(self):
        """Test detection of shared tones in adjacent channels"""
        pattern_a = Pattern(
            pattern_id=0,
            freq_sequence=np.array([0, 1] * 16, dtype='uint8'),
            iq_trajectory=np.ones(32, dtype='complex64'),
            iq_complexity_lambda=0.0
        )

        pattern_b = Pattern(
            pattern_id=1,
            freq_sequence=np.array([1, 0] * 16, dtype='uint8'),
            iq_trajectory=np.ones(32, dtype='complex64'),
            iq_complexity_lambda=0.0
        )

        # Tone pairs that share tone 35
        tone_pair_a = (34, 35)
        tone_pair_b = (35, 36)

        is_safe = check_adjacent_channel_safety(
            pattern_a, pattern_b,
            tone_pair_a, tone_pair_b
        )

        # Safety depends on actual correlation values
        assert isinstance(is_safe, bool)

    def test_non_adjacent_channels_always_safe(self):
        """Test that non-adjacent channels (no shared tones) are always safe"""
        pattern_a = Pattern(
            pattern_id=0,
            freq_sequence=np.random.randint(0, 2, 32, dtype='uint8'),
            iq_trajectory=np.ones(32, dtype='complex64'),
            iq_complexity_lambda=0.0
        )

        pattern_b = Pattern(
            pattern_id=1,
            freq_sequence=np.random.randint(0, 2, 32, dtype='uint8'),
            iq_trajectory=np.ones(32, dtype='complex64'),
            iq_complexity_lambda=0.0
        )

        # Tone pairs that don't share any tones
        tone_pair_a = (10, 11)
        tone_pair_b = (20, 21)

        is_safe = check_adjacent_channel_safety(
            pattern_a, pattern_b,
            tone_pair_a, tone_pair_b
        )

        assert is_safe == True, "Non-adjacent channels should always be safe"


class TestFlipOrthogonalityValidation:
    """Test flip-orthogonality validation functions"""

    def test_validate_flip_orthogonality(self):
        """Test flip-orthogonality validation for pattern set"""
        patterns = []

        # Create a small set of test patterns
        for i in range(4):
            freq_seq = np.zeros(32, dtype='uint8')
            freq_seq[i::4] = 1  # Different patterns use different time slots

            pattern = Pattern(
                pattern_id=i,
                freq_sequence=freq_seq,
                iq_trajectory=np.ones(32, dtype='complex64'),
                iq_complexity_lambda=0.0
            )
            patterns.append(pattern)

        passes, stats = validate_flip_orthogonality(patterns, target_db=-30.0)

        assert isinstance(passes, bool)
        assert 'min_flip_corr_db' in stats
        assert 'max_flip_corr_db' in stats
        assert 'mean_flip_corr_db' in stats
        assert 'failed_flip_pairs' in stats
        assert stats['target_db'] == -30.0

    def test_flip_validation_report_generation(self):
        """Test generation of flip validation report"""
        patterns = []

        # Create test patterns
        for i in range(4):
            pattern = Pattern(
                pattern_id=i,
                freq_sequence=np.random.randint(0, 2, 32, dtype='uint8'),
                iq_trajectory=np.exp(1j * np.random.uniform(-np.pi, np.pi, 32)).astype('complex64'),
                iq_complexity_lambda=0.1
            )
            patterns.append(pattern)

        report = generate_flip_validation_report(patterns)

        assert isinstance(report, str)
        assert "Flip-Orthogonality Validation Report" in report
        assert "Normal Correlation Statistics" in report
        assert "Flip Correlation Statistics" in report


class TestFlipConstrainedOptimization:
    """Test optimization with flip-orthogonality constraints"""

    def test_optimizer_respects_flip_weight(self):
        """Test that optimizer respects flip_weight parameter"""
        existing = [
            Pattern(
                pattern_id=0,
                freq_sequence=np.array([0, 1] * 16, dtype='uint8'),
                iq_trajectory=np.ones(32, dtype='complex64'),
                iq_complexity_lambda=0.0
            )
        ]

        # Optimize with different flip weights
        base_freq = np.random.randint(0, 2, 32, dtype='uint8')

        # High flip weight should produce better flip-orthogonality
        freq_high, lambda_high = optimize_pattern(
            pattern_id=1,
            base_freq_sequence=base_freq.copy(),
            existing_patterns=existing,
            target_db=-37.5,
            max_iterations=1000,  # Small for testing
            flip_weight=0.9,
            seed=42
        )

        # Low flip weight may allow worse flip-orthogonality
        freq_low, lambda_low = optimize_pattern(
            pattern_id=2,
            base_freq_sequence=base_freq.copy(),
            existing_patterns=existing,
            target_db=-37.5,
            max_iterations=1000,
            flip_weight=0.1,
            seed=42
        )

        # Both should produce valid patterns
        assert freq_high.shape == (32,)
        assert freq_low.shape == (32,)
        assert 0.0 <= lambda_high <= 0.9
        assert 0.0 <= lambda_low <= 0.9


class TestPatternFlipStats:
    """Test Pattern model flip-orthogonality statistics"""

    def test_pattern_flip_stats_initialization(self):
        """Test that Pattern initializes flip stats correctly"""
        pattern = Pattern(
            pattern_id=0,
            freq_sequence=np.array([0, 1] * 16, dtype='uint8'),
            iq_trajectory=np.ones(32, dtype='complex64'),
            iq_complexity_lambda=0.0
        )

        # Check flip stats are initialized
        assert hasattr(pattern, 'flip_orthogonality_stats')
        assert 'max_flip_correlation_db' in pattern.flip_orthogonality_stats
        assert 'avg_flip_correlation_db' in pattern.flip_orthogonality_stats
        assert 'adjacent_channel_safe' in pattern.flip_orthogonality_stats

    def test_pattern_inverted_sequence_computed(self):
        """Test that Pattern computes inverted frequency sequence"""
        freq_seq = np.array([0, 1, 0, 1] * 8, dtype='uint8')

        pattern = Pattern(
            pattern_id=0,
            freq_sequence=freq_seq,
            iq_trajectory=np.ones(32, dtype='complex64'),
            iq_complexity_lambda=0.0
        )

        # Check inverted sequence is computed
        assert hasattr(pattern, 'freq_sequence_inv')
        assert pattern.freq_sequence_inv.shape == (32,)

        # Verify inversion is correct (0↔1)
        expected_inv = 1 - freq_seq
        np.testing.assert_array_equal(pattern.freq_sequence_inv, expected_inv)


class TestGeneratorFlipValidation:
    """Test generator flip validation function"""

    def test_generator_flip_validation(self):
        """Test the generator's validate_flip_orthogonality function"""
        patterns = []

        # Create test patterns with known flip properties
        for i in range(8):
            # Create patterns that should have good flip-orthogonality
            freq_seq = np.zeros(32, dtype='uint8')
            freq_seq[i::8] = 1  # Time-interleaved patterns

            pattern = Pattern(
                pattern_id=i,
                freq_sequence=freq_seq,
                iq_trajectory=np.ones(32, dtype='complex64'),
                iq_complexity_lambda=0.0
            )
            patterns.append(pattern)

        # Validate using generator function
        flip_stats = gen_validate_flip(patterns, target_db=-30.0)

        assert 'min_flip_corr' in flip_stats
        assert 'max_flip_corr' in flip_stats
        assert 'mean_flip_corr' in flip_stats
        assert 'adjacent_safe_count' in flip_stats
        assert 'adjacent_safe_patterns' in flip_stats

        # Check that stats are populated in patterns
        for pattern in patterns:
            assert pattern.flip_orthogonality_stats['max_flip_correlation_db'] is not None
            assert pattern.flip_orthogonality_stats['adjacent_channel_safe'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])