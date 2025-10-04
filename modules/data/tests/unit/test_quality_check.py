"""Unit tests for quality validation module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from src.validators.quality_check import QualityCheck


class TestQualityCheck:
    """Test quality validation for recordings and samples."""

    @pytest.fixture
    def validator(self):
        """Create QualityCheck instance."""
        return QualityCheck(
            min_snr_db=10,
            max_clipping_percent=5,
            min_duration_seconds=60
        )

    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data."""
        # Generate 1 second of audio at 12 kHz
        sample_rate = 12000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Signal + noise
        signal = np.sin(2 * np.pi * 1000 * t) * 0.5
        noise = np.random.randn(len(t)) * 0.1
        return signal + noise

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.min_snr_db == 10
        assert validator.max_clipping_percent == 5
        assert validator.min_duration_seconds == 60

    def test_calculate_snr(self, validator, sample_audio):
        """Test SNR calculation."""
        snr = validator.calculate_snr(sample_audio)

        assert snr is not None
        assert isinstance(snr, float)
        assert snr > 0  # Should have positive SNR

    def test_detect_clipping(self, validator):
        """Test clipping detection."""
        # Create clipped audio
        audio = np.random.randn(1000)
        audio[100:150] = 0.99  # Clip high
        audio[200:220] = -0.99  # Clip low

        clipping_percent = validator.detect_clipping(audio)

        assert clipping_percent > 0
        assert clipping_percent < 100

    def test_validate_sample_quality(self, validator, sample_audio):
        """Test complete sample quality validation."""
        result = validator.validate_sample(
            audio_data=sample_audio,
            sample_rate=12000,
            expected_duration=1.0
        )

        assert 'is_valid' in result
        assert 'quality_score' in result
        assert 'snr_db' in result
        assert 'clipping_percent' in result
        assert 'duration_seconds' in result
        assert 'issues' in result

    def test_quality_score_calculation(self, validator, sample_audio):
        """Test quality score calculation."""
        score = validator.calculate_quality_score(
            snr_db=20,
            clipping_percent=1,
            duration_match=0.95
        )

        assert 0 <= score <= 100
        assert isinstance(score, float)

    def test_validate_recording_session(self, validator):
        """Test recording session validation."""
        session_data = {
            'id': 'session_123',
            'start_time': datetime.now(),
            'duration_seconds': 3600,
            'file_size_bytes': 1024 * 1024 * 50,  # 50 MB
            'sample_rate': 12000,
            'frequency_band': '20m'
        }

        result = validator.validate_session(session_data)

        assert 'session_valid' in result
        assert 'duration_valid' in result
        assert 'size_valid' in result
        assert 'expected_size_mb' in result

    def test_detect_dc_offset(self, validator):
        """Test DC offset detection."""
        # Audio with DC offset
        audio = np.random.randn(1000) + 0.1  # 0.1 DC offset

        dc_offset = validator.detect_dc_offset(audio)

        assert dc_offset is not None
        assert abs(dc_offset - 0.1) < 0.01  # Close to actual offset

    def test_detect_frequency_spurs(self, validator):
        """Test detection of frequency spurs."""
        sample_rate = 12000
        t = np.linspace(0, 1, sample_rate)

        # Add strong spur at 2 kHz
        signal = np.sin(2 * np.pi * 1000 * t) * 0.1
        spur = np.sin(2 * np.pi * 2000 * t) * 0.5
        audio = signal + spur

        spurs = validator.detect_spurs(audio, sample_rate)

        assert len(spurs) > 0
        assert any(abs(f - 2000) < 50 for f in spurs)  # Detect 2 kHz spur

    def test_validate_spectral_occupancy(self, validator):
        """Test spectral occupancy validation."""
        sample_rate = 12000
        audio = np.random.randn(sample_rate)

        occupancy = validator.calculate_spectral_occupancy(
            audio, sample_rate, band_edges=(300, 3000)
        )

        assert 0 <= occupancy <= 100
        assert isinstance(occupancy, float)

    def test_validate_quiet_periods(self, validator):
        """Test quiet period detection."""
        # Create audio with quiet periods
        audio = np.random.randn(10000) * 0.5
        audio[2000:3000] = np.random.randn(1000) * 0.001  # Quiet
        audio[6000:7000] = np.random.randn(1000) * 0.001  # Quiet

        quiet_periods = validator.detect_quiet_periods(
            audio, threshold_db=-40
        )

        assert len(quiet_periods) >= 2
        assert all('start' in p and 'end' in p for p in quiet_periods)

    def test_validate_propagation_quality(self, validator):
        """Test propagation signal quality validation."""
        ft8_signal = {
            'snr': 15,
            'frequency_offset': 5,
            'time_offset': 0.2,
            'decoded': True
        }

        result = validator.validate_propagation_signal(ft8_signal)

        assert 'valid' in result
        assert 'quality_class' in result
        assert result['quality_class'] in ['excellent', 'good', 'fair', 'poor']

    def test_batch_validation(self, validator):
        """Test batch validation of multiple samples."""
        samples = [np.random.randn(1000) for _ in range(10)]

        results = validator.validate_batch(samples)

        assert len(results) == 10
        assert all('is_valid' in r for r in results)
        assert all('quality_score' in r for r in results)

    def test_quarantine_decision(self, validator):
        """Test quarantine decision based on quality."""
        # Poor quality sample
        poor_result = {
            'quality_score': 25,
            'snr_db': 5,
            'clipping_percent': 15
        }

        assert validator.should_quarantine(poor_result) is True

        # Good quality sample
        good_result = {
            'quality_score': 85,
            'snr_db': 20,
            'clipping_percent': 0.5
        }

        assert validator.should_quarantine(good_result) is False

    def test_quality_statistics(self, validator):
        """Test quality statistics aggregation."""
        results = [
            {'quality_score': 80, 'snr_db': 20},
            {'quality_score': 75, 'snr_db': 18},
            {'quality_score': 60, 'snr_db': 12},
            {'quality_score': 90, 'snr_db': 25}
        ]

        stats = validator.aggregate_statistics(results)

        assert 'mean_quality' in stats
        assert 'std_quality' in stats
        assert 'mean_snr' in stats
        assert 'pass_rate' in stats

    def test_adaptive_thresholds(self, validator):
        """Test adaptive threshold adjustment."""
        # Simulate changing conditions
        validator.update_thresholds(
            recent_snr_values=[15, 18, 12, 20, 16]
        )

        # Thresholds should adapt
        assert validator.min_snr_db != 10  # Changed from initial

    def test_error_handling(self, validator):
        """Test error handling for invalid inputs."""
        # Empty audio
        with pytest.raises(ValueError):
            validator.validate_sample(np.array([]), 12000, 1.0)

        # Invalid sample rate
        with pytest.raises(ValueError):
            validator.validate_sample(np.random.randn(100), 0, 1.0)

        # NaN values
        audio_with_nan = np.random.randn(100)
        audio_with_nan[50] = np.nan
        result = validator.validate_sample(audio_with_nan, 12000, 0.01)
        assert result['is_valid'] is False
        assert 'NaN values' in result['issues'][0]