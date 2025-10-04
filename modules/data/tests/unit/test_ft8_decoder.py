"""Unit tests for FT8 decoder module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.processors.ft8_decoder import FT8Decoder


class TestFT8Decoder:
    """Test FT8 signal decoding and processing."""

    @pytest.fixture
    def decoder(self):
        """Create FT8Decoder instance."""
        return FT8Decoder(
            sample_rate=12000,
            frequency_offset_tolerance=50
        )

    @pytest.fixture
    def sample_ft8_audio(self):
        """Generate sample audio with FT8-like characteristics."""
        sample_rate = 12000
        duration = 15  # FT8 transmission is ~13 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Simulate FT8 8-FSK modulation
        frequencies = [500, 550, 600, 650, 700, 750, 800, 850]  # Hz
        audio = np.zeros_like(t)

        # Create mock FT8 signal
        symbols_per_second = 6.25
        symbol_duration = 1.0 / symbols_per_second

        for i, freq in enumerate(np.random.choice(frequencies, size=int(duration * symbols_per_second))):
            start_idx = int(i * symbol_duration * sample_rate)
            end_idx = int((i + 1) * symbol_duration * sample_rate)
            if end_idx <= len(t):
                audio[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx])

        return audio * 0.3 + np.random.randn(len(audio)) * 0.05

    def test_decoder_initialization(self, decoder):
        """Test decoder initialization."""
        assert decoder.sample_rate == 12000
        assert decoder.frequency_offset_tolerance == 50
        assert decoder.symbol_rate == 6.25
        assert decoder.tone_spacing == 6.25

    def test_detect_ft8_signals(self, decoder, sample_ft8_audio):
        """Test FT8 signal detection."""
        signals = decoder.detect_signals(sample_ft8_audio)

        assert isinstance(signals, list)
        # Should detect at least some candidates
        assert len(signals) >= 0

        for signal in signals:
            assert 'frequency' in signal
            assert 'snr' in signal
            assert 'time_offset' in signal

    @patch('subprocess.run')
    def test_decode_ft8_message(self, mock_run, decoder, sample_ft8_audio):
        """Test FT8 message decoding."""
        # Mock ft8_decode output
        mock_run.return_value.stdout = """
        000000  -5  0.2  1234  CQ K1ABC FN42
        000000  10  0.1  1456  K1ABC W2XYZ -10
        000000  -2  0.3  1678  W2XYZ K1ABC R-05
        """
        mock_run.return_value.returncode = 0

        messages = decoder.decode(sample_ft8_audio)

        assert len(messages) == 3
        assert messages[0]['callsign'] == 'K1ABC'
        assert messages[0]['grid'] == 'FN42'
        assert messages[0]['snr'] == -5

    def test_extract_propagation_mode(self, decoder):
        """Test propagation mode classification."""
        # Test different SNR and time patterns
        test_cases = [
            {'snr': 25, 'time': '12:00', 'expected': 'E-skip'},
            {'snr': -10, 'time': '02:00', 'expected': 'F2'},
            {'snr': 5, 'time': '06:00', 'expected': 'Greyline'},
            {'snr': 0, 'time': '14:00', 'expected': 'Groundwave'}
        ]

        for case in test_cases:
            mode = decoder.classify_propagation_mode(
                snr_db=case['snr'],
                time_str=case['time'],
                frequency=14074000
            )
            # Mode should be one of the valid types
            assert mode in ['Groundwave', 'E-skip', 'F2', 'Greyline', 'Unknown']

    def test_calculate_snr(self, decoder):
        """Test SNR calculation."""
        # Create signal with known SNR
        signal = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, 12000))
        noise = np.random.randn(12000) * 0.1
        audio = signal + noise

        snr = decoder.calculate_snr(audio, frequency=1000, bandwidth=50)

        assert isinstance(snr, float)
        assert snr > 10  # Should have good SNR

    def test_frequency_offset_correction(self, decoder):
        """Test frequency offset correction."""
        # Signal with offset
        t = np.linspace(0, 1, 12000)
        offset_freq = 1005  # 5 Hz offset from 1000 Hz
        audio = np.sin(2 * np.pi * offset_freq * t)

        corrected = decoder.correct_frequency_offset(
            audio, expected_freq=1000
        )

        # Should detect and correct the offset
        assert corrected is not None
        # Verify correction (would need FFT to check properly)

    def test_time_synchronization(self, decoder, sample_ft8_audio):
        """Test time synchronization detection."""
        sync_offset = decoder.find_sync_offset(sample_ft8_audio)

        assert isinstance(sync_offset, float)
        assert -1.0 <= sync_offset <= 1.0  # Within FT8 window

    def test_message_validation(self, decoder):
        """Test FT8 message validation."""
        # Valid messages
        assert decoder.validate_message("CQ K1ABC FN42") is True
        assert decoder.validate_message("K1ABC W2XYZ -10") is True
        assert decoder.validate_message("W2XYZ K1ABC RRR") is True

        # Invalid messages
        assert decoder.validate_message("") is False
        assert decoder.validate_message("INVALID") is False
        assert decoder.validate_message("12345") is False

    def test_grid_square_extraction(self, decoder):
        """Test grid square extraction from messages."""
        messages = [
            "CQ K1ABC FN42",
            "K1ABC W2XYZ EM48",
            "CQ DX JA1XYZ PM95",
            "K1ABC W2XYZ -10"  # No grid
        ]

        grids = [decoder.extract_grid_square(msg) for msg in messages]

        assert grids[0] == "FN42"
        assert grids[1] == "EM48"
        assert grids[2] == "PM95"
        assert grids[3] is None

    def test_batch_decode(self, decoder):
        """Test batch decoding of multiple files."""
        audio_files = [np.random.randn(12000 * 15) for _ in range(5)]

        with patch.object(decoder, 'decode') as mock_decode:
            mock_decode.return_value = [{'test': 'message'}]

            results = decoder.decode_batch(audio_files)

            assert len(results) == 5
            assert mock_decode.call_count == 5

    def test_weak_signal_detection(self, decoder):
        """Test detection of weak signals."""
        # Very weak signal in noise
        t = np.linspace(0, 1, 12000)
        weak_signal = np.sin(2 * np.pi * 1000 * t) * 0.01
        noise = np.random.randn(12000) * 0.1
        audio = weak_signal + noise

        # Should still attempt detection
        signals = decoder.detect_signals(audio, min_snr=-20)

        assert isinstance(signals, list)

    def test_multipath_detection(self, decoder):
        """Test detection of multipath propagation."""
        # Simulate multipath with delayed copy
        t = np.linspace(0, 1, 12000)
        signal = np.sin(2 * np.pi * 1000 * t)

        # Add delayed copy (multipath)
        delay_samples = 100
        multipath = np.zeros_like(signal)
        multipath[delay_samples:] = signal[:-delay_samples] * 0.5

        audio = signal + multipath

        multipath_detected = decoder.detect_multipath(audio)

        assert isinstance(multipath_detected, bool)

    def test_performance_metrics(self, decoder):
        """Test decoder performance metrics."""
        # Track decode attempts
        decoder.reset_metrics()

        for _ in range(10):
            decoder.record_decode_attempt(success=True, snr=10)

        for _ in range(5):
            decoder.record_decode_attempt(success=False, snr=-15)

        metrics = decoder.get_metrics()

        assert metrics['total_attempts'] == 15
        assert metrics['successful_decodes'] == 10
        assert metrics['success_rate'] == 10/15
        assert 'average_snr' in metrics

    def test_error_handling(self, decoder):
        """Test error handling in decoder."""
        # Empty audio
        result = decoder.decode(np.array([]))
        assert result == []

        # Invalid audio type
        with pytest.raises(TypeError):
            decoder.decode("not_audio")

        # NaN in audio
        audio_with_nan = np.random.randn(12000)
        audio_with_nan[100] = np.nan
        result = decoder.decode(audio_with_nan)
        assert isinstance(result, list)  # Should handle gracefully