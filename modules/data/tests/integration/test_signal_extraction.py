"""
Integration tests for FT8/WSPR signal extraction and processing.

Tests T015: End-to-end signal extraction, decoding, anonymization, and propagation analysis.
This test suite follows TDD principles and will initially fail until implementations are complete.

Tests:
1. FT8 signal detection and decoding from IQ data
2. WSPR signal detection and decoding
3. Callsign anonymization
4. Propagation mutation extraction
5. Real-time vs batch processing
6. Multiple signal handling
7. Weak signal detection
8. False positive filtering
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta, timezone
import uuid
import tempfile
import os
import json
from pathlib import Path

# These imports will fail initially (TDD approach)
try:
    from cascade_collector.processors.ft8_decoder import FT8Decoder, FT8Message
    from cascade_collector.processors.wspr_decoder import WSPRDecoder, WSPRMessage
    from cascade_collector.processors.anonymizer import CallsignAnonymizer
    from cascade_collector.models.propagation_record import PropagationRecord
    from cascade_collector.models.recording_session import RecordingSession
    from cascade_collector.storage.file_manager import FileManager
except ImportError:
    pytest.skip("Processors not yet implemented (TDD)", allow_module_level=True)


class TestFT8SignalExtraction:
    """Test FT8 signal detection and decoding from IQ data"""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = Mock()
        session.query = Mock()
        return session

    @pytest.fixture
    def ft8_decoder(self, mock_db_session):
        """Initialize FT8 decoder"""
        return FT8Decoder(db_session=mock_db_session)

    @pytest.fixture
    def sample_ft8_iq_data(self):
        """Generate synthetic FT8 IQ data with embedded signal"""
        sample_rate = 12000
        duration = 15  # 15 second FT8 message
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create baseband noise
        noise = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.01

        # Add FT8-like signal at 1500 Hz offset
        signal_freq = 1500
        ft8_signal = 0.05 * np.exp(2j * np.pi * signal_freq * t)

        # Add frequency modulation (FSK8)
        phase_mod = np.cumsum(np.random.choice([-3, -1, 1, 3], size=len(t)))
        ft8_signal *= np.exp(1j * phase_mod * 0.01)

        iq_data = noise + ft8_signal
        return iq_data.astype(np.complex64)

    @pytest.mark.asyncio
    async def test_ft8_signal_detection(self, ft8_decoder, sample_ft8_iq_data):
        """Test basic FT8 signal detection from IQ data"""
        # Detect FT8 signals in IQ stream
        signals = await ft8_decoder.detect_signals(
            iq_data=sample_ft8_iq_data,
            sample_rate=12000,
            center_frequency=14074000
        )

        # Should detect at least one FT8 signal
        assert len(signals) > 0, "Failed to detect FT8 signal"
        assert signals[0].frequency_hz > 14073000
        assert signals[0].frequency_hz < 14075000

    @pytest.mark.asyncio
    async def test_ft8_decoding_with_snr(self, ft8_decoder, sample_ft8_iq_data):
        """Test FT8 decoding with SNR calculation"""
        messages = await ft8_decoder.decode(
            iq_data=sample_ft8_iq_data,
            sample_rate=12000,
            timestamp=datetime.now(timezone.utc)
        )

        # Verify message structure
        for msg in messages:
            assert isinstance(msg, FT8Message)
            assert msg.snr_db is not None
            assert -30 <= msg.snr_db <= 50, f"Invalid SNR: {msg.snr_db}"
            assert msg.frequency_hz > 0
            assert msg.dt_seconds is not None

    @pytest.mark.asyncio
    async def test_ft8_cq_message_parsing(self, ft8_decoder):
        """Test parsing of FT8 CQ messages"""
        # Simulate decoded FT8 message text
        test_messages = [
            "CQ K1ABC FN42",
            "CQ DX W2XYZ EM00",
            "CQ TEST VE3ZZZ FN03"
        ]

        for msg_text in test_messages:
            parsed = ft8_decoder.parse_message(msg_text)

            assert parsed.callsign is not None
            assert parsed.grid_square is not None
            assert len(parsed.grid_square) == 4
            assert parsed.message_type == "CQ"

    @pytest.mark.asyncio
    async def test_ft8_qso_extraction(self, ft8_decoder):
        """Test extraction of FT8 QSO exchanges"""
        # Simulate a complete QSO sequence
        qso_messages = [
            "K1ABC W2XYZ -12",
            "W2XYZ K1ABC R-08",
            "K1ABC W2XYZ RRR",
            "W2XYZ K1ABC 73"
        ]

        qsos = []
        for msg_text in qso_messages:
            parsed = ft8_decoder.parse_message(msg_text)
            qsos.append(parsed)

        # Verify QSO structure
        assert len(qsos) == 4
        assert qsos[0].callsign == "K1ABC"
        assert qsos[0].dx_callsign == "W2XYZ"
        assert qsos[1].snr_report == -8

    @pytest.mark.asyncio
    async def test_weak_signal_detection(self, ft8_decoder):
        """Test detection of weak FT8 signals (SNR < -10 dB)"""
        # Create very weak signal
        sample_rate = 12000
        duration = 15
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Strong noise
        noise = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.1

        # Very weak signal (SNR ~ -15 dB)
        weak_signal = 0.005 * np.exp(2j * np.pi * 1500 * t)
        iq_data = noise + weak_signal

        signals = await ft8_decoder.detect_signals(
            iq_data=iq_data,
            sample_rate=sample_rate,
            center_frequency=14074000,
            min_snr_db=-20
        )

        # Should still detect weak signal
        assert len(signals) > 0, "Failed to detect weak signal"
        assert signals[0].snr_db < -10

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_signals(self, ft8_decoder):
        """Test detection of multiple simultaneous FT8 signals"""
        sample_rate = 12000
        duration = 15
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create multiple signals at different frequencies
        noise = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.01
        signal1 = 0.03 * np.exp(2j * np.pi * 1000 * t)
        signal2 = 0.04 * np.exp(2j * np.pi * 1500 * t)
        signal3 = 0.02 * np.exp(2j * np.pi * 2000 * t)

        iq_data = noise + signal1 + signal2 + signal3

        signals = await ft8_decoder.detect_signals(
            iq_data=iq_data,
            sample_rate=sample_rate,
            center_frequency=14074000
        )

        # Should detect multiple signals
        assert len(signals) >= 3, f"Expected 3+ signals, found {len(signals)}"

        # Verify frequency separation
        freqs = sorted([s.frequency_hz for s in signals])
        for i in range(len(freqs) - 1):
            assert freqs[i+1] - freqs[i] >= 400, "Signals too close in frequency"


class TestWSPRSignalExtraction:
    """Test WSPR signal detection and decoding"""

    @pytest.fixture
    def wspr_decoder(self, mock_db_session):
        """Initialize WSPR decoder"""
        return WSPRDecoder(db_session=mock_db_session)

    @pytest.fixture
    def sample_wspr_iq_data(self):
        """Generate synthetic WSPR IQ data"""
        sample_rate = 12000
        duration = 110  # WSPR message is 110.6 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create noise
        noise = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.01

        # Add WSPR-like signal (4-FSK)
        signal_freq = 1500
        wspr_signal = 0.03 * np.exp(2j * np.pi * signal_freq * t)

        # Add slow FSK modulation (1.46 baud)
        symbol_rate = 1.46
        symbols_per_sample = symbol_rate / sample_rate
        phase_mod = np.cumsum(np.random.choice([0, 1, 2, 3], size=len(t)))
        wspr_signal *= np.exp(1j * phase_mod * 0.005)

        iq_data = noise + wspr_signal
        return iq_data.astype(np.complex64)

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = Mock()
        return session

    @pytest.mark.asyncio
    async def test_wspr_signal_detection(self, wspr_decoder, sample_wspr_iq_data):
        """Test basic WSPR signal detection"""
        signals = await wspr_decoder.detect_signals(
            iq_data=sample_wspr_iq_data,
            sample_rate=12000,
            center_frequency=14097100
        )

        assert len(signals) > 0, "Failed to detect WSPR signal"
        assert signals[0].mode == "WSPR"

    @pytest.mark.asyncio
    async def test_wspr_decoding_with_power(self, wspr_decoder, sample_wspr_iq_data):
        """Test WSPR decoding including power field"""
        messages = await wspr_decoder.decode(
            iq_data=sample_wspr_iq_data,
            sample_rate=12000,
            timestamp=datetime.now(timezone.utc)
        )

        for msg in messages:
            assert isinstance(msg, WSPRMessage)
            assert msg.callsign is not None
            assert msg.grid_square is not None
            assert msg.power_dbm is not None
            assert 0 <= msg.power_dbm <= 60, f"Invalid power: {msg.power_dbm}"

    @pytest.mark.asyncio
    async def test_wspr_timing_synchronization(self, wspr_decoder):
        """Test WSPR timing synchronization to even 2-minute marks"""
        # WSPR transmissions occur on even 2-minute UTC marks
        test_times = [
            datetime(2025, 9, 29, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 9, 29, 12, 2, 0, tzinfo=timezone.utc),
            datetime(2025, 9, 29, 12, 4, 0, tzinfo=timezone.utc),
        ]

        for test_time in test_times:
            is_valid = wspr_decoder.is_wspr_transmission_time(test_time)
            assert is_valid, f"{test_time} should be valid WSPR time"

        # Test invalid times
        invalid_time = datetime(2025, 9, 29, 12, 1, 0, tzinfo=timezone.utc)
        assert not wspr_decoder.is_wspr_transmission_time(invalid_time)

    @pytest.mark.asyncio
    async def test_wspr_frequency_drift_measurement(self, wspr_decoder, sample_wspr_iq_data):
        """Test measurement of WSPR frequency drift"""
        messages = await wspr_decoder.decode(
            iq_data=sample_wspr_iq_data,
            sample_rate=12000,
            timestamp=datetime.now(timezone.utc),
            measure_drift=True
        )

        for msg in messages:
            assert hasattr(msg, 'drift_hz')
            assert msg.drift_hz is not None
            assert abs(msg.drift_hz) < 5.0, f"Excessive drift: {msg.drift_hz} Hz"


class TestCallsignAnonymization:
    """Test callsign anonymization for privacy"""

    @pytest.fixture
    def anonymizer(self):
        """Initialize anonymizer with test salt"""
        return CallsignAnonymizer(salt="test_salt_12345")

    def test_callsign_hashing_deterministic(self, anonymizer):
        """Test that callsign hashing is deterministic"""
        callsign = "K1ABC"

        hash1 = anonymizer.anonymize_callsign(callsign)
        hash2 = anonymizer.anonymize_callsign(callsign)

        assert hash1 == hash2, "Hash should be deterministic"
        assert hash1 != callsign, "Hash should not equal original"
        assert len(hash1) == 64, "SHA256 hash should be 64 hex chars"

    def test_different_callsigns_different_hashes(self, anonymizer):
        """Test that different callsigns produce different hashes"""
        callsigns = ["K1ABC", "W2XYZ", "VE3ZZZ", "G4DEF"]

        hashes = [anonymizer.anonymize_callsign(cs) for cs in callsigns]

        assert len(set(hashes)) == len(callsigns), "All hashes should be unique"

    def test_anonymize_ft8_message(self, anonymizer):
        """Test anonymization of complete FT8 message"""
        message = FT8Message(
            timestamp=datetime.now(timezone.utc),
            frequency_hz=14074500,
            snr_db=-10,
            dt_seconds=0.2,
            message="CQ K1ABC FN42",
            callsign_hash=None,
            grid_square="FN42"
        )

        anon_message = anonymizer.anonymize_message(message)

        assert anon_message.callsign_hash is not None
        assert len(anon_message.callsign_hash) == 64
        assert "K1ABC" not in str(anon_message)

    def test_anonymize_wspr_message(self, anonymizer):
        """Test anonymization of WSPR message"""
        message = WSPRMessage(
            timestamp=datetime.now(timezone.utc),
            frequency_hz=14097100,
            snr_db=-20,
            callsign="W2XYZ",
            grid_square="EM00",
            power_dbm=30
        )

        anon_message = anonymizer.anonymize_message(message)

        assert anon_message.callsign_hash is not None
        assert "W2XYZ" not in str(anon_message)

    def test_grid_square_truncation(self, anonymizer):
        """Test that grid squares are truncated to 4 characters"""
        # 6-character grid
        full_grid = "FN42aa"
        truncated = anonymizer.truncate_grid_square(full_grid)

        assert len(truncated) == 4
        assert truncated == "FN42"

    def test_anonymization_prevents_correlation(self, anonymizer):
        """Test that anonymization prevents cross-session correlation"""
        # Same callsign in different sessions should use same hash
        # but different anonymizers (salts) should produce different hashes

        callsign = "K1ABC"
        hash1 = anonymizer.anonymize_callsign(callsign)

        # Different anonymizer (different deployment)
        anonymizer2 = CallsignAnonymizer(salt="different_salt_67890")
        hash2 = anonymizer2.anonymize_callsign(callsign)

        assert hash1 != hash2, "Different salts should produce different hashes"


class TestPropagationMutationExtraction:
    """Test extraction of propagation mutations from signals"""

    @pytest.fixture
    def ft8_decoder(self, mock_db_session):
        """Initialize FT8 decoder"""
        return FT8Decoder(db_session=mock_db_session)

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = Mock()
        return session

    @pytest.mark.asyncio
    async def test_frequency_spread_extraction(self, ft8_decoder):
        """Test extraction of frequency spread per symbol"""
        # Create signal with frequency spread
        sample_rate = 12000
        duration = 15
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Add frequency modulation to simulate ionospheric effects
        freq_spread = 2.0  # Hz
        base_freq = 1500
        freq_mod = base_freq + freq_spread * np.sin(2 * np.pi * 0.5 * t)

        iq_data = 0.05 * np.exp(2j * np.pi * freq_mod * t)

        mutations = await ft8_decoder.extract_propagation_mutations(
            iq_data=iq_data,
            sample_rate=sample_rate,
            signal_frequency=1500
        )

        assert 'frequency_spread_hz' in mutations
        assert isinstance(mutations['frequency_spread_hz'], list)
        assert len(mutations['frequency_spread_hz']) > 0

    @pytest.mark.asyncio
    async def test_amplitude_fading_extraction(self, ft8_decoder):
        """Test extraction of amplitude fading patterns"""
        sample_rate = 12000
        duration = 15
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Add amplitude fading
        fading = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t)
        iq_data = fading * 0.05 * np.exp(2j * np.pi * 1500 * t)

        mutations = await ft8_decoder.extract_propagation_mutations(
            iq_data=iq_data,
            sample_rate=sample_rate,
            signal_frequency=1500
        )

        assert 'amplitude_fading' in mutations
        assert isinstance(mutations['amplitude_fading'], list)

        # Verify fading was detected
        fading_array = np.array(mutations['amplitude_fading'])
        assert fading_array.max() > 0.8
        assert fading_array.min() < 0.3

    @pytest.mark.asyncio
    async def test_multipath_delay_detection(self, ft8_decoder):
        """Test detection of multipath propagation delays"""
        sample_rate = 12000
        duration = 15
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Direct path
        direct = 0.05 * np.exp(2j * np.pi * 1500 * t)

        # Delayed path (5ms delay)
        delay_samples = int(0.005 * sample_rate)
        delayed = np.zeros_like(direct)
        delayed[delay_samples:] = 0.02 * direct[:-delay_samples]

        iq_data = direct + delayed

        mutations = await ft8_decoder.extract_propagation_mutations(
            iq_data=iq_data,
            sample_rate=sample_rate,
            signal_frequency=1500
        )

        assert 'multipath_delays_ms' in mutations
        assert len(mutations['multipath_delays_ms']) > 0
        # Should detect ~5ms delay
        assert any(4 < delay < 6 for delay in mutations['multipath_delays_ms'])

    @pytest.mark.asyncio
    async def test_doppler_spread_measurement(self, ft8_decoder):
        """Test measurement of Doppler spread"""
        sample_rate = 12000
        duration = 15
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Add Doppler spread
        doppler_shift = 1.0 * np.sin(2 * np.pi * 0.1 * t)
        freq = 1500 + doppler_shift
        iq_data = 0.05 * np.exp(2j * np.pi * freq * t)

        mutations = await ft8_decoder.extract_propagation_mutations(
            iq_data=iq_data,
            sample_rate=sample_rate,
            signal_frequency=1500
        )

        assert 'doppler_spread_hz' in mutations
        assert 0 < mutations['doppler_spread_hz'] < 5.0

    @pytest.mark.asyncio
    async def test_propagation_mode_classification(self, ft8_decoder):
        """Test classification of propagation modes (F2, Es, MS, etc.)"""
        # Create signals with different characteristics
        test_cases = [
            {
                'snr_db': -5,
                'distance_km': 2500,
                'time_of_day': 'day',
                'frequency_hz': 14074000,
                'expected_mode': 'F2'
            },
            {
                'snr_db': 10,
                'distance_km': 800,
                'time_of_day': 'day',
                'frequency_hz': 28074000,
                'expected_mode': 'Es'
            },
            {
                'snr_db': -15,
                'distance_km': 1200,
                'time_of_day': 'night',
                'frequency_hz': 50313000,
                'expected_mode': 'MS'
            }
        ]

        for case in test_cases:
            mode = await ft8_decoder.classify_propagation_mode(
                snr_db=case['snr_db'],
                distance_km=case['distance_km'],
                frequency_hz=case['frequency_hz'],
                timestamp=datetime.now(timezone.utc)
            )

            assert mode in ['F2', 'Es', 'MS', 'TEP', 'Aurora', 'Unknown']


class TestRealtimeVsBatchProcessing:
    """Test real-time vs batch signal processing"""

    @pytest.fixture
    def ft8_decoder(self, mock_db_session):
        """Initialize FT8 decoder"""
        return FT8Decoder(db_session=mock_db_session)

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = Mock()
        return session

    @pytest.mark.asyncio
    async def test_realtime_streaming_extraction(self, ft8_decoder):
        """Test real-time signal extraction from streaming data"""
        sample_rate = 12000
        chunk_size = 1024

        # Simulate streaming chunks
        async def iq_stream():
            for _ in range(100):
                chunk = (np.random.randn(chunk_size) +
                        1j * np.random.randn(chunk_size)) * 0.01
                yield chunk
                await asyncio.sleep(0.01)  # Simulate real-time

        signals = []
        async for chunk in iq_stream():
            chunk_signals = await ft8_decoder.process_streaming_chunk(
                iq_chunk=chunk,
                sample_rate=sample_rate
            )
            signals.extend(chunk_signals)

        # Should accumulate signals over time
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_batch_file_processing(self, ft8_decoder):
        """Test batch processing of recorded IQ file"""
        # Create temporary IQ file
        sample_rate = 12000
        duration = 60
        samples = int(sample_rate * duration)
        iq_data = (np.random.randn(samples) + 1j * np.random.randn(samples)) * 0.01

        with tempfile.NamedTemporaryFile(suffix='.iq', delete=False) as tmp_file:
            try:
                iq_data.tofile(tmp_file.name)

                # Process entire file
                signals = await ft8_decoder.process_file(
                    file_path=tmp_file.name,
                    sample_rate=sample_rate,
                    center_frequency=14074000
                )

                assert isinstance(signals, list)
            finally:
                os.unlink(tmp_file.name)

    @pytest.mark.asyncio
    async def test_realtime_latency(self, ft8_decoder):
        """Test that real-time processing has acceptable latency"""
        sample_rate = 12000
        chunk_size = 2048
        iq_chunk = (np.random.randn(chunk_size) +
                   1j * np.random.randn(chunk_size)) * 0.01

        import time
        start_time = time.time()

        signals = await ft8_decoder.process_streaming_chunk(
            iq_chunk=iq_chunk,
            sample_rate=sample_rate
        )

        latency = time.time() - start_time

        # Processing should be faster than data acquisition
        data_duration = chunk_size / sample_rate
        assert latency < data_duration, f"Processing too slow: {latency}s for {data_duration}s data"

    @pytest.mark.asyncio
    async def test_batch_vs_realtime_consistency(self, ft8_decoder):
        """Test that batch and real-time processing produce consistent results"""
        sample_rate = 12000
        duration = 30
        iq_data = (np.random.randn(int(sample_rate * duration)) +
                  1j * np.random.randn(int(sample_rate * duration))) * 0.01

        # Batch processing
        batch_signals = await ft8_decoder.decode(
            iq_data=iq_data,
            sample_rate=sample_rate,
            timestamp=datetime.now(timezone.utc)
        )

        # Real-time processing (same data in chunks)
        realtime_signals = []
        chunk_size = 2048
        for i in range(0, len(iq_data), chunk_size):
            chunk = iq_data[i:i+chunk_size]
            chunk_signals = await ft8_decoder.process_streaming_chunk(
                iq_chunk=chunk,
                sample_rate=sample_rate
            )
            realtime_signals.extend(chunk_signals)

        # Results should be similar (allow some tolerance)
        assert abs(len(batch_signals) - len(realtime_signals)) <= 2


class TestFalsePositiveFiltering:
    """Test filtering of false positive signal detections"""

    @pytest.fixture
    def ft8_decoder(self, mock_db_session):
        """Initialize FT8 decoder"""
        return FT8Decoder(db_session=mock_db_session)

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = Mock()
        return session

    @pytest.mark.asyncio
    async def test_noise_spike_rejection(self, ft8_decoder):
        """Test rejection of impulse noise as false positive"""
        sample_rate = 12000
        duration = 15
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Mostly noise
        iq_data = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.01

        # Add impulse noise spikes
        spike_indices = [1000, 5000, 9000]
        for idx in spike_indices:
            iq_data[idx] += 0.5 + 0.5j

        signals = await ft8_decoder.detect_signals(
            iq_data=iq_data,
            sample_rate=sample_rate,
            center_frequency=14074000,
            filter_false_positives=True
        )

        # Should not detect impulse noise as FT8
        assert len(signals) == 0, "Impulse noise incorrectly detected as signal"

    @pytest.mark.asyncio
    async def test_cw_signal_rejection(self, ft8_decoder):
        """Test rejection of CW (morse) signals"""
        sample_rate = 12000
        duration = 15
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create CW signal (continuous tone with on/off keying)
        cw_signal = np.zeros(len(t), dtype=complex)
        carrier = np.exp(2j * np.pi * 1500 * t)

        # On/off keying at ~20 WPM
        keying = np.zeros(len(t))
        for i in range(0, len(t), 600):
            keying[i:i+300] = 1.0

        cw_signal = keying * carrier * 0.05
        iq_data = cw_signal + (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.01

        signals = await ft8_decoder.detect_signals(
            iq_data=iq_data,
            sample_rate=sample_rate,
            center_frequency=14074000,
            filter_false_positives=True
        )

        # Should not detect CW as FT8
        assert len(signals) == 0, "CW signal incorrectly detected as FT8"

    @pytest.mark.asyncio
    async def test_harmonics_filtering(self, ft8_decoder):
        """Test filtering of harmonics from strong signals"""
        sample_rate = 12000
        duration = 15
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Strong signal with harmonics
        fundamental = 0.1 * np.exp(2j * np.pi * 1000 * t)
        harmonic2 = 0.03 * np.exp(2j * np.pi * 2000 * t)
        harmonic3 = 0.01 * np.exp(2j * np.pi * 3000 * t)

        iq_data = fundamental + harmonic2 + harmonic3

        signals = await ft8_decoder.detect_signals(
            iq_data=iq_data,
            sample_rate=sample_rate,
            center_frequency=14074000,
            filter_harmonics=True
        )

        # Should only report fundamental, not harmonics
        assert len(signals) <= 1, "Harmonics not properly filtered"

    @pytest.mark.asyncio
    async def test_duration_based_filtering(self, ft8_decoder):
        """Test filtering based on signal duration"""
        sample_rate = 12000

        # Too short signal (not a valid FT8 message)
        short_duration = 5  # FT8 is ~12.6 seconds
        t_short = np.linspace(0, short_duration, int(sample_rate * short_duration))
        short_signal = 0.05 * np.exp(2j * np.pi * 1500 * t_short)

        signals = await ft8_decoder.detect_signals(
            iq_data=short_signal,
            sample_rate=sample_rate,
            center_frequency=14074000,
            min_duration_seconds=10
        )

        assert len(signals) == 0, "Too-short signal not filtered"

    @pytest.mark.asyncio
    async def test_bandwidth_based_filtering(self, ft8_decoder):
        """Test filtering based on signal bandwidth"""
        sample_rate = 12000
        duration = 15
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create wide bandwidth signal (not FT8-like)
        # FT8 bandwidth is ~50 Hz, create 500 Hz wide signal
        wide_signal = np.zeros(len(t), dtype=complex)
        for freq_offset in range(-250, 250, 10):
            wide_signal += 0.001 * np.exp(2j * np.pi * (1500 + freq_offset) * t)

        signals = await ft8_decoder.detect_signals(
            iq_data=wide_signal,
            sample_rate=sample_rate,
            center_frequency=14074000,
            max_bandwidth_hz=100
        )

        assert len(signals) == 0, "Wide bandwidth signal not filtered"


class TestEndToEndPropagationAnalysis:
    """Test complete end-to-end propagation analysis workflow"""

    @pytest.fixture
    def full_pipeline(self, mock_db_session):
        """Initialize complete processing pipeline"""
        ft8_decoder = FT8Decoder(db_session=mock_db_session)
        wspr_decoder = WSPRDecoder(db_session=mock_db_session)
        anonymizer = CallsignAnonymizer(salt="integration_test_salt")

        return {
            'ft8': ft8_decoder,
            'wspr': wspr_decoder,
            'anonymizer': anonymizer
        }

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = Mock()
        session.query = Mock()
        return session

    @pytest.mark.asyncio
    async def test_complete_ft8_propagation_workflow(self, full_pipeline, mock_db_session):
        """Test complete FT8 workflow: detect -> decode -> anonymize -> store"""
        ft8_decoder = full_pipeline['ft8']
        anonymizer = full_pipeline['anonymizer']

        # Generate test data
        sample_rate = 12000
        duration = 15
        iq_data = (np.random.randn(int(sample_rate * duration)) +
                  1j * np.random.randn(int(sample_rate * duration))) * 0.01

        # Step 1: Detect and decode
        messages = await ft8_decoder.decode(
            iq_data=iq_data,
            sample_rate=sample_rate,
            timestamp=datetime.now(timezone.utc)
        )

        # Step 2: Extract propagation mutations
        for msg in messages:
            mutations = await ft8_decoder.extract_propagation_mutations(
                iq_data=iq_data,
                sample_rate=sample_rate,
                signal_frequency=msg.frequency_hz
            )
            msg.mutations = mutations

        # Step 3: Anonymize
        anon_messages = [anonymizer.anonymize_message(msg) for msg in messages]

        # Step 4: Create propagation records
        records = []
        for msg in anon_messages:
            record = PropagationRecord(
                session_id=uuid.uuid4(),
                timestamp=msg.timestamp,
                mode='FT8',
                tx_callsign_hash=msg.callsign_hash,
                tx_grid=msg.grid_square,
                frequency_hz=msg.frequency_hz,
                snr_db=msg.snr_db,
                mutation_data=msg.mutations if hasattr(msg, 'mutations') else {}
            )
            records.append(record)
            mock_db_session.add(record)

        # Verify workflow completion
        assert len(anon_messages) == len(messages)
        assert all(msg.callsign_hash is not None for msg in anon_messages)
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_mixed_ft8_wspr_processing(self, full_pipeline, mock_db_session):
        """Test processing recording with both FT8 and WSPR signals"""
        ft8_decoder = full_pipeline['ft8']
        wspr_decoder = full_pipeline['wspr']

        # Create IQ data with both signal types
        sample_rate = 12000
        duration = 120  # Long enough for both
        iq_data = (np.random.randn(int(sample_rate * duration)) +
                  1j * np.random.randn(int(sample_rate * duration))) * 0.01

        # Process as both
        ft8_messages = await ft8_decoder.decode(
            iq_data=iq_data,
            sample_rate=sample_rate,
            timestamp=datetime.now(timezone.utc)
        )

        wspr_messages = await wspr_decoder.decode(
            iq_data=iq_data,
            sample_rate=sample_rate,
            timestamp=datetime.now(timezone.utc)
        )

        # Should process both types
        assert isinstance(ft8_messages, list)
        assert isinstance(wspr_messages, list)

    @pytest.mark.asyncio
    async def test_propagation_database_storage(self, full_pipeline, mock_db_session):
        """Test storage of propagation records in database"""
        ft8_decoder = full_pipeline['ft8']
        anonymizer = full_pipeline['anonymizer']

        # Create mock session
        session_id = uuid.uuid4()
        recording_session = RecordingSession(
            session_id=session_id,
            center_frequency_hz=14074000,
            start_time=datetime.now(timezone.utc),
            band="20m"
        )

        # Create sample messages
        messages = [
            FT8Message(
                timestamp=datetime.now(timezone.utc),
                frequency_hz=14074500,
                snr_db=-10,
                dt_seconds=0.2,
                message="CQ K1ABC FN42"
            )
        ]

        # Process and store
        for msg in messages:
            anon_msg = anonymizer.anonymize_message(msg)

            record = PropagationRecord(
                session_id=session_id,
                timestamp=anon_msg.timestamp,
                mode='FT8',
                tx_callsign_hash=anon_msg.callsign_hash,
                frequency_hz=anon_msg.frequency_hz,
                snr_db=anon_msg.snr_db
            )

            mock_db_session.add(record)

        mock_db_session.commit()

        # Verify storage
        mock_db_session.add.assert_called()
        mock_db_session.commit.assert_called()


# Performance and stress tests
class TestSignalExtractionPerformance:
    """Test performance of signal extraction"""

    @pytest.fixture
    def ft8_decoder(self, mock_db_session):
        """Initialize FT8 decoder"""
        return FT8Decoder(db_session=mock_db_session)

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = Mock()
        return session

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_file_processing_performance(self, ft8_decoder):
        """Test processing performance on large files"""
        # 1 hour of data
        sample_rate = 12000
        duration = 3600
        samples = sample_rate * duration

        import time
        start_time = time.time()

        # Process in chunks to avoid memory issues
        chunk_duration = 60
        chunk_samples = sample_rate * chunk_duration

        total_signals = 0
        for _ in range(duration // chunk_duration):
            iq_chunk = (np.random.randn(chunk_samples) +
                       1j * np.random.randn(chunk_samples)) * 0.01

            signals = await ft8_decoder.detect_signals(
                iq_data=iq_chunk,
                sample_rate=sample_rate,
                center_frequency=14074000
            )
            total_signals += len(signals)

        elapsed = time.time() - start_time

        # Should process faster than real-time
        assert elapsed < duration, f"Processing too slow: {elapsed}s for {duration}s data"

    @pytest.mark.asyncio
    async def test_memory_usage_streaming(self, ft8_decoder):
        """Test memory usage during streaming processing"""
        import tracemalloc

        tracemalloc.start()

        sample_rate = 12000
        chunk_size = 2048

        # Process many chunks
        for _ in range(1000):
            iq_chunk = (np.random.randn(chunk_size) +
                       1j * np.random.randn(chunk_size)) * 0.01

            await ft8_decoder.process_streaming_chunk(
                iq_chunk=iq_chunk,
                sample_rate=sample_rate
            )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should stay reasonable (< 100 MB)
        assert peak < 100 * 1024 * 1024, f"Memory usage too high: {peak / 1024 / 1024:.1f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])