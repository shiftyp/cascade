"""Unit tests for station fingerprinting pipeline.

T080: Test station fingerprint extraction, equipment signature detection,
and privacy-preserving aggregation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.processors.station_fingerprint import (
    StationFingerprint, StationFingerprintExtractor
)
from src.processors.equipment_signature import (
    EquipmentSignature, EquipmentSignatureExtractor
)
from src.analytics.station_patterns import (
    ActivityPattern, StationPatternAnalyzer
)
from src.analytics.station_aggregator import (
    AggregatedStatistics, PrivacySafeAggregator
)
from src.validators.station_anomaly import (
    Anomaly, StationAnomalyDetector
)


class TestStationFingerprint:
    """Test StationFingerprint class and extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = StationFingerprintExtractor(salt="test_salt")

    def test_fingerprint_creation(self):
        """Test creating initial fingerprint."""
        observation = {
            'callsign_hash': 'ANON_ABC123',
            'timestamp': datetime.now().isoformat(),
            'frequency': 14074000,
            'snr': -5,
            'grid': 'FN42',
            'message_type': 'CQ',
            'power': 10
        }

        # Process observation
        result = self.extractor.process_signal(observation)
        assert result is None  # Not enough observations yet

        # Add more observations to meet minimum
        for i in range(10):
            obs = observation.copy()
            obs['snr'] = -5 + i * 0.5
            self.extractor.process_signal(obs)

        # Should now have fingerprint
        fp = self.extractor.get_fingerprint('ANON_ABC123')
        assert fp is not None
        assert fp.station_hash == 'ANON_ABC123'
        assert fp.primary_bands == ['20m']
        assert fp.primary_grid == 'FN42'
        assert fp.total_observations >= 10

    def test_frequency_band_detection(self):
        """Test frequency to band conversion."""
        test_cases = [
            (1900000, '160m'),
            (3700000, '80m'),
            (7100000, '40m'),
            (14200000, '20m'),
            (21200000, '15m'),
            (28500000, '10m'),
            (50100000, '6m'),
            (145000000, 'unknown')  # Out of HF range
        ]

        for freq, expected_band in test_cases:
            band = self.extractor._freq_to_band(freq)
            assert band == expected_band

    def test_duty_cycle_calculation(self):
        """Test duty cycle calculation."""
        station_hash = 'ANON_XYZ789'

        # Create observations over 24 hours
        start_time = datetime.now() - timedelta(hours=24)

        for hour in range(0, 24, 3):  # Active every 3 hours
            obs = {
                'callsign_hash': station_hash,
                'timestamp': (start_time + timedelta(hours=hour)).isoformat(),
                'frequency': 14074000,
                'snr': -10,
                'grid': 'FN42'
            }
            self.extractor.process_signal(obs)

        fp = self.extractor.get_fingerprint(station_hash)
        if fp:
            # Should have duty cycle between 25-35% (8 hours out of 24)
            assert 20 <= fp.duty_cycle <= 40

    def test_grid_square_tracking(self):
        """Test tracking of multiple grid squares."""
        station_hash = 'ANON_MULTI123'
        grids = ['FN42', 'FN42', 'FN43', 'FN42', 'FN41']  # FN42 most common

        for i, grid in enumerate(grids * 2):  # 10 observations
            obs = {
                'callsign_hash': station_hash,
                'timestamp': datetime.now().isoformat(),
                'frequency': 14074000,
                'snr': -10,
                'grid': grid
            }
            self.extractor.process_signal(obs)

        fp = self.extractor.get_fingerprint(station_hash)
        if fp:
            assert fp.primary_grid == 'FN42'  # Most common
            assert set(fp.grid_squares) == {'FN42', 'FN43', 'FN41'}

    def test_message_type_distribution(self):
        """Test message type tracking."""
        station_hash = 'ANON_MSG456'
        message_types = ['CQ'] * 5 + ['QSO'] * 3 + ['BEACON'] * 2

        for msg_type in message_types:
            obs = {
                'callsign_hash': station_hash,
                'timestamp': datetime.now().isoformat(),
                'frequency': 14074000,
                'snr': -10,
                'grid': 'FN42',
                'message_type': msg_type
            }
            self.extractor.process_signal(obs)

        fp = self.extractor.get_fingerprint(station_hash)
        if fp:
            assert fp.message_types['CQ'] == 5
            assert fp.message_types['QSO'] == 3
            assert fp.message_types['BEACON'] == 2

    def test_similarity_calculation(self):
        """Test similarity between fingerprints."""
        # Create two similar stations
        for station in ['ANON_SIM1', 'ANON_SIM2']:
            for i in range(10):
                obs = {
                    'callsign_hash': station,
                    'timestamp': datetime.now().isoformat(),
                    'frequency': 14074000,  # Same band
                    'snr': -10 + i * 0.1,  # Similar SNR
                    'grid': 'FN42',  # Same grid
                    'message_type': 'CQ'
                }
                self.extractor.process_signal(obs)

        # Create dissimilar station
        for i in range(10):
            obs = {
                'callsign_hash': 'ANON_DIFF',
                'timestamp': datetime.now().isoformat(),
                'frequency': 7074000,  # Different band
                'snr': 10,  # Very different SNR
                'grid': 'DM79',  # Different grid
                'message_type': 'BEACON'
            }
            self.extractor.process_signal(obs)

        # Test similarity
        similar = self.extractor.find_similar_stations('ANON_SIM1', threshold=0.5)

        # Should find SIM2 as similar
        similar_hashes = [s[0] for s in similar]
        assert 'ANON_SIM2' in similar_hashes
        assert 'ANON_DIFF' not in similar_hashes


class TestEquipmentSignature:
    """Test equipment signature extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = EquipmentSignatureExtractor(sample_rate=12000)

    def test_signature_extraction(self):
        """Test extracting equipment signature from IQ samples."""
        # Generate test IQ samples with known characteristics
        t = np.arange(0, 1, 1/12000)  # 1 second at 12 kHz
        carrier = 1000  # Hz

        # Add phase noise
        phase = 2 * np.pi * carrier * t + 0.01 * np.random.randn(len(t))
        iq_samples = np.exp(1j * phase)

        # Add amplitude variation
        iq_samples *= (1 + 0.1 * np.sin(2 * np.pi * 10 * t))

        signature = self.extractor.extract_signature(
            iq_samples,
            station_hash='ANON_TEST',
            timestamp=datetime.now().isoformat(),
            snr_db=20
        )

        assert signature is not None
        assert signature.station_hash == 'ANON_TEST'
        assert signature.confidence_score > 0.5  # Good SNR should give good confidence
        assert signature.sample_count == len(iq_samples)

    def test_phase_noise_measurement(self):
        """Test phase noise measurement."""
        # Create clean signal
        t = np.arange(0, 1, 1/12000)
        clean = np.exp(1j * 2 * np.pi * 1000 * t)

        # Add phase noise
        noisy = clean * np.exp(1j * 0.1 * np.random.randn(len(t)))

        phase_noise_clean = self.extractor._measure_phase_noise(clean)
        phase_noise_noisy = self.extractor._measure_phase_noise(noisy)

        # Noisy signal should have higher phase noise
        assert phase_noise_noisy > phase_noise_clean

    def test_frequency_drift_measurement(self):
        """Test frequency drift measurement."""
        t = np.arange(0, 1, 1/12000)

        # Signal with linear drift
        drift_rate = 100  # Hz/s
        phase = 2 * np.pi * (1000 * t + 0.5 * drift_rate * t**2)
        drifting = np.exp(1j * phase)

        # Stable signal
        stable = np.exp(1j * 2 * np.pi * 1000 * t)

        drift_measured = self.extractor._measure_frequency_drift(drifting)
        drift_stable = self.extractor._measure_frequency_drift(stable)

        # Drifting signal should show drift
        assert abs(drift_measured) > abs(drift_stable)
        # Should be close to 6000 Hz/min (100 Hz/s * 60)
        assert 5000 < abs(drift_measured) < 7000

    def test_aggregated_signatures(self):
        """Test aggregating multiple signatures."""
        station_hash = 'ANON_AGG'

        # Create multiple observations
        for i in range(5):
            t = np.arange(0, 0.1, 1/12000)  # 100ms samples
            iq = np.exp(1j * 2 * np.pi * 1000 * t)

            self.extractor.extract_signature(
                iq,
                station_hash=station_hash,
                timestamp=datetime.now().isoformat(),
                snr_db=10 + i
            )

        # Get aggregated signature
        agg_sig = self.extractor.aggregate_signatures(station_hash)

        assert agg_sig is not None
        assert agg_sig.sample_count > len(t)  # Combined samples
        assert agg_sig.confidence_score > 0  # Should have confidence


class TestStationPatterns:
    """Test station activity pattern analysis."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StationPatternAnalyzer()

    def test_pattern_analysis(self):
        """Test analyzing station patterns."""
        station_hash = 'ANON_PATTERN'

        # Create observations with pattern
        start = datetime.now() - timedelta(days=7)

        for day in range(7):
            for hour in [8, 12, 18, 22]:  # Active at specific hours
                obs = {
                    'station_hash': station_hash,
                    'timestamp': start + timedelta(days=day, hours=hour),
                    'band': '20m' if hour < 18 else '40m',  # Band change pattern
                    'message_type': 'CQ' if hour == 8 else 'QSO',
                    'snr': -10,
                    'grid': 'FN42'
                }
                self.analyzer.add_observation(obs)

        # Analyze patterns
        pattern = self.analyzer.analyze_station(station_hash)

        assert pattern is not None
        assert 8 in pattern.active_hours_utc
        assert 12 in pattern.active_hours_utc
        assert 18 in pattern.active_hours_utc
        assert 22 in pattern.active_hours_utc
        assert pattern.band_usage['20m'] > 0
        assert pattern.band_usage['40m'] > 0

    def test_qso_success_rate(self):
        """Test QSO success rate calculation."""
        station_hash = 'ANON_QSO'

        # Pattern: CQ followed by QSO (success) or CQ alone (failure)
        observations = [
            {'message_type': 'CQ'},
            {'message_type': 'QSO'},  # Success
            {'message_type': 'CQ'},
            {'message_type': 'CQ'},  # Two failures
            {'message_type': 'QSO'},  # Success
        ]

        for i, obs_template in enumerate(observations):
            obs = {
                'station_hash': station_hash,
                'timestamp': datetime.now() + timedelta(minutes=i),
                'band': '20m',
                'snr': -10,
                'grid': 'FN42',
                **obs_template
            }
            self.analyzer.add_observation(obs)

        pattern = self.analyzer.analyze_station(station_hash)
        if pattern:
            # 2 successful CQs out of 3 total
            assert 60 <= pattern.qso_success_rate <= 70


class TestPrivacyAggregation:
    """Test privacy-safe aggregation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.aggregator = PrivacySafeAggregator(k_anonymity=3, epsilon=1.0)

    def test_k_anonymity(self):
        """Test k-anonymity enforcement."""
        # Create stations with different characteristics
        stations = []

        # Group 1: 4 stations with same grid/band (meets k-anonymity)
        for i in range(4):
            stations.append({
                'station_hash': f'ANON_GROUP1_{i}',
                'primary_grid': 'FN42',
                'primary_bands': ['20m'],
                'avg_snr_db': -10,
                'total_observations': 100
            })

        # Group 2: 2 stations (below k-anonymity)
        for i in range(2):
            stations.append({
                'station_hash': f'ANON_GROUP2_{i}',
                'primary_grid': 'DM79',
                'primary_bands': ['40m'],
                'avg_snr_db': 5,
                'total_observations': 50
            })

        stats = self.aggregator.aggregate_stations(stations)

        # Group 2 should be suppressed
        assert stats.suppressed_count == 2
        assert stats.total_stations == 4  # Only group 1

    def test_differential_privacy_noise(self):
        """Test differential privacy noise addition."""
        # Create identical stations for predictable aggregation
        stations = []
        for i in range(10):
            stations.append({
                'station_hash': f'ANON_DP_{i}',
                'primary_grid': 'FN42',
                'primary_bands': ['20m'],
                'avg_snr_db': -10.0,  # All same SNR
                'total_observations': 100,
                'duty_cycle': 25.0
            })

        # Run multiple aggregations
        snr_values = []
        for _ in range(10):
            stats = self.aggregator.aggregate_stations(stations.copy())
            snr_values.append(stats.avg_snr_db)

        # With DP noise, values should vary
        assert len(set(snr_values)) > 1  # Not all identical
        assert all(-15 < snr < -5 for snr in snr_values)  # Reasonable range

    def test_grid_aggregation(self):
        """Test aggregation by grid square."""
        stations = []

        # Create stations in different grids
        grids = ['FN42', 'FN42', 'FN42', 'FN43', 'FN43', 'FN43']
        for i, grid in enumerate(grids):
            stations.append({
                'station_hash': f'ANON_GRID_{i}',
                'primary_grid': grid + 'ab',  # 6-char grid
                'primary_bands': ['20m'],
                'avg_snr_db': -10,
                'total_observations': 100
            })

        grid_stats = self.aggregator.aggregate_by_grid(stations)

        # Should have stats for both 4-char grids
        assert 'FN42' in grid_stats
        assert 'FN43' in grid_stats
        # Each grid has 3 stations (meets k=3)
        assert grid_stats['FN42'].total_stations >= 3
        assert grid_stats['FN43'].total_stations >= 3


class TestAnomalyDetection:
    """Test station anomaly detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = StationAnomalyDetector(history_window=50)

    def test_snr_anomaly_detection(self):
        """Test detecting SNR anomalies."""
        station_hash = 'ANON_ANOMALY'

        # Build normal history
        for i in range(20):
            obs = {
                'station_hash': station_hash,
                'timestamp': datetime.now() + timedelta(minutes=i),
                'snr': -10 + np.random.randn(),  # Normal variation
                'frequency': 14074000,
                'drift': 0,
                'power': 10
            }
            anomalies = self.detector.process_observation(obs)
            assert len(anomalies) == 0  # No anomalies in normal data

        # Add anomalous observation
        anomaly_obs = {
            'station_hash': station_hash,
            'timestamp': datetime.now() + timedelta(minutes=21),
            'snr': 20,  # Very high SNR (anomaly)
            'frequency': 14074000,
            'drift': 0,
            'power': 10
        }
        anomalies = self.detector.process_observation(anomaly_obs)

        assert len(anomalies) > 0
        assert anomalies[0].anomaly_type == 'snr_deviation'
        assert anomalies[0].severity in ['high', 'critical']

    def test_frequency_jump_detection(self):
        """Test detecting frequency jumps."""
        station_hash = 'ANON_FREQ'

        # Build history
        for i in range(10):
            obs = {
                'station_hash': station_hash,
                'timestamp': datetime.now() + timedelta(minutes=i),
                'snr': -10,
                'frequency': 14074000,  # Stable frequency
                'band': '20m',
                'drift': 0,
                'power': 10
            }
            self.detector.process_observation(obs)

        # Frequency jump
        jump_obs = {
            'station_hash': station_hash,
            'timestamp': datetime.now() + timedelta(minutes=11),
            'snr': -10,
            'frequency': 14076000,  # 2 kHz jump
            'band': '20m',
            'drift': 0,
            'power': 10
        }
        anomalies = self.detector.process_observation(jump_obs)

        assert len(anomalies) > 0
        freq_anomaly = [a for a in anomalies if a.anomaly_type == 'frequency_jump']
        assert len(freq_anomaly) > 0

    def test_collective_anomaly_detection(self):
        """Test detecting anomalies across multiple stations."""
        stations = ['ANON_C1', 'ANON_C2', 'ANON_C3', 'ANON_C4']

        # Create simultaneous anomalies
        for station in stations[:3]:  # 3 out of 4 stations
            for i in range(10):
                # Normal observations
                obs = {
                    'station_hash': station,
                    'timestamp': datetime.now() - timedelta(minutes=30-i),
                    'snr': -10,
                    'frequency': 14074000,
                    'drift': 0,
                    'power': 10
                }
                self.detector.process_observation(obs)

            # Anomaly
            anomaly_obs = {
                'station_hash': station,
                'timestamp': datetime.now() - timedelta(minutes=5),
                'snr': 30,  # All show high SNR
                'frequency': 14074000,
                'drift': 0,
                'power': 10
            }
            self.detector.process_observation(anomaly_obs)

        # Check for collective anomalies
        collective = self.detector.detect_collective_anomalies(stations)

        assert len(collective) > 0
        assert collective[0]['type'] == 'collective_snr_deviation'
        assert collective[0]['affected_stations'] >= 3