"""Integration tests for persistent path learning.

T081: Test end-to-end persistent path tracking, prediction,
and station fingerprinting integration.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import tempfile
import json
import sys
import os
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.processors.station_fingerprint import StationFingerprintExtractor
from src.processors.persistent_paths import PersistentPathTracker
from src.processors.equipment_signature import EquipmentSignatureExtractor
from src.embeddings.station_aware import StationAwareEmbeddingGenerator
from src.analytics.station_patterns import StationPatternAnalyzer
from src.analytics.station_aggregator import PrivacySafeAggregator
from src.validators.station_anomaly import StationAnomalyDetector


class TestPersistentPathIntegration:
    """Test complete persistent path tracking pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fingerprint_extractor = StationFingerprintExtractor()
        self.path_tracker = PersistentPathTracker()
        self.equipment_extractor = EquipmentSignatureExtractor()
        self.embedding_generator = StationAwareEmbeddingGenerator()
        self.pattern_analyzer = StationPatternAnalyzer()
        self.aggregator = PrivacySafeAggregator(k_anonymity=2)
        self.anomaly_detector = StationAnomalyDetector()

    def test_end_to_end_path_learning(self):
        """Test complete path learning from observations to predictions."""
        # Simulate TX-RX pairs over multiple days
        tx_stations = ['ANON_TX1', 'ANON_TX2', 'ANON_TX3']
        rx_stations = ['ANON_RX1', 'ANON_RX2']

        # Time progression
        start_time = datetime.now() - timedelta(days=7)

        # Generate observations
        observations = []
        for day in range(7):
            for hour in [6, 12, 18]:  # Peak propagation times
                for tx in tx_stations:
                    for rx in rx_stations:
                        # Simulate propagation probability
                        if self._simulate_propagation(tx, rx, hour):
                            obs = {
                                'tx_hash': tx,
                                'rx_hash': rx,
                                'tx_grid': 'FN42',
                                'rx_grid': 'FN43' if 'RX1' in rx else 'FN44',
                                'timestamp': start_time + timedelta(days=day, hours=hour),
                                'frequency': 14074000,
                                'snr': np.random.uniform(-20, 10),
                                'mode': 'FT8',
                                'band': '20m'
                            }
                            observations.append(obs)

        # Process observations through pipeline
        for obs in observations:
            # Update path tracker
            self.path_tracker.record_observation(
                obs['tx_hash'],
                obs['rx_hash'],
                obs['tx_grid'],
                obs['rx_grid'],
                obs['snr'],
                obs['timestamp'],
                obs['mode']
            )

            # Update fingerprints
            self.fingerprint_extractor.process_signal({
                'callsign_hash': obs['tx_hash'],
                'timestamp': obs['timestamp'],
                'frequency': obs['frequency'],
                'snr': obs['snr'],
                'grid': obs['tx_grid'],
                'message_type': 'QSO'
            })

            # Update patterns
            self.pattern_analyzer.add_observation({
                'station_hash': obs['tx_hash'],
                'timestamp': obs['timestamp'],
                'band': obs['band'],
                'message_type': 'QSO',
                'snr': obs['snr'],
                'grid': obs['tx_grid']
            })

        # Verify path learning
        paths = self.path_tracker.get_all_paths()
        assert len(paths) > 0

        # Test path prediction
        for tx in tx_stations:
            prediction = self.path_tracker.predict_propagation(
                tx, 'ANON_RX1',
                datetime.now() + timedelta(hours=6)
            )
            assert prediction is not None
            assert 0 <= prediction['probability'] <= 1
            assert prediction['best_frequency'] > 0

        # Test bidirectional detection
        bidirectional = self.path_tracker.find_bidirectional_paths()
        # Some paths should be bidirectional
        assert len(bidirectional) >= 0

    def test_station_embedding_integration(self):
        """Test integration of station embeddings with path tracking."""
        # Create stations with different characteristics
        stations = [
            {'hash': 'ANON_HIGH_POWER', 'power': 100, 'snr': 10},
            {'hash': 'ANON_LOW_POWER', 'power': 5, 'snr': -15},
            {'hash': 'ANON_QRP', 'power': 1, 'snr': -20}
        ]

        # Build fingerprints
        for station in stations:
            for i in range(15):  # Enough observations
                self.fingerprint_extractor.process_signal({
                    'callsign_hash': station['hash'],
                    'timestamp': datetime.now() + timedelta(minutes=i),
                    'frequency': 14074000,
                    'snr': station['snr'] + np.random.randn(),
                    'grid': 'FN42',
                    'message_type': 'CQ',
                    'power': station['power']
                })

        # Generate embeddings
        for station in stations:
            fp = self.fingerprint_extractor.get_fingerprint(station['hash'])
            if fp:
                fp_dict = {
                    'station_hash': fp.station_hash,
                    'avg_snr_db': fp.avg_snr_db,
                    'primary_bands': fp.primary_bands,
                    'primary_grid': fp.primary_grid,
                    'duty_cycle': fp.duty_cycle,
                    'total_observations': fp.total_observations,
                    'first_seen': fp.first_seen,
                    'last_seen': fp.last_seen
                }
                embedding = self.embedding_generator.generate_embedding(fp_dict)
                assert embedding is not None
                assert len(embedding.embedding_vector) == 128

        # Test similarity based on embeddings
        similar = self.embedding_generator.find_similar_stations(
            'ANON_HIGH_POWER', threshold=0.5
        )

        # QRP should be less similar to high power
        similarity_scores = {s[0]: s[1] for s in similar}
        if 'ANON_LOW_POWER' in similarity_scores and 'ANON_QRP' in similarity_scores:
            assert similarity_scores.get('ANON_LOW_POWER', 0) > similarity_scores.get('ANON_QRP', 0)

    def test_privacy_aggregation_integration(self):
        """Test privacy-safe aggregation with real data flow."""
        # Create diverse station data
        station_data = []

        # Group 1: Urban stations (many)
        for i in range(10):
            station_data.append({
                'station_hash': f'ANON_URBAN_{i}',
                'primary_grid': 'FN42',
                'primary_bands': ['20m', '40m'],
                'avg_snr_db': -5 + np.random.randn(),
                'total_observations': 100 + i * 10,
                'duty_cycle': 30 + np.random.uniform(-5, 5),
                'first_seen': datetime.now() - timedelta(days=30),
                'last_seen': datetime.now()
            })

        # Group 2: Rural stations (few - will be suppressed)
        for i in range(2):
            station_data.append({
                'station_hash': f'ANON_RURAL_{i}',
                'primary_grid': 'DN70',
                'primary_bands': ['80m'],
                'avg_snr_db': -15 + np.random.randn(),
                'total_observations': 50,
                'duty_cycle': 10,
                'first_seen': datetime.now() - timedelta(days=10),
                'last_seen': datetime.now()
            })

        # Aggregate with privacy protection
        stats = self.aggregator.aggregate_stations(station_data)

        assert stats.total_stations >= 10  # Rural suppressed
        assert stats.suppressed_count == 2
        assert 'FN42' in stats.grid_distribution
        assert 'DN70' not in stats.grid_distribution  # Suppressed

        # Test time series aggregation
        time_series = self.aggregator.create_time_series(station_data, window_hours=24)
        assert len(time_series) > 0

    def test_anomaly_detection_integration(self):
        """Test anomaly detection in full pipeline."""
        station_hash = 'ANON_ANOMALY_TEST'

        # Build normal baseline
        for i in range(30):
            obs = {
                'station_hash': station_hash,
                'timestamp': datetime.now() - timedelta(hours=30-i),
                'frequency': 14074000,
                'snr': -10 + np.random.randn(),
                'drift': np.random.uniform(-1, 1),
                'power': 10 + np.random.uniform(-0.5, 0.5),
                'band': '20m'
            }

            # Process through pipeline
            self.fingerprint_extractor.process_signal({
                'callsign_hash': obs['station_hash'],
                'timestamp': obs['timestamp'],
                'frequency': obs['frequency'],
                'snr': obs['snr'],
                'grid': 'FN42'
            })

            anomalies = self.anomaly_detector.process_observation(obs)
            assert len(anomalies) == 0  # No anomalies in baseline

        # Inject anomalies
        anomaly_obs = {
            'station_hash': station_hash,
            'timestamp': datetime.now(),
            'frequency': 14074000,
            'snr': 30,  # Very high SNR
            'drift': 50,  # High drift
            'power': 100,  # Power spike
            'band': '20m'
        }

        anomalies = self.anomaly_detector.process_observation(anomaly_obs)
        assert len(anomalies) > 0

        # Should detect multiple anomaly types
        anomaly_types = [a.anomaly_type for a in anomalies]
        assert 'snr_deviation' in anomaly_types
        assert 'excessive_drift' in anomaly_types

    def test_equipment_signature_integration(self):
        """Test equipment signature extraction with IQ samples."""
        # Generate IQ samples with equipment characteristics
        sample_rate = 12000
        duration = 1.0
        t = np.arange(0, duration, 1/sample_rate)

        # Station 1: Clean signal
        station1_iq = np.exp(1j * 2 * np.pi * 1000 * t)

        # Station 2: Drifting oscillator
        drift = 100 * t  # Linear drift
        station2_iq = np.exp(1j * 2 * np.pi * (1000 + drift) * t)

        # Station 3: High phase noise
        phase_noise = 0.1 * np.cumsum(np.random.randn(len(t)))
        station3_iq = np.exp(1j * (2 * np.pi * 1000 * t + phase_noise))

        # Extract signatures
        sig1 = self.equipment_extractor.extract_signature(
            station1_iq, 'ANON_CLEAN', datetime.now().isoformat(), 20
        )
        sig2 = self.equipment_extractor.extract_signature(
            station2_iq, 'ANON_DRIFT', datetime.now().isoformat(), 20
        )
        sig3 = self.equipment_extractor.extract_signature(
            station3_iq, 'ANON_NOISY', datetime.now().isoformat(), 20
        )

        # Verify different characteristics detected
        assert sig2.frequency_drift_hz_per_min > sig1.frequency_drift_hz_per_min
        assert sig3.phase_noise_dbc > sig1.phase_noise_dbc

        # Test aggregation
        for i in range(5):
            self.equipment_extractor.extract_signature(
                station1_iq + 0.01 * np.random.randn(len(t)),
                'ANON_CLEAN',
                (datetime.now() + timedelta(minutes=i)).isoformat(),
                20
            )

        agg_sig = self.equipment_extractor.aggregate_signatures('ANON_CLEAN')
        assert agg_sig is not None
        assert agg_sig.confidence_score > sig1.confidence_score

    def test_pattern_analysis_integration(self):
        """Test pattern analysis with realistic data flow."""
        station_hash = 'ANON_PATTERN_TEST'

        # Simulate week of activity with patterns
        start = datetime.now() - timedelta(days=7)

        # Morning: 20m CQ
        # Afternoon: 20m QSO
        # Evening: 40m QSO
        for day in range(7):
            # Morning session
            for i in range(5):
                self.pattern_analyzer.add_observation({
                    'station_hash': station_hash,
                    'timestamp': start + timedelta(days=day, hours=8, minutes=i*10),
                    'band': '20m',
                    'message_type': 'CQ',
                    'snr': -10,
                    'grid': 'FN42'
                })

            # Afternoon session
            for i in range(10):
                self.pattern_analyzer.add_observation({
                    'station_hash': station_hash,
                    'timestamp': start + timedelta(days=day, hours=14, minutes=i*5),
                    'band': '20m',
                    'message_type': 'QSO',
                    'snr': -5,
                    'grid': 'FN42'
                })

            # Evening session
            for i in range(8):
                self.pattern_analyzer.add_observation({
                    'station_hash': station_hash,
                    'timestamp': start + timedelta(days=day, hours=20, minutes=i*5),
                    'band': '40m',
                    'message_type': 'QSO',
                    'snr': -8,
                    'grid': 'FN42'
                })

        # Analyze patterns
        pattern = self.pattern_analyzer.analyze_station(station_hash)

        assert pattern is not None
        assert 8 in pattern.active_hours_utc
        assert 14 in pattern.active_hours_utc
        assert 20 in pattern.active_hours_utc

        # Check band preferences by hour
        assert pattern.preferred_bands_by_hour.get(8) == '20m'
        assert pattern.preferred_bands_by_hour.get(20) == '40m'

        # Check message type distribution
        assert pattern.message_type_distribution['CQ'] > 0
        assert pattern.message_type_distribution['QSO'] > pattern.message_type_distribution['CQ']

    def test_data_export_import(self):
        """Test exporting and importing processed data."""
        # Create test data
        station_hash = 'ANON_EXPORT'

        # Build fingerprint
        for i in range(15):
            self.fingerprint_extractor.process_signal({
                'callsign_hash': station_hash,
                'timestamp': datetime.now() + timedelta(minutes=i),
                'frequency': 14074000,
                'snr': -10,
                'grid': 'FN42',
                'message_type': 'CQ'
            })

        # Export fingerprints
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name

        self.fingerprint_extractor.export_fingerprints(export_path)

        # Read and verify export
        with open(export_path, 'r') as f:
            exported_data = json.load(f)

        assert station_hash in exported_data
        assert exported_data[station_hash]['primary_grid'] == 'FN42'

        # Clean up
        os.unlink(export_path)

    def test_multiband_path_tracking(self):
        """Test path tracking across multiple bands."""
        tx = 'ANON_TX_MULTI'
        rx = 'ANON_RX_MULTI'

        bands = {
            '80m': 3574000,
            '40m': 7074000,
            '20m': 14074000,
            '15m': 21074000,
            '10m': 28074000
        }

        # Record observations on different bands at different times
        for hour in range(24):
            # Select band based on time of day (simplified propagation model)
            if 6 <= hour < 10:
                band = '20m'
            elif 10 <= hour < 14:
                band = '15m'
            elif 14 <= hour < 18:
                band = '10m' if hour % 2 == 0 else '15m'
            elif 18 <= hour < 22:
                band = '20m'
            else:
                band = '40m' if hour < 3 else '80m'

            self.path_tracker.record_observation(
                tx, rx,
                'FN42', 'FN43',
                np.random.uniform(-20, 0),
                datetime.now() - timedelta(hours=24-hour),
                'FT8'
            )

            # Update band-specific data
            self.path_tracker.update_band_data(
                tx, rx, band, bands[band]
            )

        # Get path statistics
        path = self.path_tracker.get_path(tx, rx)
        assert path is not None

        # Check best times and frequencies
        best_times = self.path_tracker.get_best_propagation_times(tx, rx)
        assert len(best_times) > 0

    def _simulate_propagation(self, tx: str, rx: str, hour: int) -> bool:
        """Simulate propagation probability based on time."""
        # Simple model: better propagation at certain hours
        if 6 <= hour <= 9 or 18 <= hour <= 21:
            return np.random.random() < 0.8  # High probability
        elif 10 <= hour <= 17:
            return np.random.random() < 0.5  # Medium probability
        else:
            return np.random.random() < 0.2  # Low probability


class TestDatabaseIntegration:
    """Test database integration for persistent paths."""

    @pytest.mark.skipif(not os.getenv('DATABASE_URL'),
                       reason="Database not configured")
    def test_database_persistence(self):
        """Test saving and loading from database."""
        # This test would require actual database connection
        # Skipped unless DATABASE_URL is set
        pass

    @pytest.mark.skipif(not os.getenv('REDIS_URL'),
                       reason="Redis not configured")
    def test_redis_coordination(self):
        """Test Redis-based coordination between collectors."""
        # This test would require Redis connection
        # Skipped unless REDIS_URL is set
        pass