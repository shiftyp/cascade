"""Station anomaly detection for unusual behavior.

T079: Detect anomalous station behavior that may indicate equipment issues,
propagation anomalies, or other interesting phenomena.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from scipy import stats
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Detected anomaly in station behavior."""

    station_hash: str
    timestamp: datetime
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float  # 0-1

    # Anomaly details
    metric_name: str
    observed_value: float
    expected_value: float
    deviation_sigma: float

    # Context
    recent_history: List[float] = field(default_factory=list)
    contributing_factors: Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""


class StationAnomalyDetector:
    """Detects anomalous behavior in station operations."""

    def __init__(self, history_window: int = 100):
        """Initialize anomaly detector.

        Args:
            history_window: Number of observations to keep in history
        """
        self.history_window = history_window
        self.station_histories: Dict[str, Dict[str, deque]] = {}
        self.anomalies: List[Anomaly] = []
        self.baseline_stats: Dict[str, Dict[str, Dict]] = {}

        # Anomaly detection models
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        self.model_trained = False

    def process_observation(self, observation: Dict) -> List[Anomaly]:
        """Process a station observation for anomalies.

        Args:
            observation: Station observation with metrics

        Returns:
            List of detected anomalies
        """
        station_hash = observation.get('station_hash')
        if not station_hash:
            return []

        # Initialize history if needed
        if station_hash not in self.station_histories:
            self.station_histories[station_hash] = {
                'snr': deque(maxlen=self.history_window),
                'frequency': deque(maxlen=self.history_window),
                'drift': deque(maxlen=self.history_window),
                'power': deque(maxlen=self.history_window),
                'timestamps': deque(maxlen=self.history_window)
            }

        # Update history
        self._update_history(station_hash, observation)

        # Check for various anomaly types
        anomalies = []

        # Signal anomalies
        snr_anomaly = self._check_snr_anomaly(station_hash, observation)
        if snr_anomaly:
            anomalies.append(snr_anomaly)

        # Frequency anomalies
        freq_anomaly = self._check_frequency_anomaly(station_hash, observation)
        if freq_anomaly:
            anomalies.append(freq_anomaly)

        # Drift anomalies
        drift_anomaly = self._check_drift_anomaly(station_hash, observation)
        if drift_anomaly:
            anomalies.append(drift_anomaly)

        # Power anomalies
        power_anomaly = self._check_power_anomaly(station_hash, observation)
        if power_anomaly:
            anomalies.append(power_anomaly)

        # Pattern anomalies
        pattern_anomaly = self._check_pattern_anomaly(station_hash, observation)
        if pattern_anomaly:
            anomalies.append(pattern_anomaly)

        # Multivariate anomalies (if model trained)
        if self.model_trained:
            multivar_anomaly = self._check_multivariate_anomaly(station_hash, observation)
            if multivar_anomaly:
                anomalies.append(multivar_anomaly)

        # Store detected anomalies
        self.anomalies.extend(anomalies)

        return anomalies

    def _update_history(self, station_hash: str, observation: Dict):
        """Update station history with new observation."""
        history = self.station_histories[station_hash]

        history['snr'].append(observation.get('snr', 0))
        history['frequency'].append(observation.get('frequency', 0))
        history['drift'].append(observation.get('drift', 0))
        history['power'].append(observation.get('power', 0))
        history['timestamps'].append(observation.get('timestamp', datetime.now()))

    def _check_snr_anomaly(self, station_hash: str, observation: Dict) -> Optional[Anomaly]:
        """Check for SNR anomalies."""
        history = self.station_histories[station_hash]['snr']

        if len(history) < 10:
            return None

        current_snr = observation.get('snr', 0)
        historical_snr = list(history)[:-1]  # Exclude current

        # Calculate statistics
        mean_snr = np.mean(historical_snr)
        std_snr = np.std(historical_snr)

        if std_snr == 0:
            return None

        # Z-score
        z_score = abs((current_snr - mean_snr) / std_snr)

        # Check for anomaly
        if z_score > 3:  # 3-sigma rule
            severity = self._get_severity(z_score)

            return Anomaly(
                station_hash=station_hash,
                timestamp=datetime.now(),
                anomaly_type='snr_deviation',
                severity=severity,
                confidence=min(0.99, z_score / 5),
                metric_name='SNR',
                observed_value=current_snr,
                expected_value=mean_snr,
                deviation_sigma=z_score,
                recent_history=historical_snr[-10:],
                contributing_factors={'std_dev': std_snr},
                recommended_action=self._get_snr_recommendation(current_snr, mean_snr)
            )

        return None

    def _check_frequency_anomaly(self, station_hash: str, observation: Dict) -> Optional[Anomaly]:
        """Check for frequency stability anomalies."""
        history = self.station_histories[station_hash]['frequency']

        if len(history) < 10:
            return None

        current_freq = observation.get('frequency', 0)
        historical_freq = list(history)[:-1]

        # Check for sudden frequency jump
        if historical_freq:
            last_freq = historical_freq[-1]
            freq_change = abs(current_freq - last_freq)

            # Threshold depends on band
            band = observation.get('band', '20m')
            threshold = self._get_frequency_threshold(band)

            if freq_change > threshold:
                return Anomaly(
                    station_hash=station_hash,
                    timestamp=datetime.now(),
                    anomaly_type='frequency_jump',
                    severity='high' if freq_change > threshold * 2 else 'medium',
                    confidence=min(0.95, freq_change / (threshold * 3)),
                    metric_name='Frequency',
                    observed_value=current_freq,
                    expected_value=last_freq,
                    deviation_sigma=freq_change / threshold,
                    recent_history=historical_freq[-10:],
                    contributing_factors={'band': band, 'threshold': threshold},
                    recommended_action='Check for oscillator instability or QSY'
                )

        return None

    def _check_drift_anomaly(self, station_hash: str, observation: Dict) -> Optional[Anomaly]:
        """Check for frequency drift anomalies."""
        history = self.station_histories[station_hash]['drift']

        if len(history) < 5:
            return None

        current_drift = observation.get('drift', 0)
        historical_drift = list(history)[:-1]

        # Check for excessive drift
        max_normal_drift = 10  # Hz/min
        if abs(current_drift) > max_normal_drift:
            mean_drift = np.mean(historical_drift)

            return Anomaly(
                station_hash=station_hash,
                timestamp=datetime.now(),
                anomaly_type='excessive_drift',
                severity='high' if abs(current_drift) > max_normal_drift * 2 else 'medium',
                confidence=min(0.9, abs(current_drift) / (max_normal_drift * 3)),
                metric_name='Frequency Drift',
                observed_value=current_drift,
                expected_value=mean_drift,
                deviation_sigma=abs(current_drift - mean_drift) / (np.std(historical_drift) + 1e-6),
                recent_history=historical_drift[-10:],
                contributing_factors={'max_normal': max_normal_drift},
                recommended_action='Temperature compensation issue or aging oscillator'
            )

        return None

    def _check_power_anomaly(self, station_hash: str, observation: Dict) -> Optional[Anomaly]:
        """Check for transmit power anomalies."""
        history = self.station_histories[station_hash]['power']

        if len(history) < 10:
            return None

        current_power = observation.get('power', 0)
        historical_power = list(history)[:-1]

        mean_power = np.mean(historical_power)
        std_power = np.std(historical_power)

        # Check for sudden power change
        if std_power > 0:
            z_score = abs((current_power - mean_power) / std_power)

            if z_score > 2.5:  # Less strict than SNR
                return Anomaly(
                    station_hash=station_hash,
                    timestamp=datetime.now(),
                    anomaly_type='power_variation',
                    severity=self._get_severity(z_score),
                    confidence=min(0.85, z_score / 4),
                    metric_name='TX Power',
                    observed_value=current_power,
                    expected_value=mean_power,
                    deviation_sigma=z_score,
                    recent_history=historical_power[-10:],
                    contributing_factors={'std_dev': std_power},
                    recommended_action=self._get_power_recommendation(current_power, mean_power)
                )

        return None

    def _check_pattern_anomaly(self, station_hash: str, observation: Dict) -> Optional[Anomaly]:
        """Check for activity pattern anomalies."""
        timestamps = list(self.station_histories[station_hash]['timestamps'])

        if len(timestamps) < 20:
            return None

        # Check for unusual timing patterns
        current_time = observation.get('timestamp', datetime.now())
        if isinstance(current_time, str):
            current_time = datetime.fromisoformat(current_time)

        # Calculate inter-arrival times
        if len(timestamps) > 1:
            intervals = []
            for i in range(1, len(timestamps)):
                if isinstance(timestamps[i], str):
                    t1 = datetime.fromisoformat(timestamps[i])
                else:
                    t1 = timestamps[i]

                if isinstance(timestamps[i-1], str):
                    t0 = datetime.fromisoformat(timestamps[i-1])
                else:
                    t0 = timestamps[i-1]

                interval = (t1 - t0).total_seconds()
                intervals.append(interval)

            if intervals:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)

                # Check current interval
                if isinstance(timestamps[-1], str):
                    last_time = datetime.fromisoformat(timestamps[-1])
                else:
                    last_time = timestamps[-1]

                current_interval = (current_time - last_time).total_seconds()

                if std_interval > 0:
                    z_score = abs((current_interval - mean_interval) / std_interval)

                    if z_score > 3:
                        return Anomaly(
                            station_hash=station_hash,
                            timestamp=current_time,
                            anomaly_type='timing_pattern',
                            severity='low' if z_score < 4 else 'medium',
                            confidence=min(0.7, z_score / 5),
                            metric_name='Activity Interval',
                            observed_value=current_interval,
                            expected_value=mean_interval,
                            deviation_sigma=z_score,
                            recent_history=intervals[-10:],
                            contributing_factors={'pattern': 'irregular'},
                            recommended_action='Unusual activity timing detected'
                        )

        return None

    def _check_multivariate_anomaly(self, station_hash: str,
                                   observation: Dict) -> Optional[Anomaly]:
        """Check for anomalies using multivariate analysis."""
        # Extract feature vector
        features = self._extract_features(observation)

        if features is None:
            return None

        # Predict with isolation forest
        features_array = np.array(features).reshape(1, -1)
        anomaly_score = self.isolation_forest.decision_function(features_array)[0]

        # Negative scores indicate anomalies
        if anomaly_score < -0.1:
            return Anomaly(
                station_hash=station_hash,
                timestamp=datetime.now(),
                anomaly_type='multivariate',
                severity='medium',
                confidence=min(0.9, abs(anomaly_score)),
                metric_name='Combined Metrics',
                observed_value=anomaly_score,
                expected_value=0.0,
                deviation_sigma=abs(anomaly_score) * 10,
                contributing_factors={'features': features},
                recommended_action='Multiple metrics show unusual combination'
            )

        return None

    def train_multivariate_model(self, training_data: List[Dict]):
        """Train the multivariate anomaly detection model.

        Args:
            training_data: List of normal observations for training
        """
        if len(training_data) < 100:
            logger.warning("Insufficient data for training multivariate model")
            return

        # Extract features from training data
        features = []
        for obs in training_data:
            feature_vector = self._extract_features(obs)
            if feature_vector is not None:
                features.append(feature_vector)

        if len(features) < 50:
            logger.warning("Insufficient valid features for training")
            return

        # Train isolation forest
        X = np.array(features)
        self.isolation_forest.fit(X)
        self.model_trained = True

        logger.info(f"Trained multivariate model with {len(features)} samples")

    def _extract_features(self, observation: Dict) -> Optional[List[float]]:
        """Extract feature vector from observation."""
        try:
            features = [
                observation.get('snr', 0),
                observation.get('frequency', 0) / 1e6,  # Normalize to MHz
                observation.get('drift', 0),
                observation.get('power', 0),
                observation.get('phase_noise', 0),
                observation.get('evm', 0),
                observation.get('duty_cycle', 0)
            ]
            return features
        except:
            return None

    def _get_severity(self, z_score: float) -> str:
        """Determine anomaly severity from z-score."""
        if z_score < 3:
            return 'low'
        elif z_score < 4:
            return 'medium'
        elif z_score < 5:
            return 'high'
        else:
            return 'critical'

    def _get_frequency_threshold(self, band: str) -> float:
        """Get frequency jump threshold for band."""
        thresholds = {
            '160m': 50,  # Hz
            '80m': 100,
            '40m': 200,
            '20m': 500,
            '15m': 750,
            '10m': 1000,
            '6m': 2000
        }
        return thresholds.get(band, 500)

    def _get_snr_recommendation(self, current: float, expected: float) -> str:
        """Get recommendation for SNR anomaly."""
        if current > expected + 10:
            return "Unusually strong signal - check for local QRM or enhanced propagation"
        elif current < expected - 10:
            return "Weak signal - check antenna, propagation conditions, or QRM"
        else:
            return "Monitor signal strength variations"

    def _get_power_recommendation(self, current: float, expected: float) -> str:
        """Get recommendation for power anomaly."""
        if current > expected + 3:
            return "Power increase detected - possible amp engagement or setting change"
        elif current < expected - 3:
            return "Power decrease - check PA, SWR, or power settings"
        else:
            return "Monitor power stability"

    def detect_collective_anomalies(self, station_group: List[str]) -> List[Dict]:
        """Detect anomalies affecting multiple stations.

        Args:
            station_group: List of station hashes to analyze

        Returns:
            List of collective anomaly descriptions
        """
        collective_anomalies = []

        # Check for simultaneous anomalies
        recent_window = datetime.now() - timedelta(minutes=30)
        recent_anomalies = [a for a in self.anomalies
                          if a.timestamp >= recent_window and
                          a.station_hash in station_group]

        # Group by anomaly type
        from collections import Counter
        anomaly_types = Counter(a.anomaly_type for a in recent_anomalies)

        for anomaly_type, count in anomaly_types.items():
            if count >= len(station_group) * 0.3:  # 30% threshold
                collective_anomalies.append({
                    'type': 'collective_' + anomaly_type,
                    'affected_stations': count,
                    'total_stations': len(station_group),
                    'percentage': (count / len(station_group)) * 100,
                    'timestamp': datetime.now(),
                    'likely_cause': self._get_collective_cause(anomaly_type)
                })

        return collective_anomalies

    def _get_collective_cause(self, anomaly_type: str) -> str:
        """Determine likely cause of collective anomaly."""
        causes = {
            'snr_deviation': 'Propagation change or widespread QRM',
            'frequency_jump': 'GPS/time reference issue affecting multiple SDRs',
            'excessive_drift': 'Temperature change affecting equipment',
            'power_variation': 'Grid voltage fluctuation or propagation change',
            'timing_pattern': 'Network or coordination issue'
        }
        return causes.get(anomaly_type, 'Unknown collective phenomenon')

    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected anomalies."""
        if not self.anomalies:
            return {'total': 0}

        # Count by type
        type_counts = Counter(a.anomaly_type for a in self.anomalies)

        # Count by severity
        severity_counts = Counter(a.severity for a in self.anomalies)

        # Recent rate
        recent = datetime.now() - timedelta(hours=1)
        recent_count = sum(1 for a in self.anomalies if a.timestamp >= recent)

        return {
            'total': len(self.anomalies),
            'by_type': dict(type_counts),
            'by_severity': dict(severity_counts),
            'recent_hour': recent_count,
            'avg_confidence': np.mean([a.confidence for a in self.anomalies]),
            'unique_stations': len(set(a.station_hash for a in self.anomalies))
        }