"""Station activity pattern analysis.

T077: Analyze station operating patterns including time-of-day preferences,
band usage patterns, and behavioral characteristics.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json

logger = logging.getLogger(__name__)


@dataclass
class ActivityPattern:
    """Station activity pattern analysis results."""

    station_hash: str
    analysis_period: Tuple[datetime, datetime]

    # Temporal patterns
    active_hours_utc: List[int]  # Most active hours [0-23]
    hour_probabilities: Dict[int, float]  # Probability active each hour
    active_days: List[int]  # Most active days [0-6, 0=Monday]
    day_probabilities: Dict[int, float]

    # Band preferences
    band_usage: Dict[str, int]  # Band -> observation count
    band_sequence_patterns: List[Tuple[str, str, int]]  # Common band transitions
    preferred_bands_by_hour: Dict[int, str]  # Hour -> most used band

    # Operating patterns
    avg_session_duration_min: float
    avg_qso_duration_min: float
    avg_cq_interval_min: float
    duty_cycle_percent: float

    # Message patterns
    message_type_distribution: Dict[str, float]  # Type -> percentage
    response_time_stats: Dict[str, float]  # min, max, mean, median
    qso_success_rate: float  # CQs that lead to QSOs

    # Predictive metrics
    next_active_probability: Dict[int, float]  # Hour -> probability
    expected_next_band: Optional[str] = None
    confidence_score: float = 0.0


@dataclass
class BandTransition:
    """Band change pattern."""

    from_band: str
    to_band: str
    count: int
    avg_time_on_band_min: float
    typical_hour_utc: int


class StationPatternAnalyzer:
    """Analyzes station operating patterns from observation history."""

    def __init__(self):
        """Initialize pattern analyzer."""
        self.patterns: Dict[str, ActivityPattern] = {}
        self.observations: Dict[str, List[Dict]] = defaultdict(list)
        self.band_transitions: Dict[str, List[BandTransition]] = defaultdict(list)

    def add_observation(self, observation: Dict):
        """Add a station observation for pattern analysis.

        Args:
            observation: Dict with keys:
                - station_hash: Anonymized station ID
                - timestamp: Observation time
                - band: Operating band
                - message_type: CQ, QSO, etc.
                - snr: Signal strength
                - grid: Grid square
        """
        station_hash = observation.get('station_hash')
        if not station_hash:
            return

        self.observations[station_hash].append(observation)

        # Trigger pattern analysis if enough data
        if len(self.observations[station_hash]) % 100 == 0:
            self.analyze_station(station_hash)

    def analyze_station(self, station_hash: str) -> Optional[ActivityPattern]:
        """Analyze patterns for a specific station.

        Args:
            station_hash: Station identifier

        Returns:
            ActivityPattern or None if insufficient data
        """
        obs = self.observations.get(station_hash, [])
        if len(obs) < 10:
            return None

        # Sort observations by timestamp
        obs = sorted(obs, key=lambda x: x['timestamp'])

        # Determine analysis period
        first_time = self._parse_timestamp(obs[0]['timestamp'])
        last_time = self._parse_timestamp(obs[-1]['timestamp'])

        # Analyze temporal patterns
        hour_probs, active_hours = self._analyze_hourly_activity(obs)
        day_probs, active_days = self._analyze_daily_activity(obs)

        # Analyze band usage
        band_usage = self._analyze_band_usage(obs)
        band_sequences = self._analyze_band_sequences(obs)
        bands_by_hour = self._analyze_band_by_hour(obs)

        # Analyze operating patterns
        session_duration = self._calculate_session_duration(obs)
        qso_duration = self._calculate_qso_duration(obs)
        cq_interval = self._calculate_cq_interval(obs)
        duty_cycle = self._calculate_duty_cycle(obs, first_time, last_time)

        # Analyze message patterns
        msg_distribution = self._analyze_message_types(obs)
        response_stats = self._analyze_response_times(obs)
        qso_success = self._calculate_qso_success_rate(obs)

        # Generate predictions
        next_active = self._predict_next_activity(obs, hour_probs)
        next_band = self._predict_next_band(obs, band_sequences)

        # Calculate confidence
        confidence = self._calculate_confidence(obs, first_time, last_time)

        pattern = ActivityPattern(
            station_hash=station_hash,
            analysis_period=(first_time, last_time),
            active_hours_utc=active_hours,
            hour_probabilities=hour_probs,
            active_days=active_days,
            day_probabilities=day_probs,
            band_usage=band_usage,
            band_sequence_patterns=band_sequences,
            preferred_bands_by_hour=bands_by_hour,
            avg_session_duration_min=session_duration,
            avg_qso_duration_min=qso_duration,
            avg_cq_interval_min=cq_interval,
            duty_cycle_percent=duty_cycle,
            message_type_distribution=msg_distribution,
            response_time_stats=response_stats,
            qso_success_rate=qso_success,
            next_active_probability=next_active,
            expected_next_band=next_band,
            confidence_score=confidence
        )

        self.patterns[station_hash] = pattern
        return pattern

    def _parse_timestamp(self, ts) -> datetime:
        """Parse timestamp to datetime."""
        if isinstance(ts, str):
            return datetime.fromisoformat(ts)
        return ts

    def _analyze_hourly_activity(self, observations: List[Dict]) -> Tuple[Dict[int, float], List[int]]:
        """Analyze hourly activity patterns.

        Returns:
            Tuple of (hour_probabilities, most_active_hours)
        """
        hour_counts = Counter()

        for obs in observations:
            ts = self._parse_timestamp(obs['timestamp'])
            hour_counts[ts.hour] += 1

        total = sum(hour_counts.values())
        hour_probs = {h: count/total for h, count in hour_counts.items()}

        # Find most active hours (top 25%)
        threshold = np.percentile(list(hour_counts.values()), 75) if hour_counts else 0
        active_hours = [h for h, count in hour_counts.items() if count >= threshold]

        return hour_probs, sorted(active_hours)

    def _analyze_daily_activity(self, observations: List[Dict]) -> Tuple[Dict[int, float], List[int]]:
        """Analyze daily activity patterns.

        Returns:
            Tuple of (day_probabilities, most_active_days)
        """
        day_counts = Counter()

        for obs in observations:
            ts = self._parse_timestamp(obs['timestamp'])
            day_counts[ts.weekday()] += 1

        total = sum(day_counts.values())
        day_probs = {d: count/total for d, count in day_counts.items()}

        # Find most active days
        threshold = np.mean(list(day_counts.values())) if day_counts else 0
        active_days = [d for d, count in day_counts.items() if count >= threshold]

        return day_probs, sorted(active_days)

    def _analyze_band_usage(self, observations: List[Dict]) -> Dict[str, int]:
        """Analyze band usage patterns."""
        band_counts = Counter()

        for obs in observations:
            band = obs.get('band', 'unknown')
            band_counts[band] += 1

        return dict(band_counts)

    def _analyze_band_sequences(self, observations: List[Dict]) -> List[Tuple[str, str, int]]:
        """Analyze common band transition sequences."""
        transitions = []

        for i in range(1, len(observations)):
            prev_band = observations[i-1].get('band')
            curr_band = observations[i].get('band')

            if prev_band and curr_band and prev_band != curr_band:
                transitions.append((prev_band, curr_band))

        # Count transition frequencies
        transition_counts = Counter(transitions)

        # Return top 5 most common transitions
        return [(t[0], t[1], count) for t, count in transition_counts.most_common(5)]

    def _analyze_band_by_hour(self, observations: List[Dict]) -> Dict[int, str]:
        """Determine preferred band for each hour."""
        hour_band_counts = defaultdict(Counter)

        for obs in observations:
            ts = self._parse_timestamp(obs['timestamp'])
            band = obs.get('band', 'unknown')
            hour_band_counts[ts.hour][band] += 1

        # Find most used band per hour
        bands_by_hour = {}
        for hour, band_counts in hour_band_counts.items():
            if band_counts:
                bands_by_hour[hour] = band_counts.most_common(1)[0][0]

        return bands_by_hour

    def _calculate_session_duration(self, observations: List[Dict]) -> float:
        """Calculate average operating session duration."""
        if len(observations) < 2:
            return 0.0

        sessions = []
        session_start = self._parse_timestamp(observations[0]['timestamp'])
        last_time = session_start

        for obs in observations[1:]:
            curr_time = self._parse_timestamp(obs['timestamp'])
            time_gap = (curr_time - last_time).total_seconds() / 60  # minutes

            # Gap > 30 minutes indicates new session
            if time_gap > 30:
                session_duration = (last_time - session_start).total_seconds() / 60
                if session_duration > 0:
                    sessions.append(session_duration)
                session_start = curr_time

            last_time = curr_time

        # Add final session
        final_duration = (last_time - session_start).total_seconds() / 60
        if final_duration > 0:
            sessions.append(final_duration)

        return np.mean(sessions) if sessions else 0.0

    def _calculate_qso_duration(self, observations: List[Dict]) -> float:
        """Calculate average QSO duration."""
        qso_durations = []
        qso_start = None

        for obs in observations:
            msg_type = obs.get('message_type', '')

            if msg_type == 'QSO' and qso_start is None:
                qso_start = self._parse_timestamp(obs['timestamp'])
            elif msg_type != 'QSO' and qso_start is not None:
                qso_end = self._parse_timestamp(obs['timestamp'])
                duration = (qso_end - qso_start).total_seconds() / 60
                qso_durations.append(duration)
                qso_start = None

        return np.mean(qso_durations) if qso_durations else 0.0

    def _calculate_cq_interval(self, observations: List[Dict]) -> float:
        """Calculate average time between CQ calls."""
        cq_times = []

        for obs in observations:
            if obs.get('message_type') == 'CQ':
                cq_times.append(self._parse_timestamp(obs['timestamp']))

        if len(cq_times) < 2:
            return 0.0

        intervals = []
        for i in range(1, len(cq_times)):
            interval = (cq_times[i] - cq_times[i-1]).total_seconds() / 60
            intervals.append(interval)

        return np.mean(intervals) if intervals else 0.0

    def _calculate_duty_cycle(self, observations: List[Dict],
                             first_time: datetime, last_time: datetime) -> float:
        """Calculate duty cycle percentage."""
        if not observations or last_time <= first_time:
            return 0.0

        # Count unique active hours
        active_hours = set()
        for obs in observations:
            ts = self._parse_timestamp(obs['timestamp'])
            active_hours.add((ts.date(), ts.hour))

        # Total possible hours
        total_hours = (last_time - first_time).total_seconds() / 3600

        if total_hours > 0:
            duty_cycle = (len(active_hours) / total_hours) * 100
            return min(100.0, duty_cycle)

        return 0.0

    def _analyze_message_types(self, observations: List[Dict]) -> Dict[str, float]:
        """Analyze message type distribution."""
        msg_counts = Counter()

        for obs in observations:
            msg_type = obs.get('message_type', 'unknown')
            msg_counts[msg_type] += 1

        total = sum(msg_counts.values())
        if total == 0:
            return {}

        return {msg_type: (count/total)*100 for msg_type, count in msg_counts.items()}

    def _analyze_response_times(self, observations: List[Dict]) -> Dict[str, float]:
        """Analyze response time statistics."""
        response_times = []
        last_cq = None

        for obs in observations:
            msg_type = obs.get('message_type')
            ts = self._parse_timestamp(obs['timestamp'])

            if msg_type == 'CQ':
                last_cq = ts
            elif msg_type == 'QSO' and last_cq:
                response_time = (ts - last_cq).total_seconds()
                response_times.append(response_time)
                last_cq = None

        if not response_times:
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0}

        return {
            'min': np.min(response_times),
            'max': np.max(response_times),
            'mean': np.mean(response_times),
            'median': np.median(response_times)
        }

    def _calculate_qso_success_rate(self, observations: List[Dict]) -> float:
        """Calculate percentage of CQs that lead to QSOs."""
        cq_count = 0
        successful_cqs = 0
        last_was_cq = False

        for obs in observations:
            msg_type = obs.get('message_type')

            if msg_type == 'CQ':
                cq_count += 1
                last_was_cq = True
            elif msg_type == 'QSO' and last_was_cq:
                successful_cqs += 1
                last_was_cq = False
            else:
                last_was_cq = False

        if cq_count == 0:
            return 0.0

        return (successful_cqs / cq_count) * 100

    def _predict_next_activity(self, observations: List[Dict],
                              hour_probs: Dict[int, float]) -> Dict[int, float]:
        """Predict probability of activity in next 24 hours."""
        if not observations or not hour_probs:
            return {}

        # Get current time from last observation
        last_obs = observations[-1]
        last_time = self._parse_timestamp(last_obs['timestamp'])

        # Generate predictions for next 24 hours
        predictions = {}
        for hours_ahead in range(24):
            future_hour = (last_time.hour + hours_ahead) % 24
            base_prob = hour_probs.get(future_hour, 0)

            # Adjust based on recent activity
            recent_activity = self._get_recent_activity_factor(observations, hours_ahead)
            predictions[hours_ahead] = min(1.0, base_prob * recent_activity)

        return predictions

    def _get_recent_activity_factor(self, observations: List[Dict], hours_ahead: int) -> float:
        """Calculate activity factor based on recent observations."""
        if not observations:
            return 1.0

        # Check activity in last 24 hours
        last_time = self._parse_timestamp(observations[-1]['timestamp'])
        cutoff = last_time - timedelta(hours=24)

        recent_count = sum(1 for obs in observations
                         if self._parse_timestamp(obs['timestamp']) >= cutoff)

        # Decay factor based on time ahead
        decay = np.exp(-hours_ahead / 12)  # 12-hour half-life

        # Activity factor
        if recent_count > 10:
            return 1.5 * decay  # Recently very active
        elif recent_count > 5:
            return 1.2 * decay  # Recently active
        else:
            return 0.8 * decay  # Recently quiet

    def _predict_next_band(self, observations: List[Dict],
                          band_sequences: List[Tuple[str, str, int]]) -> Optional[str]:
        """Predict most likely next band."""
        if not observations:
            return None

        last_band = observations[-1].get('band')
        if not last_band or not band_sequences:
            return None

        # Find transitions from current band
        next_bands = {}
        for from_band, to_band, count in band_sequences:
            if from_band == last_band:
                next_bands[to_band] = count

        if next_bands:
            # Return most likely transition
            return max(next_bands, key=next_bands.get)

        return None

    def _calculate_confidence(self, observations: List[Dict],
                             first_time: datetime, last_time: datetime) -> float:
        """Calculate confidence score for pattern analysis."""
        # Factors affecting confidence
        obs_count = len(observations)
        time_span_days = (last_time - first_time).days

        # More observations = higher confidence
        obs_confidence = min(1.0, obs_count / 1000)

        # Longer time span = higher confidence
        time_confidence = min(1.0, time_span_days / 30)

        # Data recency
        last_obs_time = self._parse_timestamp(observations[-1]['timestamp'])
        days_since_last = (datetime.now() - last_obs_time).days
        recency_factor = max(0.5, 1.0 - days_since_last / 30)

        # Combined confidence
        confidence = (obs_confidence * 0.5 +
                     time_confidence * 0.3 +
                     recency_factor * 0.2)

        return confidence

    def export_patterns(self, output_path: str):
        """Export all station patterns to JSON.

        Args:
            output_path: Output file path
        """
        export_data = {}

        for station_hash, pattern in self.patterns.items():
            export_data[station_hash] = {
                'analysis_period': [
                    pattern.analysis_period[0].isoformat(),
                    pattern.analysis_period[1].isoformat()
                ],
                'active_hours_utc': pattern.active_hours_utc,
                'band_usage': pattern.band_usage,
                'duty_cycle_percent': pattern.duty_cycle_percent,
                'message_type_distribution': pattern.message_type_distribution,
                'qso_success_rate': pattern.qso_success_rate,
                'confidence_score': pattern.confidence_score
            }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(export_data)} station patterns to {output_path}")

    def get_active_stations(self, hour_utc: int) -> List[str]:
        """Get stations likely to be active at given hour.

        Args:
            hour_utc: Hour in UTC [0-23]

        Returns:
            List of station hashes likely active
        """
        active = []

        for station_hash, pattern in self.patterns.items():
            if hour_utc in pattern.active_hours_utc:
                prob = pattern.hour_probabilities.get(hour_utc, 0)
                if prob > 0.3:  # 30% probability threshold
                    active.append(station_hash)

        return active