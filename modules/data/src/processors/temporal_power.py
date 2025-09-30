"""Temporal power pattern tracking for station behavior analysis.

T099: Detect time-of-day power adjustment patterns, contest vs casual operation,
and build station-specific power profiles over time.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from collections import defaultdict
from scipy import signal
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)


@dataclass
class PowerObservation:
    """Single power observation at a point in time."""

    station_hash: str
    timestamp: datetime
    estimated_power_dbm: float
    confidence: float

    # Context
    band: str
    is_contest: bool
    is_weekend: bool
    solar_state: str  # 'day', 'night', 'sunrise', 'sunset'


@dataclass
class TemporalPowerProfile:
    """Station's temporal power usage pattern."""

    station_hash: str
    analysis_period: Tuple[datetime, datetime]

    # Hourly patterns (UTC)
    hourly_avg_power: Dict[int, float]  # hour -> avg power
    hourly_std_power: Dict[int, float]  # hour -> std deviation

    # Day of week patterns
    weekday_avg_power: float
    weekend_avg_power: float

    # Contest patterns
    contest_avg_power: float
    casual_avg_power: float
    contest_power_boost_db: float

    # Solar patterns
    daytime_avg_power: float
    nighttime_avg_power: float
    sunrise_sunset_avg_power: float

    # Detected patterns
    has_time_schedule: bool  # Consistent time-based changes
    has_contest_mode: bool   # Different power during contests
    has_solar_tracking: bool # Follows day/night cycle

    # Power switching behavior
    distinct_power_levels: List[float]  # Detected discrete levels
    power_switch_times: List[time]      # Common switching times

    confidence_score: float


class TemporalPowerTracker:
    """Tracks and analyzes temporal power patterns."""

    def __init__(self):
        """Initialize temporal power tracker."""
        self.observations: Dict[str, List[PowerObservation]] = defaultdict(list)
        self.profiles: Dict[str, TemporalPowerProfile] = {}

        # Analysis parameters
        self.min_observations_for_profile = 50
        self.power_level_threshold_db = 3  # Min difference for distinct level
        self.schedule_consistency_threshold = 0.7

    def add_observation(self, observation: PowerObservation):
        """Add a power observation.

        Args:
            observation: Power observation to track
        """
        self.observations[observation.station_hash].append(observation)

    def analyze_station(self, station_hash: str,
                       min_days: int = 7) -> Optional[TemporalPowerProfile]:
        """Analyze temporal power patterns for a station.

        Args:
            station_hash: Station to analyze
            min_days: Minimum days of data required

        Returns:
            TemporalPowerProfile or None if insufficient data
        """
        if station_hash not in self.observations:
            return None

        obs = self.observations[station_hash]

        if len(obs) < self.min_observations_for_profile:
            logger.debug(f"Insufficient observations for {station_hash}: {len(obs)}")
            return None

        # Check time span
        timestamps = [o.timestamp for o in obs]
        time_span = max(timestamps) - min(timestamps)

        if time_span.days < min_days:
            logger.debug(f"Insufficient time span for {station_hash}: {time_span.days} days")
            return None

        # Analyze hourly patterns
        hourly_avg, hourly_std = self._analyze_hourly_patterns(obs)

        # Analyze day of week patterns
        weekday_avg, weekend_avg = self._analyze_weekday_patterns(obs)

        # Analyze contest patterns
        contest_avg, casual_avg, contest_boost = self._analyze_contest_patterns(obs)

        # Analyze solar patterns
        day_avg, night_avg, twilight_avg = self._analyze_solar_patterns(obs)

        # Detect discrete power levels
        power_levels = self._detect_power_levels(obs)

        # Detect switching times
        switch_times = self._detect_switch_times(obs)

        # Determine pattern flags
        has_schedule = self._has_time_schedule(hourly_avg, hourly_std)
        has_contest = abs(contest_boost) > self.power_level_threshold_db
        has_solar = self._has_solar_tracking(day_avg, night_avg)

        # Calculate confidence
        confidence = self._calculate_confidence(obs, hourly_std)

        profile = TemporalPowerProfile(
            station_hash=station_hash,
            analysis_period=(min(timestamps), max(timestamps)),
            hourly_avg_power=hourly_avg,
            hourly_std_power=hourly_std,
            weekday_avg_power=weekday_avg,
            weekend_avg_power=weekend_avg,
            contest_avg_power=contest_avg,
            casual_avg_power=casual_avg,
            contest_power_boost_db=contest_boost,
            daytime_avg_power=day_avg,
            nighttime_avg_power=night_avg,
            sunrise_sunset_avg_power=twilight_avg,
            has_time_schedule=has_schedule,
            has_contest_mode=has_contest,
            has_solar_tracking=has_solar,
            distinct_power_levels=power_levels,
            power_switch_times=switch_times,
            confidence_score=confidence
        )

        self.profiles[station_hash] = profile
        return profile

    def _analyze_hourly_patterns(self, observations: List[PowerObservation]) -> Tuple[Dict, Dict]:
        """Analyze hourly power patterns.

        Args:
            observations: List of observations

        Returns:
            Tuple of (hourly_averages, hourly_std_devs)
        """
        hourly_powers = defaultdict(list)

        for obs in observations:
            hour = obs.timestamp.hour
            hourly_powers[hour].append(obs.estimated_power_dbm)

        hourly_avg = {}
        hourly_std = {}

        for hour in range(24):
            if hour in hourly_powers and hourly_powers[hour]:
                powers = hourly_powers[hour]
                hourly_avg[hour] = np.mean(powers)
                hourly_std[hour] = np.std(powers)
            else:
                # Interpolate missing hours
                prev_hour = (hour - 1) % 24
                next_hour = (hour + 1) % 24

                if prev_hour in hourly_avg and next_hour in hourly_avg:
                    hourly_avg[hour] = (hourly_avg[prev_hour] + hourly_avg[next_hour]) / 2
                    hourly_std[hour] = (hourly_std[prev_hour] + hourly_std[next_hour]) / 2
                else:
                    hourly_avg[hour] = 30.0  # Default
                    hourly_std[hour] = 5.0

        return hourly_avg, hourly_std

    def _analyze_weekday_patterns(self, observations: List[PowerObservation]) -> Tuple[float, float]:
        """Analyze weekday vs weekend patterns.

        Args:
            observations: List of observations

        Returns:
            Tuple of (weekday_avg, weekend_avg)
        """
        weekday_powers = []
        weekend_powers = []

        for obs in observations:
            if obs.is_weekend:
                weekend_powers.append(obs.estimated_power_dbm)
            else:
                weekday_powers.append(obs.estimated_power_dbm)

        weekday_avg = np.mean(weekday_powers) if weekday_powers else 30.0
        weekend_avg = np.mean(weekend_powers) if weekend_powers else 30.0

        return weekday_avg, weekend_avg

    def _analyze_contest_patterns(self, observations: List[PowerObservation]) -> Tuple[float, float, float]:
        """Analyze contest vs casual operation patterns.

        Args:
            observations: List of observations

        Returns:
            Tuple of (contest_avg, casual_avg, boost_db)
        """
        contest_powers = []
        casual_powers = []

        for obs in observations:
            if obs.is_contest:
                contest_powers.append(obs.estimated_power_dbm)
            else:
                casual_powers.append(obs.estimated_power_dbm)

        if not contest_powers:
            contest_avg = 30.0
        else:
            contest_avg = np.mean(contest_powers)

        if not casual_powers:
            casual_avg = 30.0
        else:
            casual_avg = np.mean(casual_powers)

        boost = contest_avg - casual_avg

        return contest_avg, casual_avg, boost

    def _analyze_solar_patterns(self, observations: List[PowerObservation]) -> Tuple[float, float, float]:
        """Analyze solar-dependent power patterns.

        Args:
            observations: List of observations

        Returns:
            Tuple of (day_avg, night_avg, twilight_avg)
        """
        day_powers = []
        night_powers = []
        twilight_powers = []

        for obs in observations:
            if obs.solar_state == 'day':
                day_powers.append(obs.estimated_power_dbm)
            elif obs.solar_state == 'night':
                night_powers.append(obs.estimated_power_dbm)
            else:  # sunrise/sunset
                twilight_powers.append(obs.estimated_power_dbm)

        day_avg = np.mean(day_powers) if day_powers else 30.0
        night_avg = np.mean(night_powers) if night_powers else 30.0
        twilight_avg = np.mean(twilight_powers) if twilight_powers else 30.0

        return day_avg, night_avg, twilight_avg

    def _detect_power_levels(self, observations: List[PowerObservation]) -> List[float]:
        """Detect discrete power levels used by station.

        Args:
            observations: List of observations

        Returns:
            List of distinct power levels in dBm
        """
        powers = [obs.estimated_power_dbm for obs in observations]

        if len(powers) < 10:
            return []

        # Use kernel density estimation to find peaks
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(powers)
        x = np.linspace(min(powers), max(powers), 200)
        density = kde(x)

        # Find peaks in density
        peaks, properties = signal.find_peaks(density, height=0.01, distance=10)

        if len(peaks) == 0:
            return []

        # Get power levels at peaks
        power_levels = [x[peak] for peak in peaks]

        # Filter out close levels
        filtered_levels = []
        for level in sorted(power_levels):
            if not filtered_levels or level - filtered_levels[-1] >= self.power_level_threshold_db:
                filtered_levels.append(level)

        return filtered_levels

    def _detect_switch_times(self, observations: List[PowerObservation]) -> List[time]:
        """Detect common power switching times.

        Args:
            observations: List of observations

        Returns:
            List of common switching times
        """
        if len(observations) < 20:
            return []

        # Sort by time
        sorted_obs = sorted(observations, key=lambda x: x.timestamp)

        # Detect significant power changes
        switch_times = []
        for i in range(1, len(sorted_obs)):
            power_change = abs(sorted_obs[i].estimated_power_dbm -
                             sorted_obs[i-1].estimated_power_dbm)

            if power_change >= self.power_level_threshold_db:
                switch_time = sorted_obs[i].timestamp.time()
                switch_times.append(switch_time)

        if not switch_times:
            return []

        # Cluster switch times by hour
        hour_counts = defaultdict(int)
        for st in switch_times:
            hour_counts[st.hour] += 1

        # Find most common switch hours
        common_hours = [hour for hour, count in hour_counts.items()
                       if count >= len(switch_times) * 0.1]  # At least 10% of switches

        return [time(hour=h) for h in sorted(common_hours)]

    def _has_time_schedule(self, hourly_avg: Dict[int, float],
                          hourly_std: Dict[int, float]) -> bool:
        """Determine if station has consistent time-based schedule.

        Args:
            hourly_avg: Hourly average powers
            hourly_std: Hourly standard deviations

        Returns:
            True if consistent schedule detected
        """
        if not hourly_avg or not hourly_std:
            return False

        # Check for significant variation across hours
        powers = list(hourly_avg.values())
        hour_variation = np.std(powers)

        if hour_variation < 2:  # Less than 2 dB variation
            return False

        # Check for consistency (low std dev within hours)
        avg_hourly_std = np.mean(list(hourly_std.values()))

        # Schedule exists if: variation across hours > variation within hours
        return hour_variation > avg_hourly_std * 1.5

    def _has_solar_tracking(self, day_avg: float, night_avg: float) -> bool:
        """Determine if station follows solar cycle.

        Args:
            day_avg: Average daytime power
            night_avg: Average nighttime power

        Returns:
            True if solar tracking detected
        """
        return abs(day_avg - night_avg) >= self.power_level_threshold_db

    def _calculate_confidence(self, observations: List[PowerObservation],
                            hourly_std: Dict[int, float]) -> float:
        """Calculate confidence in temporal analysis.

        Args:
            observations: List of observations
            hourly_std: Hourly standard deviations

        Returns:
            Confidence score 0-1
        """
        # Factor 1: Number of observations
        obs_factor = min(1.0, len(observations) / 200)

        # Factor 2: Time coverage
        timestamps = [o.timestamp for o in observations]
        time_span = (max(timestamps) - min(timestamps)).days
        coverage_factor = min(1.0, time_span / 30)  # 30 days = full confidence

        # Factor 3: Consistency (low variance)
        if hourly_std:
            avg_std = np.mean(list(hourly_std.values()))
            consistency_factor = max(0.5, 1.0 - avg_std / 10)  # 10 dB std = 0.5 confidence
        else:
            consistency_factor = 0.5

        # Factor 4: Data distribution across hours
        hours_covered = len(set(o.timestamp.hour for o in observations))
        distribution_factor = hours_covered / 24

        # Weighted average
        confidence = (obs_factor * 0.25 +
                     coverage_factor * 0.25 +
                     consistency_factor * 0.25 +
                     distribution_factor * 0.25)

        return confidence

    def identify_operating_patterns(self) -> Dict[str, List[str]]:
        """Identify stations with specific operating patterns.

        Returns:
            Dictionary mapping pattern types to station lists
        """
        patterns = {
            'scheduled': [],      # Regular time-based schedule
            'contest': [],        # Contest power boost
            'solar': [],          # Solar-dependent
            'multi_level': [],    # Multiple discrete power levels
            'qrp_only': [],      # Always low power
            'qro_only': []       # Always high power
        }

        for station_hash, profile in self.profiles.items():
            if profile.has_time_schedule:
                patterns['scheduled'].append(station_hash)

            if profile.has_contest_mode:
                patterns['contest'].append(station_hash)

            if profile.has_solar_tracking:
                patterns['solar'].append(station_hash)

            if len(profile.distinct_power_levels) >= 3:
                patterns['multi_level'].append(station_hash)

            # Check for QRP/QRO only
            avg_power = np.mean(list(profile.hourly_avg_power.values()))
            if avg_power <= 37:  # 5W or less
                patterns['qrp_only'].append(station_hash)
            elif avg_power >= 57:  # 500W or more
                patterns['qro_only'].append(station_hash)

        return patterns

    def predict_power(self, station_hash: str, timestamp: datetime) -> Tuple[float, float]:
        """Predict likely power at given time based on patterns.

        Args:
            station_hash: Station to predict
            timestamp: Time to predict for

        Returns:
            Tuple of (predicted_power_dbm, uncertainty_db)
        """
        if station_hash not in self.profiles:
            return 30.0, 10.0  # Default with high uncertainty

        profile = self.profiles[station_hash]

        # Start with hourly average
        hour = timestamp.hour
        base_power = profile.hourly_avg_power.get(hour, 30.0)
        uncertainty = profile.hourly_std_power.get(hour, 5.0)

        # Adjust for day of week
        is_weekend = timestamp.weekday() >= 5
        if is_weekend and profile.weekend_avg_power != profile.weekday_avg_power:
            adjustment = profile.weekend_avg_power - profile.weekday_avg_power
            base_power += adjustment * 0.5  # Partial adjustment

        # Adjust for contest (would need contest calendar)
        # This is simplified - real implementation would check actual contest schedule

        # Adjust for solar state
        solar_state = self._get_solar_state(timestamp)
        if profile.has_solar_tracking:
            if solar_state == 'night':
                base_power = profile.nighttime_avg_power
            elif solar_state == 'day':
                base_power = profile.daytime_avg_power

        # Snap to nearest discrete level if applicable
        if profile.distinct_power_levels:
            distances = [abs(base_power - level) for level in profile.distinct_power_levels]
            nearest_idx = np.argmin(distances)
            if distances[nearest_idx] < 3:  # Within 3 dB
                base_power = profile.distinct_power_levels[nearest_idx]
                uncertainty *= 0.7  # More certain when at discrete level

        # Adjust uncertainty based on confidence
        uncertainty *= (2 - profile.confidence_score)  # Low confidence = higher uncertainty

        return base_power, uncertainty

    def _get_solar_state(self, timestamp: datetime) -> str:
        """Determine solar state for given time.

        Simplified - real implementation would use actual sunrise/sunset.
        """
        hour = timestamp.hour

        if 8 <= hour < 18:
            return 'day'
        elif 20 <= hour or hour < 6:
            return 'night'
        else:
            return 'twilight'

    def generate_report(self, station_hash: str) -> str:
        """Generate human-readable temporal analysis report.

        Args:
            station_hash: Station to report on

        Returns:
            Text report
        """
        if station_hash not in self.profiles:
            return f"No temporal profile for station {station_hash[:8]}..."

        profile = self.profiles[station_hash]

        report = f"Temporal Power Profile for {station_hash[:8]}...\n"
        report += "=" * 50 + "\n\n"

        # Time period
        start, end = profile.analysis_period
        report += f"Analysis Period: {start.date()} to {end.date()}\n"
        report += f"Confidence: {profile.confidence_score:.0%}\n\n"

        # Detected patterns
        report += "Detected Patterns:\n"
        if profile.has_time_schedule:
            report += "  ✓ Regular time-based schedule\n"
        if profile.has_contest_mode:
            report += f"  ✓ Contest mode (+{profile.contest_power_boost_db:.1f} dB)\n"
        if profile.has_solar_tracking:
            report += "  ✓ Solar-dependent operation\n"

        if not (profile.has_time_schedule or profile.has_contest_mode or profile.has_solar_tracking):
            report += "  No specific patterns detected\n"

        report += "\n"

        # Power levels
        if profile.distinct_power_levels:
            report += f"Discrete Power Levels: "
            watts = [10**((p-30)/10) for p in profile.distinct_power_levels]
            report += ", ".join([f"{w:.0f}W" for w in watts])
            report += "\n"

        # Time schedule
        if profile.has_time_schedule:
            report += "\nHourly Power Schedule (UTC):\n"
            for hour in range(0, 24, 3):  # Show every 3 hours
                power = profile.hourly_avg_power.get(hour, 0)
                watts = 10**((power-30)/10)
                report += f"  {hour:02d}:00: {power:.1f} dBm ({watts:.0f}W)\n"

        # Day/night difference
        if profile.has_solar_tracking:
            day_w = 10**((profile.daytime_avg_power-30)/10)
            night_w = 10**((profile.nighttime_avg_power-30)/10)
            report += f"\nSolar Pattern:\n"
            report += f"  Day:   {profile.daytime_avg_power:.1f} dBm ({day_w:.0f}W)\n"
            report += f"  Night: {profile.nighttime_avg_power:.1f} dBm ({night_w:.0f}W)\n"

        # Switching times
        if profile.power_switch_times:
            report += f"\nCommon Switch Times: "
            report += ", ".join([t.strftime("%H:00") for t in profile.power_switch_times])
            report += " UTC\n"

        return report

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall temporal analysis statistics.

        Returns:
            Statistics dictionary
        """
        if not self.profiles:
            return {'analyzed_stations': 0}

        patterns = self.identify_operating_patterns()

        avg_confidence = np.mean([p.confidence_score for p in self.profiles.values()])

        contest_stations = [p for p in self.profiles.values() if p.has_contest_mode]
        avg_contest_boost = np.mean([p.contest_power_boost_db for p in contest_stations]) if contest_stations else 0

        return {
            'analyzed_stations': len(self.profiles),
            'scheduled_operators': len(patterns['scheduled']),
            'contest_operators': len(patterns['contest']),
            'solar_dependent': len(patterns['solar']),
            'multi_level_operators': len(patterns['multi_level']),
            'qrp_only': len(patterns['qrp_only']),
            'qro_only': len(patterns['qro_only']),
            'avg_confidence': avg_confidence,
            'avg_contest_boost_db': avg_contest_boost
        }