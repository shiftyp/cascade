"""Station fingerprint extraction from signal history.

T072: Extract unique station characteristics while preserving privacy.
Uses hashed callsigns to build anonymous station profiles.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class StationFingerprint:
    """Anonymous station fingerprint based on signal characteristics."""

    station_hash: str  # Anonymized station ID
    first_seen: datetime
    last_seen: datetime
    total_observations: int

    # Frequency characteristics
    primary_bands: List[str]
    frequency_stability_ppm: float
    frequency_drift_hz_per_min: float

    # Signal characteristics
    avg_snr_db: float
    snr_variance: float
    typical_power_dbm: float

    # Timing patterns
    active_hours_utc: List[int]  # Hours when station is typically active
    active_days: List[int]  # Days of week (0=Monday)
    duty_cycle: float  # Percentage of time active

    # Technical signature
    phase_noise_db: Optional[float] = None
    imd3_db: Optional[float] = None  # 3rd order intermodulation
    keying_profile: Optional[Dict[str, float]] = None
    modulation_quality: Optional[float] = None

    # Geographic consistency
    grid_squares: List[str] = field(default_factory=list)
    primary_grid: Optional[str] = None

    # Behavioral patterns
    message_types: Dict[str, int] = field(default_factory=dict)
    qso_duration_avg_min: Optional[float] = None
    response_time_avg_sec: Optional[float] = None

    # Propagation characteristics
    typical_paths: List[Tuple[str, str]] = field(default_factory=list)  # (rx_hash, distance_km)
    max_distance_km: float = 0
    median_distance_km: float = 0


class StationFingerprintExtractor:
    """Extracts and maintains anonymous station fingerprints."""

    def __init__(self, salt: str = "cascade_station_salt"):
        """Initialize fingerprint extractor.

        Args:
            salt: Cryptographic salt for consistent hashing
        """
        self.salt = salt
        self.fingerprints: Dict[str, StationFingerprint] = {}
        self.observation_buffer: Dict[str, List[Dict]] = defaultdict(list)
        self.min_observations = 10  # Minimum observations before creating fingerprint

    def process_signal(self, signal_data: Dict[str, Any]) -> Optional[str]:
        """Process a signal observation and update fingerprints.

        Args:
            signal_data: Signal observation data including:
                - callsign_hash: Already anonymized callsign
                - frequency: Operating frequency
                - snr: Signal-to-noise ratio
                - timestamp: Observation time
                - grid: Grid square (preserved)
                - message_type: Type of message (CQ, QSO, etc.)

        Returns:
            Station hash if fingerprint was updated
        """
        station_hash = signal_data.get('callsign_hash')
        if not station_hash:
            return None

        # Buffer observations
        self.observation_buffer[station_hash].append(signal_data)

        # Check if we have enough observations
        if len(self.observation_buffer[station_hash]) >= self.min_observations:
            self._update_fingerprint(station_hash)
            return station_hash

        return None

    def _update_fingerprint(self, station_hash: str):
        """Update or create fingerprint from buffered observations.

        Args:
            station_hash: Anonymous station identifier
        """
        observations = self.observation_buffer[station_hash]
        if not observations:
            return

        # Get or create fingerprint
        if station_hash in self.fingerprints:
            fp = self.fingerprints[station_hash]
        else:
            fp = self._create_initial_fingerprint(station_hash, observations[0])
            self.fingerprints[station_hash] = fp

        # Update with all observations
        for obs in observations:
            self._update_with_observation(fp, obs)

        # Clear processed observations (keep last few for continuity)
        self.observation_buffer[station_hash] = observations[-5:]

        # Calculate derived metrics
        self._calculate_derived_metrics(fp)

        logger.debug(f"Updated fingerprint for station {station_hash[:8]}...")

    def _create_initial_fingerprint(self, station_hash: str, first_obs: Dict) -> StationFingerprint:
        """Create initial fingerprint from first observation.

        Args:
            station_hash: Anonymous station ID
            first_obs: First observation data

        Returns:
            New StationFingerprint
        """
        timestamp = datetime.fromisoformat(first_obs['timestamp']) if isinstance(
            first_obs['timestamp'], str) else first_obs['timestamp']

        return StationFingerprint(
            station_hash=station_hash,
            first_seen=timestamp,
            last_seen=timestamp,
            total_observations=0,
            primary_bands=[self._freq_to_band(first_obs.get('frequency', 0))],
            frequency_stability_ppm=0,
            frequency_drift_hz_per_min=0,
            avg_snr_db=first_obs.get('snr', 0),
            snr_variance=0,
            typical_power_dbm=first_obs.get('power', 0),
            active_hours_utc=[],
            active_days=[],
            duty_cycle=0
        )

    def _update_with_observation(self, fp: StationFingerprint, obs: Dict):
        """Update fingerprint with new observation.

        Args:
            fp: StationFingerprint to update
            obs: Observation data
        """
        fp.total_observations += 1

        # Update timestamps
        timestamp = datetime.fromisoformat(obs['timestamp']) if isinstance(
            obs['timestamp'], str) else obs['timestamp']
        fp.last_seen = max(fp.last_seen, timestamp)

        # Update frequency data
        band = self._freq_to_band(obs.get('frequency', 0))
        if band and band not in fp.primary_bands:
            fp.primary_bands.append(band)

        # Update SNR (running average)
        if 'snr' in obs:
            alpha = 0.1  # Exponential moving average factor
            fp.avg_snr_db = (1 - alpha) * fp.avg_snr_db + alpha * obs['snr']

        # Track activity patterns
        hour = timestamp.hour
        if hour not in fp.active_hours_utc:
            fp.active_hours_utc.append(hour)

        day = timestamp.weekday()
        if day not in fp.active_days:
            fp.active_days.append(day)

        # Track grid squares
        if 'grid' in obs and obs['grid']:
            if obs['grid'] not in fp.grid_squares:
                fp.grid_squares.append(obs['grid'])

        # Track message types
        msg_type = obs.get('message_type', 'unknown')
        fp.message_types[msg_type] = fp.message_types.get(msg_type, 0) + 1

    def _calculate_derived_metrics(self, fp: StationFingerprint):
        """Calculate derived metrics for fingerprint.

        Args:
            fp: StationFingerprint to update
        """
        # Calculate duty cycle
        if fp.first_seen and fp.last_seen:
            total_hours = (fp.last_seen - fp.first_seen).total_seconds() / 3600
            if total_hours > 0:
                active_hours = len(fp.active_hours_utc)
                fp.duty_cycle = min(100, (active_hours / total_hours) * 100)

        # Determine primary grid
        if fp.grid_squares:
            # Most common grid
            from collections import Counter
            grid_counts = Counter(fp.grid_squares)
            fp.primary_grid = grid_counts.most_common(1)[0][0]

        # Sort activity patterns
        fp.active_hours_utc.sort()
        fp.active_days.sort()

    def _freq_to_band(self, frequency: float) -> str:
        """Convert frequency to amateur band.

        Args:
            frequency: Frequency in Hz

        Returns:
            Band name (e.g., "20m", "40m")
        """
        freq_mhz = frequency / 1_000_000

        bands = {
            (1.8, 2.0): "160m",
            (3.5, 4.0): "80m",
            (7.0, 7.3): "40m",
            (10.1, 10.15): "30m",
            (14.0, 14.35): "20m",
            (18.068, 18.168): "17m",
            (21.0, 21.45): "15m",
            (24.89, 24.99): "12m",
            (28.0, 29.7): "10m",
            (50.0, 54.0): "6m"
        }

        for (low, high), band in bands.items():
            if low <= freq_mhz <= high:
                return band
        return "unknown"

    def extract_equipment_signature(self, iq_samples: np.ndarray,
                                   station_hash: str,
                                   sample_rate: int) -> Dict[str, float]:
        """Extract equipment-specific characteristics from IQ samples.

        Args:
            iq_samples: Complex IQ samples
            station_hash: Station identifier
            sample_rate: Sample rate in Hz

        Returns:
            Equipment signature metrics
        """
        signature = {}

        # Calculate phase noise
        phase = np.unwrap(np.angle(iq_samples))
        phase_diff = np.diff(phase)
        signature['phase_noise_db'] = 10 * np.log10(np.var(phase_diff) + 1e-10)

        # Estimate frequency stability
        inst_freq = np.diff(phase) * sample_rate / (2 * np.pi)
        signature['freq_stability_hz'] = np.std(inst_freq)

        # Detect keying characteristics (for CW/FSK)
        envelope = np.abs(iq_samples)
        edges = np.diff(envelope > 0.5 * np.max(envelope))
        rise_times = []
        fall_times = []

        # Simple edge detection
        for i in range(1, len(edges)-1):
            if edges[i] > edges[i-1]:  # Rising edge
                rise_times.append(i)
            elif edges[i] < edges[i-1]:  # Falling edge
                fall_times.append(i)

        if rise_times:
            signature['avg_rise_time_ms'] = np.mean(rise_times) * 1000 / sample_rate
        if fall_times:
            signature['avg_fall_time_ms'] = np.mean(fall_times) * 1000 / sample_rate

        # Update fingerprint if exists
        if station_hash in self.fingerprints:
            fp = self.fingerprints[station_hash]
            fp.phase_noise_db = signature.get('phase_noise_db')
            fp.keying_profile = {
                'rise_time': signature.get('avg_rise_time_ms', 0),
                'fall_time': signature.get('avg_fall_time_ms', 0)
            }

        return signature

    def get_fingerprint(self, station_hash: str) -> Optional[StationFingerprint]:
        """Get fingerprint for a station.

        Args:
            station_hash: Anonymous station ID

        Returns:
            StationFingerprint or None
        """
        return self.fingerprints.get(station_hash)

    def find_similar_stations(self, station_hash: str,
                            threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find stations with similar characteristics.

        Args:
            station_hash: Reference station
            threshold: Similarity threshold (0-1)

        Returns:
            List of (station_hash, similarity_score) tuples
        """
        reference = self.fingerprints.get(station_hash)
        if not reference:
            return []

        similar = []

        for other_hash, other_fp in self.fingerprints.items():
            if other_hash == station_hash:
                continue

            similarity = self._calculate_similarity(reference, other_fp)
            if similarity >= threshold:
                similar.append((other_hash, similarity))

        return sorted(similar, key=lambda x: x[1], reverse=True)

    def _calculate_similarity(self, fp1: StationFingerprint,
                            fp2: StationFingerprint) -> float:
        """Calculate similarity between two fingerprints.

        Args:
            fp1, fp2: Fingerprints to compare

        Returns:
            Similarity score (0-1)
        """
        scores = []

        # Band overlap
        band_overlap = len(set(fp1.primary_bands) & set(fp2.primary_bands))
        band_union = len(set(fp1.primary_bands) | set(fp2.primary_bands))
        if band_union > 0:
            scores.append(band_overlap / band_union)

        # SNR similarity (normalized difference)
        snr_diff = abs(fp1.avg_snr_db - fp2.avg_snr_db)
        scores.append(max(0, 1 - snr_diff / 20))  # 20 dB range

        # Activity pattern overlap
        hour_overlap = len(set(fp1.active_hours_utc) & set(fp2.active_hours_utc))
        hour_union = len(set(fp1.active_hours_utc) | set(fp2.active_hours_utc))
        if hour_union > 0:
            scores.append(hour_overlap / hour_union)

        # Grid square similarity
        if fp1.primary_grid and fp2.primary_grid:
            # Simple check: same or adjacent grid
            if fp1.primary_grid == fp2.primary_grid:
                scores.append(1.0)
            elif abs(ord(fp1.primary_grid[0]) - ord(fp2.primary_grid[0])) <= 1:
                scores.append(0.5)  # Adjacent
            else:
                scores.append(0.0)

        return np.mean(scores) if scores else 0.0

    def export_fingerprints(self, output_path: str):
        """Export fingerprints to JSON file.

        Args:
            output_path: Output file path
        """
        export_data = {}

        for station_hash, fp in self.fingerprints.items():
            export_data[station_hash] = {
                'first_seen': fp.first_seen.isoformat(),
                'last_seen': fp.last_seen.isoformat(),
                'total_observations': fp.total_observations,
                'primary_bands': fp.primary_bands,
                'avg_snr_db': fp.avg_snr_db,
                'active_hours_utc': fp.active_hours_utc,
                'primary_grid': fp.primary_grid,
                'duty_cycle': fp.duty_cycle,
                'message_types': fp.message_types
            }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(export_data)} station fingerprints to {output_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected fingerprints.

        Returns:
            Statistics dictionary
        """
        if not self.fingerprints:
            return {'total_stations': 0}

        total_obs = sum(fp.total_observations for fp in self.fingerprints.values())
        avg_obs = total_obs / len(self.fingerprints) if self.fingerprints else 0

        band_counts = defaultdict(int)
        for fp in self.fingerprints.values():
            for band in fp.primary_bands:
                band_counts[band] += 1

        return {
            'total_stations': len(self.fingerprints),
            'total_observations': total_obs,
            'avg_observations_per_station': avg_obs,
            'stations_in_buffer': len(self.observation_buffer),
            'band_distribution': dict(band_counts),
            'avg_duty_cycle': np.mean([fp.duty_cycle for fp in self.fingerprints.values()])
        }