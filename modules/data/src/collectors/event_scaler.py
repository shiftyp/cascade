"""
Event-based SDR scaling logic for dynamic collection adjustment
Implements FR-023, FR-024, FR-041
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import json

from ..models.space_weather_data import SpaceWeatherData
from ..models.collection_schedule import CollectionSchedule
from ..external.noaa_client import NOAASpaceWeatherClient as NOAAClient
from .sdr_manager import SDRManager
from .hybrid_sdr_selector import HybridSDRSelector
from ..config.redis_config import RedisKeys
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that trigger scaling"""
    GEOMAGNETIC_STORM = "geomagnetic_storm"
    SOLAR_FLARE = "solar_flare"
    PROTON_EVENT = "proton_event"
    AURORA = "aurora"
    GRAYLINE = "grayline"
    CONTEST = "contest"
    SOLAR_MINIMUM = "solar_minimum"
    EQUINOX = "equinox"


class ScalingMode(Enum):
    """Collection scaling modes"""
    BASELINE = "baseline"       # 6 stations (normal)
    ENHANCED = "enhanced"       # 12 stations (K≥5)
    STORM = "storm"            # 20+ stations (K≥7)
    MAXIMUM = "maximum"        # All available (X-class flare)
    MINIMUM = "minimum"        # 1-2 stations (resource conservation)


@dataclass
class ScalingEvent:
    """Represents a scaling event"""
    event_type: EventType
    severity: int  # 1-10 scale
    start_time: datetime
    end_time: Optional[datetime]
    affected_bands: List[str]
    scaling_mode: ScalingMode
    metadata: Dict


class EventScaler:
    """Event-based SDR scaling for space weather and propagation events"""
    """
    Manages dynamic SDR scaling based on space weather and propagation events
    """

    def __init__(self, db_session, redis_client: aioredis.Redis):
        self.db_session = db_session
        self.redis = redis_client
        self.noaa_client = NOAAClient()
        self.sdr_manager = SDRManager(db_session)
        self.hybrid_selector = HybridSDRSelector(db_session)

        # Scaling thresholds
        self.k_index_thresholds = {
            3: ScalingMode.BASELINE,
            5: ScalingMode.ENHANCED,
            7: ScalingMode.STORM,
            9: ScalingMode.MAXIMUM
        }

        self.xray_class_thresholds = {
            'C': ScalingMode.BASELINE,
            'M': ScalingMode.ENHANCED,
            'X': ScalingMode.MAXIMUM
        }

        # Station counts by mode
        self.station_counts = {
            ScalingMode.MINIMUM: 2,
            ScalingMode.BASELINE: 6,
            ScalingMode.ENHANCED: 12,
            ScalingMode.STORM: 20,
            ScalingMode.MAXIMUM: 50  # All available
        }

        self.current_mode = ScalingMode.BASELINE
        self.active_events: List[ScalingEvent] = []
        self.scaling_history: List[Dict] = []

    async def monitor_events(self):
        """
        Main event monitoring loop
        """
        logger.info("Starting event-based scaling monitor")

        while True:
            try:
                # Check multiple event sources
                events = await self._detect_events()

                # Determine required scaling
                required_mode = self._calculate_scaling_mode(events)

                # Apply scaling if needed
                if required_mode != self.current_mode:
                    await self._apply_scaling(required_mode, events)

                # Update active events
                self.active_events = events

                # Wait before next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Event monitoring error: {e}")
                await asyncio.sleep(300)  # Back off on error

    async def _detect_events(self) -> List[ScalingEvent]:
        """
        Detect all active events requiring scaling
        """
        events = []
        now = datetime.utcnow()

        # Check space weather
        space_weather = await self._get_latest_space_weather()
        if space_weather:
            # Geomagnetic storms
            if space_weather.k_index >= 5:
                events.append(ScalingEvent(
                    event_type=EventType.GEOMAGNETIC_STORM,
                    severity=space_weather.k_index,
                    start_time=space_weather.timestamp,
                    end_time=None,
                    affected_bands=self._get_storm_affected_bands(space_weather.k_index),
                    scaling_mode=self._k_index_to_mode(space_weather.k_index),
                    metadata={'k_index': space_weather.k_index, 'ap_index': space_weather.ap_index}
                ))

            # Solar flares
            if space_weather.xray_class and space_weather.xray_class[0] in ['M', 'X']:
                events.append(ScalingEvent(
                    event_type=EventType.SOLAR_FLARE,
                    severity=self._xray_class_to_severity(space_weather.xray_class),
                    start_time=space_weather.timestamp,
                    end_time=space_weather.timestamp + timedelta(hours=2),
                    affected_bands=['10m', '15m', '20m'],  # Higher bands affected
                    scaling_mode=self._xray_class_to_mode(space_weather.xray_class),
                    metadata={'xray_class': space_weather.xray_class, 'xray_flux': space_weather.xray_flux}
                ))

            # Aurora activity
            if space_weather.aurora_power and space_weather.aurora_power > 50:
                events.append(ScalingEvent(
                    event_type=EventType.AURORA,
                    severity=min(10, space_weather.aurora_power // 10),
                    start_time=space_weather.timestamp,
                    end_time=None,
                    affected_bands=['6m', '10m'],  # VHF/low HF
                    scaling_mode=ScalingMode.ENHANCED,
                    metadata={'aurora_power': space_weather.aurora_power}
                ))

            # Solar minimum detection
            if space_weather.solar_flux_index < 70:
                events.append(ScalingEvent(
                    event_type=EventType.SOLAR_MINIMUM,
                    severity=3,
                    start_time=now,
                    end_time=None,
                    affected_bands=['80m', '40m'],  # Lower bands better during minimum
                    scaling_mode=ScalingMode.ENHANCED,
                    metadata={'sfi': space_weather.solar_flux_index}
                ))

        # Check gray-line propagation
        grayline_event = await self._check_grayline()
        if grayline_event:
            events.append(grayline_event)

        # Check contest calendar
        contest_event = await self._check_contests()
        if contest_event:
            events.append(contest_event)

        # Check equinoctial enhancement (March/September)
        if now.month in [3, 9] and 10 <= now.day <= 25:
            events.append(ScalingEvent(
                event_type=EventType.EQUINOX,
                severity=4,
                start_time=now.replace(hour=0, minute=0, second=0),
                end_time=now.replace(hour=23, minute=59, second=59),
                affected_bands=['20m', '15m', '10m'],
                scaling_mode=ScalingMode.ENHANCED,
                metadata={'equinox_type': 'vernal' if now.month == 3 else 'autumnal'}
            ))

        return events

    async def _get_latest_space_weather(self) -> Optional[SpaceWeatherData]:
        """
        Get latest space weather data from database or NOAA
        """
        # Check cache first
        cached = await self.redis.get(RedisKeys.CACHE_SPACE_WEATHER)
        if cached:
            data = json.loads(cached)
            # Convert to SpaceWeatherData object
            return SpaceWeatherData(**data)

        # Fetch from NOAA
        try:
            weather_data = await self.noaa_client.get_current_conditions()

            # Cache for 5 minutes
            await self.redis.setex(
                RedisKeys.CACHE_SPACE_WEATHER,
                300,
                json.dumps(weather_data)
            )

            return SpaceWeatherData(**weather_data)
        except Exception as e:
            logger.error(f"Failed to fetch space weather: {e}")
            return None

    async def _check_grayline(self) -> Optional[ScalingEvent]:
        """
        Check if gray-line propagation is active
        """
        from ..events.grayline import GraylineCalculator

        calculator = GraylineCalculator()
        grayline_sdrs = await calculator.get_grayline_sdrs()

        if len(grayline_sdrs) >= 2:
            return ScalingEvent(
                event_type=EventType.GRAYLINE,
                severity=5,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow() + timedelta(hours=2),
                affected_bands=['40m', '30m', '20m'],
                scaling_mode=ScalingMode.ENHANCED,
                metadata={'grayline_sdrs': [s.grid_square for s in grayline_sdrs]}
            )
        return None

    async def _check_contests(self) -> Optional[ScalingEvent]:
        """
        Check if major contest is active
        """
        from ..events.contest_calendar import ContestCalendar

        calendar = ContestCalendar()
        active_contest = await calendar.get_active_contest()

        if active_contest:
            return ScalingEvent(
                event_type=EventType.CONTEST,
                severity=6,
                start_time=active_contest['start'],
                end_time=active_contest['end'],
                affected_bands=active_contest['bands'],
                scaling_mode=ScalingMode.ENHANCED,
                metadata={'contest_name': active_contest['name']}
            )
        return None

    def _calculate_scaling_mode(self, events: List[ScalingEvent]) -> ScalingMode:
        """
        Determine required scaling mode from active events
        """
        if not events:
            return ScalingMode.BASELINE

        # Use highest severity mode
        modes = [e.scaling_mode for e in events]
        mode_priority = {
            ScalingMode.MINIMUM: 0,
            ScalingMode.BASELINE: 1,
            ScalingMode.ENHANCED: 2,
            ScalingMode.STORM: 3,
            ScalingMode.MAXIMUM: 4
        }

        highest_mode = max(modes, key=lambda m: mode_priority[m])
        return highest_mode

    async def _apply_scaling(self, new_mode: ScalingMode, events: List[ScalingEvent]):
        """
        Apply the scaling changes
        """
        logger.info(f"Scaling from {self.current_mode} to {new_mode}")

        # Calculate station delta
        current_count = self.station_counts[self.current_mode]
        target_count = self.station_counts[new_mode]
        delta = target_count - current_count

        if delta > 0:
            # Scale up
            await self._scale_up(delta, events)
        elif delta < 0:
            # Scale down
            await self._scale_down(abs(delta))

        # Update mode
        self.current_mode = new_mode

        # Record scaling event
        self.scaling_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'from_mode': self.current_mode.value,
            'to_mode': new_mode.value,
            'events': [e.event_type.value for e in events],
            'station_count': target_count
        })

        # Publish event for monitoring
        await self.redis.publish(
            RedisKeys.EVENT_SCALING,
            json.dumps({
                'mode': new_mode.value,
                'stations': target_count,
                'reason': [e.event_type.value for e in events]
            })
        )

    async def _scale_up(self, additional_stations: int, events: List[ScalingEvent]):
        """
        Scale up by adding more stations
        """
        logger.info(f"Scaling up: adding {additional_stations} stations")

        # Determine which bands need coverage
        affected_bands = set()
        for event in events:
            affected_bands.update(event.affected_bands)

        # Select additional SDRs
        new_sdrs = []
        for band in affected_bands:
            needed = min(3, additional_stations // len(affected_bands))

            # Use hybrid selector for optimal mix
            sdrs = await self.hybrid_selector.select_for_event(
                event_type=events[0].event_type.value if events else 'manual',
                k_index=events[0].metadata.get('k_index', 3) if events else 3,
                required_stations=needed
            )
            new_sdrs.extend(sdrs)

        # Start new recording sessions
        for sdr in new_sdrs[:additional_stations]:
            await self._start_recording_session(sdr)

    async def _scale_down(self, reduce_stations: int):
        """
        Scale down by stopping some stations
        """
        logger.info(f"Scaling down: removing {reduce_stations} stations")

        # Get current active sessions
        active_sessions = await self._get_active_sessions()

        # Stop lowest priority sessions
        sessions_to_stop = sorted(
            active_sessions,
            key=lambda s: s.priority
        )[:reduce_stations]

        for session in sessions_to_stop:
            await self._stop_recording_session(session)

    async def _start_recording_session(self, sdr):
        """
        Start a new recording session on an SDR
        """
        # Queue recording task
        task = {
            'sdr_id': sdr.sdr_id,
            'sdr_url': sdr.url,
            'frequency_hz': self._select_frequency_for_sdr(sdr),
            'duration_seconds': 1800,  # 30 minutes
            'priority': 5,
            'assigned_at': datetime.utcnow().isoformat()
        }

        await self.redis.lpush(
            RedisKeys.SDR_ASSIGNMENT_QUEUE,
            json.dumps(task)
        )

    async def _stop_recording_session(self, session):
        """
        Stop an active recording session
        """
        # Send stop command
        await self.redis.publish(
            f"cascade:control:session:{session.session_id}",
            json.dumps({'command': 'stop'})
        )

    async def _get_active_sessions(self):
        """
        Get list of currently active recording sessions
        """
        # Query database for active sessions
        from ..models.recording_session import RecordingSession

        sessions = self.db_session.query(RecordingSession).filter(
            RecordingSession.processing_status == 'recording'
        ).all()

        return sessions

    def _select_frequency_for_sdr(self, sdr) -> int:
        """
        Select appropriate frequency based on current conditions
        """
        # Default center frequencies
        frequencies = {
            '80m': 3576000,
            '40m': 7080000,
            '20m': 14080000,
            '15m': 21080000,
            '10m': 28080000,
            '6m': 50303000
        }

        # TODO: Implement smart frequency selection based on:
        # - Current propagation conditions
        # - Band coverage gaps
        # - SDR location

        return frequencies['20m']  # Default to 20m

    def _k_index_to_mode(self, k_index: int) -> ScalingMode:
        """Convert K-index to scaling mode"""
        for threshold, mode in sorted(self.k_index_thresholds.items(), reverse=True):
            if k_index >= threshold:
                return mode
        return ScalingMode.BASELINE

    def _xray_class_to_mode(self, xray_class: str) -> ScalingMode:
        """Convert X-ray class to scaling mode"""
        if not xray_class:
            return ScalingMode.BASELINE

        class_letter = xray_class[0].upper()
        return self.xray_class_thresholds.get(class_letter, ScalingMode.BASELINE)

    def _xray_class_to_severity(self, xray_class: str) -> int:
        """Convert X-ray class to severity (1-10)"""
        if not xray_class:
            return 1

        severity_map = {'A': 1, 'B': 2, 'C': 3, 'M': 6, 'X': 9}
        return severity_map.get(xray_class[0].upper(), 1)

    def _get_storm_affected_bands(self, k_index: int) -> List[str]:
        """Determine which bands are affected by geomagnetic storm"""
        if k_index >= 7:
            # Major storm affects all bands
            return ['80m', '40m', '20m', '15m', '10m', '6m']
        elif k_index >= 5:
            # Minor storm mainly affects higher bands
            return ['20m', '15m', '10m', '6m']
        else:
            return ['10m', '6m']

    async def get_scaling_status(self) -> Dict:
        """
        Get current scaling status for monitoring
        """
        return {
            'current_mode': self.current_mode.value,
            'station_count': self.station_counts[self.current_mode],
            'active_events': [
                {
                    'type': e.event_type.value,
                    'severity': e.severity,
                    'bands': e.affected_bands,
                    'start': e.start_time.isoformat(),
                    'end': e.end_time.isoformat() if e.end_time else None
                }
                for e in self.active_events
            ],
            'scaling_history': self.scaling_history[-10:]  # Last 10 events
        }

    async def force_scaling(self, mode: ScalingMode, duration_minutes: int = 60):
        """
        Force manual scaling for testing or special events
        """
        logger.info(f"Manual scaling to {mode} for {duration_minutes} minutes")

        # Create manual event
        manual_event = ScalingEvent(
            event_type=EventType.CONTEST,  # Use contest as generic event
            severity=5,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(minutes=duration_minutes),
            affected_bands=['80m', '40m', '20m', '15m', '10m', '6m'],
            scaling_mode=mode,
            metadata={'manual': True, 'reason': 'Manual override'}
        )

        await self._apply_scaling(mode, [manual_event])
# Alias for compatibility
EventBasedScaler = EventScaler
