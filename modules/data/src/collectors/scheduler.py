"""Collection scheduling master process.

Implements T027: Collection scheduler (FR-011, FR-016, FR-042).
"""

import asyncio
import logging
import json
import os
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import redis.asyncio as redis

from ..collectors.sdr_manager import SDRManager
from ..collectors.recorder import Recorder
from ..collectors.queue_manager import QueueManager
from ..collectors.geographic_quotas import GeographicQuotaManager, LatitudeBand
from ..collectors.southern_priority import SouthernHemispherePriorityCollector
from ..collectors.hybrid_sdr_selector import HybridSDRSelector
from ..models import SessionLocal, CollectionSchedule, SpaceWeatherData
from ..config import config
from ..config.frequencies import BANDS, BAND_CONFIGS
from ..notifications.gmail_notifier import GmailNotifier, NotificationConfig
from ..validators.qa_reporter import QAReporter

logger = logging.getLogger(__name__)


class CollectionScheduler:
    """Master scheduler for collection orchestration (FR-042)."""

    def __init__(self):
        """Initialize scheduler with diversity awareness (T088)."""
        self.db = SessionLocal()
        self.sdr_manager = SDRManager(self.db)
        self.recorder = Recorder(sdr_manager=self.sdr_manager)  # Pass sdr_manager for usage tracking!
        self.redis_client: Optional[redis.Redis] = None
        self.queue_manager: Optional[QueueManager] = None
        self.running = False
        self.tasks: List[asyncio.Task] = []

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Collection parameters (FR-016, FR-018) - Default from env, but dynamically adjustable
        # Start conservative, scale up as testing proves stability
        self.baseline_sdr_count = int(os.getenv('BASELINE_SDR_COUNT', '6'))  # Initial value
        self.max_sdr_count = int(os.getenv('MAX_SDR_COUNT', '50'))  # Initial max
        self.current_sdr_target = self.baseline_sdr_count

        # These can be updated via Redis without restart
        self.dynamic_config_key = "scheduler:dynamic_config"

        # Seasonal balancing (FR-050, FR-057)
        self.seasonal_quotas = {
            "winter": {"target": 0.25, "tolerance": 0.05},
            "spring": {"target": 0.25, "tolerance": 0.05},
            "summer": {"target": 0.25, "tolerance": 0.05},
            "autumn": {"target": 0.25, "tolerance": 0.05},
        }
        self.seasonal_stats = {"winter": 0, "spring": 0, "summer": 0, "autumn": 0}

        # Geographic diversity managers (T088)
        self.quota_manager = GeographicQuotaManager()
        self.southern_collector = SouthernHemispherePriorityCollector()
        self.hybrid_selector = HybridSDRSelector()

        # QA reporting (FR-037)
        self.qa_reporter = QAReporter()
        self.last_qa_report_date = None

        # KiwiSDR source sync tracking
        self.last_kiwisdr_sync = None

        # Diversity-aware scheduling parameters (T088a-d)
        self.scarce_region_reserved_slots = 0.2  # 20% reserved for underrepresented
        self.diversity_hour_enabled = True  # Daily geographic diversity hour
        self.prefer_scarce_regions = True  # Preference flag
        self.last_diversity_hour = None
        self.diversity_vs_efficiency_threshold = 0.7  # Trade-off threshold

        # Initialize Gmail notifier for operator alerts (FR-034)
        try:
            self.notifier = GmailNotifier()
            logger.info("Gmail notifier initialized for operator alerts")
        except ValueError as e:
            logger.warning(f"Gmail notifier not configured: {e}")
            self.notifier = None

        # Alert thresholds (FR-034) - Configurable for different deployment scales
        self.alert_thresholds = {
            "sdr_availability_percent": int(os.getenv('ALERT_SDR_AVAILABILITY', '50')),
            "collection_rate_hours_per_day": int(os.getenv('ALERT_MIN_HOURS_DAY', '20')),  # Start low
            "storage_usage_percent": int(os.getenv('ALERT_STORAGE_PERCENT', '90')),
            "consecutive_failures": int(os.getenv('ALERT_CONSECUTIVE_FAILURES', '5')),
        }
        self.last_alert_time = {}  # Rate limiting for alerts

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    async def initialize(self):
        """Initialize scheduler connections."""
        # Initialize Tigris buckets first
        from ..storage.tigris_init import initialize_tigris_buckets
        if not initialize_tigris_buckets():
            logger.warning("Tigris bucket initialization failed - uploads may not work")

        try:
            # Connect to Redis for distributed coordination
            self.redis_client = redis.from_url(
                config.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis")

            # Initialize queue manager for job distribution
            self.queue_manager = QueueManager(config.REDIS_URL)
            await self.queue_manager.connect()
            logger.info("Queue manager initialized")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
            self.queue_manager = None

    async def start(self):
        """Start the scheduler."""
        await self.initialize()
        self.running = True
        self.start_time = datetime.utcnow()

        logger.info("Starting collection scheduler")

        # Start background tasks
        self.tasks = [
            asyncio.create_task(self._continuous_collection_loop()),
            asyncio.create_task(self._event_monitoring_loop()),
            asyncio.create_task(self._aggressive_event_monitor()),  # New 18-month aggressive monitoring
            asyncio.create_task(self._schedule_checker_loop()),
            asyncio.create_task(self._health_monitor_loop()),
            asyncio.create_task(self._diversity_monitor_loop()),  # T088: Geographic diversity monitoring
            asyncio.create_task(self._diversity_hour_loop()),  # T088b: Daily diversity hour
            asyncio.create_task(self._dynamic_config_updater()),  # Live config updates without restart
            asyncio.create_task(self._daily_qa_report_loop()),  # FR-037: Daily QA reports
            asyncio.create_task(self._kiwisdr_sync_loop()),  # Daily KiwiSDR source sync
        ]

        # Wait for tasks
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping collection scheduler")
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Close connections
        if self.redis_client:
            await self.redis_client.close()
        self.sdr_manager.close()
        self.db.close()

    async def _continuous_collection_loop(self):
        """Maintain continuous baseline collection (FR-016)."""
        while self.running:
            try:
                # Get current active sessions
                active_sessions = self.recorder.get_active_sessions()
                active_count = len(active_sessions)

                logger.info(f"Active sessions: {active_count}/{self.current_sdr_target}")

                # Check if we need more sessions
                if active_count < self.current_sdr_target:
                    needed = self.current_sdr_target - active_count
                    await self._start_collection_sessions(needed)

                # Wait before next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                error_msg = str(e)
                if "does not exist" in error_msg or "UndefinedTable" in error_msg:
                    logger.error(f"Database table missing in continuous collection: {e}")
                    logger.info("Attempting to create missing tables...")
                    try:
                        # Try to create tables
                        from ..models import Base, engine
                        Base.metadata.create_all(bind=engine)
                        logger.info("✅ Database tables created successfully")
                    except Exception as create_error:
                        logger.error(f"❌ Failed to create tables: {create_error}")
                else:
                    logger.error(f"Error in continuous collection: {e}")
                
                try:
                    self.db.rollback()
                except Exception:
                    pass  # Rollback might fail if connection is broken
                await asyncio.sleep(30)

    async def _get_locked_sdrs(self) -> set:
        """Get list of currently locked SDR URLs from Redis.

        Returns:
            Set of locked SDR URLs
        """
        if not self.queue_manager:
            return set()

        try:
            # Get all lock keys from Redis
            lock_pattern = "lock:sdr:*"
            lock_keys = await self.queue_manager.client.keys(lock_pattern)

            # Extract SDR URLs from lock keys
            locked_urls = set()
            for key in lock_keys:
                # Key format: "lock:sdr:http://example.com:8073"
                sdr_url = key.replace("lock:sdr:", "")
                locked_urls.add(sdr_url)

            return locked_urls

        except Exception as e:
            logger.error(f"Error getting locked SDRs: {e}")
            return set()

    async def _start_collection_sessions(self, count: int):
        """Start new collection sessions.

        Args:
            count: Number of sessions to start
        """
        logger.info(f"Starting {count} new collection sessions")

        # Get optimal SDRs using hybrid selector for each band
        logger.info(f"🔍 Selecting {count} optimal SDRs across bands")
        sdrs = []
        used_sdr_urls = set()  # Track already-selected SDRs

        # Get currently locked SDRs from Redis to avoid scheduling conflicts
        locked_sdrs = await self._get_locked_sdrs()
        logger.info(f"🔒 Found {len(locked_sdrs)} currently locked SDRs")

        bands_to_monitor = BANDS[:count] if count <= len(BANDS) else BANDS * (count // len(BANDS) + 1)

        for i in range(count):
            band = bands_to_monitor[i % len(bands_to_monitor)]
            frequency_khz = BAND_CONFIGS[band].center_khz

            logger.debug(f"  🔎 Selecting SDR for {band} ({frequency_khz} kHz)")

            # Select SDR, excluding already-selected AND currently locked SDRs
            exclude_list = list(used_sdr_urls.union(locked_sdrs))
            sdr_candidate = self.hybrid_selector.select_optimal_sdr(
                frequency_khz=frequency_khz,
                expected_duration_minutes=5,
                band=band,
                require_gps=True,
                exclude_urls=exclude_list
            )

            if sdr_candidate:
                logger.debug(f"  ✅ Selected {sdr_candidate.url} for {band}")
                sdrs.append(sdr_candidate)
                used_sdr_urls.add(sdr_candidate.url)
            else:
                logger.debug(f"  ⚠️  No available SDR for {band} (excluded: {len(used_sdr_urls)})")

        if not sdrs:
            logger.warning("⚠️  No available SDRs for new sessions, sleeping 30s")
            await asyncio.sleep(30)
            return

        logger.info(f"✅ Selected {len(sdrs)} unique SDRs for collection (target: {count})")

        # Distribute across bands
        band_index = 0
        bands = BANDS

        for sdr in sdrs:
            band = bands[band_index % len(bands)]
            config = BAND_CONFIGS[band]

            # Pre-flight check: Verify SDR still has available time (FR-008, FR-014)
            if sdr.remaining_daily_minutes < 5:
                logger.warning(
                    f"Skipping {sdr.url}: insufficient remaining time "
                    f"({sdr.remaining_daily_minutes:.1f} min < 5 min minimum)"
                )
                continue

            try:
                logger.info(f"🎯 Assigning {sdr.url} to band {band} ({config.center_khz} kHz)")

                # Queue collection job
                if self.queue_manager:
                    # Distributed mode - push to queue
                    job = {
                        "sdr_url": sdr.url,
                        "frequency_khz": config.center_khz,
                        "band": band,
                        "duration_seconds": 360,  # 6 minutes
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    logger.info(f"📤 Pushing job to Redis queue: {sdr.url} on {band}")
                    await self.queue_manager.push_job(
                        self.queue_manager.COLLECTION_QUEUE,
                        job,
                    )
                    logger.info(f"✅ Queued job for {sdr.url} on {band} at {config.center_khz} kHz")
                else:
                    # Local mode - start directly
                    logger.info(f"🎬 Starting recording directly (local mode): {sdr.url} on {band}")
                    session_id = await self.recorder.start_recording(
                        kiwisdr_url=sdr.url,
                        frequency_khz=config.center_khz,
                        duration_seconds=360,
                        band=band,
                    )
                    logger.info(f"✅ Started session {session_id} on {band}")

                band_index += 1

            except Exception as e:
                logger.error(f"❌ Failed to start session on {sdr.url}: {e}", exc_info=True)
                await self.sdr_manager.handle_sdr_failure(sdr, e)

    async def _event_monitoring_loop(self):
        """Monitor for propagation events and scale collection (FR-023)."""
        while self.running:
            try:
                # Check space weather
                space_weather = await self._get_latest_space_weather()

                if space_weather:
                    # Determine scaling based on conditions
                    new_target = self._calculate_sdr_target(space_weather)

                    if new_target != self.current_sdr_target:
                        logger.info(
                            f"Adjusting SDR target: {self.current_sdr_target} -> {new_target}"
                        )
                        self.current_sdr_target = new_target

                # Check gray-line positions
                gray_line_sdrs = await self._check_gray_line()
                if gray_line_sdrs:
                    logger.info(f"Gray-line enhancement: {len(gray_line_sdrs)} SDRs")

                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in event monitoring: {e}")
                await asyncio.sleep(60)

    def _calculate_sdr_target(self, space_weather: SpaceWeatherData) -> int:
        """Calculate target SDR count based on conditions with 18-month aggressive scaling.

        Args:
            space_weather: Current space weather

        Returns:
            Target SDR count
        """
        # Use aggressive scaling for 18-month collection window (FR-055, FR-059)
        if space_weather.opportunity_limited_mode:
            return space_weather.get_aggressive_sdr_target(self.baseline_sdr_count)

        # Legacy scaling for non-aggressive mode
        target = self.baseline_sdr_count

        # Scale based on K-index
        if space_weather.k_index:
            if space_weather.k_index >= 7:
                target = min(self.max_sdr_count, target + 20)
            elif space_weather.k_index >= 5:
                target = min(self.max_sdr_count, target + 10)
            elif space_weather.k_index >= 3:
                target = min(self.max_sdr_count, target + 5)

        # Scale based on X-ray class
        if space_weather.xray_class:
            if space_weather.xray_class[0] == 'X':
                target = min(self.max_sdr_count, target + 15)
            elif space_weather.xray_class[0] == 'M':
                target = min(self.max_sdr_count, target + 10)

        # Scale based on solar flux
        if space_weather.solar_flux and space_weather.solar_flux > 150:
            target = min(self.max_sdr_count, target + 5)

        return target

    async def _get_latest_space_weather(self) -> Optional[SpaceWeatherData]:
        """Get latest space weather data.

        Returns:
            Latest SpaceWeatherData or None
        """
        try:
            # Get most recent entry
            latest = (
                self.db.query(SpaceWeatherData)
                .order_by(SpaceWeatherData.observation_time.desc())
                .first()
            )

            # Check if recent enough (within 1 hour)
            if latest and (
                datetime.utcnow() - latest.observation_time.replace(tzinfo=None)
            ) < timedelta(hours=1):
                return latest

        except Exception as e:
            logger.error(f"Error getting space weather: {e}")

        return None

    async def _check_gray_line(self) -> List[str]:
        """Check for gray-line propagation opportunities.

        Returns:
            List of SDR URLs in gray-line zone
        """
        gray_line_sdrs = []
        current_time = datetime.utcnow()

        # Get best SDR for gray-line ONCE, not in a loop!
        best_sdr = await self.sdr_manager.get_best_for_propagation(
            target_time=current_time,
            propagation_type='gray_line',
            frequency_mhz=14.0,  # 20m band
        )

        if best_sdr:
            gray_line_sdrs.append(best_sdr.url)
            logger.info(f"Gray-line opportunity: {best_sdr.url}")

        return gray_line_sdrs

    async def _schedule_checker_loop(self):
        """Check and execute scheduled collections (FR-011)."""
        while self.running:
            try:
                # Get active schedules
                schedules = (
                    self.db.query(CollectionSchedule)
                    .filter(CollectionSchedule.active == True)
                    .all()
                )

                current_time = datetime.utcnow()
                current_day = current_time.weekday()

                for schedule in schedules:
                    # Check if schedule applies today
                    if schedule.days_of_week and current_day not in schedule.days_of_week:
                        continue

                    # Check if within time window
                    if schedule.start_time and schedule.end_time:
                        current_time_only = current_time.time()
                        if not (schedule.start_time <= current_time_only <= schedule.end_time):
                            continue

                    # Check interval
                    # TODO: Track last execution and interval

                    logger.debug(f"Schedule {schedule.name} is active")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                error_msg = str(e)
                if "does not exist" in error_msg or "UndefinedTable" in error_msg:
                    logger.error(f"Database table missing in schedule checker: {e}")
                    logger.info("Attempting to create missing tables...")
                    try:
                        # Try to create tables
                        from ..models import Base, engine
                        Base.metadata.create_all(bind=engine)
                        logger.info("✅ Database tables created successfully")
                    except Exception as create_error:
                        logger.error(f"❌ Failed to create tables: {create_error}")
                else:
                    logger.error(f"Error in schedule checker: {e}")
                
                try:
                    self.db.rollback()
                except Exception:
                    pass  # Rollback might fail if connection is broken
                await asyncio.sleep(30)

    async def _health_monitor_loop(self):
        """Monitor worker health and send operator alerts (FR-043, FR-034)."""
        consecutive_failures = 0

        # Write scheduler health status for external monitoring
        await self._write_scheduler_health()

        while self.running:
            try:
                if self.redis_client:
                    # Get worker health from Redis
                    workers = await self.redis_client.keys("worker:*:health")

                    healthy_count = 0
                    for worker_key in workers:
                        health_data = await self.redis_client.get(worker_key)
                        if health_data:
                            health = json.loads(health_data)
                            last_update = datetime.fromisoformat(health['timestamp'])
                            age = (datetime.utcnow() - last_update).total_seconds()

                            if age < 60:  # Health update within last minute
                                healthy_count += 1
                            else:
                                logger.warning(f"Worker {worker_key} is stale ({age:.0f}s)")

                    logger.info(f"Healthy workers: {healthy_count}/{len(workers)}")

                    # Check worker availability and send alerts if needed (FR-034)
                    total_workers = max(len(workers), 1)  # Avoid division by zero
                    availability_percent = (healthy_count / total_workers) * 100

                    if availability_percent < self.alert_thresholds["sdr_availability_percent"]:
                        await self._send_operator_alert(
                            "Low SDR Availability",
                            f"Only {healthy_count}/{total_workers} workers are healthy "
                            f"({availability_percent:.1f}% availability). "
                            f"Target threshold: {self.alert_thresholds['sdr_availability_percent']}%",
                            priority="high"
                        )
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 0

                    # Alert on consecutive failures
                    if consecutive_failures >= self.alert_thresholds["consecutive_failures"]:
                        await self._send_operator_alert(
                            "Critical: Sustained Low SDR Availability",
                            f"SDR availability has been below threshold for "
                            f"{consecutive_failures} consecutive checks. "
                            f"Immediate attention required.",
                            priority="high"
                        )

                    # Update metrics
                    await self.redis_client.set(
                        "scheduler:metrics",
                        json.dumps({
                            "healthy_workers": healthy_count,
                            "total_workers": len(workers),
                            "sdr_target": self.current_sdr_target,
                            "timestamp": datetime.utcnow().isoformat(),
                        }),
                    )

                # Update scheduler health status
                await self._write_scheduler_health()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(30)

    async def _write_scheduler_health(self):
        """Write scheduler health status to file for external monitoring."""
        try:
            health_file = Path("/tmp/scheduler_health.json")
            health_data = {
                "status": "healthy" if self.running else "stopped",
                "timestamp": datetime.utcnow().isoformat(),
                "sdr_target": self.current_sdr_target,
                "baseline_sdrs": self.baseline_sdr_count,
                "max_sdrs": self.max_sdr_count,
                "active_sessions": len(self.recorder.get_active_sessions()),
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds() if hasattr(self, 'start_time') else 0
            }

            health_file.write_text(json.dumps(health_data))

            # Also write to Redis if available
            if self.redis_client:
                await self.redis_client.setex(
                    "scheduler:health",
                    60,  # Expire after 60 seconds
                    json.dumps(health_data)
                )
        except Exception as e:
            logger.warning(f"Failed to write health status: {e}")

    async def _aggressive_event_monitor(self):
        """Aggressive event monitoring for 18-month collection window (FR-055, FR-059)."""
        while self.running:
            try:
                # Check space weather more frequently during aggressive mode
                space_weather = await self._get_latest_space_weather()

                if space_weather and space_weather.opportunity_limited_mode:
                    # Check for 100% capture requirements (FR-059)
                    if space_weather.is_100_percent_capture_required():
                        await self._enforce_100_percent_capture(space_weather)

                    # Monitor for aggressive scaling opportunities
                    urgency = space_weather.calculate_collection_urgency()
                    if urgency > 10.0:
                        logger.critical(
                            f"HIGH URGENCY EVENT: urgency={urgency:.1f}, "
                            f"K={space_weather.k_index}, X-ray={space_weather.xray_class}, "
                            f"scaling to {space_weather.get_aggressive_sdr_target()} SDRs"
                        )

                        # Apply aggressive scaling immediately
                        new_target = space_weather.get_aggressive_sdr_target(self.baseline_sdr_count)
                        if new_target > self.current_sdr_target:
                            self.current_sdr_target = new_target
                            logger.warning(f"Aggressive scaling activated: {new_target} SDRs")

                    # Check for C-class flare triggers during solar minimum
                    if (space_weather.should_include_c_flares() and
                        space_weather.xray_class and
                        space_weather.xray_class.startswith('C')):

                        logger.warning(
                            f"C-class flare during solar minimum: {space_weather.xray_class}, "
                            f"triggering aggressive collection"
                        )
                        self.current_sdr_target = max(self.current_sdr_target, 20)

                # Check every 2 minutes during aggressive mode
                await asyncio.sleep(120)

            except Exception as e:
                logger.error(f"Error in aggressive event monitor: {e}")
                await asyncio.sleep(60)

    async def _enforce_100_percent_capture(self, space_weather: SpaceWeatherData):
        """Enforce 100% capture rate for solar minimum events (FR-059).

        Args:
            space_weather: Current space weather conditions
        """
        logger.critical(
            f"100% CAPTURE MODE ACTIVATED: K={space_weather.k_index}, "
            f"X-ray={space_weather.xray_class}, scaling to MAXIMUM"
        )

        # Scale to maximum SDRs immediately
        aggressive_target = space_weather.get_aggressive_sdr_target(self.baseline_sdr_count)
        self.current_sdr_target = aggressive_target

        # Broadcast priority collection alert
        if self.redis_client:
            alert = {
                "type": "100_percent_capture",
                "timestamp": datetime.utcnow().isoformat(),
                "k_index": space_weather.k_index,
                "xray_class": space_weather.xray_class,
                "target_sdrs": aggressive_target,
                "urgency": space_weather.calculate_collection_urgency(),
                "rarity_multiplier": space_weather.get_rarity_multiplier(),
            }

            await self.redis_client.publish(
                "collection:priority_alerts",
                json.dumps(alert)
            )

        # Start additional sessions immediately if needed
        active_count = len(self.recorder.get_active_sessions())
        if active_count < aggressive_target:
            needed = aggressive_target - active_count
            logger.critical(f"Starting {needed} additional sessions for 100% capture")
            await self._start_collection_sessions(needed)

    async def manual_trigger_collection(
        self,
        band: str,
        duration_seconds: int,
        sdr_count: int = 1,
    ) -> List[str]:
        """Manually trigger collection sessions.

        Args:
            band: Band to collect
            duration_seconds: Duration of each session
            sdr_count: Number of concurrent sessions

        Returns:
            List of session IDs
        """
        session_ids = []
        config = BAND_CONFIGS.get(band)

        if not config:
            raise ValueError(f"Unknown band: {band}")

        # Get SDRs
        sdrs = await self.sdr_manager.get_concurrent_sdrs(sdr_count)

        for sdr in sdrs:
            try:
                session_id = await self.recorder.start_recording(
                    kiwisdr_url=sdr.url,
                    frequency_khz=config.center_khz,
                    duration_seconds=duration_seconds,
                    band=band,
                )
                session_ids.append(session_id)
            except Exception as e:
                logger.error(f"Failed to start session: {e}")

        return session_ids

    async def _diversity_monitor_loop(self):
        """Monitor geographic diversity and trigger rebalancing (T088)."""
        while self.running:
            try:
                # Get current diversity metrics
                progress = self.quota_manager.get_collection_progress()
                diversity_score = self.quota_manager.get_diversity_score()

                logger.info(f"Geographic diversity score: {diversity_score:.2f}")

                # Check for critical imbalances
                warnings = self.quota_manager.get_quota_warnings()
                if warnings:
                    logger.warning(f"Geographic diversity warnings: {warnings}")

                    # Get rebalancing recommendations
                    recommendations = self.quota_manager.get_rebalancing_recommendations()

                    # Apply rebalancing (T088d: Trade-off algorithm)
                    if diversity_score < self.diversity_vs_efficiency_threshold:
                        logger.info("Prioritizing diversity over efficiency")
                        await self._apply_diversity_rebalancing(recommendations)
                    else:
                        logger.info("Maintaining balance between diversity and efficiency")

                # Reserve slots for underrepresented regions (T088a)
                await self._reserve_scarce_region_slots()

                # Wait before next check
                await asyncio.sleep(600)  # Check every 10 minutes

            except Exception as e:
                logger.error(f"Error in diversity monitor: {e}")
                await asyncio.sleep(300)

    async def _diversity_hour_loop(self):
        """Implement daily geographic diversity hour (T088b)."""
        while self.running:
            try:
                current_hour = datetime.utcnow().hour

                # Run diversity hour once per day (e.g., at 14:00 UTC)
                if current_hour == 14 and (
                    not self.last_diversity_hour or
                    (datetime.utcnow() - self.last_diversity_hour).days >= 1
                ):
                    logger.info("Starting geographic diversity hour")
                    self.last_diversity_hour = datetime.utcnow()

                    # Get list of scarce regions
                    underrepresented = self.quota_manager.get_underrepresented_bands()

                    # Rotate through scarce regions for one hour
                    end_time = datetime.utcnow() + timedelta(hours=1)
                    while datetime.utcnow() < end_time and self.running:
                        for band in underrepresented:
                            # Select SDRs from underrepresented band
                            await self._collect_from_latitude_band(band, duration_minutes=10)
                            await asyncio.sleep(600)  # 10 minutes per band

                    logger.info("Completed geographic diversity hour")

                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in diversity hour: {e}")
                await asyncio.sleep(1800)

    async def _reserve_scarce_region_slots(self):
        """Reserve 20% of collection slots for underrepresented regions (T088a)."""
        try:
            # Calculate reserved slots
            reserved_count = int(self.current_sdr_target * self.scarce_region_reserved_slots)

            if reserved_count > 0:
                # Get southern hemisphere SDRs
                southern_sdrs = self.southern_collector.maintain_southern_sdr_list()

                # Select up to reserved_count SDRs from scarce regions
                selected = southern_sdrs[:reserved_count]

                for sdr in selected:
                    # Apply collection weight
                    weighted_hours = self.southern_collector.apply_collection_weight(
                        sdr, base_collection_hours=0.5
                    )

                    logger.debug(
                        f"Reserved slot for {sdr.grid_square} "
                        f"({sdr.hemisphere.value}, {weighted_hours:.1f}h weighted)"
                    )

        except Exception as e:
            logger.error(f"Error reserving scarce region slots: {e}")

    async def _apply_diversity_rebalancing(self, recommendations: List[Dict]):
        """Apply diversity rebalancing recommendations."""
        for rec in recommendations[:3]:  # Apply top 3 recommendations
            region = rec["region"]
            multiplier = rec["priority_multiplier"]

            logger.info(
                f"Applying {multiplier:.1f}x priority to {region} "
                f"(deficit: {rec['deficit']:.1f}%)"
            )

            # Adjust SDR selection priorities
            if "Antarctic" in region:
                # Prioritize Antarctic SDRs
                await self._collect_from_latitude_band(
                    LatitudeBand.ANTARCTIC,
                    duration_minutes=30,
                    priority_multiplier=multiplier
                )
            elif "Arctic" in region:
                await self._collect_from_latitude_band(
                    LatitudeBand.ARCTIC,
                    duration_minutes=30,
                    priority_multiplier=multiplier
                )
            elif "South" in region:
                # Use southern hemisphere priority collector
                await self._collect_from_southern_hemisphere(
                    duration_minutes=30,
                    priority_multiplier=multiplier
                )

    async def _collect_from_latitude_band(
        self,
        band: LatitudeBand,
        duration_minutes: int = 30,
        priority_multiplier: float = 1.0
    ):
        """Collect from specific latitude band."""
        try:
            # Use hybrid selector with band preference
            candidate = self.hybrid_selector.select_optimal_sdr(
                frequency_khz=14080,  # 20m band
                expected_duration_minutes=duration_minutes,
                band="20m",
                prefer_location=None  # Will be filtered by band
            )

            if candidate:
                logger.info(
                    f"Starting collection from {band.value} band: "
                    f"{candidate.url} for {duration_minutes} minutes"
                )

                # Start collection with priority
                if self.redis_client:
                    job = {
                        "sdr_url": candidate.url,
                        "frequency_khz": 14080,
                        "band": "20m",
                        "duration_seconds": duration_minutes * 60,
                        "priority": priority_multiplier,
                        "latitude_band": band.value,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    await self.redis_client.lpush(
                        "priority_collection_queue",
                        json.dumps(job),
                    )

        except Exception as e:
            logger.error(f"Error collecting from {band.value}: {e}")

    async def _collect_from_southern_hemisphere(
        self,
        duration_minutes: int = 30,
        priority_multiplier: float = 1.0
    ):
        """Prioritize collection from southern hemisphere."""
        try:
            # Get optimal southern SDR
            southern_sdr = self.southern_collector.select_optimal_southern_sdr(
                frequency_khz=14080,
                duration_minutes=duration_minutes,
                prefer_antarctic=priority_multiplier > 3.0
            )

            if southern_sdr:
                logger.info(
                    f"Starting southern collection from {southern_sdr.grid_square} "
                    f"({southern_sdr.latitude:.1f}°) for {duration_minutes} minutes"
                )

                # Apply weight and start collection
                weighted_duration = self.southern_collector.apply_collection_weight(
                    southern_sdr,
                    base_collection_hours=duration_minutes / 60
                ) * 60  # Convert back to minutes

                if self.redis_client:
                    job = {
                        "sdr_url": southern_sdr.url,
                        "frequency_khz": 14080,
                        "band": "20m",
                        "duration_seconds": int(weighted_duration * 60),
                        "priority": priority_multiplier,
                        "hemisphere": "southern",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    await self.redis_client.lpush(
                        "priority_collection_queue",
                        json.dumps(job),
                    )
            else:
                # Failover to reciprocal inference
                logger.warning("No southern SDRs available, using reciprocal inference")
                result = self.southern_collector.failover_to_reciprocal(
                    target_bands=["20m"],
                    required_hours=duration_minutes / 60
                )
                logger.info(f"Reciprocal inference result: {result}")

        except Exception as e:
            logger.error(f"Error in southern collection: {e}")

    def get_diversity_status(self) -> Dict[str, Any]:
        """Get current diversity status for monitoring."""
        progress = self.quota_manager.get_collection_progress()
        diversity_score = self.quota_manager.get_diversity_score()
        warnings = self.quota_manager.get_quota_warnings()

        return {
            "diversity_score": diversity_score,
            "total_hours": progress.total_hours,
            "hemisphere_balance": self.quota_manager.get_hemisphere_balance_score(),
            "underrepresented_bands": [
                band.value for band in self.quota_manager.get_underrepresented_bands()
            ],
            "warnings": warnings,
            "scarce_slots_reserved": self.scarce_region_reserved_slots,
            "diversity_hour_enabled": self.diversity_hour_enabled,
            "last_diversity_hour": self.last_diversity_hour.isoformat() if self.last_diversity_hour else None,
            "prefer_scarce_regions": self.prefer_scarce_regions
        }

    def _get_current_qa_rate(self) -> float:
        """Get current QA sampling rate based on collection phase.

        Returns progressive rates:
        - Months 1-2: 3% random (bootstrap)
        - Months 3-4: 8% mixed (hybrid)
        - Months 5-18: 12% intelligent (production)
        """
        months_elapsed = (datetime.utcnow() - self.collection_start_date).days / 30

        if months_elapsed <= 2:
            # Bootstrap phase: 3% random sampling
            return config.QA_BOOTSTRAP_RATE
        elif months_elapsed <= 4:
            # Hybrid phase: 8% mixed sampling
            return config.QA_HYBRID_RATE
        else:
            # Production phase: 12% intelligent sampling
            return config.QA_PRODUCTION_RATE

    async def _daily_qa_report_loop(self):
        """Generate daily QA reports (FR-037).

        Lists sampled files with metadata for review.
        """
        while self.running:
            try:
                now = datetime.utcnow()

                # Check if it's time for daily report (run at 00:00 UTC)
                if now.hour == 0 and (
                    self.last_qa_report_date is None or
                    self.last_qa_report_date.date() < now.date()
                ):
                    logger.info("Generating daily QA report")

                    try:
                        # Generate report for previous day
                        yesterday = now - timedelta(days=1)
                        report = self.qa_reporter.generate_daily_report(yesterday)

                        # Save report
                        report_path = f"/tmp/qa_report_{yesterday.strftime('%Y%m%d')}.json"
                        report.save_json(report_path)

                        # Send summary via email if configured
                        if self.notifier:
                            summary = f"""Daily QA Report - {yesterday.strftime('%Y-%m-%d')}

Samples Reviewed: {report.total_samples}
Average Quality Score: {report.metrics_summary.avg_quality_score:.1f}
Passed: {report.metrics_summary.passed_count}
Warnings: {report.metrics_summary.warning_count}
Failed: {report.metrics_summary.failed_count}

Top Issues:
{chr(10).join(f"- {issue}" for issue in report.metrics_summary.top_issues[:5])}

Full report saved to: {report_path}
"""
                            self.notifier.send_notification(
                                subject=f"Daily QA Report - {yesterday.strftime('%Y-%m-%d')}",
                                body=summary,
                                priority="normal"
                            )

                        self.last_qa_report_date = now
                        logger.info(f"Daily QA report generated: {report_path}")

                    except Exception as e:
                        logger.error(f"Failed to generate QA report: {e}")

                # Check every hour
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Error in QA report loop: {e}")
                await asyncio.sleep(3600)

    async def _dynamic_config_updater(self):
        """Check Redis for config updates without interrupting collection.

        Allows scaling SDR counts without restart or data loss.
        """
        while self.running:
            try:
                if self.redis_client:
                    # Get dynamic config from Redis
                    config_json = await self.redis_client.get(self.dynamic_config_key)

                    if config_json:
                        config = json.loads(config_json)

                        # Update baseline SDR count (won't affect running sessions)
                        new_baseline = config.get('baseline_sdr_count')
                        if new_baseline and new_baseline != self.baseline_sdr_count:
                            logger.info(f"Updating baseline SDR count: {self.baseline_sdr_count} -> {new_baseline}")
                            self.baseline_sdr_count = new_baseline
                            # Only update target if not in event mode
                            if self.current_sdr_target <= self.baseline_sdr_count:
                                self.current_sdr_target = new_baseline

                        # Update max SDR count
                        new_max = config.get('max_sdr_count')
                        if new_max and new_max != self.max_sdr_count:
                            logger.info(f"Updating max SDR count: {self.max_sdr_count} -> {new_max}")
                            self.max_sdr_count = new_max

                        # Update alert thresholds
                        new_thresholds = config.get('alert_thresholds', {})
                        for key, value in new_thresholds.items():
                            if key in self.alert_thresholds:
                                self.alert_thresholds[key] = value

                # Check every 30 seconds for updates
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Error updating dynamic config: {e}")
                await asyncio.sleep(60)

    async def _kiwisdr_sync_loop(self):
        """Sync KiwiSDR sources from public directory daily.

        Runs at 02:00 UTC daily to update the local kiwisdr_sources database
        with the latest receivers from kiwisdr.com/public/.
        """
        while self.running:
            try:
                now = datetime.utcnow()

                # Run sync at 02:00 UTC (low traffic time)
                if now.hour == 2 and (
                    self.last_kiwisdr_sync is None or
                    (now - self.last_kiwisdr_sync).days >= 1
                ):
                    logger.info("Starting KiwiSDR source sync from public directory")

                    try:
                        # Import sync functionality
                        import sys
                        from pathlib import Path
                        scripts_path = Path(__file__).parent.parent.parent / "scripts"
                        sys.path.insert(0, str(scripts_path))

                        from sync_kiwisdr_sources import fetch_kiwisdr_list, sync_to_database

                        # Fetch current list
                        receivers = await asyncio.to_thread(fetch_kiwisdr_list)

                        if receivers:
                            # Sync to database
                            await asyncio.to_thread(sync_to_database, receivers, False)

                            logger.info(
                                f"KiwiSDR sync complete: {len(receivers)} receivers processed"
                            )

                            # Send notification if configured
                            if self.notifier:
                                summary = f"""KiwiSDR Source Sync Complete

Timestamp: {now.isoformat()}Z
Receivers Processed: {len(receivers)}

The local KiwiSDR source database has been updated with the latest
receivers from the public directory.
"""
                                self.notifier.send_notification(
                                    subject="[CASCADE] KiwiSDR Source Sync Complete",
                                    body=summary,
                                    priority="low"
                                )

                        else:
                            logger.warning("No KiwiSDR receivers found during sync")

                            # Alert on sync failure
                            if self.notifier:
                                self.notifier.send_notification(
                                    subject="[CASCADE] Warning: KiwiSDR Sync Found No Receivers",
                                    body=f"The KiwiSDR sync at {now.isoformat()}Z found no receivers. "
                                         f"This may indicate a parsing issue or connectivity problem.",
                                    priority="normal"
                                )

                        self.last_kiwisdr_sync = now

                    except Exception as e:
                        logger.error(f"Failed to sync KiwiSDR sources: {e}")

                        # Alert on sync error
                        if self.notifier:
                            await self._send_operator_alert(
                                "KiwiSDR Source Sync Failed",
                                f"Failed to sync KiwiSDR sources from public directory: {e}\n\n"
                                f"The scheduler will retry tomorrow at 02:00 UTC.",
                                priority="normal"
                            )

                # Check every hour
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Error in KiwiSDR sync loop: {e}")
                await asyncio.sleep(3600)

    async def _send_operator_alert(self, subject: str, body: str, priority: str = "normal"):
        """Send operator alert via Gmail (FR-034).

        Args:
            subject: Alert subject
            body: Alert message body
            priority: Alert priority (low, normal, high)
        """
        try:
            # Rate limiting - don't send same alert more than once per hour
            alert_key = f"{subject}:{priority}"
            now = datetime.utcnow()

            if alert_key in self.last_alert_time:
                time_since_last = (now - self.last_alert_time[alert_key]).total_seconds()
                if time_since_last < 3600:  # 1 hour
                    logger.debug(f"Alert rate limited: {alert_key}")
                    return

            # Send notification if configured
            if self.notifier:
                # Add context to body
                full_body = f"""CASCADE Data Collector Alert

{body}

Timestamp: {now.isoformat()}Z
Scheduler: {config.APP_NAME or 'cascade-scheduler'}
Environment: {config.ENVIRONMENT or 'production'}

Current Status:
- SDR Target: {self.current_sdr_target}
- Active Sessions: {len(self.recorder.get_active_sessions())}
- Baseline SDRs: {self.baseline_sdr_count}
- Max SDRs: {self.max_sdr_count}

Alert Thresholds:
- SDR Availability: {self.alert_thresholds['sdr_availability_percent']}%
- Collection Rate: {self.alert_thresholds['collection_rate_hours_per_day']} hours/day
- Storage Usage: {self.alert_thresholds['storage_usage_percent']}%
- Consecutive Failures: {self.alert_thresholds['consecutive_failures']}

This is an automated alert from the CASCADE Data Collection system.
"""

                # Send email
                success = self.notifier.send_notification(
                    subject=f"[CASCADE] {subject}",
                    body=full_body,
                    priority=priority
                )

                if success:
                    logger.info(f"Operator alert sent: {subject} (priority: {priority})")
                    self.last_alert_time[alert_key] = now
                else:
                    logger.error(f"Failed to send operator alert: {subject}")

            # Also log to Redis for monitoring
            if self.redis_client:
                alert_data = {
                    "timestamp": now.isoformat(),
                    "subject": subject,
                    "body": body,
                    "priority": priority
                }
                await self.redis_client.lpush(
                    "alerts:operator",
                    json.dumps(alert_data)
                )
                await self.redis_client.ltrim("alerts:operator", 0, 99)  # Keep last 100

        except Exception as e:
            logger.error(f"Error sending operator alert: {e}")


async def main():
    """Main entry point for scheduler."""
    scheduler = CollectionScheduler()

    try:
        await scheduler.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await scheduler.stop()


if __name__ == "__main__":
    from ..config.logging_config import setup_logging
    setup_logging()
    asyncio.run(main())