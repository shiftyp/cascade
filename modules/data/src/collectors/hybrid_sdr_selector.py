"""Hybrid SDR selection algorithm for KiwiSDR and WebSDR coordination.

Implements T028e: Hybrid SDR selection algorithm (FR-067).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math
import asyncio
import aiohttp

from ..models import SessionLocal, KiwiSDRSource  # , WebSDRSource  # TODO: Re-enable WebSDR
from .geographic_quotas import GeographicQuotaManager, GridSquareClassifier

logger = logging.getLogger(__name__)


class SDRType(Enum):
    """SDR receiver types."""
    KIWISDR = "kiwisdr"
    WEBSDR = "websdr"


class InstitutionType(Enum):
    """Institution types for WebSDR receivers."""
    INDIVIDUAL = "individual"
    UNIVERSITY = "university"
    RESEARCH_INSTITUTE = "research_institute"
    AMATEUR_CLUB = "amateur_club"


class UsagePolicy(Enum):
    """SDR usage policies."""
    PUBLIC_LIMITED = "public_limited"         # KiwiSDR typical: 30-90 min daily
    RESEARCH_AGREEMENT = "research_agreement" # WebSDR with research coordination
    COOPERATIVE = "cooperative"               # Amateur club with member access
    RESTRICTED = "restricted"                 # Private or invitation-only


@dataclass
class SDRCandidate:
    """SDR candidate for selection."""

    sdr_id: str
    sdr_type: SDRType
    url: str
    institution_type: Optional[InstitutionType]
    usage_policy: UsagePolicy
    grid_square: Optional[str]
    daily_limit_minutes: float
    session_limit_minutes: int
    remaining_daily_minutes: float
    reliability_score: float
    has_gps: bool
    geographic_score: float
    load_score: float
    relationship_score: float
    total_score: float


class HybridSDRSelector:
    """Selects optimal SDR (KiwiSDR or WebSDR) based on requirements."""

    def __init__(self):
        """Initialize hybrid selector."""
        self.db = SessionLocal()

        # Selection weights for different factors
        self.weights = {
            "duration_suitability": 0.25,  # Important
            "reliability": 0.20,
            "geographic_diversity": 0.25,  # Increased for bias mitigation (T084)
            "load_balance": 0.10,
            "relationship_status": 0.10,
            "scarcity_bonus": 0.10,  # Added for T084
        }

        # Initialize geographic quota manager (T084)
        self.quota_manager = GeographicQuotaManager()
        self.grid_classifier = GridSquareClassifier()

        # SDR density cache for scarcity scoring (T084a)
        self.sdr_density_cache = {}
        self.density_cache_time = None

        # Initialize empty websdr_registry for backwards compatibility
        self.websdr_registry = []

        # Registry update tracking
        self.last_kiwisdr_sync = None
        self.last_websdr_sync = None
        self.sync_interval = timedelta(hours=24)  # Sync daily to reduce memory usage

    def select_optimal_sdr(
        self,
        frequency_khz: float,
        expected_duration_minutes: int,
        band: str,
        prefer_location: Optional[str] = None,
        require_gps: bool = True,
        exclude_urls: Optional[List[str]] = None,
    ) -> Optional[SDRCandidate]:
        """Select optimal SDR for given requirements (FR-067).

        Args:
            frequency_khz: Required frequency
            expected_duration_minutes: Expected session duration
            band: Amateur band
            prefer_location: Preferred grid square prefix
            require_gps: Whether GPS timing is required

        Returns:
            Best SDR candidate or None
        """
        logger.info(
            f"Selecting SDR for {frequency_khz} kHz, {expected_duration_minutes} min, band {band}"
        )

        # Get available SDRs
        kiwisdrs = self._get_available_kiwisdrs(frequency_khz, require_gps, exclude_urls or [])
        # TODO: Re-enable WebSDR
        # websdrs = self._get_available_websdrs(frequency_khz)

        # Score all candidates
        candidates = []

        # Score KiwiSDRs
        for kiwisdr in kiwisdrs:
            candidate = self._score_kiwisdr(
                kiwisdr, expected_duration_minutes, prefer_location
            )
            if candidate:
                candidates.append(candidate)

        # TODO: Re-enable WebSDR
        # # Score WebSDRs
        # for websdr in websdrs:
        #     candidate = self._score_websdr(
        #         websdr, expected_duration_minutes, prefer_location
        #     )
        #     if candidate:
        #         candidates.append(candidate)

        if not candidates:
            logger.warning("No suitable SDRs found")
            return None

        # Apply hybrid selection logic (FR-067)
        best_candidate = self._apply_hybrid_selection_rules(
            candidates, expected_duration_minutes
        )

        if best_candidate:
            logger.info(
                f"Selected {best_candidate.sdr_type.value}: {best_candidate.url}, "
                f"score: {best_candidate.total_score:.2f}, "
                f"duration fit: {expected_duration_minutes}/{best_candidate.session_limit_minutes} min"
            )

        return best_candidate

    def _get_available_kiwisdrs(
        self, frequency_khz: float, require_gps: bool, exclude_urls: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Get available KiwiSDR receivers with auto-sync.

        Args:
            frequency_khz: Required frequency
            require_gps: GPS requirement
            exclude_urls: URLs to exclude from selection

        Returns:
            List of available KiwiSDRs
        """
        # Skip sync on first call to prevent OOM - scheduler handles it in background
        # Only sync if it's been more than 24 hours and we're not in initial startup
        if self.last_kiwisdr_sync is None:
            # Set to now to prevent immediate sync - background task will handle it
            self.last_kiwisdr_sync = datetime.now(timezone.utc)
        elif datetime.now(timezone.utc) - self.last_kiwisdr_sync > timedelta(hours=24):
            # Daily sync only, not during every selection
            try:
                self._sync_kiwisdr_registry()
            except Exception as e:
                logger.error(f"Background sync failed: {e}")

        query = self.db.query(KiwiSDRSource).filter(
            KiwiSDRSource.active == True,
            KiwiSDRSource.failure_count < 5,
            KiwiSDRSource.min_freq_khz <= frequency_khz,
            KiwiSDRSource.max_freq_khz >= frequency_khz,
        )

        if require_gps:
            query = query.filter(KiwiSDRSource.has_gps == True)

        # Exclude already-selected SDRs
        if exclude_urls:
            query = query.filter(~KiwiSDRSource.url.in_(exclude_urls))

        sdrs = query.all()

        # Filter by remaining daily usage and update stale records
        available = []
        stale_sdrs = []  # Track stale SDRs for async health check

        for sdr in sdrs:
            # Update usage if needed
            if sdr.should_reset_usage():
                sdr.daily_usage_minutes = 0
                sdr.last_usage_reset = datetime.now(timezone.utc)
                self.db.commit()

            # Skip stale SDRs (not seen in 6+ hours) to avoid blocking
            # They'll be checked asynchronously in background
            if sdr.last_seen and sdr.last_seen < datetime.now(timezone.utc) - timedelta(hours=6):
                stale_sdrs.append(sdr)
                continue

            if sdr.remaining_daily_minutes > 5:  # At least 5 minutes remaining
                available.append({
                    "sdr_id": str(sdr.kiwisdr_id),
                    "sdr_type": SDRType.KIWISDR,
                    "url": sdr.url,
                    "grid_square": sdr.grid_square,
                    "daily_limit_minutes": sdr.daily_limit_minutes,
                    "session_limit_minutes": min(sdr.daily_limit_minutes, 90),
                    "remaining_daily_minutes": sdr.remaining_daily_minutes,
                    "reliability_score": sdr.reliability_score or 0.5,
                    "has_gps": sdr.has_gps,
                    "usage_policy": UsagePolicy.PUBLIC_LIMITED,
                    "institution_type": InstitutionType.INDIVIDUAL,
                })

        # Schedule async health checks for stale SDRs (fire and forget)
        if stale_sdrs:
            logger.info(f"Skipped {len(stale_sdrs)} stale SDRs (will check async)")
            asyncio.create_task(self._check_stale_sdrs_async(stale_sdrs))

        return available

    def _score_kiwisdr(
        self,
        sdr_data: Dict[str, Any],
        expected_duration_minutes: int,
        prefer_location: Optional[str] = None
    ) -> Optional[SDRCandidate]:
        """Score a KiwiSDR candidate.

        Args:
            sdr_data: KiwiSDR data dictionary
            expected_duration_minutes: Expected session duration
            prefer_location: Preferred grid square

        Returns:
            Scored SDRCandidate or None if unsuitable
        """
        # Check if duration fits within session limit
        session_limit = sdr_data.get("session_limit_minutes", 90)
        if expected_duration_minutes > session_limit:
            return None

        # Calculate scores
        availability_score = min(
            1.0,
            sdr_data.get("remaining_daily_minutes", 0) / 60.0
        )
        reliability_score = sdr_data.get("reliability_score", 0.5)

        # Duration fit score (prefer SDRs where session fits comfortably)
        remaining = sdr_data.get("remaining_daily_minutes", 0)
        if remaining > 0:
            duration_fit = min(1.0, remaining / expected_duration_minutes)
        else:
            duration_fit = 0.0

        # Location preference bonus
        location_bonus = 0.0
        if prefer_location and sdr_data.get("grid_square") == prefer_location:
            location_bonus = 0.2

        # Calculate component scores matching dataclass
        geographic_score = location_bonus  # 0.0 or 0.2 based on location match
        load_score = availability_score  # How much capacity is available
        relationship_score = duration_fit  # How well duration fits

        # Calculate total score
        total_score = (
            load_score * 0.3 +
            reliability_score * 0.4 +
            relationship_score * 0.2 +
            geographic_score * 0.1
        )

        return SDRCandidate(
            sdr_id=sdr_data["sdr_id"],
            sdr_type=SDRType.KIWISDR,
            url=sdr_data["url"],
            institution_type=sdr_data.get("institution_type"),
            usage_policy=sdr_data.get("usage_policy", UsagePolicy.PUBLIC_LIMITED),
            grid_square=sdr_data.get("grid_square"),
            daily_limit_minutes=sdr_data["daily_limit_minutes"],
            session_limit_minutes=session_limit,
            remaining_daily_minutes=sdr_data.get("remaining_daily_minutes", 0),
            reliability_score=reliability_score,
            has_gps=sdr_data.get("has_gps", False),
            geographic_score=geographic_score,
            load_score=load_score,
            relationship_score=relationship_score,
            total_score=total_score
        )

    def _apply_hybrid_selection_rules(
        self,
        candidates: List[SDRCandidate],
        expected_duration_minutes: int
    ) -> Optional[SDRCandidate]:
        """Apply hybrid selection rules to choose best candidate.

        Args:
            candidates: List of scored candidates
            expected_duration_minutes: Expected session duration

        Returns:
            Best candidate or None
        """
        if not candidates:
            return None

        # Sort by total score (highest first)
        candidates.sort(key=lambda c: c.total_score, reverse=True)

        # Return highest scoring candidate
        return candidates[0]

    # TODO: Re-enable WebSDR
    # def _get_available_websdrs(self, frequency_khz: float) -> List[Dict[str, Any]]:
    #     """Get available WebSDR receivers from database with auto-sync.
    #
    #     Args:
    #         frequency_khz: Required frequency
    #
    #     Returns:
    #         List of available WebSDRs
    #     """
    #     # Check if we need to sync WebSDR registry
    #     if (self.last_websdr_sync is None or
    #         datetime.now(timezone.utc) - self.last_websdr_sync > self.sync_interval):
    #         self._sync_websdr_registry()
    #
    #     # Query WebSDR database
    #     websdrs = self.db.query(WebSDRSource).filter(
    #         WebSDRSource.active == True,
    #         WebSDRSource.failure_count < 5,
    #         WebSDRSource.min_freq_khz <= frequency_khz,
    #         WebSDRSource.max_freq_khz >= frequency_khz,
    #     ).all()
    #
    #     available = []
    #     for websdr in websdrs:
    #         # Check if WebSDR is stale (not connected recently)
    #         if (websdr.last_connected and
    #             websdr.last_connected < datetime.now(timezone.utc) - timedelta(hours=12)):
    #             logger.warning(f"WebSDR {websdr.url} not connected for 12+ hours, checking status")
    #             self._check_websdr_status(websdr)
    #
    #         # Check daily usage if limited
    #         if websdr.daily_limit_minutes:
    #             if websdr.daily_usage_minutes >= websdr.daily_limit_minutes:
    #                 continue  # Skip if daily limit reached
    #
    #         available.append({
    #             "sdr_id": str(websdr.websdr_id),
    #             "sdr_type": SDRType.WEBSDR,
    #             "url": websdr.url,
    #             "grid_square": websdr.grid_square,
    #             "daily_limit_minutes": websdr.daily_limit_minutes or 999999,
    #             "session_limit_minutes": websdr.session_limit_minutes or 180,
    #             "remaining_daily_minutes": (websdr.daily_limit_minutes - websdr.daily_usage_minutes) if websdr.daily_limit_minutes else 999999,
    #             "reliability_score": websdr.reliability_score or 0.8,
    #             "has_gps": True,  # Most WebSDRs have accurate timing
    #             "usage_policy": UsagePolicy.RESEARCH_AGREEMENT if websdr.has_research_agreement else UsagePolicy.PUBLIC_LIMITED,
    #             "institution_type": websdr.institution_type,
    #             "preferred_for_long": websdr.preferred_for_long_sessions,
    #         })
    #
    #     return available

    def _sync_kiwisdr_registry(self):
        """Sync KiwiSDR registry from public directory."""
        try:
            logger.info("Syncing KiwiSDR registry from public directory")

            # Fetch from KiwiSDR public directory
            import requests
            from bs4 import BeautifulSoup
            import re

            response = requests.get(
                "http://kiwisdr.com/public/",
                timeout=30,
                headers={"User-Agent": "CASCADE Data Collector/1.0"}
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Track which SDRs we've seen in this sync
            added_count = 0
            updated_count = 0
            batch_size = 50
            current_batch = 0

            # Find all receiver entries
            # Based on dyatlov parser: look for <div class='cl-info'>
            all_divs = soup.find_all('div', class_='cl-info')

            # Process in batches to reduce memory usage
            for batch_start in range(0, len(all_divs), batch_size):
                batch_divs = all_divs[batch_start:batch_start + batch_size]

                for div in batch_divs:
                    try:
                        # Extract data from HTML comments (format: <!-- fieldname=value -->)
                        comments = div.find_all(string=lambda text: isinstance(text, str) and '<!--' in str(text.parent))

                        data = {}
                        for comment in comments:
                            comment_text = str(comment).strip()
                            if comment_text.startswith('<!--') and comment_text.endswith('-->'):
                                # Parse <!-- fieldname=value -->
                                match = re.match(r'<!--\s*(\w+)=(.+?)\s*-->', comment_text)
                                if match:
                                    field, value = match.groups()
                                    data[field] = value.strip()

                        # Extract URL from link
                        link = div.find('a')
                        if not link or not link.get('href'):
                            continue

                        url = link.get('href').strip()
                        if not url:
                            continue

                        # Clean up URL (remove protocol if present)
                        url = re.sub(r'^https?://', '', url)
                        url = url.rstrip('/')

                        # Extract grid square (usually in 'grid' or 'gridsq' field)
                        grid_square = data.get('gridsq') or data.get('grid')

                        # Extract name
                        name = data.get('name') or link.text.strip()

                        # Extract antenna
                        antenna = data.get('ant') or data.get('antenna')

                        # Check if SDR exists in database
                        existing = self.db.query(KiwiSDRSource).filter(
                            KiwiSDRSource.url == url
                        ).first()

                        if existing:
                            # Update existing SDR
                            existing.last_seen = datetime.now(timezone.utc)
                            existing.active = True  # Mark as active since it's in directory
                            if grid_square:
                                existing.grid_square = grid_square[:6]  # Limit to 6 chars
                            if name and not existing.name:
                                existing.name = name[:255]
                            if antenna and not existing.antenna_type:
                                existing.antenna_type = antenna[:100]
                            updated_count += 1
                        else:
                            # Add new SDR
                            new_sdr = KiwiSDRSource(
                                url=url,
                                name=name[:255] if name else None,
                                grid_square=grid_square[:6] if grid_square else None,
                                antenna_type=antenna[:100] if antenna else None,
                                last_seen=datetime.now(timezone.utc),
                                active=True,
                                reliability_score=0.5,  # Start with neutral score
                                min_freq_khz=10,  # Default KiwiSDR range
                                max_freq_khz=30000,
                                has_gps=True,  # Most KiwiSDRs have GPS
                            )
                            self.db.add(new_sdr)
                            added_count += 1

                    except Exception as e:
                        logger.warning(f"Error parsing KiwiSDR entry: {e}")
                        continue

                # Commit batch and clear session to free memory
                self.db.commit()
                self.db.expire_all()  # Clear SQLAlchemy session cache
                current_batch += 1
                logger.debug(f"Processed batch {current_batch}: {len(batch_divs)} SDRs")

            # Mark old SDRs as inactive (use efficient query instead of loading all)
            # Only check SDRs not seen in this sync that are very stale (7+ days)
            stale_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            self.db.query(KiwiSDRSource).filter(
                KiwiSDRSource.last_seen < stale_cutoff,
                KiwiSDRSource.active == True
            ).update(
                {"active": False},
                synchronize_session=False
            )

            self.db.commit()
            self.last_kiwisdr_sync = datetime.now(timezone.utc)

            # Get total count efficiently
            total_count = self.db.query(KiwiSDRSource).count()
            logger.info(
                f"KiwiSDR registry sync completed: {added_count} added, "
                f"{updated_count} updated, {total_count} total in database"
            )

        except Exception as e:
            logger.error(f"Error syncing KiwiSDR registry: {e}")
            self.db.rollback()

    def _sync_websdr_registry(self):
        """Sync WebSDR registry from known sources."""
        # TODO: Re-enable WebSDR
        logger.info("WebSDR sync disabled for now")
        self.last_websdr_sync = datetime.now(timezone.utc)
        # try:
        #     logger.info("Syncing WebSDR registry from known sources")
        #
        #     # This would check known WebSDR URLs and update status
        #     active_websdrs = self.db.query(WebSDRSource).filter(
        #         WebSDRSource.active == True
        #     ).all()
        #
        #     # In a real implementation, you would:
        #     # 1. Check each WebSDR URL for availability
        #     # 2. Update connection status and capabilities
        #     # 3. Add new WebSDRs from discovery
        #
        #     self.last_websdr_sync = datetime.now(timezone.utc)
        #     logger.info(f"WebSDR registry sync completed, {len(active_websdrs)} SDRs in database")
        #
        # except Exception as e:
        #     logger.error(f"Error syncing WebSDR registry: {e}")

    async def _check_kiwisdr_status(self, sdr: KiwiSDRSource):
        """Check if a KiwiSDR is still online and update status (async)."""
        try:
            # Clean URL - remove both http:// and https:// prefixes
            clean_url = sdr.url.replace('https://', '').replace('http://', '')

            # Determine protocol from original URL
            protocol = 'https' if 'https://' in sdr.url else 'http'

            # Use aiohttp for async HTTP check
            timeout = aiohttp.ClientTimeout(total=5)  # Reduced timeout for faster checks
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{protocol}://{clean_url}") as response:
                    if response.status == 200:
                        # SDR is online
                        sdr.last_seen = datetime.now(timezone.utc)
                        sdr.failure_count = max(0, sdr.failure_count - 1)
                        if sdr.reliability_score:
                            sdr.reliability_score = min(1.0, sdr.reliability_score + 0.05)
                    else:
                        # SDR returned error
                        sdr.failure_count += 1
                        if sdr.reliability_score:
                            sdr.reliability_score = max(0.1, sdr.reliability_score - 0.1)

        except Exception as e:
            # Connection failed (don't log at warning level to reduce noise)
            logger.debug(f"KiwiSDR {sdr.url} connection failed: {e}")
            sdr.failure_count += 1
            if sdr.reliability_score:
                sdr.reliability_score = max(0.1, sdr.reliability_score - 0.1)

            # Mark as inactive if too many failures
            if sdr.failure_count >= 10:
                sdr.active = False

    async def _check_stale_sdrs_async(self, stale_sdrs: List[KiwiSDRSource]):
        """Check multiple stale SDRs concurrently in background."""
        logger.debug(f"Starting async health check for {len(stale_sdrs)} stale SDRs")

        # Run all checks concurrently
        tasks = [self._check_kiwisdr_status(sdr) for sdr in stale_sdrs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Commit changes to database
        try:
            self.db.commit()
            online_count = sum(1 for sdr in stale_sdrs if sdr.failure_count == 0)
            logger.info(f"Health check complete: {online_count}/{len(stale_sdrs)} stale SDRs are online")
        except Exception as e:
            logger.error(f"Error committing health check results: {e}")
            self.db.rollback()

    # TODO: Re-enable WebSDR
    # def _check_websdr_status(self, websdr: WebSDRSource):
    #     """Check if a WebSDR is still online and update status."""
    #     try:
    #         import requests
    #
    #         # Try to connect to the WebSDR
    #         response = requests.get(websdr.url, timeout=15)
    #
    #         if response.status_code == 200 and "websdr" in response.text.lower():
    #             # WebSDR is online and responding
    #             websdr.last_connected = datetime.now(timezone.utc)
    #             websdr.failure_count = max(0, websdr.failure_count - 1)
    #             if websdr.reliability_score:
    #                 websdr.reliability_score = min(1.0, websdr.reliability_score + 0.05)
    #         else:
    #             # WebSDR returned error or unexpected content
    #             websdr.failure_count += 1
    #             if websdr.reliability_score:
    #                 websdr.reliability_score = max(0.1, websdr.reliability_score - 0.1)
    #
    #     except Exception as e:
    #         # Connection failed
    #         logger.warning(f"WebSDR {websdr.url} connection failed: {e}")
    #         websdr.failure_count += 1
    #         if websdr.reliability_score:
    #             websdr.reliability_score = max(0.1, websdr.reliability_score - 0.1)
    #
    #         # Mark as inactive if too many failures
    #         if websdr.failure_count >= 5:  # Lower threshold for WebSDRs
    #             websdr.active = False
    #             logger.warning(f"Marking WebSDR {websdr.url} as inactive due to repeated failures")
    #
    #     self.db.commit()

    def update_sdr_usage(self, sdr_id: str, sdr_type: SDRType, usage_minutes: float, success: bool):
        """Update SDR usage statistics after a session.

        Args:
            sdr_id: SDR identifier
            sdr_type: Type of SDR
            usage_minutes: Minutes used in session
            success: Whether session was successful
        """
        try:
            if sdr_type == SDRType.KIWISDR:
                sdr = self.db.query(KiwiSDRSource).filter(
                    KiwiSDRSource.kiwisdr_id == sdr_id
                ).first()
            # TODO: Re-enable WebSDR
            # else:
            #     sdr = self.db.query(WebSDRSource).filter(
            #         WebSDRSource.websdr_id == sdr_id
            #     ).first()
            else:
                logger.warning(f"WebSDR usage tracking disabled")
                return

            if sdr:
                # Update usage statistics
                sdr.daily_usage_minutes += usage_minutes
                sdr.total_usage_minutes += usage_minutes
                
                if success:
                    # Successful session
                    if sdr_type == SDRType.KIWISDR:
                        sdr.last_seen = datetime.now(timezone.utc)
                    else:
                        sdr.last_connected = datetime.now(timezone.utc)
                    
                    # Improve reliability score
                    if sdr.reliability_score:
                        sdr.reliability_score = min(1.0, sdr.reliability_score + 0.02)
                    else:
                        sdr.reliability_score = 0.8
                        
                    # Reset failure count on success
                    sdr.failure_count = max(0, sdr.failure_count - 1)
                else:
                    # Failed session
                    sdr.failure_count += 1
                    if sdr.reliability_score:
                        sdr.reliability_score = max(0.1, sdr.reliability_score - 0.05)
                
                self.db.commit()
                logger.info(f"Updated {sdr_type.value} {sdr_id} usage: +{usage_minutes:.1f}min, success={success}")
            
        except Exception as e:
            logger.error(f"Error updating SDR usage: {e}")

    def force_registry_sync(self):
        """Force immediate sync of both SDR registries."""
        logger.info("Forcing immediate registry sync")
        self.last_kiwisdr_sync = None
        self.last_websdr_sync = None
        self._sync_kiwisdr_registry()
        self._sync_websdr_registry()

    def get_registry_status(self) -> Dict[str, Any]:
        """Get status of SDR registries and sync information."""
        try:
            kiwi_total = self.db.query(KiwiSDRSource).count()
            kiwi_active = self.db.query(KiwiSDRSource).filter(KiwiSDRSource.active == True).count()
            kiwi_recent = self.db.query(KiwiSDRSource).filter(
                KiwiSDRSource.last_seen > datetime.now(timezone.utc) - timedelta(hours=24)
            ).count()

            # TODO: Re-enable WebSDR
            # web_total = self.db.query(WebSDRSource).count()
            # web_active = self.db.query(WebSDRSource).filter(WebSDRSource.active == True).count()
            # web_recent = self.db.query(WebSDRSource).filter(
            #     WebSDRSource.last_connected > datetime.now(timezone.utc) - timedelta(hours=24)
            # ).count()

            return {
                "sync_status": {
                    "last_kiwisdr_sync": self.last_kiwisdr_sync.isoformat() if self.last_kiwisdr_sync else None,
                    "last_websdr_sync": "disabled",  # TODO: Re-enable WebSDR
                    "sync_interval_hours": self.sync_interval.total_seconds() / 3600,
                },
                "kiwisdr": {
                    "total_count": kiwi_total,
                    "active_count": kiwi_active,
                    "seen_24h": kiwi_recent,
                    "active_percentage": (kiwi_active / kiwi_total * 100) if kiwi_total > 0 else 0,
                },
                "websdr": {
                    "status": "disabled",  # TODO: Re-enable WebSDR
                    # "total_count": web_total,
                    # "active_count": web_active,
                    # "connected_24h": web_recent,
                    # "active_percentage": (web_active / web_total * 100) if web_total > 0 else 0,
                },
                "recommendations": {
                    "needs_kiwisdr_sync": self.last_kiwisdr_sync is None or datetime.now(timezone.utc) - self.last_kiwisdr_sync > self.sync_interval,
                    "needs_websdr_sync": False,  # TODO: Re-enable WebSDR
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting registry status: {e}")
            return {"error": str(e)}

    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()


# Utility functions
async def test_hybrid_selection() -> Dict[str, Any]:
    """Test hybrid selection algorithm.

    Returns:
        Test results
    """
    selector = HybridSDRSelector()

    try:
        # Test short session selection
        short_candidate = selector.select_optimal_sdr(
            frequency_khz=14080,
            expected_duration_minutes=30,
            band="20m"
        )

        # Test long session selection
        long_candidate = selector.select_optimal_sdr(
            frequency_khz=14080,
            expected_duration_minutes=120,
            band="20m"
        )

        # Get statistics
        stats = selector.get_selection_statistics()

        return {
            "short_session_result": {
                "candidate": short_candidate.sdr_type.value if short_candidate else None,
                "score": short_candidate.total_score if short_candidate else 0,
            },
            "long_session_result": {
                "candidate": long_candidate.sdr_type.value if long_candidate else None,
                "score": long_candidate.total_score if long_candidate else 0,
            },
            "statistics": stats,
        }

    finally:
        selector.close()