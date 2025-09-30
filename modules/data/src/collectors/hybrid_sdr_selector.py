"""Hybrid SDR selection algorithm for KiwiSDR and WebSDR coordination.

Implements T028e: Hybrid SDR selection algorithm (FR-067).
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

from ..models import SessionLocal, KiwiSDRSource, WebSDRSource
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
    daily_limit_minutes: int
    session_limit_minutes: int
    remaining_daily_minutes: float
    reliability_score: float
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

    def select_optimal_sdr(
        self,
        frequency_khz: float,
        expected_duration_minutes: int,
        band: str,
        prefer_location: Optional[str] = None,
        require_gps: bool = True
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
        kiwisdrs = self._get_available_kiwisdrs(frequency_khz, require_gps)
        websdrs = self._get_available_websdrs(frequency_khz)

        # Score all candidates
        candidates = []

        # Score KiwiSDRs
        for kiwisdr in kiwisdrs:
            candidate = self._score_kiwisdr(
                kiwisdr, expected_duration_minutes, prefer_location
            )
            if candidate:
                candidates.append(candidate)

        # Score WebSDRs
        for websdr in websdrs:
            candidate = self._score_websdr(
                websdr, expected_duration_minutes, prefer_location
            )
            if candidate:
                candidates.append(candidate)

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
        self, frequency_khz: float, require_gps: bool
    ) -> List[Dict[str, Any]]:
        """Get available KiwiSDR receivers.

        Args:
            frequency_khz: Required frequency
            require_gps: GPS requirement

        Returns:
            List of available KiwiSDRs
        """
        query = self.db.query(KiwiSDRSource).filter(
            KiwiSDRSource.active == True,
            KiwiSDRSource.failure_count < 5,
            KiwiSDRSource.frequency_min_khz <= frequency_khz,
            KiwiSDRSource.frequency_max_khz >= frequency_khz,
        )

        if require_gps:
            query = query.filter(KiwiSDRSource.has_gps == True)

        sdrs = query.all()

        # Filter by remaining daily usage
        available = []
        for sdr in sdrs:
            if sdr.should_reset_usage():
                sdr.daily_usage_minutes = 0
                sdr.last_usage_reset = datetime.utcnow()
                self.db.commit()

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

        return available

    def _get_available_websdrs(self, frequency_khz: float) -> List[Dict[str, Any]]:
        """Get available WebSDR receivers from database (FR-065, FR-067).

        Args:
            frequency_khz: Required frequency

        Returns:
            List of available WebSDRs
        """
        # Query WebSDR database
        websdrs = self.db.query(WebSDRSource).filter(
            WebSDRSource.active == True,
            WebSDRSource.failure_count < 5,
            WebSDRSource.min_freq_khz <= frequency_khz,
            WebSDRSource.max_freq_khz >= frequency_khz,
        ).all()

        available = []
        for websdr in websdrs:
            # Check daily usage if limited
            if websdr.daily_limit_minutes:
                if websdr.daily_usage_minutes >= websdr.daily_limit_minutes:
                    continue  # Skip if daily limit reached

            available.append({
                "sdr_id": str(websdr.websdr_id),
                "sdr_type": SDRType.WEBSDR,
                "url": websdr.url,
                "grid_square": websdr.grid_square,
                "daily_limit_minutes": websdr.daily_limit_minutes or 999999,
                "session_limit_minutes": websdr.session_limit_minutes or 180,
                "remaining_daily_minutes": (websdr.daily_limit_minutes - websdr.daily_usage_minutes) if websdr.daily_limit_minutes else 999999,
                "reliability_score": websdr.reliability_score or 0.8,
                "has_gps": True,  # Most WebSDRs have accurate timing
                "usage_policy": UsagePolicy.RESEARCH_AGREEMENT if websdr.has_research_agreement else UsagePolicy.PUBLIC_LIMITED,
                "institution_type": websdr.institution_type,
                "preferred_for_long": websdr.preferred_for_long_sessions,
            })

        # Also check hardcoded registry for backwards compatibility
        for websdr in self.websdr_registry:
            capabilities = websdr.get("capabilities", {})
            policy = websdr.get("usage_policy", {})

            # Check frequency support
            min_freq = capabilities.get("frequency_min_khz", 0)
            max_freq = capabilities.get("frequency_max_khz", 30000)

            if min_freq <= frequency_khz <= max_freq:
                available.append({
                    "sdr_id": websdr["url"],  # Use URL as ID for WebSDRs
                    "sdr_type": SDRType.WEBSDR,
                    "url": websdr["url"],
                    "grid_square": capabilities.get("grid_square", "XX00"),
                    "daily_limit_minutes": policy.get("daily_limit_minutes", 0),
                    "session_limit_minutes": policy.get("session_limit_minutes", 180),
                    "remaining_daily_minutes": 999999,  # Effectively unlimited
                    "reliability_score": 0.8,  # Assume good reliability
                    "has_gps": capabilities.get("has_gps", True),
                    "usage_policy": UsagePolicy.RESEARCH_AGREEMENT,
                    "institution_type": InstitutionType.UNIVERSITY,
                })

        return available

    def _score_kiwisdr(
        self,
        kiwisdr: Dict[str, Any],
        duration_minutes: int,
        prefer_location: Optional[str]
    ) -> Optional[SDRCandidate]:
        """Score KiwiSDR candidate with scarcity bonus (T084).

        Args:
            kiwisdr: KiwiSDR information
            duration_minutes: Required duration
            prefer_location: Preferred location

        Returns:
            Scored candidate or None
        """
        # Duration suitability
        remaining = kiwisdr["remaining_daily_minutes"]
        session_limit = kiwisdr["session_limit_minutes"]

        if duration_minutes > remaining or duration_minutes > session_limit:
            return None  # Can't meet duration requirement

        # Score duration fit (better if we use less of the available time)
        duration_score = min(1.0, remaining / (duration_minutes * 2))

        # Reliability score
        reliability_score = kiwisdr["reliability_score"]

        # Geographic score (includes quota priority boost)
        geographic_score = self._calculate_geographic_score(
            kiwisdr["grid_square"], prefer_location
        )

        # Load score (usage-based)
        usage_ratio = (kiwisdr["daily_limit_minutes"] - remaining) / kiwisdr["daily_limit_minutes"]
        load_score = 1.0 - usage_ratio

        # Relationship score (public KiwiSDRs have no special relationship)
        relationship_score = 0.5

        # Scarcity score (T084a)
        scarcity_score = self._calculate_scarcity_score(kiwisdr["grid_square"])

        # Calculate total score with scarcity bonus
        total_score = (
            duration_score * self.weights["duration_suitability"] +
            reliability_score * self.weights["reliability"] +
            geographic_score * self.weights["geographic_diversity"] +
            load_score * self.weights["load_balance"] +
            relationship_score * self.weights["relationship_status"] +
            scarcity_score * self.weights["scarcity_bonus"]
        )

        return SDRCandidate(
            sdr_id=kiwisdr["sdr_id"],
            sdr_type=SDRType.KIWISDR,
            url=kiwisdr["url"],
            institution_type=kiwisdr["institution_type"],
            usage_policy=kiwisdr["usage_policy"],
            daily_limit_minutes=kiwisdr["daily_limit_minutes"],
            session_limit_minutes=session_limit,
            remaining_daily_minutes=remaining,
            reliability_score=reliability_score,
            geographic_score=geographic_score,
            load_score=load_score,
            relationship_score=relationship_score,
            total_score=total_score,
        )

    def _score_websdr(
        self,
        websdr: Dict[str, Any],
        duration_minutes: int,
        prefer_location: Optional[str]
    ) -> Optional[SDRCandidate]:
        """Score WebSDR candidate with scarcity bonus (T084).

        Args:
            websdr: WebSDR information
            duration_minutes: Required duration
            prefer_location: Preferred location

        Returns:
            Scored candidate or None
        """
        session_limit = websdr["session_limit_minutes"]

        if duration_minutes > session_limit:
            return None  # Can't meet duration requirement

        # Duration suitability (WebSDRs excel at longer sessions)
        if duration_minutes > 90:
            duration_score = 1.0  # Perfect for long sessions
        elif duration_minutes > 30:
            duration_score = 0.8  # Good for medium sessions
        else:
            duration_score = 0.6  # OK for short sessions

        # Reliability score (assume good for institutional receivers)
        reliability_score = websdr["reliability_score"]

        # Geographic score (includes quota priority boost)
        geographic_score = self._calculate_geographic_score(
            websdr["grid_square"], prefer_location
        )

        # Load score (WebSDRs typically handle load better)
        load_score = 0.9  # Assume good load handling

        # Relationship score (higher for institutional relationships)
        if websdr["institution_type"] == InstitutionType.UNIVERSITY:
            relationship_score = 0.9  # Strong research relationship
        elif websdr["institution_type"] == InstitutionType.RESEARCH_INSTITUTE:
            relationship_score = 0.95  # Excellent research relationship
        else:
            relationship_score = 0.7  # Good amateur relationship

        # Scarcity score (T084a) - WebSDRs often in scarce locations
        scarcity_score = self._calculate_scarcity_score(websdr["grid_square"])

        # Calculate total score with scarcity bonus
        total_score = (
            duration_score * self.weights["duration_suitability"] +
            reliability_score * self.weights["reliability"] +
            geographic_score * self.weights["geographic_diversity"] +
            load_score * self.weights["load_balance"] +
            relationship_score * self.weights["relationship_status"] +
            scarcity_score * self.weights["scarcity_bonus"]
        )

        return SDRCandidate(
            sdr_id=websdr["sdr_id"],
            sdr_type=SDRType.WEBSDR,
            url=websdr["url"],
            institution_type=websdr["institution_type"],
            usage_policy=websdr["usage_policy"],
            daily_limit_minutes=websdr["daily_limit_minutes"],
            session_limit_minutes=session_limit,
            remaining_daily_minutes=websdr["remaining_daily_minutes"],
            reliability_score=reliability_score,
            geographic_score=geographic_score,
            load_score=load_score,
            relationship_score=relationship_score,
            total_score=total_score,
        )

    def _apply_hybrid_selection_rules(
        self, candidates: List[SDRCandidate], duration_minutes: int
    ) -> Optional[SDRCandidate]:
        """Apply hybrid selection rules with diversity bias (FR-067, T084).

        Args:
            candidates: All scored candidates
            duration_minutes: Required duration

        Returns:
            Best candidate after applying hybrid rules
        """
        if not candidates:
            return None

        # Apply ocean/land path balancing (T084c)
        candidates = self._check_ocean_path_balance(candidates)

        # FR-067: Prioritize WebSDR for longer sessions (>90 minutes)
        if duration_minutes > 90:
            websdr_candidates = [c for c in candidates if c.sdr_type == SDRType.WEBSDR]
            if websdr_candidates:
                # Prefer WebSDR for long sessions
                websdr_candidates.sort(key=lambda c: c.total_score, reverse=True)
                logger.info(f"Long session ({duration_minutes} min): preferring WebSDR")
                return websdr_candidates[0]

        # For shorter sessions, consider both types but apply preferences
        all_candidates = sorted(candidates, key=lambda c: c.total_score, reverse=True)

        # If duration is short (<30 min), prefer KiwiSDR
        if duration_minutes < 30:
            kiwisdr_candidates = [c for c in all_candidates if c.sdr_type == SDRType.KIWISDR]
            if kiwisdr_candidates:
                logger.info(f"Short session ({duration_minutes} min): preferring KiwiSDR")
                return kiwisdr_candidates[0]

        # Medium duration (30-90 min): choose best regardless of type
        logger.info(f"Medium session ({duration_minutes} min): choosing best available")

        # Log if we selected a scarce region SDR (T084 tracking)
        if all_candidates[0]:
            selected = all_candidates[0]
            scarcity = self._calculate_scarcity_score(
                selected.sdr_id[:4] if len(selected.sdr_id) >= 4 else ""
            )
            if scarcity > 0.7:
                logger.info(f"Selected SDR from scarce region (scarcity score: {scarcity:.2f})")

        return all_candidates[0]

    def _calculate_geographic_score(
        self, grid_square: str, prefer_location: Optional[str]
    ) -> float:
        """Calculate geographic preference score with diversity bias (T084).

        Args:
            grid_square: SDR grid square
            prefer_location: Preferred grid square prefix

        Returns:
            Geographic score (0.0-1.0)
        """
        base_score = 0.5  # Neutral

        # Basic geographic preference
        if prefer_location and grid_square:
            # Exact grid square match
            if grid_square == prefer_location:
                base_score = 1.0
            # Same grid square prefix (e.g., FN)
            elif len(prefer_location) >= 2 and grid_square.startswith(prefer_location[:2]):
                base_score = 0.8
            else:
                base_score = 0.3

        # Apply quota-based priority boost (T084)
        quota_priority = self.quota_manager.should_prioritize(grid_square)

        # Combine base score with quota priority
        # Higher quota priority increases the geographic score
        combined_score = base_score * 0.6 + min(1.0, quota_priority / 3.0) * 0.4

        return combined_score

    def _calculate_scarcity_score(self, grid_square: str) -> float:
        """Calculate scarcity score based on SDR density (T084a).

        Args:
            grid_square: SDR grid square

        Returns:
            Scarcity score (0.0-1.0, higher = more scarce)
        """
        # Update density cache if stale
        if (self.density_cache_time is None or
            datetime.utcnow() - self.density_cache_time > timedelta(hours=1)):
            self._update_sdr_density_cache()

        # Get grid prefix (first 2 characters)
        if not grid_square or len(grid_square) < 2:
            return 0.5

        grid_prefix = grid_square[:2].upper()

        # Get density for this grid square
        density = self.sdr_density_cache.get(grid_prefix, 5)  # Default to 5 SDRs

        # Calculate scarcity score (T084b)
        # < 5 SDRs per grid square prefix = scarce
        if density < 5:
            # 2x-3x weight multiplier for scarce regions
            scarcity_score = 1.0 - (density / 10.0)  # 0 SDRs = 1.0, 5 SDRs = 0.5
        else:
            # Not scarce
            scarcity_score = max(0.0, 0.5 - (density - 5) / 20.0)

        return scarcity_score

    def _update_sdr_density_cache(self):
        """Update SDR density cache for scarcity scoring (T084a)."""
        try:
            # Count SDRs per grid square prefix
            all_sdrs = self.db.query(KiwiSDRSource).filter(
                KiwiSDRSource.active == True
            ).all()

            density_map = {}
            for sdr in all_sdrs:
                if sdr.grid_square and len(sdr.grid_square) >= 2:
                    prefix = sdr.grid_square[:2].upper()
                    density_map[prefix] = density_map.get(prefix, 0) + 1

            self.sdr_density_cache = density_map
            self.density_cache_time = datetime.utcnow()

            logger.info(f"Updated SDR density cache: {len(density_map)} grid prefixes")

        except Exception as e:
            logger.error(f"Error updating SDR density cache: {e}")
            # Use default values if update fails
            self.sdr_density_cache = {}
            self.density_cache_time = datetime.utcnow()

    def _check_ocean_path_balance(self, candidates: List[SDRCandidate]) -> List[SDRCandidate]:
        """Apply ocean/land path balancing (T084c).

        Args:
            candidates: List of SDR candidates

        Returns:
            Potentially reordered list prioritizing ocean paths if needed
        """
        # Calculate current ocean path percentage
        progress = self.quota_manager.get_collection_progress()
        ocean_percentage = progress.ocean_path_percentage

        # If ocean paths are underrepresented (< 30% minimum)
        if ocean_percentage < 30.0:
            # Boost ocean-capable SDRs
            ocean_candidates = []
            land_candidates = []

            for candidate in candidates:
                # Check if this SDR is likely to capture ocean paths
                is_ocean_capable = self.grid_classifier.is_ocean_grid(
                    candidate.sdr_id[:4] if len(candidate.sdr_id) >= 4 else ""
                )

                if is_ocean_capable:
                    # Apply 1.3x boost to ocean-capable SDRs
                    candidate.total_score *= 1.3
                    ocean_candidates.append(candidate)
                else:
                    land_candidates.append(candidate)

            # Recombine with ocean SDRs prioritized
            candidates = sorted(ocean_candidates, key=lambda c: c.total_score, reverse=True)
            candidates.extend(sorted(land_candidates, key=lambda c: c.total_score, reverse=True))

        return candidates

    async def allocate_sdrs_for_event(
        self,
        event_type: str,
        target_count: int,
        frequency_list: List[float],
        duration_minutes: int = 60
    ) -> List[SDRCandidate]:
        """Allocate SDRs for space weather event using hybrid strategy.

        Args:
            event_type: Type of event (storm, flare, etc.)
            target_count: Number of SDRs needed
            frequency_list: List of frequencies to monitor
            duration_minutes: Expected session duration

        Returns:
            List of allocated SDR candidates
        """
        logger.info(
            f"Allocating {target_count} SDRs for {event_type} event, "
            f"duration: {duration_minutes} min"
        )

        allocated = []
        used_urls = set()

        for frequency_khz in frequency_list:
            if len(allocated) >= target_count:
                break

            # Determine optimal strategy for this frequency/duration
            if duration_minutes > 90:
                # Long sessions: prefer WebSDR
                prefer_types = [SDRType.WEBSDR, SDRType.KIWISDR]
            elif duration_minutes < 30:
                # Short sessions: prefer KiwiSDR
                prefer_types = [SDRType.KIWISDR, SDRType.WEBSDR]
            else:
                # Medium sessions: no preference
                prefer_types = [SDRType.WEBSDR, SDRType.KIWISDR]

            for sdr_type in prefer_types:
                candidate = self.select_optimal_sdr(
                    frequency_khz, duration_minutes, f"{frequency_khz/1000:.0f}m"
                )

                if (candidate and
                    candidate.sdr_type == sdr_type and
                    candidate.url not in used_urls):

                    allocated.append(candidate)
                    used_urls.add(candidate.url)
                    break

        # If we need more SDRs, relax preferences
        remaining_needed = target_count - len(allocated)
        if remaining_needed > 0:
            logger.info(f"Need {remaining_needed} more SDRs, relaxing preferences")

            for frequency_khz in frequency_list:
                if len(allocated) >= target_count:
                    break

                candidate = self.select_optimal_sdr(
                    frequency_khz, duration_minutes, f"{frequency_khz/1000:.0f}m"
                )

                if candidate and candidate.url not in used_urls:
                    allocated.append(candidate)
                    used_urls.add(candidate.url)

        logger.info(
            f"Allocated {len(allocated)}/{target_count} SDRs for {event_type}: "
            f"{sum(1 for c in allocated if c.sdr_type == SDRType.KIWISDR)} KiwiSDR, "
            f"{sum(1 for c in allocated if c.sdr_type == SDRType.WEBSDR)} WebSDR"
        )

        return allocated

    def get_hybrid_collection_strategy(
        self, total_hours_target: int, time_budget_days: int
    ) -> Dict[str, Any]:
        """Calculate optimal hybrid collection strategy.

        Args:
            total_hours_target: Total hours to collect
            time_budget_days: Days available for collection

        Returns:
            Recommended hybrid strategy
        """
        hours_per_day_needed = total_hours_target / time_budget_days

        # Calculate optimal mix based on session characteristics
        if hours_per_day_needed > 200:
            # High volume: need both types
            kiwisdr_percentage = 0.6  # Short sessions
            websdr_percentage = 0.4   # Long sessions
            recommended_strategy = "AGGRESSIVE_HYBRID"
        elif hours_per_day_needed > 100:
            # Medium volume: moderate hybrid
            kiwisdr_percentage = 0.7
            websdr_percentage = 0.3
            recommended_strategy = "BALANCED_HYBRID"
        else:
            # Low volume: mostly KiwiSDR
            kiwisdr_percentage = 0.8
            websdr_percentage = 0.2
            recommended_strategy = "KIWISDR_PRIMARY"

        # Calculate session allocations
        avg_kiwisdr_session_hours = 1.0  # ~60 minutes average
        avg_websdr_session_hours = 2.5   # ~150 minutes average

        kiwisdr_hours_per_day = hours_per_day_needed * kiwisdr_percentage
        websdr_hours_per_day = hours_per_day_needed * websdr_percentage

        kiwisdr_sessions_per_day = kiwisdr_hours_per_day / avg_kiwisdr_session_hours
        websdr_sessions_per_day = websdr_hours_per_day / avg_websdr_session_hours

        return {
            "strategy": recommended_strategy,
            "target_hours_per_day": hours_per_day_needed,
            "kiwisdr_allocation": {
                "percentage": kiwisdr_percentage * 100,
                "hours_per_day": kiwisdr_hours_per_day,
                "sessions_per_day": kiwisdr_sessions_per_day,
                "concurrent_sdrs": int(kiwisdr_sessions_per_day / 24 * avg_kiwisdr_session_hours),
            },
            "websdr_allocation": {
                "percentage": websdr_percentage * 100,
                "hours_per_day": websdr_hours_per_day,
                "sessions_per_day": websdr_sessions_per_day,
                "concurrent_sdrs": int(websdr_sessions_per_day / 24 * avg_websdr_session_hours),
            },
            "total_concurrent_sdrs": int(
                kiwisdr_sessions_per_day / 24 * avg_kiwisdr_session_hours +
                websdr_sessions_per_day / 24 * avg_websdr_session_hours
            ),
            "advantages": {
                "kiwisdr": "Geographic diversity, individual operator relationships",
                "websdr": "Longer sessions, institutional backing, higher capacity"
            }
        }

    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get hybrid selection statistics.

        Returns:
            Selection statistics
        """
        try:
            # Count available SDRs by type
            kiwisdrs = self._get_available_kiwisdrs(14080, True)  # Test frequency
            websdrs = self._get_available_websdrs(14080)

            # Calculate capacity
            total_kiwisdr_capacity = sum(
                sdr["remaining_daily_minutes"] for sdr in kiwisdrs
            )
            total_websdr_capacity = sum(
                sdr["session_limit_minutes"] for sdr in websdrs
            ) * 10  # Estimate 10 sessions per day per WebSDR

            return {
                "available_sdrs": {
                    "kiwisdr_count": len(kiwisdrs),
                    "websdr_count": len(websdrs),
                    "total_count": len(kiwisdrs) + len(websdrs),
                },
                "daily_capacity_minutes": {
                    "kiwisdr_total": total_kiwisdr_capacity,
                    "websdr_total": total_websdr_capacity,
                    "combined_total": total_kiwisdr_capacity + total_websdr_capacity,
                },
                "daily_capacity_hours": {
                    "kiwisdr_hours": total_kiwisdr_capacity / 60,
                    "websdr_hours": total_websdr_capacity / 60,
                    "combined_hours": (total_kiwisdr_capacity + total_websdr_capacity) / 60,
                },
                "strategy_recommendations": {
                    "short_sessions": "Prefer KiwiSDR (30-90 min limits)",
                    "long_sessions": "Prefer WebSDR (180+ min capability)",
                    "high_volume": "Use aggressive hybrid strategy",
                    "research_access": "Coordinate with WebSDR institutions"
                }
            }

        except Exception as e:
            logger.error(f"Error getting selection statistics: {e}")
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