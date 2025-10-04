"""SpaceWeatherData model with X-ray classification.

Implements T022: SpaceWeatherData model with xray_class and xray_flux.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, Integer, String, JSON, Boolean, Enum
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
import enum

from .base import Base


class SolarCyclePhase(enum.Enum):
    """Solar cycle phases."""
    MINIMUM = "minimum"
    RISING = "rising"
    MAXIMUM = "maximum"
    DECLINING = "declining"


class QBOPhase(enum.Enum):
    """Quasi-Biennial Oscillation phases."""
    EASTERLY = "easterly"
    WESTERLY = "westerly"
    TRANSITION = "transition"


class Season(enum.Enum):
    """Astronomical seasons."""
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"


class SpaceWeatherData(Base):
    """Solar and geomagnetic conditions for event correlation."""

    __tablename__ = "space_weather_data"

    # Primary key
    weather_id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        comment="Unique identifier",
    )

    # Timestamp
    observation_time = Column(
        DateTime(timezone=True),
        nullable=False,
        unique=True,
        index=True,
        comment="Observation timestamp (UTC)",
    )

    # Solar indices
    solar_flux = Column(
        Float,
        nullable=True,
        comment="10.7cm solar flux (SFU)",
    )

    sunspot_number = Column(
        Integer,
        nullable=True,
        comment="Sunspot number",
    )

    # X-ray data (FR-027)
    xray_class = Column(
        String(5),
        nullable=True,
        comment="X-ray flare class (A, B, C, M, X)",
    )

    xray_flux = Column(
        Float,
        nullable=True,
        comment="X-ray flux in W/m²",
    )

    xray_flare_start = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Flare start time",
    )

    xray_flare_peak = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Flare peak time",
    )

    # Geomagnetic indices
    k_index = Column(
        Integer,
        nullable=True,
        comment="Planetary K-index (0-9)",
    )

    a_index = Column(
        Integer,
        nullable=True,
        comment="Planetary A-index",
    )

    dst_index = Column(
        Integer,
        nullable=True,
        comment="Disturbance storm time index (nT)",
    )

    # Solar wind
    solar_wind_speed = Column(
        Float,
        nullable=True,
        comment="Solar wind speed (km/s)",
    )

    solar_wind_density = Column(
        Float,
        nullable=True,
        comment="Solar wind density (p/cm³)",
    )

    bz_component = Column(
        Float,
        nullable=True,
        comment="IMF Bz component (nT)",
    )

    # Propagation impact
    muf_3000 = Column(
        Float,
        nullable=True,
        comment="Maximum usable frequency for 3000km path (MHz)",
    )

    fof2 = Column(
        Float,
        nullable=True,
        comment="F2 layer critical frequency (MHz)",
    )

    # Event flags
    storm_level = Column(
        String(10),
        nullable=True,
        comment="Storm level: G1-G5, S1-S5, R1-R5",
    )

    aurora_visible = Column(
        Float,
        nullable=True,
        comment="Lowest latitude for aurora visibility",
    )

    # Raw data
    raw_data = Column(
        JSON,
        nullable=True,
        comment="Complete raw data from NOAA",
    )

    # Natural cycle tracking (FR-052, FR-053, FR-054)
    solar_cycle_phase = Column(
        Enum(SolarCyclePhase),
        nullable=True,
        comment="Current solar cycle phase",
    )

    solar_cycle_number = Column(
        Integer,
        nullable=True,
        default=25,
        comment="Solar cycle number (currently 25)",
    )

    qbo_index = Column(
        Float,
        nullable=True,
        comment="Quasi-Biennial Oscillation index (-40 to +40)",
    )

    qbo_phase = Column(
        Enum(QBOPhase),
        nullable=True,
        comment="QBO phase for equatorial propagation",
    )

    lunar_phase = Column(
        Float,
        nullable=True,
        comment="Lunar phase (0.0=new moon, 0.5=full moon, 1.0=new moon)",
    )

    lunar_age_days = Column(
        Integer,
        nullable=True,
        comment="Days since new moon (0-29)",
    )

    # Seasonal balancing (FR-050, FR-051, FR-056, FR-057)
    season = Column(
        Enum(Season),
        nullable=True,
        comment="Astronomical season",
    )

    seasonal_balance_factor = Column(
        Float,
        nullable=True,
        default=1.0,
        comment="Collection weighting factor for seasonal balance (0.8-1.3)",
    )

    equinoctial_enhancement = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="True during equinoctial periods (Mar 15-Apr 15, Sep 15-Oct 15)",
    )

    cycle_metadata = Column(
        JSON,
        nullable=True,
        comment="Additional natural cycle tracking data",
    )

    # 18-month collection window aggressive strategies (FR-055, FR-059)
    collection_window_factor = Column(
        Float,
        nullable=True,
        default=1.0,
        comment="18-month window urgency multiplier (1.0-10.0)",
    )

    opportunity_limited_mode = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Aggressive collection mode for limited 18-month window",
    )

    rarity_multiplier = Column(
        Float,
        nullable=True,
        default=1.0,
        comment="Applied rarity multiplier (5x-10x during solar minimum)",
    )

    # Metadata
    source = Column(
        String(50),
        nullable=False,
        default="NOAA",
        comment="Data source",
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="Record creation time",
    )

    def __repr__(self):
        return f"<SpaceWeatherData(time={self.observation_time}, K={self.k_index}, X-ray={self.xray_class})>"

    @property
    def is_storm_conditions(self) -> bool:
        """Check if storm conditions present."""
        return (
            (self.k_index and self.k_index >= 5)
            or (self.storm_level and self.storm_level[0] in ['G', 'S'])
            or (self.xray_class and self.xray_class[0] in ['M', 'X'])
        )

    @property
    def propagation_impact_level(self) -> str:
        """Assess propagation impact level."""
        if self.k_index:
            if self.k_index >= 7:
                return "severe"
            elif self.k_index >= 5:
                return "moderate"
            elif self.k_index >= 3:
                return "minor"
        return "quiet"

    def get_xray_scale(self) -> float:
        """Convert X-ray class to numerical scale."""
        if not self.xray_class:
            return 0.0

        scales = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
        base = scales.get(self.xray_class[0], 0)

        if len(self.xray_class) > 1:
            try:
                multiplier = float(self.xray_class[1:])
                return base * multiplier
            except ValueError:
                pass

        return base

    def is_equinoctial_period(self) -> bool:
        """Check if current time is during equinoctial enhancement periods."""
        if not self.observation_time:
            return False

        month = self.observation_time.month
        day = self.observation_time.day

        # March 15 - April 15
        if month == 3 and day >= 15:
            return True
        if month == 4 and day <= 15:
            return True

        # September 15 - October 15
        if month == 9 and day >= 15:
            return True
        if month == 10 and day <= 15:
            return True

        return False

    def get_seasonal_balance_factor(self) -> float:
        """Calculate seasonal balance factor based on current season."""
        if not self.season:
            return 1.0

        # Winter gets 20% boost (FR-051)
        # Other seasons get slightly reduced to maintain 25% balance
        if self.season == Season.WINTER:
            base_factor = 1.2  # 20% higher for winter
        else:
            base_factor = 0.93  # Slightly lower for other seasons to balance

        # Equinoctial periods get 30% boost (FR-056)
        if self.equinoctial_enhancement:
            base_factor *= 1.3

        # Constrain to valid range
        return max(0.8, min(1.3, base_factor))

    def is_solar_minimum_compensation_active(self) -> bool:
        """Check if solar minimum compensation should be applied (FR-055)."""
        if not self.solar_cycle_phase:
            return False

        # Apply during solar minimum and early rising phase
        return self.solar_cycle_phase in [SolarCyclePhase.MINIMUM, SolarCyclePhase.RISING] and (
            self.solar_flux and self.solar_flux < 120
        )

    def get_storm_threshold(self) -> int:
        """Get K-index threshold based on solar cycle phase (FR-055, FR-059)."""
        if self.is_solar_minimum_compensation_active() and self.opportunity_limited_mode:
            return 3  # Aggressive K≥3 threshold during 18-month solar minimum window
        elif self.is_solar_minimum_compensation_active():
            return 4  # Moderate threshold during solar minimum
        return 5  # Normal threshold

    def should_include_c_flares(self) -> bool:
        """Check if C-class flares should trigger collection (FR-055, FR-059)."""
        return self.is_solar_minimum_compensation_active() and self.opportunity_limited_mode

    def get_rarity_multiplier(self) -> float:
        """Calculate rarity multiplier for 18-month collection window (FR-055)."""
        if not self.opportunity_limited_mode:
            return 1.0

        # Aggressive multipliers during solar minimum (5x-10x)
        if self.is_solar_minimum_compensation_active():
            # Maximum boost for very rare events during solar minimum
            if (self.k_index and self.k_index >= 7) or (self.xray_class and self.xray_class.startswith('X')):
                return 10.0  # Ultra-rare events get maximum boost
            elif (self.k_index and self.k_index >= 5) or (self.xray_class and self.xray_class.startswith('M')):
                return 7.0   # Moderate events get high boost
            elif (self.k_index and self.k_index >= 3) or (self.xray_class and self.xray_class.startswith('C')):
                return 5.0   # Minor events get baseline aggressive boost

        # Moderate boost during other solar cycle phases
        elif self.solar_cycle_phase == SolarCyclePhase.RISING:
            return 2.0

        return self.rarity_multiplier or 1.0

    def is_100_percent_capture_required(self) -> bool:
        """Check if 100% capture rate is required (FR-059)."""
        if not self.opportunity_limited_mode:
            return False

        # 100% capture for any activity during solar minimum
        if self.is_solar_minimum_compensation_active():
            return (
                (self.k_index and self.k_index >= 3) or
                (self.xray_class and self.xray_class[0] in ['C', 'M', 'X']) or
                (self.storm_level and self.storm_level.startswith('G'))
            )

        return False

    def calculate_collection_urgency(self) -> float:
        """Calculate collection urgency for 18-month window."""
        if not self.opportunity_limited_mode:
            return 1.0

        urgency = 1.0

        # Base urgency during solar minimum
        if self.is_solar_minimum_compensation_active():
            urgency = 5.0  # High base urgency during solar minimum

        # Scale by event rarity
        rarity_mult = self.get_rarity_multiplier()
        urgency *= min(rarity_mult, 10.0)  # Cap at 10x

        # 18-month window time pressure
        if self.opportunity_limited_mode:
            urgency *= 1.5  # Additional urgency for limited collection window

        return min(urgency, 50.0)  # Cap at 50x maximum urgency

    def get_aggressive_sdr_target(self, baseline_count: int = 6) -> int:
        """Calculate aggressive SDR target for 18-month collection (FR-055, FR-059)."""
        if not self.opportunity_limited_mode:
            return baseline_count

        # 100% capture mode during solar minimum
        if self.is_100_percent_capture_required():
            if (self.k_index and self.k_index >= 7) or (self.xray_class and self.xray_class.startswith('X')):
                return 50  # Maximum deployment for extreme events
            elif (self.k_index and self.k_index >= 5) or (self.xray_class and self.xray_class.startswith('M')):
                return 40  # High deployment for major events
            elif (self.k_index and self.k_index >= 3) or self.should_include_c_flares():
                return 30  # Aggressive deployment for minor events during solar minimum

        # Enhanced scaling during other phases
        urgency = self.calculate_collection_urgency()
        if urgency > 10:
            return min(baseline_count * 6, 50)  # Scale up to 6x baseline
        elif urgency > 5:
            return min(baseline_count * 4, 40)  # Scale up to 4x baseline
        elif urgency > 2:
            return min(baseline_count * 2, 20)  # Scale up to 2x baseline

        return baseline_count

    def get_qbo_propagation_factor(self) -> float:
        """Get QBO-based propagation enhancement factor."""
        if not self.qbo_phase:
            return 1.0

        # Easterly phase enhances trans-equatorial propagation
        if self.qbo_phase == QBOPhase.EASTERLY:
            return 1.1
        # Westerly phase reduces it slightly
        elif self.qbo_phase == QBOPhase.WESTERLY:
            return 0.95

        return 1.0  # Transition phase