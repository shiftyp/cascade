"""NOAA space weather API client.

Implements T046: NOAA space weather API client with X-ray data.
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import httpx

from ..models import SessionLocal, SpaceWeatherData, SolarCyclePhase, QBOPhase, Season
from ..config import config

logger = logging.getLogger(__name__)


class NOAASpaceWeatherClient:
    """Client for NOAA Space Weather Prediction Center API."""

    BASE_URL = "https://services.swpc.noaa.gov"
    QBO_URL = "https://www.cpc.ncep.noaa.gov/data/indices/qbo.u30.index"

    def __init__(self):
        """Initialize NOAA client."""
        self.db = SessionLocal()
        self.client = httpx.AsyncClient(timeout=30.0)

    async def fetch_current_conditions(self) -> Dict[str, Any]:
        """Fetch current space weather conditions.

        Returns:
            Dictionary of current conditions
        """
        conditions = {}

        # Fetch different data products
        try:
            # Solar flux
            flux_data = await self._fetch_solar_flux()
            if flux_data:
                conditions.update(flux_data)

            # K-index
            k_index_data = await self._fetch_k_index()
            if k_index_data:
                conditions.update(k_index_data)

            # X-ray flux (FR-027)
            xray_data = await self._fetch_xray_flux()
            if xray_data:
                conditions.update(xray_data)

            # Solar wind
            wind_data = await self._fetch_solar_wind()
            if wind_data:
                conditions.update(wind_data)

            # QBO data (FR-053)
            qbo_data = await self._fetch_qbo_data()
            if qbo_data:
                conditions.update(qbo_data)

            # Calculate natural cycles
            cycle_data = self._calculate_natural_cycles()
            conditions.update(cycle_data)

        except Exception as e:
            logger.error(f"Error fetching NOAA data: {e}")

        return conditions

    async def _fetch_solar_flux(self) -> Optional[Dict[str, Any]]:
        """Fetch 10.7cm solar flux data.

        Returns:
            Solar flux data
        """
        try:
            url = f"{self.BASE_URL}/json/f107_cm_flux.json"
            response = await self.client.get(url)
            response.raise_for_status()

            data = response.json()
            if data:
                latest = data[-1]  # Most recent entry
                return {
                    "solar_flux": float(latest["flux"]),
                    "flux_time": latest["time_tag"],
                }
        except Exception as e:
            logger.error(f"Failed to fetch solar flux: {e}")

        return None

    async def _fetch_k_index(self) -> Optional[Dict[str, Any]]:
        """Fetch planetary K-index.

        Returns:
            K-index data
        """
        try:
            url = f"{self.BASE_URL}/json/planetary_k_index_1m.json"
            response = await self.client.get(url)
            response.raise_for_status()

            data = response.json()
            if data:
                latest = data[-1]
                return {
                    "k_index": int(latest["kp_index"]),
                    "a_index": int(latest.get("a_index", 0)),
                    "k_time": latest["time_tag"],
                }
        except Exception as e:
            logger.error(f"Failed to fetch K-index: {e}")

        return None

    async def _fetch_xray_flux(self) -> Optional[Dict[str, Any]]:
        """Fetch X-ray flux data (FR-027).

        Returns:
            X-ray flux data with classification
        """
        try:
            url = f"{self.BASE_URL}/json/goes/primary/xrays-6-hour.json"
            response = await self.client.get(url)
            response.raise_for_status()

            data = response.json()
            if data:
                latest = data[-1]

                # Parse X-ray flux values
                short_flux = float(latest.get("flux", 0))  # 0.5-4.0 Angstrom
                long_flux = float(latest.get("energy", 0))  # 1.0-8.0 Angstrom

                # Classify X-ray level
                xray_class = self._classify_xray(long_flux)

                return {
                    "xray_flux": long_flux,
                    "xray_flux_short": short_flux,
                    "xray_class": xray_class,
                    "xray_time": latest["time_tag"],
                }
        except Exception as e:
            logger.error(f"Failed to fetch X-ray flux: {e}")

        return None

    def _classify_xray(self, flux: float) -> str:
        """Classify X-ray flux level.

        Args:
            flux: X-ray flux in W/m²

        Returns:
            X-ray class (A, B, C, M, X)
        """
        if flux < 1e-8:
            return "A"
        elif flux < 1e-7:
            level = int(flux / 1e-8)
            return f"A{level}"
        elif flux < 1e-6:
            level = int(flux / 1e-7)
            return f"B{level}"
        elif flux < 1e-5:
            level = int(flux / 1e-6)
            return f"C{level}"
        elif flux < 1e-4:
            level = int(flux / 1e-5)
            return f"M{level}"
        else:
            level = int(flux / 1e-4)
            return f"X{level}"

    async def _fetch_solar_wind(self) -> Optional[Dict[str, Any]]:
        """Fetch solar wind data.

        Returns:
            Solar wind data
        """
        try:
            url = f"{self.BASE_URL}/json/rtsw/rtsw_wind_1m.json"
            response = await self.client.get(url)
            response.raise_for_status()

            data = response.json()
            if data:
                latest = data[-1]
                return {
                    "solar_wind_speed": float(latest.get("proton_speed", 0)),
                    "solar_wind_density": float(latest.get("proton_density", 0)),
                    "bz_component": float(latest.get("bz_gsm", 0)),
                    "wind_time": latest["time_tag"],
                }
        except Exception as e:
            logger.error(f"Failed to fetch solar wind: {e}")

        return None

    async def _fetch_qbo_data(self) -> Optional[Dict[str, Any]]:
        """Fetch Quasi-Biennial Oscillation data (FR-053).

        Returns:
            QBO index and phase data
        """
        try:
            response = await self.client.get(self.QBO_URL)
            response.raise_for_status()

            # Parse QBO data (text format with monthly values)
            lines = response.text.strip().split('\n')

            # Get most recent complete month
            current_data = None
            current_time = datetime.utcnow()

            for line in reversed(lines):
                if line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 13:  # Year + 12 months
                        year = int(parts[0])
                        month = current_time.month

                        # Get current month's QBO value
                        if year == current_time.year and month <= 12:
                            qbo_value = float(parts[month])

                            # Determine QBO phase
                            if qbo_value > 5:
                                qbo_phase = QBOPhase.WESTERLY
                            elif qbo_value < -5:
                                qbo_phase = QBOPhase.EASTERLY
                            else:
                                qbo_phase = QBOPhase.TRANSITION

                            return {
                                "qbo_index": qbo_value,
                                "qbo_phase": qbo_phase,
                            }

        except Exception as e:
            logger.error(f"Failed to fetch QBO data: {e}")

        return None

    def _calculate_natural_cycles(self) -> Dict[str, Any]:
        """Calculate natural cycle data (FR-052, FR-054).

        Returns:
            Natural cycle information
        """
        current_time = datetime.utcnow()

        # Calculate lunar phase (FR-054)
        lunar_data = self._calculate_lunar_phase(current_time)

        # Calculate season (FR-050)
        season = self._calculate_season(current_time)

        # Calculate solar cycle phase (FR-052)
        solar_cycle_data = self._estimate_solar_cycle_phase()

        # Check equinoctial enhancement (FR-056)
        equinoctial = self._is_equinoctial_period(current_time)

        # Calculate seasonal balance factor (FR-051, FR-057)
        seasonal_factor = self._calculate_seasonal_factor(season, equinoctial)

        # Calculate 18-month window aggressive factors (FR-055, FR-059)
        aggressive_data = self._calculate_18_month_urgency(solar_cycle_data["phase"])

        return {
            "lunar_phase": lunar_data["phase"],
            "lunar_age_days": lunar_data["age_days"],
            "season": season,
            "solar_cycle_phase": solar_cycle_data["phase"],
            "solar_cycle_number": solar_cycle_data["number"],
            "equinoctial_enhancement": equinoctial,
            "seasonal_balance_factor": seasonal_factor,
            # 18-month collection window aggressive strategies
            "collection_window_factor": aggressive_data["window_factor"],
            "opportunity_limited_mode": aggressive_data["limited_mode"],
            "rarity_multiplier": aggressive_data["rarity_multiplier"],
            "cycle_metadata": {
                "calculation_time": current_time.isoformat(),
                "solar_rotation_number": self._calculate_solar_rotation_number(current_time),
                "days_since_solar_min": self._days_since_solar_minimum(),
                "collection_window_months_remaining": aggressive_data["months_remaining"],
            }
        }

    def _calculate_lunar_phase(self, dt: datetime) -> Dict[str, Any]:
        """Calculate lunar phase and age.

        Args:
            dt: Date/time to calculate for

        Returns:
            Dictionary with phase (0.0-1.0) and age in days
        """
        # Reference new moon: January 6, 2000, 18:14 UTC
        ref_new_moon = datetime(2000, 1, 6, 18, 14)
        lunar_cycle_days = 29.53059  # Average synodic month

        # Days since reference new moon
        days_since_ref = (dt - ref_new_moon).total_seconds() / 86400

        # Lunar cycles since reference
        cycles = days_since_ref / lunar_cycle_days

        # Current cycle position (0.0-1.0)
        phase = cycles - int(cycles)

        # Age in days (0-29)
        age_days = int(phase * lunar_cycle_days)

        return {
            "phase": phase,
            "age_days": age_days
        }

    def _calculate_season(self, dt: datetime) -> Season:
        """Calculate astronomical season.

        Args:
            dt: Date/time to calculate for

        Returns:
            Season enum value
        """
        # Approximate season boundaries (Northern Hemisphere)
        month = dt.month
        day = dt.day

        if month == 12 or month <= 2:
            return Season.WINTER
        elif (month == 3 and day >= 20) or month in [4, 5] or (month == 6 and day < 21):
            return Season.SPRING
        elif (month == 6 and day >= 21) or month in [7, 8] or (month == 9 and day < 22):
            return Season.SUMMER
        else:
            return Season.AUTUMN

    def _estimate_solar_cycle_phase(self) -> Dict[str, Any]:
        """Estimate current solar cycle phase based on historical data.

        Returns:
            Solar cycle phase and number
        """
        # Solar Cycle 25 started in December 2019
        cycle_25_start = datetime(2019, 12, 1)
        current_time = datetime.utcnow()

        months_since_start = (current_time - cycle_25_start).days / 30.44

        # Solar cycles average 11 years (132 months)
        # Rough phases: 0-24 months rising, 24-60 maximum, 60-108 declining, 108-132 minimum

        if months_since_start < 24:
            phase = SolarCyclePhase.RISING
        elif months_since_start < 60:
            phase = SolarCyclePhase.MAXIMUM
        elif months_since_start < 108:
            phase = SolarCyclePhase.DECLINING
        else:
            phase = SolarCyclePhase.MINIMUM

        return {
            "phase": phase,
            "number": 25
        }

    def _is_equinoctial_period(self, dt: datetime) -> bool:
        """Check if date is in equinoctial enhancement period (FR-056).

        Args:
            dt: Date to check

        Returns:
            True if in equinoctial period
        """
        month = dt.month
        day = dt.day

        # March 15 - April 15
        if (month == 3 and day >= 15) or (month == 4 and day <= 15):
            return True

        # September 15 - October 15
        if (month == 9 and day >= 15) or (month == 10 and day <= 15):
            return True

        return False

    def _calculate_seasonal_factor(self, season: Season, equinoctial: bool) -> float:
        """Calculate seasonal balance factor (FR-051, FR-056).

        Args:
            season: Current season
            equinoctial: Is equinoctial period

        Returns:
            Seasonal balance factor (0.8-1.3)
        """
        # Winter gets 20% boost (FR-051)
        if season == Season.WINTER:
            base_factor = 1.2
        else:
            base_factor = 1.0

        # Equinoctial periods get 30% boost (FR-056)
        if equinoctial:
            base_factor *= 1.3

        # Constrain to specification limits
        return max(0.8, min(1.3, base_factor))

    def _calculate_18_month_urgency(self, solar_cycle_phase: SolarCyclePhase) -> Dict[str, Any]:
        """Calculate 18-month collection window urgency factors (FR-055, FR-059).

        Args:
            solar_cycle_phase: Current solar cycle phase

        Returns:
            Dictionary with urgency factors
        """
        # Collection window started Dec 2024, ends June 2026 (18 months)
        collection_start = datetime(2024, 12, 1)
        collection_end = datetime(2026, 6, 1)
        current_time = datetime.utcnow()

        # Calculate months remaining in collection window
        months_remaining = (collection_end - current_time).days / 30.44
        months_elapsed = (current_time - collection_start).days / 30.44

        # Base window factor increases as time runs out
        if months_remaining > 12:
            window_factor = 1.5  # Early collection period
        elif months_remaining > 6:
            window_factor = 2.5  # Mid collection period
        elif months_remaining > 3:
            window_factor = 5.0  # Late collection period
        else:
            window_factor = 10.0  # Critical final months

        # Aggressive mode during solar minimum (FR-055, FR-059)
        limited_mode = True  # Always enabled during 18-month window

        # Base rarity multiplier depends on solar cycle phase
        if solar_cycle_phase == SolarCyclePhase.MINIMUM:
            base_rarity = 5.0  # Aggressive baseline during solar minimum
        elif solar_cycle_phase == SolarCyclePhase.RISING:
            base_rarity = 3.0  # Moderate during rising phase
        else:
            base_rarity = 1.5  # Conservative during maximum/declining

        # Scale rarity by time pressure
        time_pressure_factor = max(1.0, 3.0 - (months_remaining / 6.0))
        rarity_multiplier = min(base_rarity * time_pressure_factor, 10.0)

        return {
            "window_factor": window_factor,
            "limited_mode": limited_mode,
            "rarity_multiplier": rarity_multiplier,
            "months_remaining": max(0, months_remaining),
            "months_elapsed": months_elapsed,
            "time_pressure_factor": time_pressure_factor,
        }

    def _calculate_solar_rotation_number(self, dt: datetime) -> int:
        """Calculate Carrington solar rotation number.

        Args:
            dt: Date to calculate for

        Returns:
            Solar rotation number
        """
        # Carrington Rotation 1 started Nov 9, 1853
        ref_date = datetime(1853, 11, 9)
        rotation_period_days = 27.2753  # Synodic period

        days_since = (dt - ref_date).total_seconds() / 86400
        return int(days_since / rotation_period_days) + 1

    def _days_since_solar_minimum(self) -> int:
        """Calculate days since Solar Cycle 25 minimum.

        Returns:
            Days since solar minimum
        """
        # Solar Cycle 25 minimum was December 2019
        solar_min = datetime(2019, 12, 1)
        return (datetime.utcnow() - solar_min).days

    async def store_conditions(self, conditions: Dict[str, Any]) -> Optional[SpaceWeatherData]:
        """Store space weather conditions in database.

        Args:
            conditions: Weather conditions dictionary

        Returns:
            Created SpaceWeatherData object
        """
        try:
            # Parse observation time
            obs_time = datetime.utcnow()
            if "flux_time" in conditions:
                try:
                    obs_time = datetime.fromisoformat(conditions["flux_time"].replace("Z", "+00:00"))
                except:
                    pass

            # Check if we already have data for this time
            existing = self.db.query(SpaceWeatherData).filter(
                SpaceWeatherData.observation_time == obs_time
            ).first()

            if existing:
                logger.debug(f"Space weather data already exists for {obs_time}")
                return existing

            # Create new record
            weather = SpaceWeatherData(
                observation_time=obs_time,
                solar_flux=conditions.get("solar_flux"),
                k_index=conditions.get("k_index"),
                a_index=conditions.get("a_index"),
                xray_flux=conditions.get("xray_flux"),
                xray_class=conditions.get("xray_class"),
                solar_wind_speed=conditions.get("solar_wind_speed"),
                solar_wind_density=conditions.get("solar_wind_density"),
                bz_component=conditions.get("bz_component"),
                # Natural cycle data
                solar_cycle_phase=conditions.get("solar_cycle_phase"),
                solar_cycle_number=conditions.get("solar_cycle_number"),
                qbo_index=conditions.get("qbo_index"),
                qbo_phase=conditions.get("qbo_phase"),
                lunar_phase=conditions.get("lunar_phase"),
                lunar_age_days=conditions.get("lunar_age_days"),
                season=conditions.get("season"),
                seasonal_balance_factor=conditions.get("seasonal_balance_factor"),
                equinoctial_enhancement=conditions.get("equinoctial_enhancement", False),
                cycle_metadata=conditions.get("cycle_metadata"),
                # 18-month aggressive collection fields
                collection_window_factor=conditions.get("collection_window_factor"),
                opportunity_limited_mode=conditions.get("opportunity_limited_mode", True),
                rarity_multiplier=conditions.get("rarity_multiplier"),
                raw_data=conditions,
            )

            self.db.add(weather)
            self.db.commit()

            logger.info(f"Stored space weather: K={weather.k_index}, X-ray={weather.xray_class}")
            return weather

        except Exception as e:
            logger.error(f"Failed to store space weather: {e}")
            self.db.rollback()
            return None

    async def check_for_alerts(self, conditions: Dict[str, Any]) -> List[str]:
        """Check conditions for alert triggers.

        Args:
            conditions: Current conditions

        Returns:
            List of alert messages
        """
        alerts = []

        # Check K-index
        k_index = conditions.get("k_index", 0)
        if k_index >= 7:
            alerts.append(f"SEVERE: Geomagnetic storm K={k_index}")
        elif k_index >= 5:
            alerts.append(f"MODERATE: Geomagnetic activity K={k_index}")

        # Check X-ray class
        xray_class = conditions.get("xray_class", "")
        if xray_class.startswith("X"):
            alerts.append(f"SEVERE: X-class solar flare {xray_class}")
        elif xray_class.startswith("M"):
            alerts.append(f"MODERATE: M-class solar flare {xray_class}")

        # Check solar wind
        wind_speed = conditions.get("solar_wind_speed", 0)
        if wind_speed > 700:
            alerts.append(f"HIGH: Solar wind speed {wind_speed:.0f} km/s")

        return alerts

    async def update_loop(self, interval_minutes: int = 15):
        """Continuous update loop for space weather.

        Args:
            interval_minutes: Update interval in minutes
        """
        logger.info(f"Starting NOAA update loop (interval: {interval_minutes} min)")

        while True:
            try:
                # Fetch current conditions
                conditions = await self.fetch_current_conditions()

                if conditions:
                    # Store in database
                    await self.store_conditions(conditions)

                    # Check for alerts
                    alerts = await self.check_for_alerts(conditions)
                    for alert in alerts:
                        logger.warning(f"Space weather alert: {alert}")

                # Wait for next update
                await asyncio.sleep(interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def get_propagation_forecast(self) -> Dict[str, Any]:
        """Get propagation forecast based on current conditions.

        Returns:
            Propagation forecast
        """
        conditions = await self.fetch_current_conditions()

        forecast = {
            "timestamp": datetime.utcnow().isoformat(),
            "conditions": "unknown",
            "recommendations": [],
        }

        if not conditions:
            return forecast

        k_index = conditions.get("k_index", 0)
        solar_flux = conditions.get("solar_flux", 100)

        # Determine overall conditions
        if k_index >= 7:
            forecast["conditions"] = "very_poor"
            forecast["recommendations"].append("Focus on lower bands (80m, 160m)")
        elif k_index >= 5:
            forecast["conditions"] = "poor"
            forecast["recommendations"].append("Expect enhanced auroral propagation")
        elif solar_flux > 150:
            forecast["conditions"] = "excellent"
            forecast["recommendations"].append("Good conditions on higher bands (10m, 6m)")
        elif solar_flux > 100:
            forecast["conditions"] = "good"
            forecast["recommendations"].append("Normal propagation on all bands")
        else:
            forecast["conditions"] = "fair"
            forecast["recommendations"].append("Limited propagation on higher bands")

        return forecast

    async def close(self):
        """Close client connections."""
        await self.client.aclose()
        self.db.close()


async def main():
    """Main entry point for testing."""
    client = NOAASpaceWeatherClient()

    try:
        # Fetch and display current conditions
        conditions = await client.fetch_current_conditions()
        print("Current Space Weather Conditions:")
        for key, value in conditions.items():
            print(f"  {key}: {value}")

        # Store in database
        await client.store_conditions(conditions)

        # Get forecast
        forecast = await client.get_propagation_forecast()
        print("\nPropagation Forecast:")
        print(f"  Conditions: {forecast['conditions']}")
        for rec in forecast["recommendations"]:
            print(f"  - {rec}")

    finally:
        await client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())