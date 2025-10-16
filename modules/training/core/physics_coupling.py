"""
Physics-coupled parameter calculation for CASCADE training.

All propagation effects, noise characteristics, and channel conditions are derived
from the same core physical drivers to ensure realistic coupling.

Key principle: QRN, propagation mode, absorption, multipath, etc. are NOT independent.
They all emerge from the same physical state (solar flux, geomagnetic activity,
time of day, season, latitude, weather).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from enum import Enum

# Import ITU-R P.533 model for accurate MUF/absorption
try:
    from itu_r_p533 import ITU_R_P533
    ITU_MODEL_AVAILABLE = True
except ImportError:
    ITU_MODEL_AVAILABLE = False


class PropagationMode(Enum):
    """HF propagation modes."""
    AWGN = "awgn"  # Very rare, only local noise
    RAYLEIGH = "rayleigh"  # Deep fading, multiple equal paths
    RICIAN = "rician"  # Dominant path + multipath
    MULTIPATH_SPARSE = "multipath_sparse"  # 2-3 paths
    MULTIPATH_DENSE = "multipath_dense"  # 4+ paths
    SPORADIC_E = "sporadic_e"  # Enhanced E-layer propagation (skip distance)


class QRNType(Enum):
    """Atmospheric noise types."""
    QUIET = "quiet"  # Thermal noise dominated
    STATIC = "static"  # Continuous atmospheric noise
    CRACKLING = "crackling"  # Lightning crashes
    HISS = "hiss"  # Auroral/ionospheric
    POPCORN = "popcorn"  # Sporadic impulses
    FLUTTERY = "fluttery"  # Rapid fading/modulation
    THUNDERSTORM = "thunderstorm"  # Severe static crashes
    AURORAL = "auroral"  # Polar auroral noise


@dataclass
class CorePhysicalDrivers:
    """
    Independent physical state variables.
    All other effects are derived from these.
    """
    # Solar conditions
    sfi: float  # Solar Flux Index (50-300, continuous)
    sunspot_number: float  # Smoothed sunspot number (0-250, continuous)

    # Geomagnetic conditions
    k_index: float  # Planetary K-index (0-9, continuous via distribution)
    a_index: float  # Planetary A-index (0-400, continuous)
    dst_index: float  # Disturbance Storm Time (-500 to +50 nT, continuous)

    # Time and location
    utc_hour: float  # UTC hour (0-24, continuous)
    day_of_year: int  # Day of year (1-365)
    latitude: float  # Station latitude (-90 to +90)
    longitude: float  # Station longitude (-180 to +180)

    # Local weather (affects atmospheric noise)
    thunderstorm_activity: float  # 0-1, continuous (local + regional)
    precipitation_rate: float  # mm/h, continuous

    # Band
    frequency_mhz: float  # Operating frequency

    # Solar activity events (optional, for SID modeling)
    solar_flare_probability: float = 0.0  # 0-1, probability of X-ray flare
    flare_intensity: float = 0.0  # If flare: X1, X5, X10 class (1, 5, 10)


@dataclass
class DerivedConditions:
    """
    All propagation and noise effects derived from core drivers.
    These are NOT independent - they're all coupled through physics.
    """
    # Propagation
    muf_mhz: float  # Maximum Usable Frequency
    luf_mhz: float  # Lowest Usable Frequency
    d_layer_absorption_db: float  # D-layer absorption
    e_layer_present: bool  # Sporadic E
    f_layer_height_km: float  # F-layer virtual height

    propagation_mode: PropagationMode
    multipath_delay_spread_ms: float
    doppler_spread_hz: float
    rician_k_factor: float  # K-factor if Rician

    # Noise
    dominant_qrn_type: QRNType
    qrn_components: Dict[QRNType, float]  # Mix of noise types with weights
    atmospheric_noise_db: float  # Above thermal floor
    man_made_noise_db: float  # QRM (depends on time/location)

    # Signal quality
    effective_snr_db: float  # What we'd measure
    signal_strength_s_units: int  # S-meter reading (1-9+)

    # Solar events (new)
    solar_flare_active: bool = False  # SID in progress
    sid_absorption_db: float = 0.0  # Additional absorption from SID

    # Pre-computed noise generation parameters (for GPU efficiency)
    atmospheric_burst_rate: float = 0.5  # Bursts/sec (K-index scaled)
    has_qrm: bool = False  # Whether this stream has QRM
    has_atmospheric_qrn: bool = False  # Whether has atmospheric QRN
    has_impulsive_qrn: bool = False  # Whether has impulsive QRN


class CoupledPhysicsCalculator:
    """
    Calculate all propagation and noise effects from core physical drivers.

    This ensures realistic coupling: high K-index affects BOTH propagation
    (auroral absorption, multipath) AND noise (auroral hiss, static).
    """

    def __init__(self, seed: Optional[int] = None, use_itu_model: bool = True):
        if seed is not None:
            np.random.seed(seed)

        # Initialize ITU model if available
        self.use_itu = use_itu_model and ITU_MODEL_AVAILABLE
        if self.use_itu:
            self.itu_model = ITU_R_P533()
        else:
            self.itu_model = None

    def calculate_all_effects(self, drivers: CorePhysicalDrivers) -> DerivedConditions:
        """
        Calculate all coupled effects from core drivers.

        This is the key function: everything flows from the same physical state.
        """
        # Calculate ionospheric parameters
        muf = self._calculate_muf(drivers)
        luf = self._calculate_luf(drivers)
        d_absorption = self._calculate_d_layer_absorption(drivers)
        e_layer = self._calculate_sporadic_e(drivers)
        f_height = self._calculate_f_layer_height(drivers)

        # Check for solar flare effects (Sudden Ionospheric Disturbance)
        flare_active, sid_absorption = self._calculate_solar_flare_effects(drivers)
        if flare_active:
            d_absorption += sid_absorption  # SID increases D-layer absorption

        # Calculate propagation mode (depends on frequency vs MUF, K-index, time, Sporadic E)
        prop_mode = self._calculate_propagation_mode(drivers, muf)

        # Calculate multipath characteristics (coupled to K-index, time, MUF)
        delay_spread = self._calculate_delay_spread(drivers, prop_mode, f_height)
        doppler_spread = self._calculate_doppler_spread(drivers, prop_mode)
        k_factor = self._calculate_rician_k(drivers, prop_mode)

        # Calculate QRN (coupled to K-index, weather, time, latitude)
        qrn_type, qrn_mix = self._calculate_qrn_characteristics(drivers)
        atm_noise = self._calculate_atmospheric_noise(drivers, qrn_mix)
        man_noise = self._calculate_man_made_noise(drivers)

        # Calculate overall signal quality
        snr = self._calculate_effective_snr(drivers, d_absorption, atm_noise, man_noise)
        s_units = self._snr_to_s_units(snr)

        # PRE-COMPUTE noise generation parameters (for GPU efficiency)
        # Atmospheric burst rate (K-index dependent)
        atm_burst_rate = 0.5 * (1.0 + drivers.k_index / 9.0)  # 0.5-1.0 bursts/sec

        # Determine which noise types this stream will have (random but pre-computed)
        has_qrm = np.random.random() < 0.2  # 20% have QRM
        has_atmospheric = qrn_type in [QRNType.CRACKLING, QRNType.THUNDERSTORM, QRNType.STATIC] or np.random.random() < 0.3
        has_impulsive = np.random.random() < 0.15  # 15% have powerline noise

        return DerivedConditions(
            muf_mhz=muf,
            luf_mhz=luf,
            d_layer_absorption_db=d_absorption,
            e_layer_present=e_layer,
            f_layer_height_km=f_height,
            propagation_mode=prop_mode,
            multipath_delay_spread_ms=delay_spread,
            doppler_spread_hz=doppler_spread,
            rician_k_factor=k_factor,
            dominant_qrn_type=qrn_type,
            qrn_components=qrn_mix,
            atmospheric_noise_db=atm_noise,
            man_made_noise_db=man_noise,
            effective_snr_db=snr,
            signal_strength_s_units=s_units,
            solar_flare_active=flare_active,
            sid_absorption_db=sid_absorption,
            # Pre-computed noise parameters
            atmospheric_burst_rate=atm_burst_rate,
            has_qrm=has_qrm,
            has_atmospheric_qrn=has_atmospheric,
            has_impulsive_qrn=has_impulsive
        )

    def _calculate_muf(self, d: CorePhysicalDrivers) -> float:
        """Maximum Usable Frequency (coupled to SFI, time, season, latitude)."""
        if self.use_itu and self.itu_model is not None:
            # Use ITU-R P.533 model (industry standard)
            month = ((d.day_of_year - 1) // 30) + 1  # Approximate month from day
            month = min(12, max(1, month))

            muf = self.itu_model.calculate_muf(
                ssn=d.sunspot_number,
                month=month,
                latitude=d.latitude,
                longitude=d.longitude,
                utc_hour=d.utc_hour,
                frequency=d.frequency_mhz
            )

            # Geomagnetic disturbance reduces MUF
            k_factor = 1.0 - 0.05 * d.k_index
            muf *= k_factor

            return muf
        else:
            # Fallback to simplified model
            base_muf = 5.0 + (d.sfi - 70) * 0.08

            local_hour = (d.utc_hour + d.longitude / 15.0) % 24
            if 6 <= local_hour <= 18:
                time_factor = 1.0 + 0.3 * np.sin(np.pi * (local_hour - 6) / 12)
            else:
                time_factor = 0.5 + 0.2 * np.cos(np.pi * (local_hour - 18) / 12)

            season_factor = 1.0 + 0.1 * np.cos(2 * np.pi * d.day_of_year / 365.25)
            k_factor = 1.0 - 0.05 * d.k_index

            muf = base_muf * time_factor * season_factor * k_factor
            return max(3.0, min(50.0, muf))

    def _calculate_luf(self, d: CorePhysicalDrivers) -> float:
        """Lowest Usable Frequency (coupled to absorption)."""
        # LUF rises with D-layer absorption
        base_luf = 1.5  # MHz

        # Daytime raises LUF (D-layer absorption)
        local_hour = (d.utc_hour + d.longitude / 15.0) % 24
        if 8 <= local_hour <= 16:
            luf_factor = 2.5
        elif 6 <= local_hour <= 18:
            luf_factor = 1.5
        else:
            luf_factor = 1.0

        # High K-index raises LUF (auroral absorption)
        k_factor = 1.0 + 0.15 * d.k_index

        luf = base_luf * luf_factor * k_factor
        return max(1.0, min(10.0, luf))

    def _calculate_d_layer_absorption(self, d: CorePhysicalDrivers) -> float:
        """D-layer absorption in dB (coupled to time, frequency, K-index)."""
        local_hour = (d.utc_hour + d.longitude / 15.0) % 24

        if self.use_itu and self.itu_model is not None:
            # Use ITU-R P.533 absorption model
            absorption = self.itu_model.calculate_absorption(
                frequency=d.frequency_mhz,
                local_time=local_hour,
                ssn=d.sunspot_number,
                k_index=d.k_index,
                latitude=d.latitude
            )
            return absorption
        else:
            # Fallback to simplified model
            latitude_rad = np.radians(d.latitude)

            if 6 <= local_hour <= 18:
                zenith_factor = np.cos(np.pi * (local_hour - 12) / 12) * np.cos(latitude_rad)
                zenith_factor = max(0, zenith_factor)
            else:
                zenith_factor = 0.0

            base_absorption = 0.5 * (d.frequency_mhz / 7.0) ** 1.5
            day_absorption = base_absorption * zenith_factor * 10.0

            if abs(d.latitude) > 55:
                auroral_absorption = d.k_index * 2.0 * (abs(d.latitude) - 55) / 35
            else:
                auroral_absorption = 0.0

            total_absorption = day_absorption + auroral_absorption
            return max(0.0, min(30.0, total_absorption))

    def _calculate_sporadic_e(self, d: CorePhysicalDrivers) -> bool:
        """Sporadic E presence (more common in summer, mid-latitudes)."""
        # Peak in June/July (day ~170)
        season_prob = 0.15 * (1 + 0.5 * np.cos(2 * np.pi * (d.day_of_year - 170) / 365.25))

        # More common at mid-latitudes (30-50 deg)
        lat_prob = 0.3 * np.exp(-((abs(d.latitude) - 40) / 20) ** 2)

        # Slightly more during day
        local_hour = (d.utc_hour + d.longitude / 15.0) % 24
        time_prob = 1.2 if 10 <= local_hour <= 16 else 0.8

        total_prob = min(0.4, season_prob * lat_prob * time_prob)
        return np.random.random() < total_prob

    def _calculate_f_layer_height(self, d: CorePhysicalDrivers) -> float:
        """F-layer virtual height in km (affects multipath delay)."""
        # Base height
        base_height = 250.0  # km

        # Higher at night (F-layer rises)
        local_hour = (d.utc_hour + d.longitude / 15.0) % 24
        if local_hour < 6 or local_hour > 18:
            night_factor = 1.3
        else:
            night_factor = 1.0

        # Solar activity affects height
        sfi_factor = 1.0 + (d.sfi - 150) / 1000

        # Geomagnetic storms raise height
        k_factor = 1.0 + 0.02 * d.k_index

        height = base_height * night_factor * sfi_factor * k_factor
        return max(200.0, min(400.0, height))

    def _calculate_propagation_mode(self, d: CorePhysicalDrivers, muf: float) -> PropagationMode:
        """
        Propagation mode (coupled to freq/MUF ratio, K-index, time, Sporadic E).

        High K-index → more multipath, Rayleigh fading
        Near MUF → Rician (dominant sky wave)
        Below LUF → heavy absorption, possible ground wave (AWGN-like)
        Sporadic E → Enhanced short-skip propagation
        """
        freq_ratio = d.frequency_mhz / muf

        # Check for Sporadic E (can support higher frequencies)
        if self._calculate_sporadic_e(d):
            # Sporadic E supports VHF/low-HF with strong signal
            if d.frequency_mhz > 20 or freq_ratio > 1.2:
                return PropagationMode.SPORADIC_E

        # Very low frequency or very high K-index → Rayleigh
        if freq_ratio < 0.3 or d.k_index > 7:
            if d.frequency_mhz < 4.0:
                return PropagationMode.RICIAN  # Ground wave dominant
            else:
                return PropagationMode.RAYLEIGH

        # Near MUF with low K-index → Rician (strong direct path)
        if 0.7 <= freq_ratio <= 0.95 and d.k_index < 3:
            return PropagationMode.RICIAN

        # High K-index → dense multipath
        if d.k_index > 5:
            return PropagationMode.MULTIPATH_DENSE

        # Mid-range with moderate K-index → sparse multipath
        if 0.4 <= freq_ratio <= 0.8 and 3 <= d.k_index <= 5:
            return PropagationMode.MULTIPATH_SPARSE

        return PropagationMode.RICIAN

    def _calculate_delay_spread(self, d: CorePhysicalDrivers, mode: PropagationMode, f_height: float) -> float:
        """Multipath delay spread in ms (coupled to K-index, F-layer height)."""
        if mode == PropagationMode.AWGN:
            return 0.0

        # Base delay from F-layer height variation
        base_delay = (f_height - 250) / 100  # 0-1.5 ms

        # K-index increases delay spread (disturbed ionosphere)
        k_factor = 1.0 + 0.3 * d.k_index

        # Mode-specific multiplier
        mode_multiplier = {
            PropagationMode.RICIAN: 0.3,
            PropagationMode.RAYLEIGH: 1.5,
            PropagationMode.MULTIPATH_SPARSE: 1.0,
            PropagationMode.MULTIPATH_DENSE: 2.5,
            PropagationMode.SPORADIC_E: 0.5  # Sporadic E has moderate delay spread
        }.get(mode, 1.0)  # Default to 1.0 if mode not found

        delay = base_delay * k_factor * mode_multiplier
        return max(0.0, min(10.0, delay))

    def _calculate_doppler_spread(self, d: CorePhysicalDrivers, mode: PropagationMode) -> float:
        """Doppler spread in Hz (coupled to K-index, auroral activity)."""
        # Base Doppler from ionospheric motion
        base_doppler = 0.1  # Hz

        # K-index causes ionospheric motion
        k_factor = 1.0 + 0.5 * d.k_index

        # Auroral zones have more Doppler
        if abs(d.latitude) > 60:
            auroral_factor = 2.0 + d.k_index
        else:
            auroral_factor = 1.0

        # Mode-specific
        mode_multiplier = {
            PropagationMode.AWGN: 0.1,
            PropagationMode.RICIAN: 0.5,
            PropagationMode.RAYLEIGH: 2.0,
            PropagationMode.MULTIPATH_SPARSE: 1.5,
            PropagationMode.MULTIPATH_DENSE: 3.0,
            PropagationMode.SPORADIC_E: 1.0  # Sporadic E has moderate Doppler
        }.get(mode, 1.0)  # Default to 1.0 if mode not found

        doppler = base_doppler * k_factor * auroral_factor * mode_multiplier
        return max(0.0, min(5.0, doppler))

    def _calculate_rician_k(self, d: CorePhysicalDrivers, mode: PropagationMode) -> float:
        """Rician K-factor in dB (ratio of dominant to scattered power)."""
        if mode != PropagationMode.RICIAN:
            return 0.0  # Not applicable

        # Base K-factor (strong direct path)
        base_k = 10.0  # dB

        # K-index reduces K-factor (more scattering)
        k_reduction = d.k_index * 1.5

        k_factor_db = base_k - k_reduction
        return max(0.0, min(15.0, k_factor_db))

    def _calculate_qrn_characteristics(self, d: CorePhysicalDrivers) -> Tuple[QRNType, Dict[QRNType, float]]:
        """
        QRN type and component mix (coupled to K-index, weather, latitude, time).

        High K-index → auroral hiss (high lat) or disturbed static
        Thunderstorms → crackling/popcorn
        Quiet conditions → low static
        """
        qrn_weights = {qtype: 0.0 for qtype in QRNType}

        # Base quiet level
        qrn_weights[QRNType.QUIET] = 1.0

        # Atmospheric static (always present at some level)
        local_hour = (d.utc_hour + d.longitude / 15.0) % 24

        # Higher in tropics, summer, night
        tropical_factor = max(0, 1.0 - abs(d.latitude) / 60)
        season_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (d.day_of_year - 80) / 365.25)
        night_factor = 1.5 if local_hour < 6 or local_hour > 20 else 1.0

        static_level = 0.3 * tropical_factor * season_factor * night_factor
        qrn_weights[QRNType.STATIC] = static_level

        # Thunderstorm activity → crackling and popcorn
        if d.thunderstorm_activity > 0.1:
            qrn_weights[QRNType.CRACKLING] = d.thunderstorm_activity * 2.0
            qrn_weights[QRNType.POPCORN] = d.thunderstorm_activity * 1.5
            if d.thunderstorm_activity > 0.7:
                qrn_weights[QRNType.THUNDERSTORM] = d.thunderstorm_activity

        # High K-index effects
        if d.k_index > 4:
            # Auroral hiss at high latitudes
            if abs(d.latitude) > 55:
                auroral_strength = (d.k_index - 4) / 5 * (abs(d.latitude) - 55) / 35
                qrn_weights[QRNType.AURORAL] = auroral_strength
                qrn_weights[QRNType.HISS] = auroral_strength * 0.7
                qrn_weights[QRNType.FLUTTERY] = auroral_strength * 0.5
            else:
                # Disturbed conditions at lower latitudes
                qrn_weights[QRNType.STATIC] += (d.k_index - 4) / 5 * 0.5
                qrn_weights[QRNType.FLUTTERY] = (d.k_index - 4) / 5 * 0.3

        # Normalize weights
        total_weight = sum(qrn_weights.values())
        if total_weight > 0:
            qrn_weights = {k: v / total_weight for k, v in qrn_weights.items()}

        # Dominant type
        dominant_type = max(qrn_weights.items(), key=lambda x: x[1])[0]

        return dominant_type, qrn_weights

    def _calculate_atmospheric_noise(self, d: CorePhysicalDrivers, qrn_mix: Dict[QRNType, float]) -> float:
        """Atmospheric noise level in dB above thermal floor."""
        # Base atmospheric noise (frequency dependent: ~f^-2)
        base_noise = 20.0 * (7.0 / d.frequency_mhz) ** 2  # 0-20 dB range

        # QRN component contributions
        qrn_noise = 0.0
        noise_levels = {
            QRNType.QUIET: 0.0,
            QRNType.STATIC: 5.0,
            QRNType.CRACKLING: 15.0,
            QRNType.HISS: 8.0,
            QRNType.POPCORN: 12.0,
            QRNType.FLUTTERY: 10.0,
            QRNType.THUNDERSTORM: 25.0,
            QRNType.AURORAL: 18.0
        }

        for qtype, weight in qrn_mix.items():
            qrn_noise += noise_levels[qtype] * weight

        # Thunderstorm activity adds impulse noise
        thunderstorm_noise = d.thunderstorm_activity * 15.0

        total_noise = base_noise + qrn_noise + thunderstorm_noise
        return max(0.0, min(40.0, total_noise))

    def _calculate_man_made_noise(self, d: CorePhysicalDrivers) -> float:
        """Man-made noise (QRM) in dB - depends on time and location."""
        # Lower at night, lower at rural locations (simplified)
        local_hour = (d.utc_hour + d.longitude / 15.0) % 24

        # Time of day effect
        if 0 <= local_hour <= 6:
            time_factor = 0.3  # Quiet at night
        elif 7 <= local_hour <= 9 or 17 <= local_hour <= 20:
            time_factor = 1.0  # Peak commute hours
        else:
            time_factor = 0.7  # Daytime baseline

        # Simplified location effect (mid-latitudes more populated)
        location_factor = 1.0 - 0.3 * abs(abs(d.latitude) - 45) / 45

        base_qrm = 10.0 * time_factor * location_factor

        # Add some random variation for specific QRM sources
        qrm_variation = np.random.normal(0, 3.0)

        return max(0.0, min(30.0, base_qrm + qrm_variation))

    def _calculate_effective_snr(self, d: CorePhysicalDrivers, absorption: float,
                                 atm_noise: float, man_noise: float) -> float:
        """
        Effective SNR considering all factors.

        This is what we'd actually measure at the receiver.
        """
        # Baseline signal strength (typical amateur HF: -73 dBm minimum detectable)
        # With good antennas and typical power (100W): S9 = 0 dB SNR
        baseline_snr = 30.0  # dB (increased from 20 to give realistic range)

        # Path loss (simplified: depends on frequency, distance)
        # Typical HF path loss: 100-140 dB for 1000-3000 km
        # But we want SNR not path loss, so use relative loss
        path_loss = 5.0 + (d.frequency_mhz - 7.0) * 0.3  # Reduced from 0.5

        # Total noise floor (atmospheric + man-made)
        total_noise = 10 * np.log10(10**(atm_noise/10) + 10**(man_noise/10))

        # Effective SNR
        snr = baseline_snr - path_loss - absorption - total_noise

        # Add some measurement uncertainty
        snr += np.random.normal(0, 1.5)

        return max(-30.0, min(40.0, snr))

    def _calculate_solar_flare_effects(self, d: CorePhysicalDrivers) -> Tuple[bool, float]:
        """
        Calculate Sudden Ionospheric Disturbance (SID) from solar flares.

        X-ray flares cause rapid increase in D-layer ionization → absorption spike.

        Args:
            d: Core physical drivers (with optional flare parameters)

        Returns:
            Tuple of (flare_active, sid_absorption_db)
        """
        # Check if flare is occurring (from driver parameters)
        if hasattr(d, 'solar_flare_probability') and d.solar_flare_probability > 0:
            flare_active = np.random.random() < d.solar_flare_probability

            if flare_active and hasattr(d, 'flare_intensity'):
                # SID absorption depends on flare class
                # X1 flare: +5 dB, X10 flare: +20 dB
                intensity = d.flare_intensity
                sid_absorption = 5.0 * np.log10(intensity + 1) * 2.0  # 0-20 dB range

                # SID only on sunlit side
                local_hour = (d.utc_hour + d.longitude / 15.0) % 24
                if 6 <= local_hour <= 18:
                    return True, sid_absorption
                else:
                    return False, 0.0
            else:
                return False, 0.0
        else:
            return False, 0.0

    def _snr_to_s_units(self, snr_db: float) -> int:
        """Convert SNR to S-meter units (1-9+)."""
        # S9 ≈ 0 dB SNR, 6 dB per S-unit
        s_units = int(9 + snr_db / 6)
        return max(1, min(9, s_units))


def demo_physics_coupling():
    """Demonstrate how all effects are coupled through core drivers."""
    calc = CoupledPhysicsCalculator(seed=42)

    print("=" * 80)
    print("PHYSICS COUPLING DEMONSTRATION")
    print("=" * 80)

    # Scenario 1: Excellent conditions
    print("\n### SCENARIO 1: Excellent Conditions ###")
    drivers1 = CorePhysicalDrivers(
        sfi=200.0, sunspot_number=150.0,
        k_index=1.2, a_index=5.0, dst_index=-10.0,
        utc_hour=14.0, day_of_year=180, latitude=40.0, longitude=-75.0,
        thunderstorm_activity=0.0, precipitation_rate=0.0,
        frequency_mhz=14.1
    )
    conditions1 = calc.calculate_all_effects(drivers1)
    print(f"MUF: {conditions1.muf_mhz:.1f} MHz")
    print(f"D-layer absorption: {conditions1.d_layer_absorption_db:.1f} dB")
    print(f"Propagation mode: {conditions1.propagation_mode.value}")
    print(f"Dominant QRN: {conditions1.dominant_qrn_type.value}")
    print(f"Atmospheric noise: {conditions1.atmospheric_noise_db:.1f} dB")
    print(f"Effective SNR: {conditions1.effective_snr_db:.1f} dB (S{conditions1.signal_strength_s_units})")

    # Scenario 2: Geomagnetic storm
    print("\n### SCENARIO 2: Severe Geomagnetic Storm ###")
    drivers2 = CorePhysicalDrivers(
        sfi=120.0, sunspot_number=80.0,
        k_index=7.8, a_index=150.0, dst_index=-180.0,  # Severe storm
        utc_hour=2.0, day_of_year=80, latitude=65.0, longitude=25.0,  # High latitude
        thunderstorm_activity=0.0, precipitation_rate=0.0,
        frequency_mhz=7.1
    )
    conditions2 = calc.calculate_all_effects(drivers2)
    print(f"MUF: {conditions2.muf_mhz:.1f} MHz (reduced by storm)")
    print(f"D-layer absorption: {conditions2.d_layer_absorption_db:.1f} dB (auroral)")
    print(f"Propagation mode: {conditions2.propagation_mode.value}")
    print(f"Delay spread: {conditions2.multipath_delay_spread_ms:.2f} ms")
    print(f"Doppler spread: {conditions2.doppler_spread_hz:.2f} Hz")
    print(f"Dominant QRN: {conditions2.dominant_qrn_type.value}")
    print(f"QRN mix: {', '.join([f'{k.value}: {v:.2f}' for k, v in conditions2.qrn_components.items() if v > 0.05])}")
    print(f"Atmospheric noise: {conditions2.atmospheric_noise_db:.1f} dB")
    print(f"Effective SNR: {conditions2.effective_snr_db:.1f} dB (S{conditions2.signal_strength_s_units})")

    # Scenario 3: Thunderstorm
    print("\n### SCENARIO 3: Local Thunderstorms ###")
    drivers3 = CorePhysicalDrivers(
        sfi=180.0, sunspot_number=120.0,
        k_index=2.5, a_index=15.0, dst_index=-20.0,
        utc_hour=20.0, day_of_year=200, latitude=25.0, longitude=80.0,  # Tropical
        thunderstorm_activity=0.85, precipitation_rate=25.0,  # Active storms
        frequency_mhz=3.6
    )
    conditions3 = calc.calculate_all_effects(drivers3)
    print(f"MUF: {conditions3.muf_mhz:.1f} MHz")
    print(f"Propagation mode: {conditions3.propagation_mode.value}")
    print(f"Dominant QRN: {conditions3.dominant_qrn_type.value}")
    print(f"QRN mix: {', '.join([f'{k.value}: {v:.2f}' for k, v in conditions3.qrn_components.items() if v > 0.05])}")
    print(f"Atmospheric noise: {conditions3.atmospheric_noise_db:.1f} dB")
    print(f"Effective SNR: {conditions3.effective_snr_db:.1f} dB (S{conditions3.signal_strength_s_units})")

    print("\n" + "=" * 80)
    print("Note: All effects are derived from the SAME physical state.")
    print("High K-index affects MUF, absorption, propagation, AND QRN together.")
    print("=" * 80)


if __name__ == "__main__":
    demo_physics_coupling()
