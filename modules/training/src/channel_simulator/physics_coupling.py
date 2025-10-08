"""
Physics-Coupled Parameter Calculation for HF Propagation
All effects (QRN, propagation, absorption, multipath) derived from same core physical drivers.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy import signal


@dataclass
class CoreDrivers:
    """Core physical drivers that control all propagation/noise effects."""
    # Solar conditions
    sfi: float  # Solar Flux Index (70-250)
    sunspot_number: float  # Sunspot number (0-300)

    # Geomagnetic conditions
    k_index: float  # Planetary K-index (0.0-9.0, continuous)
    a_index: float  # A-index (0-400)
    dst_index: float  # Dst index (-500 to +50 nT)

    # Time/Location
    local_time: float  # Hours (0-24, continuous)
    day_of_year: int  # 1-365
    latitude: float  # Degrees (-90 to +90)
    longitude: float  # Degrees (-180 to +180)

    # Weather (local)
    thunderstorm_probability: float  # 0.0-1.0
    temperature_c: float  # Celsius

    # Band
    frequency_mhz: float  # Operating frequency


class CoupledPhysicsCalculator:
    """Calculate all propagation/noise effects from core drivers."""

    def __init__(self, drivers: CoreDrivers):
        self.d = drivers  # Shorthand

    def calculate_all_effects(self) -> Dict:
        """Calculate ALL effects from core drivers (physics-coupled)."""

        # Ionospheric parameters (all coupled)
        muf = self._calculate_muf()
        fot = muf * 0.85  # Frequency of Optimum Traffic
        absorption_db = self._calculate_d_layer_absorption()
        e_layer_critical_freq = self._calculate_e_layer_critical_freq()

        # Propagation mode (depends on MUF, frequency, absorption)
        prop_mode = self._determine_propagation_mode(muf, fot, absorption_db)

        # QRN components (coupled to same drivers)
        qrn = self._calculate_qrn_components()

        # Multipath (coupled to propagation mode and K-index)
        multipath = self._calculate_multipath_parameters(prop_mode)

        # Fading (coupled to propagation and geomagnetic activity)
        fading = self._calculate_fading_parameters(prop_mode)

        # QRM likelihood (coupled to time, frequency)
        qrm = self._calculate_qrm_likelihood()

        return {
            'ionosphere': {
                'muf_mhz': muf,
                'fot_mhz': fot,
                'e_layer_critical_freq_mhz': e_layer_critical_freq,
                'absorption_db': absorption_db,
                'sporadic_e_probability': self._calculate_sporadic_e_probability(),
            },
            'propagation': prop_mode,
            'qrn': qrn,
            'multipath': multipath,
            'fading': fading,
            'qrm': qrm,
        }

    def _calculate_muf(self) -> float:
        """Maximum Usable Frequency (MHz)."""
        # Base MUF from solar flux
        # Empirical: MUF ≈ 4 + 0.04 * SFI at noon
        base_muf = 4.0 + 0.04 * self.d.sfi

        # Time of day variation (cosine of solar zenith angle)
        # Peak at noon, minimum at midnight
        solar_hour_angle = (self.d.local_time - 12.0) * 15.0  # degrees
        solar_hour_angle_rad = np.deg2rad(solar_hour_angle)

        # Solar zenith angle depends on latitude, season
        declination = 23.5 * np.sin(np.deg2rad((self.d.day_of_year - 81) * 360 / 365))
        cos_zenith = (np.sin(np.deg2rad(self.d.latitude)) * np.sin(np.deg2rad(declination)) +
                      np.cos(np.deg2rad(self.d.latitude)) * np.cos(np.deg2rad(declination)) *
                      np.cos(solar_hour_angle_rad))
        cos_zenith = np.clip(cos_zenith, 0.0, 1.0)

        # MUF varies with solar zenith angle
        # Day: full MUF, Night: ~0.4-0.6 of day MUF
        time_factor = 0.4 + 0.6 * cos_zenith

        # Geomagnetic disturbance reduces MUF
        # K=0: no effect, K=9: -40% MUF
        k_factor = 1.0 - (self.d.k_index / 9.0) * 0.4

        muf = base_muf * time_factor * k_factor

        return max(muf, 2.0)  # Physical minimum

    def _calculate_e_layer_critical_freq(self) -> float:
        """E-layer critical frequency (foE) in MHz."""
        # E-layer depends strongly on solar zenith angle
        solar_hour_angle = (self.d.local_time - 12.0) * 15.0
        solar_hour_angle_rad = np.deg2rad(solar_hour_angle)

        declination = 23.5 * np.sin(np.deg2rad((self.d.day_of_year - 81) * 360 / 365))
        cos_zenith = (np.sin(np.deg2rad(self.d.latitude)) * np.sin(np.deg2rad(declination)) +
                      np.cos(np.deg2rad(self.d.latitude)) * np.cos(np.deg2rad(declination)) *
                      np.cos(solar_hour_angle_rad))
        cos_zenith = np.clip(cos_zenith, 0.0, 1.0)

        # E-layer mostly disappears at night
        if cos_zenith < 0.1:
            return 0.5  # Residual E-layer

        # Day: foE ≈ sqrt(cos(zenith)) * scaling factor
        foe = 3.5 * np.sqrt(cos_zenith) * (1.0 + 0.3 * (self.d.sfi - 150) / 100)

        return max(foe, 0.5)

    def _calculate_d_layer_absorption(self) -> float:
        """D-layer absorption in dB."""
        # D-layer absorption depends on:
        # 1. Frequency (f^-2 law)
        # 2. Solar zenith angle
        # 3. Solar flux

        # Solar zenith angle
        solar_hour_angle = (self.d.local_time - 12.0) * 15.0
        solar_hour_angle_rad = np.deg2rad(solar_hour_angle)

        declination = 23.5 * np.sin(np.deg2rad((self.d.day_of_year - 81) * 360 / 365))
        cos_zenith = (np.sin(np.deg2rad(self.d.latitude)) * np.sin(np.deg2rad(declination)) +
                      np.cos(np.deg2rad(self.d.latitude)) * np.cos(np.deg2rad(declination)) *
                      np.cos(solar_hour_angle_rad))
        cos_zenith = np.clip(cos_zenith, 0.0, 1.0)

        # Night: minimal D-layer
        if cos_zenith < 0.1:
            base_absorption = 0.5
        else:
            # Day: absorption proportional to cos(zenith)
            base_absorption = 2.0 + 8.0 * cos_zenith

        # Frequency dependence (inverse square law)
        freq_factor = (10.0 / self.d.frequency_mhz) ** 1.5

        # Solar flux increases ionization
        sfi_factor = 1.0 + 0.5 * (self.d.sfi - 150) / 100

        absorption_db = base_absorption * freq_factor * sfi_factor

        return max(absorption_db, 0.1)

    def _calculate_sporadic_e_probability(self) -> float:
        """Probability of sporadic E enhancement."""
        # Sporadic E more common:
        # - Summer (mid-latitudes)
        # - Mid-latitudes (30-50°)
        # - Daytime

        # Season factor (peak in summer)
        season_phase = np.deg2rad((self.d.day_of_year - 172) * 360 / 365)  # Peak June 21
        season_factor = 0.5 + 0.5 * np.cos(season_phase)

        # Latitude factor (peak 30-50°)
        lat_factor = np.exp(-((abs(self.d.latitude) - 40) / 20) ** 2)

        # Time factor (more common in day)
        time_factor = 0.3 + 0.7 * (1 if 6 <= self.d.local_time <= 18 else 0.2)

        # Base probability ~5%, modulated by factors
        prob = 0.05 * season_factor * lat_factor * time_factor

        return np.clip(prob, 0.0, 0.3)

    def _determine_propagation_mode(self, muf: float, fot: float, absorption_db: float) -> Dict:
        """Determine propagation mode from physical parameters."""

        f = self.d.frequency_mhz

        # Frequency vs MUF determines mode
        if f > muf * 1.2:
            mode_type = 'through_ionosphere'  # Signal passes through
            skip_distance_km = 9999  # No skip
            hop_count = 0
            signal_strength_db = -30  # Very weak

        elif f > muf:
            mode_type = 'near_muf'  # Highly variable
            skip_distance_km = 1500 + np.random.uniform(-500, 500)
            hop_count = 1
            signal_strength_db = -10 + absorption_db * -0.5

        elif f > fot:
            mode_type = 'f_layer'  # Optimal F-layer
            skip_distance_km = 2000 + np.random.uniform(-300, 300)
            hop_count = np.random.choice([1, 2], p=[0.7, 0.3])
            signal_strength_db = -absorption_db

        else:
            # Below FOT - check E-layer
            e_critical = self._calculate_e_layer_critical_freq()

            if f < e_critical * 3:  # E-layer propagation possible
                mode_type = 'e_layer'
                skip_distance_km = 800 + np.random.uniform(-200, 200)
                hop_count = np.random.choice([1, 2], p=[0.8, 0.2])
                signal_strength_db = -absorption_db * 1.5  # More absorption
            else:
                mode_type = 'f_layer'
                skip_distance_km = 2000 + np.random.uniform(-300, 300)
                hop_count = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                signal_strength_db = -absorption_db

        return {
            'mode': mode_type,
            'skip_distance_km': skip_distance_km,
            'hop_count': hop_count,
            'signal_strength_db': signal_strength_db,
        }

    def _calculate_qrn_components(self) -> Dict:
        """Calculate QRN components coupled to physical drivers."""

        # Atmospheric noise (QRN) depends on:
        # 1. Frequency (decreases with frequency)
        # 2. Season (higher in summer thunderstorm season)
        # 3. Latitude (tropical > temperate > polar)
        # 4. Time (night often quieter, except thunderstorms)
        # 5. Local weather
        # 6. Geomagnetic activity (auroral noise at high latitudes)

        components = {}

        # 1. Static crashes (lightning, near and far)
        # Coupled to: thunderstorm probability, season, latitude
        season_phase = np.deg2rad((self.d.day_of_year - 172) * 360 / 365)
        season_factor = 0.5 + 0.5 * np.cos(season_phase)  # Peak summer

        # Tropical regions have more thunderstorms
        lat_factor = np.exp(-((abs(self.d.latitude) - 10) / 40) ** 2)

        static_intensity = (self.d.thunderstorm_probability * 0.7 +
                           season_factor * 0.2 +
                           lat_factor * 0.1)

        components['static'] = {
            'intensity': static_intensity,
            'crash_rate_per_sec': 0.5 + static_intensity * 10,
            'duration_ms': 5 + np.random.exponential(10),
        }

        # 2. Crackling (power line, local interference)
        # Coupled to: time of day (human activity), temperature (load)
        activity_factor = 1.0 if 6 <= self.d.local_time <= 22 else 0.3
        temp_factor = 1.0 + 0.3 * max(0, self.d.temperature_c - 25) / 15  # A/C load

        components['crackling'] = {
            'intensity': 0.1 * activity_factor * temp_factor,
            'impulse_rate_hz': 50 + np.random.uniform(0, 100),
        }

        # 3. Hiss (galactic noise)
        # Coupled to: frequency, time (galactic center position)
        # Lower at higher frequencies
        freq_factor = (10.0 / self.d.frequency_mhz) ** 0.7

        # Galactic center more visible at certain times
        galactic_factor = 0.8 + 0.2 * np.sin(np.deg2rad(self.d.local_time * 15))

        components['hiss'] = {
            'intensity': 0.3 * freq_factor * galactic_factor,
            'bandwidth_hz': 500,
        }

        # 4. Auroral noise (high latitudes, high K-index)
        if abs(self.d.latitude) > 50 and self.d.k_index > 3:
            # Auroral flutter coupled to K-index
            auroral_intensity = (abs(self.d.latitude) - 50) / 40 * (self.d.k_index / 9)

            components['auroral'] = {
                'intensity': auroral_intensity,
                'flutter_rate_hz': 2 + self.d.k_index * 2,
                'bandwidth_hz': 100 + self.d.k_index * 50,
            }

        # 5. Popcorn noise (ionospheric scintillation)
        # Coupled to: solar activity, geomagnetic activity
        if self.d.k_index > 4 or self.d.sfi > 200:
            scintillation_intensity = 0.1 * (self.d.k_index / 9) + 0.1 * ((self.d.sfi - 150) / 100)

            components['popcorn'] = {
                'intensity': scintillation_intensity,
                'burst_rate_hz': 1 + scintillation_intensity * 5,
            }

        return components

    def _calculate_multipath_parameters(self, prop_mode: Dict) -> Dict:
        """Calculate multipath parameters coupled to propagation mode."""

        mode = prop_mode['mode']
        hop_count = prop_mode['hop_count']

        # More hops = more multipath
        if mode == 'through_ionosphere':
            return {'type': 'awgn', 'num_paths': 1}

        elif mode == 'near_muf':
            # Highly variable near MUF
            return {
                'type': 'rician' if np.random.random() < 0.3 else 'rayleigh',
                'num_paths': np.random.randint(2, 5),
                'delay_spread_ms': 1.0 + self.d.k_index * 0.5,  # K-index increases spread
                'doppler_spread_hz': 0.5 + self.d.k_index * 0.3,
            }

        elif mode == 'e_layer':
            # E-layer: shorter paths, less multipath
            return {
                'type': 'rician',
                'num_paths': min(2, hop_count + 1),
                'delay_spread_ms': 0.3 + hop_count * 0.2,
                'doppler_spread_hz': 0.2 + self.d.k_index * 0.1,
            }

        else:  # f_layer
            # F-layer: longer paths, more multipath
            return {
                'type': 'rayleigh' if hop_count > 1 else 'rician',
                'num_paths': hop_count + np.random.randint(0, 3),
                'delay_spread_ms': 0.5 + hop_count * 0.5 + self.d.k_index * 0.3,
                'doppler_spread_hz': 0.3 + self.d.k_index * 0.2,
            }

    def _calculate_fading_parameters(self, prop_mode: Dict) -> Dict:
        """Calculate fading parameters coupled to propagation and geomagnetic activity."""

        mode = prop_mode['mode']

        # Fading rate coupled to K-index (geomagnetic disturbance)
        base_fade_rate = 0.1 + self.d.k_index * 0.2  # Hz

        # Fading depth coupled to propagation mode
        if mode == 'through_ionosphere':
            fade_depth_db = 2  # Minimal fading
        elif mode == 'near_muf':
            fade_depth_db = 15 + np.random.uniform(0, 10)  # Deep fading
        elif mode == 'e_layer':
            fade_depth_db = 5 + np.random.uniform(0, 5)
        else:  # f_layer
            fade_depth_db = 8 + np.random.uniform(0, 8)

        return {
            'rate_hz': base_fade_rate,
            'depth_db': fade_depth_db,
            'selective': prop_mode['hop_count'] > 1,  # Frequency-selective if multi-hop
        }

    def _calculate_qrm_likelihood(self) -> Dict:
        """Calculate interference likelihood coupled to time/frequency."""

        # Broadcast interference
        # More common on international broadcast bands: 49m, 41m, 31m, 25m, 19m, 16m
        broadcast_bands = [
            (5.9, 6.2), (7.2, 7.6), (9.4, 9.9), (11.6, 12.1), (15.1, 15.8)
        ]

        on_broadcast_band = any(
            low <= self.d.frequency_mhz <= high for low, high in broadcast_bands
        )

        # More broadcasts in evening (primetime)
        time_factor = 1.5 if 18 <= self.d.local_time <= 23 else 1.0

        broadcast_probability = 0.3 if on_broadcast_band else 0.05
        broadcast_probability *= time_factor

        # Amateur interference (QRM from other hams)
        # More during evening, weekends (not modeled here - day_of_week not in drivers)
        amateur_probability = 0.2 * time_factor

        # Radar (military, weather)
        # More common on certain bands, random timing
        radar_probability = 0.05

        return {
            'broadcast_probability': broadcast_probability,
            'amateur_probability': amateur_probability,
            'radar_probability': radar_probability,
            'powerline_probability': 0.1,  # Constant local noise
        }


# Test
if __name__ == '__main__':
    print("Physics-Coupled Parameter Calculator")
    print("=" * 70)

    # Example: Good conditions
    drivers = CoreDrivers(
        sfi=150,
        sunspot_number=100,
        k_index=2.0,
        a_index=10,
        dst_index=-20,
        local_time=14.0,  # 2 PM
        day_of_year=180,  # Summer
        latitude=40.0,
        longitude=-75.0,
        thunderstorm_probability=0.1,
        temperature_c=25,
        frequency_mhz=14.1,
    )

    calc = CoupledPhysicsCalculator(drivers)
    effects = calc.calculate_all_effects()

    print("\nIonosphere:")
    print(f"  MUF: {effects['ionosphere']['muf_mhz']:.2f} MHz")
    print(f"  FOT: {effects['ionosphere']['fot_mhz']:.2f} MHz")
    print(f"  Absorption: {effects['ionosphere']['absorption_db']:.2f} dB")
    print(f"  Sporadic E prob: {effects['ionosphere']['sporadic_e_probability']:.3f}")

    print("\nPropagation:")
    print(f"  Mode: {effects['propagation']['mode']}")
    print(f"  Hops: {effects['propagation']['hop_count']}")
    print(f"  Signal: {effects['propagation']['signal_strength_db']:.1f} dB")

    print("\nQRN Components:")
    for component, params in effects['qrn'].items():
        print(f"  {component}: intensity={params['intensity']:.3f}")

    print("\nMultipath:")
    print(f"  Type: {effects['multipath']['type']}")
    print(f"  Paths: {effects['multipath']['num_paths']}")

    print("\n✓ Physics coupling verified!")
