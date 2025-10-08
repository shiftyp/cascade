"""
Enhanced HF Channel Simulator with Real-world Impairments
Adds hardware effects, ionospheric physics, and realistic interference patterns.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class HardwareImpairments:
    """Hardware impairments configuration."""
    iq_imbalance_db: float = 0.5  # Typical: 0.1-1.0 dB
    iq_phase_error_deg: float = 2.0  # Typical: 0.5-5 degrees
    dc_offset: float = 0.01  # Fraction of signal amplitude
    phase_noise_dbc_hz: float = -90  # Phase noise at 1 kHz offset
    adc_bits: int = 14  # ADC resolution
    frequency_offset_hz: float = 50  # Crystal accuracy
    frequency_drift_hz_per_sec: float = 0.1  # Thermal drift


@dataclass
class IonosphericConditions:
    """Ionospheric propagation conditions."""
    solar_flux_index: int = 150  # SFI: 50-300
    k_index: int = 2  # Geomagnetic: 0-9
    time_of_day: str = 'day'  # 'day', 'night', 'greyline'
    season: str = 'summer'  # 'winter', 'spring', 'summer', 'fall'
    latitude: float = 40.0  # Degrees

    def get_muf_mhz(self, frequency_mhz: float) -> float:
        """Calculate Maximum Usable Frequency."""
        # Simplified MUF calculation
        base_muf = 10.0 + (self.solar_flux_index - 100) * 0.05

        # Time of day factor
        tod_factor = {'night': 0.6, 'greyline': 1.2, 'day': 1.0}[self.time_of_day]

        # Geomagnetic disturbance
        k_factor = 1.0 - (self.k_index / 20.0)

        return base_muf * tod_factor * k_factor

    def get_absorption_db(self, frequency_mhz: float) -> float:
        """Calculate D-layer absorption."""
        # Higher at day, lower frequencies more absorbed
        if self.time_of_day == 'night':
            return 1.0 + 3.0 * (10.0 / frequency_mhz)
        elif self.time_of_day == 'greyline':
            return 0.5 + 1.5 * (10.0 / frequency_mhz)
        else:  # day
            return 2.0 + 5.0 * (10.0 / frequency_mhz)


class RealisticHFChannel:
    """Enhanced HF channel with real-world effects."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.hardware = HardwareImpairments()
        self.iono = IonosphericConditions()

    def apply_hardware_impairments(self, signal_iq: np.ndarray) -> np.ndarray:
        """Apply realistic hardware impairments."""

        # 1. I/Q imbalance
        amplitude_imbalance = 10**(self.hardware.iq_imbalance_db / 20.0)
        phase_imbalance_rad = np.deg2rad(self.hardware.iq_phase_error_deg)

        # Create imbalance matrix
        i_signal = signal_iq.real
        q_signal = signal_iq.imag

        i_imbalanced = i_signal
        q_imbalanced = amplitude_imbalance * (
            q_signal * np.cos(phase_imbalance_rad) +
            i_signal * np.sin(phase_imbalance_rad)
        )

        signal_iq = i_imbalanced + 1j * q_imbalanced

        # 2. DC offset (slowly varying)
        dc_drift_rate = 0.1  # Hz
        t = np.arange(len(signal_iq)) / self.sample_rate
        dc_i = self.hardware.dc_offset * np.sin(2 * np.pi * dc_drift_rate * t)
        dc_q = self.hardware.dc_offset * np.cos(2 * np.pi * dc_drift_rate * t)
        signal_iq += (dc_i + 1j * dc_q)

        # 3. Phase noise (from local oscillator)
        phase_noise_std = np.sqrt(10**(self.hardware.phase_noise_dbc_hz / 10.0))
        phase_noise = np.cumsum(np.random.randn(len(signal_iq))) * phase_noise_std
        signal_iq *= np.exp(1j * phase_noise)

        # 4. Frequency offset and drift
        freq_offset = self.hardware.frequency_offset_hz + \
                     self.hardware.frequency_drift_hz_per_sec * t
        signal_iq *= np.exp(1j * 2 * np.pi * freq_offset * t)

        # 5. ADC quantization
        max_amplitude = np.max(np.abs(signal_iq))
        levels = 2 ** self.hardware.adc_bits
        signal_iq = np.round(signal_iq / max_amplitude * levels) / levels * max_amplitude

        return signal_iq

    def apply_ionospheric_effects(self, signal_iq: np.ndarray,
                                  frequency_mhz: float) -> Tuple[np.ndarray, Dict]:
        """Apply ionospheric propagation effects."""

        metadata = {}

        # 1. Check if frequency is above/below MUF
        muf = self.iono.get_muf_mhz(frequency_mhz)
        metadata['muf_mhz'] = muf

        if frequency_mhz > muf * 1.2:
            # Signal likely passes through ionosphere (weak/no sky wave)
            metadata['propagation_type'] = 'through_ionosphere'
            # Severe attenuation
            signal_iq *= 0.01
        elif frequency_mhz > muf:
            # Near MUF - highly variable
            metadata['propagation_type'] = 'near_muf'
            # Add strong fading
            flutter_rate = np.random.uniform(2, 10)
            t = np.arange(len(signal_iq)) / self.sample_rate
            fade = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * flutter_rate * t))
            signal_iq *= fade
        else:
            metadata['propagation_type'] = 'below_muf'

        # 2. D-layer absorption
        absorption_db = self.iono.get_absorption_db(frequency_mhz)
        absorption_linear = 10**(-absorption_db / 20.0)
        signal_iq *= absorption_linear
        metadata['absorption_db'] = absorption_db

        # 3. Sporadic E (random enhancement)
        if np.random.random() < 0.05:  # 5% chance
            # Sporadic E provides strong, short-duration enhancement
            duration = np.random.uniform(0.5, 2.0)  # seconds
            duration_samples = int(duration * self.sample_rate)
            start = np.random.randint(0, max(1, len(signal_iq) - duration_samples))

            # 10-20 dB enhancement
            enhancement_db = np.random.uniform(10, 20)
            enhancement = 10**(enhancement_db / 20.0)

            signal_iq[start:start+duration_samples] *= enhancement
            metadata['sporadic_e'] = True
            metadata['sporadic_e_enhancement_db'] = enhancement_db
        else:
            metadata['sporadic_e'] = False

        # 4. Auroral effects (if high K-index)
        if self.iono.k_index >= 5:
            # Auroral flutter (rapid fading)
            flutter_freq = np.random.uniform(5, 20)  # Hz
            t = np.arange(len(signal_iq)) / self.sample_rate
            flutter = 1.0 + 0.5 * np.sin(2 * np.pi * flutter_freq * t)
            signal_iq *= flutter
            metadata['auroral_flutter'] = True

        return signal_iq, metadata

    def add_realistic_qrm(self, signal_iq: np.ndarray,
                         interference_type: str = 'broadcast') -> Tuple[np.ndarray, Dict]:
        """Add realistic interference based on time and frequency."""

        metadata = {'qrm_type': interference_type}

        if interference_type == 'broadcast':
            # Real broadcast stations have specific characteristics
            num_stations = np.random.randint(1, 4)

            for i in range(num_stations):
                # Broadcast frequencies are on 5 kHz channels
                carrier_freq = np.random.choice([
                    5900, 6000, 7200, 9500, 11700, 15200  # Common broadcast freqs
                ]) + np.random.randint(-50, 50)  # Within passband

                # AM modulated voice
                voice_freqs = np.random.uniform(300, 3000, 5)
                voice = np.zeros(len(signal_iq))
                for vf in voice_freqs:
                    t = np.arange(len(signal_iq)) / self.sample_rate
                    voice += np.sin(2 * np.pi * vf * t)

                # AM modulation
                carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
                am_signal = carrier * (1.0 + 0.5 * voice)

                # Fading (broadcast also fades)
                fade_rate = np.random.uniform(0.1, 0.5)
                fade = 1.0 + 0.3 * np.sin(2 * np.pi * fade_rate * t)

                signal_iq += am_signal * fade * 0.3

            metadata['num_stations'] = num_stations

        elif interference_type == 'radar':
            # Pulsed radar interference
            pulse_rate = np.random.uniform(100, 500)  # PRF in Hz
            pulse_width = np.random.uniform(1e-6, 10e-6)  # seconds

            pulse_period_samples = int(self.sample_rate / pulse_rate)
            pulse_width_samples = int(pulse_width * self.sample_rate)

            radar_signal = np.zeros(len(signal_iq), dtype=complex)
            carrier_freq = np.random.uniform(1000, 2000)

            for i in range(0, len(signal_iq), pulse_period_samples):
                if i + pulse_width_samples < len(signal_iq):
                    t = np.arange(pulse_width_samples) / self.sample_rate
                    pulse = np.exp(1j * 2 * np.pi * carrier_freq * t)
                    radar_signal[i:i+pulse_width_samples] = pulse

            # Radar power is often much stronger
            radar_power = 10**(np.random.uniform(5, 15) / 20.0)
            signal_iq += radar_signal * radar_power

            metadata['pulse_rate_hz'] = pulse_rate
            metadata['pulse_width_us'] = pulse_width * 1e6

        elif interference_type == 'powerline_real':
            # Real power line noise has specific harmonic structure
            line_freq = 60  # Hz (50 in Europe)

            # Model actual transformer saturation harmonics
            # Odd harmonics are stronger (3rd, 5th, 7th)
            harmonics = []
            for n in [3, 5, 7, 9, 11, 13, 15]:
                amplitude = 1.0 / n**1.5  # Decay rate
                harmonics.append((n * line_freq, amplitude))

            # Also add intermodulation products
            for n in [2, 4, 6, 8]:
                amplitude = 1.0 / n**2
                harmonics.append((n * line_freq, amplitude))

            noise = np.zeros(len(signal_iq), dtype=complex)
            t = np.arange(len(signal_iq)) / self.sample_rate

            for freq, amp in harmonics:
                if freq < self.sample_rate / 2:
                    # Add amplitude modulation (flicker)
                    flicker = 1.0 + 0.1 * np.sin(2 * np.pi * np.random.uniform(1, 5) * t)
                    noise += amp * flicker * np.exp(1j * 2 * np.pi * freq * t)

            signal_iq += noise * 0.5
            metadata['line_freq'] = line_freq
            metadata['num_harmonics'] = len(harmonics)

        return signal_iq, metadata


# Test the realistic channel
if __name__ == '__main__':
    print("Realistic HF Channel Simulator")
    print("=" * 60)

    # Create test signal
    sample_rate = 48000
    duration = 1.0
    t = np.arange(int(duration * sample_rate)) / sample_rate
    test_signal = np.exp(1j * 2 * np.pi * 1500 * t)

    # Apply realistic channel
    channel = RealisticHFChannel(sample_rate=sample_rate)

    # Hardware impairments
    print("\n1. Applying hardware impairments...")
    impaired = channel.apply_hardware_impairments(test_signal.copy())
    print(f"   I/Q imbalance: {channel.hardware.iq_imbalance_db} dB")
    print(f"   Phase error: {channel.hardware.iq_phase_error_deg}°")
    print(f"   ADC bits: {channel.hardware.adc_bits}")

    # Ionospheric effects
    print("\n2. Applying ionospheric effects...")
    channel.iono.solar_flux_index = 120
    channel.iono.k_index = 3
    channel.iono.time_of_day = 'day'

    iono_signal, iono_meta = channel.apply_ionospheric_effects(impaired, frequency_mhz=14.1)
    print(f"   MUF: {iono_meta['muf_mhz']:.1f} MHz")
    print(f"   Absorption: {iono_meta['absorption_db']:.1f} dB")
    print(f"   Sporadic E: {iono_meta['sporadic_e']}")

    # Realistic QRM
    print("\n3. Adding realistic interference...")
    final_signal, qrm_meta = channel.add_realistic_qrm(iono_signal, 'powerline_real')
    print(f"   QRM type: {qrm_meta['qrm_type']}")

    print("\n✓ Realistic channel simulation complete!")
