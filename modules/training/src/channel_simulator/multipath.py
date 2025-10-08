"""Multipath fading simulator for CASCADE V2.

Implements frequency-selective multipath fading for HF ionospheric propagation:
- Tapped delay line model
- Rayleigh and Rician fading
- Doppler spread and frequency dispersion
- Time-varying channel characteristics

Source: Watterson ITU-R ionospheric channel model and HF propagation theory
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class MultipathProfile:
    """Multipath channel profile specification.

    Attributes:
        delays: Path delays in seconds (list)
        powers: Path powers in linear scale (list, should sum to ~1.0)
        doppler_shifts: Doppler shift for each path in Hz (list)
        k_factors: Rician K-factor for each path (0 = Rayleigh, >0 = Rician)
    """
    delays: List[float]
    powers: List[float]
    doppler_shifts: List[float]
    k_factors: List[float]

    def __post_init__(self):
        """Validate profile consistency."""
        n_paths = len(self.delays)
        if len(self.powers) != n_paths:
            raise ValueError(f"Powers length {len(self.powers)} != delays length {n_paths}")
        if len(self.doppler_shifts) != n_paths:
            raise ValueError(f"Doppler shifts length {len(self.doppler_shifts)} != delays length {n_paths}")
        if len(self.k_factors) != n_paths:
            raise ValueError(f"K-factors length {len(self.k_factors)} != delays length {n_paths}")


def watterson_hf_profile(delay_spread_ms: float = 2.0,
                         doppler_spread_hz: float = 0.5) -> MultipathProfile:
    """Create Watterson ITU-R HF ionospheric channel profile.

    Standard 2-path model for HF ionospheric propagation.

    Args:
        delay_spread_ms: RMS delay spread in milliseconds (default: 2.0)
        doppler_spread_hz: Doppler spread in Hz (default: 0.5)

    Returns:
        MultipathProfile: 2-path Watterson model
    """
    # Two paths: direct and one reflected
    delays = [0.0, delay_spread_ms / 1000]  # Convert to seconds
    powers = [0.7, 0.3]  # 70% direct, 30% reflected
    doppler_shifts = [0.0, doppler_spread_hz]  # Direct path stable, reflected varies
    k_factors = [5.0, 0.0]  # Direct path Rician (strong LOS), reflected Rayleigh

    return MultipathProfile(delays, powers, doppler_shifts, k_factors)


def severe_multipath_profile(num_paths: int = 5,
                             max_delay_ms: float = 5.0,
                             max_doppler_hz: float = 2.0) -> MultipathProfile:
    """Create severe multipath profile with many paths.

    Models disturbed ionospheric conditions with multiple reflections.

    Args:
        num_paths: Number of propagation paths (default: 5)
        max_delay_ms: Maximum delay spread in ms (default: 5.0)
        max_doppler_hz: Maximum Doppler shift in Hz (default: 2.0)

    Returns:
        MultipathProfile: Multi-path fading profile
    """
    # Exponentially decaying delays
    delays = [max_delay_ms * (i / num_paths) / 1000 for i in range(num_paths)]

    # Exponentially decaying powers (normalize to sum to 1)
    powers = [np.exp(-2 * i) for i in range(num_paths)]
    powers = [p / sum(powers) for p in powers]

    # Random Doppler shifts
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    doppler_shifts = [rng.uniform(-max_doppler_hz, max_doppler_hz)
                     for _ in range(num_paths)]

    # First path Rician (partial LOS), others Rayleigh
    k_factors = [3.0] + [0.0] * (num_paths - 1)

    return MultipathProfile(delays, powers, doppler_shifts, k_factors)


def apply_multipath_fading(signal: np.ndarray, profile: MultipathProfile,
                           sample_rate: int = 48000, seed: Optional[int] = None) -> np.ndarray:
    """Apply multipath fading to signal using tapped delay line model.

    Args:
        signal: Clean IQ signal (complex64)
        profile: Multipath channel profile
        sample_rate: Sample rate in Hz (default: 48000)
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Signal with multipath fading applied, complex64
    """
    if signal.dtype != np.complex64 and signal.dtype != np.complex128:
        raise ValueError(f"Signal must be complex, got dtype={signal.dtype}")

    rng = np.random.default_rng(seed)
    num_samples = len(signal)

    # Initialize output signal
    faded_signal = np.zeros(num_samples, dtype=np.complex128)

    # Process each path
    for path_idx in range(len(profile.delays)):
        delay = profile.delays[path_idx]
        power = profile.powers[path_idx]
        doppler = profile.doppler_shifts[path_idx]
        k_factor = profile.k_factors[path_idx]

        # Convert delay to samples
        delay_samples = int(delay * sample_rate)

        # Generate fading for this path
        if k_factor == 0:
            # Rayleigh fading (no LOS component)
            fading = _generate_rayleigh_fading(num_samples, doppler,
                                              sample_rate, rng.integers(0, 2**31))
        else:
            # Rician fading (with LOS component)
            fading = _generate_rician_fading(num_samples, doppler, k_factor,
                                            sample_rate, rng.integers(0, 2**31))

        # Scale by path power
        fading *= np.sqrt(power)

        # Apply delay
        if delay_samples > 0:
            delayed_signal = np.zeros(num_samples, dtype=np.complex128)
            delayed_signal[delay_samples:] = signal[:-delay_samples]
        else:
            delayed_signal = signal.copy()

        # Apply fading and accumulate
        faded_signal += fading * delayed_signal

    return faded_signal.astype(np.complex64)


def _generate_rayleigh_fading(num_samples: int, doppler_hz: float,
                              sample_rate: int, seed: int) -> np.ndarray:
    """Generate Rayleigh fading envelope.

    Uses Jakes model for time-correlated fading.

    Args:
        num_samples: Number of samples
        doppler_hz: Doppler spread in Hz
        sample_rate: Sample rate in Hz
        seed: Random seed

    Returns:
        np.ndarray: Complex fading envelope
    """
    rng = np.random.default_rng(seed)

    if doppler_hz == 0:
        # Static channel (no fading)
        return np.ones(num_samples, dtype=np.complex128)

    # Jakes model: sum of sinusoids with random phases
    # This creates time-correlated Rayleigh fading
    num_oscillators = 20
    t = np.arange(num_samples) / sample_rate

    i_component = np.zeros(num_samples)
    q_component = np.zeros(num_samples)

    for n in range(num_oscillators):
        phase = rng.uniform(0, 2 * np.pi)
        angle = 2 * np.pi * n / num_oscillators

        # Doppler frequency for this oscillator
        fd = doppler_hz * np.cos(angle)

        i_component += np.cos(2 * np.pi * fd * t + phase)
        q_component += np.sin(2 * np.pi * fd * t + phase)

    # Normalize to unit power
    i_component /= np.sqrt(num_oscillators / 2)
    q_component /= np.sqrt(num_oscillators / 2)

    fading = i_component + 1j * q_component

    return fading.astype(np.complex128)


def _generate_rician_fading(num_samples: int, doppler_hz: float, k_factor: float,
                           sample_rate: int, seed: int) -> np.ndarray:
    """Generate Rician fading envelope.

    Rician fading = LOS component + Rayleigh scattering component

    Args:
        num_samples: Number of samples
        doppler_hz: Doppler spread in Hz
        k_factor: Rician K-factor (ratio of LOS to scattered power)
        sample_rate: Sample rate in Hz
        seed: Random seed

    Returns:
        np.ndarray: Complex fading envelope
    """
    # Generate Rayleigh component
    rayleigh = _generate_rayleigh_fading(num_samples, doppler_hz, sample_rate, seed)

    # Add LOS component (static, unit amplitude)
    # K-factor = LOS_power / scattered_power
    # Total power = LOS_power + scattered_power = 1
    # Therefore: LOS_power = K / (K + 1), scattered_power = 1 / (K + 1)

    los_amplitude = np.sqrt(k_factor / (k_factor + 1))
    scattered_amplitude = np.sqrt(1 / (k_factor + 1))

    # Combine LOS and scattered
    rician = los_amplitude + scattered_amplitude * rayleigh

    return rician.astype(np.complex128)


def estimate_delay_spread(signal_clean: np.ndarray, signal_faded: np.ndarray,
                          sample_rate: int = 48000) -> float:
    """Estimate RMS delay spread from clean and faded signals.

    Uses cross-correlation to estimate channel impulse response.

    Args:
        signal_clean: Clean signal (complex64)
        signal_faded: Faded signal (complex64)
        sample_rate: Sample rate in Hz

    Returns:
        float: Estimated RMS delay spread in milliseconds
    """
    # Compute cross-correlation (channel impulse response estimate)
    xcorr = np.correlate(signal_faded, signal_clean, mode='full')
    xcorr_power = np.abs(xcorr) ** 2

    # Find center
    center_idx = len(xcorr) // 2
    max_delay_samples = len(signal_clean) // 4  # Look within ±25% of signal length

    # Extract relevant portion
    start_idx = max(0, center_idx - max_delay_samples)
    end_idx = min(len(xcorr), center_idx + max_delay_samples)
    cir = xcorr_power[start_idx:end_idx]

    # Normalize
    cir = cir / np.sum(cir)

    # Compute delay profile
    delays = np.arange(len(cir)) / sample_rate

    # RMS delay spread
    mean_delay = np.sum(delays * cir)
    rms_delay = np.sqrt(np.sum((delays - mean_delay) ** 2 * cir))

    return rms_delay * 1000  # Convert to milliseconds


def generate_time_varying_multipath(signal: np.ndarray, profile: MultipathProfile,
                                   sample_rate: int = 48000,
                                   variation_rate_hz: float = 0.1,
                                   seed: Optional[int] = None) -> np.ndarray:
    """Apply time-varying multipath fading.

    Models slowly time-varying ionospheric channel where path parameters
    change gradually over time.

    Args:
        signal: Clean IQ signal (complex64)
        profile: Initial multipath profile
        sample_rate: Sample rate in Hz
        variation_rate_hz: Rate of channel variation in Hz (default: 0.1 = 10s period)
        seed: Random seed

    Returns:
        np.ndarray: Signal with time-varying multipath, complex64
    """
    rng = np.random.default_rng(seed)
    num_samples = len(signal)

    # Split signal into chunks
    chunk_duration = 1.0 / variation_rate_hz  # Duration per chunk
    chunk_samples = int(chunk_duration * sample_rate)
    num_chunks = (num_samples + chunk_samples - 1) // chunk_samples

    faded_signal = np.zeros(num_samples, dtype=np.complex128)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_samples
        end_idx = min((chunk_idx + 1) * chunk_samples, num_samples)

        # Vary profile parameters slightly
        varied_profile = MultipathProfile(
            delays=profile.delays[:],
            powers=[p * rng.uniform(0.8, 1.2) for p in profile.powers],
            doppler_shifts=[d + rng.uniform(-0.2, 0.2) for d in profile.doppler_shifts],
            k_factors=profile.k_factors[:]
        )

        # Normalize powers
        total_power = sum(varied_profile.powers)
        varied_profile.powers = [p / total_power for p in varied_profile.powers]

        # Apply fading to this chunk
        chunk_signal = signal[start_idx:end_idx]
        chunk_faded = apply_multipath_fading(chunk_signal, varied_profile,
                                            sample_rate, rng.integers(0, 2**31))

        faded_signal[start_idx:end_idx] = chunk_faded

    return faded_signal.astype(np.complex64)
