"""QRN (Atmospheric Noise) generators for CASCADE V2.

Implements various types of atmospheric noise for realistic HF channel simulation:
- Crackling/static bursts
- Continuous atmospheric static
- Lightning crashes (impulse noise)
- Power line noise (50/60 Hz harmonics)

Source: HF radio propagation characteristics, training data specification
"""

import numpy as np
from typing import Optional
from scipy import signal


def generate_crackling_noise(num_samples: int, sample_rate: int = 48000,
                             burst_rate: float = 5.0, burst_power: float = 10.0,
                             seed: Optional[int] = None) -> np.ndarray:
    """Generate crackling/popping atmospheric noise.

    Simulates impulsive QRN from distant lightning and atmospheric discharges.

    Args:
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz (default: 48000)
        burst_rate: Average bursts per second (default: 5.0)
        burst_power: Power ratio of bursts to background (default: 10.0 = 10 dB)
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Complex noise signal with crackling bursts, complex64
    """
    rng = np.random.default_rng(seed)

    # Generate base noise
    base_power = 0.1
    noise_i = rng.normal(0, np.sqrt(base_power / 2), num_samples)
    noise_q = rng.normal(0, np.sqrt(base_power / 2), num_samples)
    noise = noise_i + 1j * noise_q

    # Add crackling bursts
    duration = num_samples / sample_rate
    num_bursts = int(burst_rate * duration)

    for _ in range(num_bursts):
        # Random burst position
        burst_pos = rng.integers(0, num_samples - 100)

        # Burst duration: 0.5-5 ms
        burst_duration_samples = rng.integers(
            int(0.0005 * sample_rate),
            int(0.005 * sample_rate)
        )

        # Burst amplitude (exponential decay envelope)
        t_burst = np.arange(burst_duration_samples)
        decay_time = burst_duration_samples / 3
        envelope = np.exp(-t_burst / decay_time)

        # Burst signal (random phase, decaying)
        burst_phase = rng.uniform(0, 2 * np.pi)
        burst_carrier = rng.uniform(500, 2000)  # Carrier frequency
        burst_sig = envelope * np.exp(1j * (2 * np.pi * burst_carrier * t_burst / sample_rate + burst_phase))

        # Scale to desired power
        burst_sig *= np.sqrt(burst_power * base_power)

        # Add to noise
        end_pos = min(burst_pos + burst_duration_samples, num_samples)
        actual_duration = end_pos - burst_pos
        noise[burst_pos:end_pos] += burst_sig[:actual_duration]

    return noise.astype(np.complex64)


def generate_continuous_static(num_samples: int, sample_rate: int = 48000,
                               color: str = 'pink', power: float = 1.0,
                               seed: Optional[int] = None) -> np.ndarray:
    """Generate continuous atmospheric static noise.

    Args:
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        color: Noise color - 'white', 'pink', 'brown' (default: 'pink')
        power: Noise power (default: 1.0)
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Complex colored noise, complex64

    Note:
        Atmospheric noise typically has 1/f (pink) characteristic in HF bands.
    """
    rng = np.random.default_rng(seed)

    # Generate white noise
    noise_i = rng.normal(0, 1, num_samples)
    noise_q = rng.normal(0, 1, num_samples)

    # Color the noise
    if color == 'white':
        # Already white, no filtering needed
        colored_i = noise_i
        colored_q = noise_q

    elif color == 'pink':
        # 1/f noise: Apply pink filter (-10 dB/decade = -3 dB/octave)
        # Use IIR filter approximation of 1/f characteristic
        # Designed using Voss-McCartney algorithm approximation
        colored_i = _apply_pink_filter(noise_i, sample_rate)
        colored_q = _apply_pink_filter(noise_q, sample_rate)

    elif color == 'brown':
        # 1/f² noise (Brownian): Integrate white noise
        colored_i = np.cumsum(noise_i)
        colored_q = np.cumsum(noise_q)

        # Detrend to remove DC drift
        colored_i -= np.mean(colored_i)
        colored_q -= np.mean(colored_q)

    else:
        raise ValueError(f"Unknown noise color: {color}. Use 'white', 'pink', or 'brown'")

    # Combine to complex
    noise = colored_i + 1j * colored_q

    # Normalize and scale to desired power
    noise = noise / np.std(noise) * np.sqrt(power / 2)

    return noise.astype(np.complex64)


def _apply_pink_filter(data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply pink noise filter (1/f characteristic).

    Uses IIR filter approximation for computational efficiency.
    """
    # Simple pink filter using cascaded first-order filters
    # This approximates 1/f over ~3 decades
    b, a = signal.butter(1, 1000, 'low', fs=sample_rate)
    filtered = signal.filtfilt(b, a, data)

    return filtered


def generate_lightning_crashes(num_samples: int, sample_rate: int = 48000,
                               crash_rate: float = 0.5, crash_power: float = 100.0,
                               background_power: float = 0.1,
                               seed: Optional[int] = None) -> np.ndarray:
    """Generate lightning crash impulse noise.

    Simulates strong nearby lightning strikes causing large impulse noise.

    Args:
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        crash_rate: Average crashes per second (default: 0.5 = one per 2 seconds)
        crash_power: Power of lightning crash relative to background (default: 100.0 = 20 dB)
        background_power: Background noise power (default: 0.1)
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Complex noise with lightning crashes, complex64
    """
    rng = np.random.default_rng(seed)

    # Generate pink background noise
    noise = generate_continuous_static(num_samples, sample_rate, 'pink',
                                      background_power, seed=seed)

    # Add lightning crashes
    duration = num_samples / sample_rate
    num_crashes = max(1, int(crash_rate * duration))

    for _ in range(num_crashes):
        # Random crash position
        crash_pos = rng.integers(0, num_samples - 500)

        # Crash duration: 5-50 ms
        crash_duration_samples = rng.integers(
            int(0.005 * sample_rate),
            int(0.050 * sample_rate)
        )

        # Double exponential envelope (fast rise, slower decay)
        t_crash = np.arange(crash_duration_samples) / sample_rate
        rise_time = 0.001  # 1 ms rise
        decay_time = 0.010  # 10 ms decay

        rise = 1 - np.exp(-t_crash / rise_time)
        decay = np.exp(-t_crash / decay_time)
        envelope = rise * decay

        # Crash signal (broadband, impulsive)
        crash_i = rng.normal(0, 1, crash_duration_samples) * envelope
        crash_q = rng.normal(0, 1, crash_duration_samples) * envelope
        crash_sig = crash_i + 1j * crash_q

        # Scale to desired power
        crash_sig *= np.sqrt(crash_power * background_power)

        # Add to noise
        end_pos = min(crash_pos + crash_duration_samples, num_samples)
        actual_duration = end_pos - crash_pos
        noise[crash_pos:end_pos] += crash_sig[:actual_duration]

    return noise.astype(np.complex64)


def generate_powerline_noise(num_samples: int, sample_rate: int = 48000,
                             line_freq: float = 60.0, num_harmonics: int = 5,
                             power: float = 0.5, seed: Optional[int] = None) -> np.ndarray:
    """Generate power line interference noise.

    Simulates AC power line noise with harmonics (50 or 60 Hz).

    Args:
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        line_freq: Power line frequency in Hz (50 or 60, default: 60)
        num_harmonics: Number of harmonics to include (default: 5)
        power: Total noise power (default: 0.5)
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Complex noise with power line harmonics, complex64
    """
    rng = np.random.default_rng(seed)

    t = np.arange(num_samples) / sample_rate

    # Initialize signal
    noise = np.zeros(num_samples, dtype=np.complex64)

    # Add fundamental and harmonics
    for h in range(1, num_harmonics + 1):
        freq = line_freq * h
        phase = rng.uniform(0, 2 * np.pi)

        # Amplitude decreases with harmonic number (roughly 1/h²)
        amplitude = 1.0 / (h * h)

        # Add harmonic (complex exponential for both sidebands)
        harmonic = amplitude * np.exp(1j * (2 * np.pi * freq * t + phase))

        noise += harmonic

    # Normalize to desired power
    noise = noise / np.std(noise) * np.sqrt(power / 2)

    # Add slow amplitude modulation (power line hum varies)
    mod_freq = 0.5  # 0.5 Hz modulation
    modulation = 1.0 + 0.2 * np.sin(2 * np.pi * mod_freq * t)
    noise *= modulation

    return noise.astype(np.complex64)


def generate_mixed_qrn(num_samples: int, sample_rate: int = 48000,
                      static_power: float = 0.3, crackling_rate: float = 3.0,
                      lightning_rate: float = 0.2, powerline_power: float = 0.1,
                      seed: Optional[int] = None) -> np.ndarray:
    """Generate mixed atmospheric noise with multiple QRN types.

    Combines continuous static, crackling, lightning, and power line noise
    for realistic HF atmospheric noise.

    Args:
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        static_power: Continuous static power (default: 0.3)
        crackling_rate: Crackling bursts per second (default: 3.0)
        lightning_rate: Lightning crashes per second (default: 0.2)
        powerline_power: Power line noise power (default: 0.1)
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Complex mixed QRN signal, complex64
    """
    # Generate separate seeds for each component
    rng = np.random.default_rng(seed)
    seeds = [rng.integers(0, 2**31) for _ in range(4)]

    # Generate components
    static = generate_continuous_static(num_samples, sample_rate, 'pink',
                                       static_power, seed=seeds[0])

    crackling = generate_crackling_noise(num_samples, sample_rate,
                                        crackling_rate, burst_power=5.0,
                                        seed=seeds[1])

    lightning = generate_lightning_crashes(num_samples, sample_rate,
                                          lightning_rate, crash_power=50.0,
                                          background_power=0.0,  # No background (already in static)
                                          seed=seeds[2])

    powerline = generate_powerline_noise(num_samples, sample_rate,
                                        line_freq=60.0, num_harmonics=5,
                                        power=powerline_power, seed=seeds[3])

    # Mix all components
    mixed = static + crackling + lightning + powerline

    return mixed.astype(np.complex64)
