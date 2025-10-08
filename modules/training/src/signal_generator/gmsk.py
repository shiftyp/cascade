"""GMSK pulse shaping for CASCADE V2 signal generator.

Implements Gaussian Minimum Shift Keying (GMSK) modulation with BT=0.3.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple


def generate_gaussian_filter(BT: float = 0.3, span_symbols: int = 4,
                            samples_per_symbol: int = 240) -> np.ndarray:
    """Generate Gaussian filter for GMSK pulse shaping.

    Args:
        BT: Bandwidth-time product (0.3 for CASCADE V2)
        span_symbols: Filter span in symbols (typically 4)
        samples_per_symbol: Samples per symbol (48000 Hz / 200 sym/s = 240)

    Returns:
        np.ndarray: Normalized Gaussian filter coefficients
    """
    # Time vector for filter span
    span_samples = span_symbols * samples_per_symbol
    t = np.arange(-span_samples // 2, span_samples // 2) / samples_per_symbol

    # Gaussian pulse shape
    # g(t) = (1 / (sqrt(2*pi) * sigma)) * exp(-t^2 / (2 * sigma^2))
    # where sigma = sqrt(ln(2)) / (2 * pi * BT)
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * BT)
    gaussian_pulse = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-t**2 / (2 * sigma**2))

    # Normalize to unit energy
    gaussian_pulse /= np.sum(gaussian_pulse)

    return gaussian_pulse


def generate_gmsk_fsk(pattern_bits: np.ndarray,
                     tone_a_hz: float,
                     tone_b_hz: float,
                     sample_rate: int = 48000,
                     symbol_rate: int = 200) -> np.ndarray:
    """Generate GMSK-modulated 2-FSK signal from pattern bits.

    This implements the CASCADE V2 pattern layer:
    - Binary pattern selects between tone A and tone B
    - GMSK pulse shaping (BT=0.3) for spectral containment
    - Constant envelope property maintained

    Args:
        pattern_bits: Binary pattern, shape (pattern_length,), values {0, 1}
        tone_a_hz: Frequency for bit 0 (e.g., 1300 Hz)
        tone_b_hz: Frequency for bit 1 (e.g., 1320 Hz)
        sample_rate: IQ sample rate in Hz (default 48000)
        symbol_rate: Symbol rate in symbols/second (default 200)

    Returns:
        np.ndarray: Complex IQ samples, shape (num_samples,), dtype complex64
                   where num_samples = pattern_length * samples_per_symbol

    Example:
        >>> pattern = np.array([0, 1, 1, 0, 1])
        >>> iq = generate_gmsk_fsk(pattern, 1300, 1320, 48000, 200)
        >>> iq.shape
        (1200,)  # 5 symbols * 240 samples/symbol
    """
    pattern_length = len(pattern_bits)
    samples_per_symbol = sample_rate // symbol_rate  # 240 for CASCADE V2

    # Validate inputs
    if not np.all((pattern_bits == 0) | (pattern_bits == 1)):
        raise ValueError("pattern_bits must contain only 0 and 1")

    # Convert bits to NRZ: 0 → -1, 1 → +1
    nrz = 2 * pattern_bits.astype(np.float32) - 1

    # Upsample to sample rate (insert zeros between symbols)
    nrz_upsampled = np.zeros(pattern_length * samples_per_symbol, dtype=np.float32)
    nrz_upsampled[::samples_per_symbol] = nrz

    # Apply Gaussian filter
    gaussian_filter = generate_gaussian_filter(BT=0.3, span_symbols=4,
                                              samples_per_symbol=samples_per_symbol)
    filtered = np.convolve(nrz_upsampled, gaussian_filter, mode='same')

    # Normalize filtered signal to ensure full frequency deviation
    # The Gaussian filter reduces amplitude - need to scale back to ±1
    # Use 95th percentile instead of max to avoid scaling by overshoot
    steady_state_level = np.percentile(np.abs(filtered), 95)
    if steady_state_level > 0:
        filtered = filtered / steady_state_level

    # Generate 2-FSK carrier with GMSK phase modulation
    # Center frequency is midpoint between tones
    center_freq = (tone_a_hz + tone_b_hz) / 2
    freq_deviation = (tone_b_hz - tone_a_hz) / 2

    # Time vector
    dt = 1 / sample_rate
    t = np.arange(len(filtered)) / sample_rate

    # Frequency modulation: f(t) = fc + fd * filtered(t)
    # where fc = center frequency, fd = frequency deviation, filtered varies ±1
    instantaneous_freq = center_freq + freq_deviation * filtered

    # Convert to phase: θ(t) = 2π ∫ f(τ) dτ
    # Integrate instantaneous frequency directly to get phase
    instantaneous_phase = 2 * np.pi * np.cumsum(instantaneous_freq) * dt

    # Generate complex IQ signal with constant envelope
    iq_signal = np.exp(1j * instantaneous_phase).astype(np.complex64)

    # Verify constant envelope (debug check)
    envelope = np.abs(iq_signal)
    if not np.allclose(envelope, 1.0, atol=0.01):
        # Warning: envelope not constant (could indicate numerical issues)
        envelope_var = np.var(envelope)
        if envelope_var > 0.01:
            import warnings
            warnings.warn(
                f"GMSK envelope not constant: variance={envelope_var:.6f}. "
                f"Expected constant envelope for GMSK."
            )

    return iq_signal


def generate_gmsk_3fsk(pattern_symbols: np.ndarray,
                       tone_a_hz: float,
                       tone_b_hz: float,
                       tone_c_hz: float,
                       sample_rate: int = 48000,
                       symbol_rate: int = 200) -> np.ndarray:
    """Generate GMSK-modulated 3-FSK signal from ternary pattern.

    This implements the CASCADE V2.1 pattern layer with frequency diversity:
    - Ternary pattern selects between tone A, B, or C
    - GMSK pulse shaping (BT=0.3) for spectral containment
    - Constant envelope property maintained
    - Better frequency-selective fading resilience than 2-FSK

    Args:
        pattern_symbols: Ternary pattern, shape (pattern_length,), values {0, 1, 2}
        tone_a_hz: Frequency for symbol 0 (e.g., 1300 Hz)
        tone_b_hz: Frequency for symbol 1 (e.g., 1320 Hz)
        tone_c_hz: Frequency for symbol 2 (e.g., 1340 Hz)
        sample_rate: IQ sample rate in Hz (default 48000)
        symbol_rate: Symbol rate in symbols/second (default 200)

    Returns:
        np.ndarray: Complex IQ samples, shape (num_samples,), dtype complex64
                   where num_samples = pattern_length * samples_per_symbol

    Example:
        >>> pattern = np.array([0, 1, 2, 1, 0, 2])
        >>> iq = generate_gmsk_3fsk(pattern, 1300, 1320, 1340, 48000, 200)
        >>> iq.shape
        (1440,)  # 6 symbols * 240 samples/symbol
    """
    pattern_length = len(pattern_symbols)
    samples_per_symbol = sample_rate // symbol_rate  # 240 for CASCADE V2

    # Validate inputs
    if not np.all((pattern_symbols >= 0) & (pattern_symbols <= 2)):
        raise ValueError("pattern_symbols must contain only 0, 1, and 2")

    # Convert ternary to 3-level: 0 → -1, 1 → 0, 2 → +1
    # This maps to three frequency states evenly spaced
    nrz = pattern_symbols.astype(np.float32) - 1

    # Upsample to sample rate (insert zeros between symbols)
    nrz_upsampled = np.zeros(pattern_length * samples_per_symbol, dtype=np.float32)
    nrz_upsampled[::samples_per_symbol] = nrz

    # Apply Gaussian filter
    gaussian_filter = generate_gaussian_filter(BT=0.3, span_symbols=4,
                                              samples_per_symbol=samples_per_symbol)
    filtered = np.convolve(nrz_upsampled, gaussian_filter, mode='same')

    # Normalize filtered signal to ensure full frequency deviation
    # For 3-FSK, filtered varies between -1, 0, +1
    steady_state_level = np.percentile(np.abs(filtered), 95)
    if steady_state_level > 0:
        filtered = filtered / steady_state_level

    # Generate 3-FSK carrier with GMSK phase modulation
    # Center frequency is middle tone (tone_b)
    center_freq = tone_b_hz
    # Frequency deviation: tone_a is -fd, tone_b is 0, tone_c is +fd
    freq_deviation = (tone_c_hz - tone_a_hz) / 2

    # Time vector
    dt = 1 / sample_rate
    t = np.arange(len(filtered)) / sample_rate

    # Frequency modulation: f(t) = fc + fd * filtered(t)
    # where fc = center frequency (tone_b), fd = half spacing, filtered varies -1, 0, +1
    instantaneous_freq = center_freq + freq_deviation * filtered

    # Convert to phase: θ(t) = 2π ∫ f(τ) dτ
    instantaneous_phase = 2 * np.pi * np.cumsum(instantaneous_freq) * dt

    # Generate complex IQ signal with constant envelope
    iq_signal = np.exp(1j * instantaneous_phase).astype(np.complex64)

    # Verify constant envelope (debug check)
    envelope = np.abs(iq_signal)
    if not np.allclose(envelope, 1.0, atol=0.01):
        envelope_var = np.var(envelope)
        if envelope_var > 0.01:
            import warnings
            warnings.warn(
                f"GMSK 3-FSK envelope not constant: variance={envelope_var:.6f}. "
                f"Expected constant envelope for GMSK."
            )

    return iq_signal


def measure_gmsk_bandwidth(iq_signal: np.ndarray, sample_rate: int = 48000,
                          threshold_db: float = -40) -> float:
    """Measure occupied bandwidth of GMSK signal at given threshold.

    Args:
        iq_signal: Complex IQ samples
        sample_rate: Sample rate in Hz
        threshold_db: Threshold below peak in dB (e.g., -40 dB)

    Returns:
        float: Occupied bandwidth in Hz at the threshold

    Note:
        CASCADE V2 requires GMSK bandwidth < 30 Hz at -40 dB
    """
    # Compute power spectral density
    freqs, psd = scipy_signal.welch(iq_signal, fs=sample_rate, nperseg=1024)

    # Convert to dB
    psd_db = 10 * np.log10(psd + 1e-12)  # Add small value to avoid log(0)
    peak_db = np.max(psd_db)

    # Find frequencies above threshold
    above_threshold = psd_db >= (peak_db + threshold_db)
    occupied_freqs = freqs[above_threshold]

    if len(occupied_freqs) == 0:
        return 0.0

    bandwidth = occupied_freqs[-1] - occupied_freqs[0]
    return bandwidth


def verify_constant_envelope(iq_signal: np.ndarray, tolerance: float = 0.05) -> bool:
    """Verify that IQ signal has constant envelope (GMSK property).

    Args:
        iq_signal: Complex IQ samples
        tolerance: Maximum allowed envelope variation (default 5%)

    Returns:
        bool: True if envelope is constant within tolerance

    Note:
        GMSK should have constant envelope |I² + Q²| ≈ 1
    """
    envelope = np.abs(iq_signal)
    mean_envelope = np.mean(envelope)

    # Check if envelope is close to 1.0 and has low variance
    envelope_normalized = envelope / mean_envelope
    variance = np.var(envelope_normalized)

    return variance < tolerance
