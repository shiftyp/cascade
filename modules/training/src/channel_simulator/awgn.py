"""Additive White Gaussian Noise (AWGN) generator for CASCADE V2.

Implements AWGN channel for synthetic training data generation.
Supports configurable SNR and proper power normalization.

Source: CASCADE V2 spec - Expert dataset generation with controlled SNR
"""

import numpy as np
from typing import Optional


def calculate_noise_power(signal: np.ndarray, snr_db: float) -> float:
    """Calculate noise power for target SNR.

    Args:
        signal: Clean IQ signal (complex64)
        snr_db: Target Signal-to-Noise Ratio in dB

    Returns:
        float: Noise power (variance) to achieve target SNR

    Note:
        SNR (dB) = 10 * log10(signal_power / noise_power)
        Therefore: noise_power = signal_power / 10^(SNR_dB / 10)
    """
    # Calculate signal power (average magnitude squared)
    signal_power = np.mean(np.abs(signal) ** 2)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate required noise power
    noise_power = signal_power / snr_linear

    return noise_power


def generate_awgn(signal: np.ndarray, snr_db: float,
                  seed: Optional[int] = None) -> np.ndarray:
    """Add AWGN to clean signal at specified SNR.

    Args:
        signal: Clean IQ signal, shape (num_samples,), complex64
        snr_db: Target SNR in dB (can be negative for harsh conditions)
        seed: Random seed for reproducibility (default: None)

    Returns:
        np.ndarray: Noisy signal (signal + noise), complex64

    Example:
        >>> clean_signal = generate_clean_signal()
        >>> noisy_signal = generate_awgn(clean_signal, snr_db=-10)
        >>> # Noisy signal now has SNR of -10 dB
    """
    if signal.dtype != np.complex64 and signal.dtype != np.complex128:
        raise ValueError(f"Signal must be complex, got dtype={signal.dtype}")

    if len(signal.shape) != 1:
        raise ValueError(f"Signal must be 1D array, got shape={signal.shape}")

    # Set random seed if provided
    rng = np.random.default_rng(seed)

    # Calculate required noise power
    noise_power = calculate_noise_power(signal, snr_db)

    # Generate complex Gaussian noise
    # For complex noise: I ~ N(0, σ²/2), Q ~ N(0, σ²/2)
    # Total power = σ²/2 + σ²/2 = σ²
    noise_std = np.sqrt(noise_power)

    # Generate I (real) and Q (imaginary) components independently
    noise_i = rng.normal(0, noise_std / np.sqrt(2), len(signal))
    noise_q = rng.normal(0, noise_std / np.sqrt(2), len(signal))

    # Combine into complex noise
    noise = noise_i + 1j * noise_q

    # Add noise to signal
    noisy_signal = signal + noise

    return noisy_signal.astype(np.complex64)


def measure_snr(signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """Measure actual SNR between clean and noisy signals.

    Args:
        signal: Clean signal (complex64)
        noisy_signal: Noisy signal (complex64)

    Returns:
        float: Measured SNR in dB

    Note:
        This computes SNR as: 10 * log10(signal_power / noise_power)
        where noise_power = E[|noisy - clean|²]
    """
    if len(signal) != len(noisy_signal):
        raise ValueError("Signal and noisy_signal must have same length")

    # Calculate signal power
    signal_power = np.mean(np.abs(signal) ** 2)

    # Calculate noise (difference between noisy and clean)
    noise = noisy_signal - signal
    noise_power = np.mean(np.abs(noise) ** 2)

    # Avoid log(0)
    if noise_power < 1e-20:
        return np.inf

    # Calculate SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)

    return snr_db


def generate_awgn_noise_only(num_samples: int, noise_power: float,
                             seed: Optional[int] = None) -> np.ndarray:
    """Generate AWGN noise samples without signal.

    Useful for creating noise-only scenarios or background noise.

    Args:
        num_samples: Number of samples to generate
        noise_power: Noise power (variance)
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Complex Gaussian noise, shape (num_samples,), complex64
    """
    rng = np.random.default_rng(seed)

    noise_std = np.sqrt(noise_power)

    # Generate I and Q components
    noise_i = rng.normal(0, noise_std / np.sqrt(2), num_samples)
    noise_q = rng.normal(0, noise_std / np.sqrt(2), num_samples)

    noise = noise_i + 1j * noise_q

    return noise.astype(np.complex64)


def sweep_snr_range(signal: np.ndarray, snr_range_db: np.ndarray,
                    seed: Optional[int] = None) -> list[tuple[float, np.ndarray]]:
    """Generate multiple noisy versions of signal across SNR range.

    Args:
        signal: Clean signal (complex64)
        snr_range_db: Array of SNR values in dB to generate
        seed: Random seed for reproducibility

    Returns:
        List of (snr_db, noisy_signal) tuples

    Example:
        >>> clean_signal = generate_clean_signal()
        >>> snr_sweep = np.arange(-20, 21, 5)  # -20 to +20 dB in 5 dB steps
        >>> results = sweep_snr_range(clean_signal, snr_sweep, seed=42)
        >>> for snr, noisy in results:
        ...     print(f"SNR: {snr} dB, Power: {np.mean(np.abs(noisy)**2):.3f}")
    """
    results = []

    # Use separate seeds for each SNR to ensure independence
    rng = np.random.default_rng(seed)
    seeds = [rng.integers(0, 2**31) for _ in range(len(snr_range_db))]

    for snr_db, subseed in zip(snr_range_db, seeds):
        noisy_signal = generate_awgn(signal, snr_db, seed=subseed)
        results.append((float(snr_db), noisy_signal))

    return results


def estimate_noise_floor(signal: np.ndarray, percentile: float = 10.0) -> float:
    """Estimate noise floor from signal using percentile method.

    Useful for quality assessment and SNR estimation without ground truth.

    Args:
        signal: IQ signal (possibly noisy)
        percentile: Percentile to use for noise floor estimation (default: 10%)

    Returns:
        float: Estimated noise floor power

    Note:
        This assumes the lowest-power portions of the signal are mostly noise.
        Works best for signals with distinct on/off periods.
    """
    # Calculate instantaneous power
    inst_power = np.abs(signal) ** 2

    # Use percentile as noise floor estimate
    noise_floor = np.percentile(inst_power, percentile)

    return noise_floor
