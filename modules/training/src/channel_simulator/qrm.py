"""QRM (Man-made interference) generators for CASCADE V2.

Implements various types of radio interference for realistic training data:
- CW (Morse code)
- SSB voice
- FT8/WSPR digital modes
- Other digital modes (RTTY, PSK31, etc.)

Source: HF band interference characteristics for neural network training
"""

import numpy as np
from typing import Optional, List


def generate_cw_interference(num_samples: int, sample_rate: int = 48000,
                            center_freq: float = 1500.0, wpm: int = 20,
                            power: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """Generate CW (Morse code) interference.

    Args:
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        center_freq: CW tone frequency in Hz (default: 1500)
        wpm: Words per minute (default: 20)
        power: Signal power (default: 1.0)
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Complex CW signal, complex64
    """
    rng = np.random.default_rng(seed)

    # Calculate timing (PARIS standard: 50 dit-time units per word)
    dit_duration = 60.0 / (wpm * 50)  # seconds per dit
    dit_samples = int(dit_duration * sample_rate)

    # Generate on/off pattern (simplified - not real Morse)
    # Randomly alternate between on (dit/dah) and off (space)
    t = np.arange(num_samples) / sample_rate
    carrier = np.exp(2j * np.pi * center_freq * t)

    # Create keying envelope
    envelope = np.zeros(num_samples)
    pos = 0

    while pos < num_samples:
        # Key down duration (dit or dah)
        if rng.random() < 0.7:
            # Dit (1 unit)
            key_down = dit_samples
        else:
            # Dah (3 units)
            key_down = 3 * dit_samples

        # Key up duration (inter-element space)
        key_up = dit_samples  # 1 unit space

        # Apply keying with rise/fall time
        rise_samples = int(0.003 * sample_rate)  # 3 ms rise time

        if pos + key_down < num_samples:
            # Rise
            envelope[pos:pos + rise_samples] = np.linspace(0, 1, rise_samples)
            # On
            envelope[pos + rise_samples:pos + key_down - rise_samples] = 1.0
            # Fall
            if pos + key_down < num_samples:
                envelope[pos + key_down - rise_samples:pos + key_down] = np.linspace(1, 0, rise_samples)

        pos += key_down + key_up

    # Apply envelope to carrier
    cw_signal = carrier * envelope

    # Scale to desired power
    cw_signal *= np.sqrt(power)

    return cw_signal.astype(np.complex64)


def generate_ssb_voice(num_samples: int, sample_rate: int = 48000,
                      center_freq: float = 0.0, power: float = 1.0,
                      seed: Optional[int] = None) -> np.ndarray:
    """Generate SSB voice interference.

    Simulates single-sideband voice transmission with typical speech characteristics.

    Args:
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        center_freq: SSB center frequency in Hz (default: 0 = baseband)
        power: Signal power (default: 1.0)
        seed: Random seed

    Returns:
        np.ndarray: Complex SSB signal, complex64
    """
    rng = np.random.default_rng(seed)

    # Generate speech-like envelope
    # Speech has high peak-to-average ratio (~10-15 dB)
    duration = num_samples / sample_rate

    # Low-frequency modulation (syllable rate ~3-5 Hz)
    syllable_rate = rng.uniform(3, 5)
    t = np.arange(num_samples) / sample_rate
    syllable_envelope = 0.3 + 0.7 * (np.sin(2 * np.pi * syllable_rate * t) ** 2)

    # Add pauses (silence periods)
    num_pauses = int(duration * 0.5)  # ~2 pauses per second
    for _ in range(num_pauses):
        pause_start = rng.integers(0, num_samples - int(0.3 * sample_rate))
        pause_duration = rng.integers(int(0.1 * sample_rate), int(0.3 * sample_rate))
        syllable_envelope[pause_start:pause_start + pause_duration] = 0.0

    # Generate speech baseband (300-3000 Hz for voice)
    # Use sum of formants (resonances in speech)
    formant_freqs = [600, 1200, 2400]  # Typical formants
    speech_baseband = np.zeros(num_samples, dtype=np.float64)

    for freq in formant_freqs:
        phase = rng.uniform(0, 2 * np.pi)
        # Add frequency modulation (pitch variation)
        pitch_variation = 50 * np.sin(2 * np.pi * 5 * t)  # ±50 Hz at 5 Hz rate
        instantaneous_freq = freq + pitch_variation
        speech_baseband += np.sin(2 * np.pi * instantaneous_freq * t + phase)

    # Normalize baseband
    speech_baseband /= len(formant_freqs)

    # Apply envelope
    speech_modulated = speech_baseband * syllable_envelope

    # Convert to SSB (use Hilbert transform for analytic signal)
    # For simplicity, just use baseband as I component, Q=0
    # Real SSB would use Hilbert transform
    ssb_signal = speech_modulated.astype(np.complex64)

    # Shift to center frequency
    if center_freq != 0:
        carrier = np.exp(2j * np.pi * center_freq * t)
        ssb_signal *= carrier

    # Scale to desired power
    actual_power = np.mean(np.abs(ssb_signal) ** 2)
    if actual_power > 0:
        ssb_signal *= np.sqrt(power / actual_power)

    return ssb_signal.astype(np.complex64)


def generate_ft8_interference(num_samples: int, sample_rate: int = 48000,
                             center_freq: float = 1500.0, power: float = 1.0,
                             seed: Optional[int] = None) -> np.ndarray:
    """Generate FT8 digital mode interference.

    FT8: 15-second transmissions, 8-FSK, 6.25 Hz tone spacing, 50 Hz bandwidth.

    Args:
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        center_freq: FT8 center frequency in Hz (default: 1500)
        power: Signal power (default: 1.0)
        seed: Random seed

    Returns:
        np.ndarray: Complex FT8-like signal, complex64
    """
    rng = np.random.default_rng(seed)

    # FT8 parameters
    symbol_rate = 6.25  # symbols per second
    tone_spacing = 6.25  # Hz
    num_tones = 8  # 8-FSK

    # FT8 message: 79 symbols, 15 seconds total
    ft8_duration = 15.0  # seconds
    num_ft8_symbols = 79
    symbol_duration = ft8_duration / num_ft8_symbols
    samples_per_symbol = int(symbol_duration * sample_rate)

    # Generate FT8 transmissions (may have multiple in signal duration)
    duration = num_samples / sample_rate
    num_transmissions = int(np.ceil(duration / ft8_duration))

    signal = np.zeros(num_samples, dtype=np.complex64)
    t = np.arange(num_samples) / sample_rate

    for tx_idx in range(num_transmissions):
        tx_start_sample = int(tx_idx * ft8_duration * sample_rate)

        # Random FT8 message (79 tones)
        tones = rng.integers(0, num_tones, num_ft8_symbols)

        # Generate 8-FSK signal
        for sym_idx, tone in enumerate(tones):
            sym_start = tx_start_sample + sym_idx * samples_per_symbol
            sym_end = min(sym_start + samples_per_symbol, num_samples)

            if sym_start >= num_samples:
                break

            # Tone frequency
            tone_freq = center_freq + tone * tone_spacing

            # Generate tone
            t_sym = np.arange(sym_end - sym_start) / sample_rate
            signal[sym_start:sym_end] = np.exp(2j * np.pi * tone_freq * t_sym)

    # Scale to desired power
    actual_power = np.mean(np.abs(signal) ** 2)
    if actual_power > 0:
        signal *= np.sqrt(power / actual_power)

    return signal.astype(np.complex64)


def generate_rtty_interference(num_samples: int, sample_rate: int = 48000,
                              center_freq: float = 1500.0, baud_rate: int = 45,
                              shift: float = 170.0, power: float = 1.0,
                              seed: Optional[int] = None) -> np.ndarray:
    """Generate RTTY (Radioteletype) interference.

    RTTY uses FSK with typically 170 Hz shift at 45 baud.

    Args:
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        center_freq: RTTY center frequency in Hz (default: 1500)
        baud_rate: Baud rate (default: 45.45 baud)
        shift: Frequency shift in Hz (default: 170)
        power: Signal power (default: 1.0)
        seed: Random seed

    Returns:
        np.ndarray: Complex RTTY signal, complex64
    """
    rng = np.random.default_rng(seed)

    # RTTY uses 5-bit Baudot code
    samples_per_bit = int(sample_rate / baud_rate)
    num_bits = num_samples // samples_per_bit

    # Generate random bit pattern
    bits = rng.integers(0, 2, num_bits)

    # Generate FSK signal
    mark_freq = center_freq + shift / 2  # Bit 1
    space_freq = center_freq - shift / 2  # Bit 0

    signal = np.zeros(num_samples, dtype=np.complex64)

    for bit_idx, bit in enumerate(bits):
        start_sample = bit_idx * samples_per_bit
        end_sample = min(start_sample + samples_per_bit, num_samples)

        freq = mark_freq if bit == 1 else space_freq
        t_bit = np.arange(end_sample - start_sample) / sample_rate

        signal[start_sample:end_sample] = np.exp(2j * np.pi * freq * t_bit)

    # Scale to desired power
    signal *= np.sqrt(power)

    return signal.astype(np.complex64)


def generate_psk31_interference(num_samples: int, sample_rate: int = 48000,
                               center_freq: float = 1500.0, power: float = 1.0,
                               seed: Optional[int] = None) -> np.ndarray:
    """Generate PSK31 digital mode interference.

    PSK31 uses BPSK at 31.25 baud with varicode encoding.

    Args:
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        center_freq: PSK31 center frequency in Hz (default: 1500)
        power: Signal power (default: 1.0)
        seed: Random seed

    Returns:
        np.ndarray: Complex PSK31 signal, complex64
    """
    rng = np.random.default_rng(seed)

    # PSK31 parameters
    baud_rate = 31.25
    samples_per_symbol = int(sample_rate / baud_rate)
    num_symbols = num_samples // samples_per_symbol

    # Generate random BPSK symbols (0, 180 degrees)
    phases = rng.choice([0, np.pi], num_symbols)

    # Generate carrier
    t = np.arange(num_samples) / sample_rate
    carrier = np.exp(2j * np.pi * center_freq * t)

    # Generate BPSK signal with raised cosine shaping
    signal = np.zeros(num_samples, dtype=np.complex64)

    for sym_idx, phase in enumerate(phases):
        start_sample = sym_idx * samples_per_symbol
        end_sample = min(start_sample + samples_per_symbol, num_samples)

        # Raised cosine pulse shape (smooth transitions)
        t_sym = np.arange(end_sample - start_sample) / sample_rate
        envelope = 0.5 * (1 + np.cos(2 * np.pi * baud_rate * t_sym))

        signal[start_sample:end_sample] = envelope * np.exp(1j * phase)

    # Apply carrier
    signal *= carrier

    # Scale to desired power
    signal *= np.sqrt(power)

    return signal.astype(np.complex64)


def generate_mixed_qrm(num_samples: int, sample_rate: int = 48000,
                      interference_types: Optional[List[str]] = None,
                      freq_range: tuple = (300, 3000),
                      power: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """Generate mixed QRM with multiple interference types.

    Args:
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        interference_types: List of interference types (default: all types)
        freq_range: Frequency range for interference in Hz (default: 300-3000)
        power: Total power (distributed among interference sources)
        seed: Random seed

    Returns:
        np.ndarray: Complex mixed QRM signal, complex64
    """
    if interference_types is None:
        interference_types = ['cw', 'ssb', 'ft8', 'rtty', 'psk31']

    rng = np.random.default_rng(seed)

    # Distribute power among interference types
    per_type_power = power / len(interference_types)

    signal = np.zeros(num_samples, dtype=np.complex64)

    for idx, interference_type in enumerate(interference_types):
        # Random frequency within range
        freq = rng.uniform(freq_range[0], freq_range[1])

        # Generate subseed
        subseed = rng.integers(0, 2**31)

        if interference_type == 'cw':
            qrm = generate_cw_interference(num_samples, sample_rate, freq,
                                          power=per_type_power, seed=subseed)
        elif interference_type == 'ssb':
            qrm = generate_ssb_voice(num_samples, sample_rate, freq,
                                    power=per_type_power, seed=subseed)
        elif interference_type == 'ft8':
            qrm = generate_ft8_interference(num_samples, sample_rate, freq,
                                           power=per_type_power, seed=subseed)
        elif interference_type == 'rtty':
            qrm = generate_rtty_interference(num_samples, sample_rate, freq,
                                            power=per_type_power, seed=subseed)
        elif interference_type == 'psk31':
            qrm = generate_psk31_interference(num_samples, sample_rate, freq,
                                             power=per_type_power, seed=subseed)
        else:
            continue

        signal += qrm

    return signal.astype(np.complex64)
