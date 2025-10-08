"""Main CASCADE V2 signal generator.

Combines pattern loading, GMSK modulation, constellation mapping, and Polar encoding
to generate clean V2-compliant IQ signals.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from pathlib import Path

from . import pattern_loader
from . import gmsk
from . import modulation
from . import polar_codec


def raised_cosine_filter(beta=0.35, span_symbols=8, samples_per_symbol=240):
    """Generate raised cosine filter for pulse shaping.

    Args:
        beta: Roll-off factor (0-1), controls bandwidth
        span_symbols: Filter span in symbol periods
        samples_per_symbol: Samples per symbol

    Returns:
        np.ndarray: Normalized filter coefficients
    """
    span_samples = span_symbols * samples_per_symbol
    t = np.arange(-span_samples // 2, span_samples // 2) / samples_per_symbol

    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2)
        # Handle singular points
        singular_points = np.abs(np.abs(t) - 1/(2*beta)) < 1e-10
        h[singular_points] = np.pi / 4 * np.sinc(1 / (2 * beta))
        h[np.abs(t) < 1e-10] = 1.0

    # Normalize for unit energy
    h /= np.sqrt(np.sum(h**2))
    return h


def apply_carrier_ramps(signal, ramp_duration_ms, sample_rate):
    """Apply smooth preamble/postamble ramps to prevent spectral splatter.

    Args:
        signal: Complex signal to ramp
        ramp_duration_ms: Ramp duration in milliseconds
        sample_rate: Sample rate in Hz

    Returns:
        np.ndarray: Ramped signal
    """
    ramp_samples = int(ramp_duration_ms * sample_rate / 1000)

    if len(signal) <= 2 * ramp_samples:
        return signal  # Too short to ramp

    # Create raised cosine ramps
    ramp_up = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples))
    ramp_down = 0.5 * (1 + np.cos(np.pi * np.arange(ramp_samples) / ramp_samples))

    # Apply ramps
    ramped = signal.copy()
    ramped[:ramp_samples] *= ramp_up
    ramped[-ramp_samples:] *= ramp_down

    return ramped


@dataclass
class KernelParameters:
    """Discrete portion of CASCADE kernel (TX parameters)."""
    pattern_id: int  # 0-3 (4 patterns with 3 tones each)
    frequency_triple: int  # 0-42 (3-FSK: 3 adjacent channels)
    modulation: str  # 'BPSK', 'QPSK', '8-PSK', '16-APSK'
    polar_rate: Tuple[int, int]  # (k, n) e.g., (2, 3)


@dataclass
class CleanIQSignal:
    """Clean CASCADE V2 signal with metadata."""
    iq_samples: np.ndarray  # Complex64, shape (num_samples,)
    sample_rate: int  # 48000 Hz
    pattern_symbol_rate: int  # 75 symbols/second (pattern layer)
    data_symbol_rate: int  # 200 symbols/second (data layer, variable)
    pattern_length: int  # 64, 128, 256, 512, 1024, or 2048
    kernel_params: KernelParameters
    tone_a_hz: float  # 3-FSK: First tone of triple
    tone_b_hz: float  # 3-FSK: Second tone of triple
    tone_c_hz: float  # 3-FSK: Third tone of triple
    polar_codeword: np.ndarray  # uint8, shape (pattern_length,)
    generation_timestamp: str  # ISO 8601
    generator_version: str  # "2.0.0"
    has_carrier_ramps: bool  # True if preamble/postamble ramps applied


class SignalGenerator:
    """CASCADE V2 signal generator.

    Generates clean IQ signals following CASCADE V2 specification:
    - Dual-layer modulation (GMSK 3-FSK + BPSK/QPSK/8-PSK/16-APSK)
    - 129-channel grid, 43 frequency triples (300-2860 Hz with 140 Hz guard band)
    - Pattern layer: 75 sym/s (optimized for low power detection)
    - Data layer: 200 sym/s (variable based on SNR)
    - Sample rate: 48 kHz
    - Polar error correction
    - 15ms preamble/postamble ramps for spectral containment
    - Raised cosine pulse shaping (β=0.35) on data layer
    """

    # CASCADE V2 constants
    SAMPLE_RATE = 48000  # Hz
    PATTERN_SYMBOL_RATE = 25  # Pattern layer symbols/second (optimized for low power)
    DATA_SYMBOL_RATE = 200  # Data layer symbols/second (variable based on SNR)
    MIN_FREQ = 300  # Hz (140 Hz guard band on right: 2860-3000 Hz)
    MAX_FREQ = 2860  # Hz (usable bandwidth: 300-2860 Hz = 2.56 kHz)
    TONE_SPACING = 20  # Hz
    NUM_CHANNELS = 129  # 300-2860 Hz in 20 Hz steps
    NUM_FREQUENCY_TRIPLES = 43  # 129 channels / 3 = 43 triples for 3-FSK
    MAX_PATTERN_LENGTH = 4096  # Maximum pattern length with looping (supports partial < 1024)
    RAMP_DURATION_MS = 15  # Preamble/postamble ramp duration for spectral containment

    def __init__(self, patterns_dir: Optional[Path] = None):
        """Initialize signal generator.

        Args:
            patterns_dir: Directory containing pattern .pkl files
        """
        self.pattern_loader = pattern_loader.PatternLoader(patterns_dir)

        # Pre-load all patterns for performance
        try:
            loaded = self.pattern_loader.load_all_patterns()
            if loaded > 0:
                print(f"Loaded {loaded}/4 master patterns (1024 symbols, partial patterns supported)")
        except Exception as e:
            print(f"Warning: Could not pre-load patterns: {e}")

    def validate_parameters(self, pattern_id: int, frequency_triple: int,
                           modulation_scheme: str, polar_rate: Tuple[int, int]) -> None:
        """Validate kernel parameters.

        Args:
            pattern_id: Pattern ID (0-3, 4 patterns with 3 tones each)
            frequency_triple: Frequency triple index (0-42)
            modulation_scheme: Modulation type
            polar_rate: Polar code rate (k, n)

        Raises:
            ValueError: If any parameter is invalid
        """
        if not (0 <= pattern_id <= 3):
            raise ValueError(f"pattern_id must be 0-3 (4 patterns), got {pattern_id}")

        if not (0 <= frequency_triple <= 42):
            raise ValueError(f"frequency_triple must be 0-42, got {frequency_triple}")

        if modulation_scheme not in ['BPSK', 'QPSK', '8-PSK', '16-APSK']:
            raise ValueError(
                f"modulation must be BPSK/QPSK/8-PSK/16-APSK, got {modulation_scheme}"
            )

        polar_codec.validate_rate(polar_rate[0], polar_rate[1])

    def get_tone_triple_frequencies(self, frequency_triple: int) -> Tuple[float, float, float]:
        """Calculate tone frequencies for a frequency triple (3-FSK).

        CASCADE V2 uses 129 channels at 20 Hz spacing (300-2860 Hz).
        Channels are organized into 43 frequency triples (129 / 3 = 43).
        140 Hz guard band on right (2860-3000 Hz reserved).

        Args:
            frequency_triple: Triple index (0-42)

        Returns:
            Tuple[float, float, float]: (tone_a_hz, tone_b_hz, tone_c_hz)

        Example:
            >>> gen = SignalGenerator()
            >>> gen.get_tone_triple_frequencies(0)
            (300.0, 320.0, 340.0)
            >>> gen.get_tone_triple_frequencies(21)
            (1560.0, 1580.0, 1600.0)
            >>> gen.get_tone_triple_frequencies(42)
            (2820.0, 2840.0, 2860.0)
        """
        if not (0 <= frequency_triple <= 42):
            raise ValueError(f"frequency_triple must be 0-42, got {frequency_triple}")

        # Each triple consists of three adjacent channels
        channel_a = 3 * frequency_triple
        channel_b = 3 * frequency_triple + 1
        channel_c = 3 * frequency_triple + 2

        tone_a = self.MIN_FREQ + channel_a * self.TONE_SPACING
        tone_b = self.MIN_FREQ + channel_b * self.TONE_SPACING
        tone_c = self.MIN_FREQ + channel_c * self.TONE_SPACING

        return (float(tone_a), float(tone_b), float(tone_c))

    def estimate_message_capacity(self, pattern_length: int,
                                  polar_rate: Tuple[int, int],
                                  modulation_scheme: str) -> int:
        """Estimate maximum data bits that fit in a pattern.

        Args:
            pattern_length: Pattern length in symbols
            polar_rate: Polar code rate (k, n)
            modulation_scheme: Modulation type

        Returns:
            int: Maximum data bits before Polar encoding
        """
        # Bits per symbol from modulation
        bits_per_symbol = modulation.get_bits_per_symbol(modulation_scheme)

        # Total raw bits in pattern
        raw_bits = pattern_length * bits_per_symbol

        # After Polar encoding (rate k/n), data bits = raw_bits * (k/n)
        k, n = polar_rate
        data_bits = int(raw_bits * (k / n))

        return data_bits

    def get_required_pattern_length(self, message_bits: int,
                                    polar_rate: Tuple[int, int],
                                    modulation_scheme: str) -> int:
        """Determine required pattern length for message size.

        Calculates minimum pattern length needed to fit message after Polar encoding.
        Pattern can loop if needed (up to 4096 symbols).

        Args:
            message_bits: Number of data bits to transmit
            polar_rate: Polar code rate (k, n)
            modulation_scheme: Modulation type

        Returns:
            int: Minimum pattern length in symbols (rounded up to power of 2 for efficiency)

        Raises:
            ValueError: If message too large for maximum pattern length
        """
        # Calculate required pattern length
        bits_per_symbol = modulation.get_bits_per_symbol(modulation_scheme)
        k, n = polar_rate

        # After Polar encoding: encoded_bits = message_bits * (n/k)
        # Pattern needs: pattern_length * bits_per_symbol >= encoded_bits
        # So: pattern_length >= (message_bits * n/k) / bits_per_symbol

        min_length = int(np.ceil((message_bits * n / k) / bits_per_symbol))

        # Round up to next power of 2 for efficiency (better for FFT, etc.)
        pattern_length = 2 ** int(np.ceil(np.log2(max(64, min_length))))

        # Check against maximum
        if pattern_length > self.MAX_PATTERN_LENGTH:
            max_capacity = self.estimate_message_capacity(
                self.MAX_PATTERN_LENGTH, polar_rate, modulation_scheme
            )
            raise ValueError(
                f"Message ({message_bits} bits) exceeds maximum capacity "
                f"({max_capacity} bits) for longest pattern ({self.MAX_PATTERN_LENGTH} symbols)"
            )

        return pattern_length

    def generate(self, pattern_id: int, frequency_triple: int,
                modulation_scheme: str, polar_rate: Tuple[int, int],
                message: bytes, seed: Optional[int] = None) -> Tuple[CleanIQSignal, Dict]:
        """Generate clean CASCADE V2 signal with 3-FSK modulation.

        Args:
            pattern_id: Pattern ID (0-7)
            frequency_triple: Frequency triple (0-42)
            modulation_scheme: 'BPSK', 'QPSK', '8-PSK', or '16-APSK'
            polar_rate: Polar code rate (k, n)
            message: Message bytes to transmit
            seed: Random seed for deterministic generation

        Returns:
            Tuple[CleanIQSignal, Dict]: Signal and metadata dict

        Example:
            >>> gen = SignalGenerator()
            >>> params = (3, 21, 'QPSK', (2, 3))
            >>> signal, metadata = gen.generate(*params, b"Hello CASCADE", seed=42)
            >>> signal.iq_samples.shape
            (349440,)  # 512 symbols × 640 samples/symbol (pattern) + ramps
            >>> signal.has_carrier_ramps
            True
        """
        if seed is not None:
            np.random.seed(seed)

        # Validate parameters
        self.validate_parameters(pattern_id, frequency_triple, modulation_scheme, polar_rate)

        # Convert message to bits
        message_bytes = np.frombuffer(message, dtype=np.uint8)
        message_bits = np.unpackbits(message_bytes)

        # Determine required pattern length
        pattern_length = self.get_required_pattern_length(
            len(message_bits), polar_rate, modulation_scheme
        )

        # Load pattern (ternary symbols for 3-FSK)
        pattern_symbols = self.pattern_loader.load_pattern(pattern_id, pattern_length)

        # Calculate bits per symbol for modulation
        bits_per_symbol = modulation.get_bits_per_symbol(modulation_scheme)

        # Polar code block length = total bits available in pattern
        polar_block_length = pattern_length * bits_per_symbol

        # Encode message with Polar code
        polar_codeword = polar_codec.encode(message_bits, polar_rate, polar_block_length)

        # Pad if needed to fill pattern
        if len(polar_codeword) < polar_block_length:
            padded = np.zeros(polar_block_length, dtype=np.uint8)
            padded[:len(polar_codeword)] = polar_codeword
            polar_codeword = padded

        # Modulate data with constellation
        data_symbols = modulation.map_to_constellation(polar_codeword, modulation_scheme)

        # Generate GMSK 3-FSK for pattern layer (25 sym/s, optimized for low power)
        tone_a, tone_b, tone_c = self.get_tone_triple_frequencies(frequency_triple)

        # Add extra samples for preamble/postamble ramps
        ramp_samples = int(self.RAMP_DURATION_MS * self.SAMPLE_RATE / 1000)
        base_samples = pattern_length * (self.SAMPLE_RATE // self.PATTERN_SYMBOL_RATE)
        total_samples = base_samples + 2 * ramp_samples

        # Generate GMSK carrier for full duration (including ramps)
        num_pattern_symbols = int(total_samples / self.SAMPLE_RATE * self.PATTERN_SYMBOL_RATE)

        # Extract ternary pattern symbols for 3-FSK
        # Pattern is already ternary (0, 1, 2) from genetic algorithm
        pattern_symbols_extended = pattern_symbols[:num_pattern_symbols]
        if len(pattern_symbols_extended) < num_pattern_symbols:
            # Pad if needed (wrap around pattern)
            pattern_symbols_extended = np.pad(
                pattern_symbols_extended,
                (0, num_pattern_symbols - len(pattern_symbols_extended)),
                mode='wrap'
            )

        gmsk_signal = gmsk.generate_gmsk_3fsk(
            pattern_symbols_extended, tone_a, tone_b, tone_c,
            self.SAMPLE_RATE, self.PATTERN_SYMBOL_RATE
        )

        # Apply carrier ramps (preamble/postamble for spectral containment)
        gmsk_signal = apply_carrier_ramps(gmsk_signal, self.RAMP_DURATION_MS, self.SAMPLE_RATE)

        # Apply raised cosine pulse shaping to data symbols (data layer at 200 sym/s)
        samples_per_data_symbol = self.SAMPLE_RATE // self.DATA_SYMBOL_RATE

        # Upsample data symbols with raised cosine filtering
        data_upsampled = np.zeros(len(gmsk_signal), dtype=np.complex64)
        for i, sym in enumerate(data_symbols):
            start_idx = i * samples_per_data_symbol
            if start_idx < len(data_upsampled):
                data_upsampled[start_idx] = sym

        # Apply raised cosine filter to I and Q separately
        rc_filter = raised_cosine_filter(beta=0.35, span_symbols=8, samples_per_symbol=samples_per_data_symbol)
        i_shaped = np.convolve(data_upsampled.real, rc_filter, mode='same')
        q_shaped = np.convolve(data_upsampled.imag, rc_filter, mode='same')
        data_shaped = i_shaped + 1j * q_shaped

        # Ensure same length
        min_len = min(len(gmsk_signal), len(data_shaped))
        gmsk_signal = gmsk_signal[:min_len]
        data_shaped = data_shaped[:min_len]

        # Combine layers: multiply ramped GMSK carrier with pulse-shaped data
        iq_signal = gmsk_signal * data_shaped

        # Create result objects
        kernel_params = KernelParameters(
            pattern_id=pattern_id,
            frequency_triple=frequency_triple,
            modulation=modulation_scheme,
            polar_rate=polar_rate
        )

        from datetime import datetime
        timestamp = datetime.utcnow().isoformat() + 'Z'

        clean_signal = CleanIQSignal(
            iq_samples=iq_signal.astype(np.complex64),
            sample_rate=self.SAMPLE_RATE,
            pattern_symbol_rate=self.PATTERN_SYMBOL_RATE,
            data_symbol_rate=self.DATA_SYMBOL_RATE,
            pattern_length=pattern_length,
            kernel_params=kernel_params,
            tone_a_hz=tone_a,
            tone_b_hz=tone_b,
            tone_c_hz=tone_c,
            polar_codeword=polar_codeword[:pattern_length],
            generation_timestamp=timestamp,
            generator_version="2.0.0",
            has_carrier_ramps=True
        )

        metadata = {
            'pattern_id': pattern_id,
            'frequency_triple': frequency_triple,
            'modulation': modulation_scheme,
            'polar_rate': f"{polar_rate[0]}/{polar_rate[1]}",
            'pattern_length': pattern_length,
            'message_bytes': len(message),
            'duration_seconds': len(iq_signal) / self.SAMPLE_RATE,
            'num_samples': len(iq_signal)
        }

        return clean_signal, metadata

    def generate_from_params(self, kernel_params: KernelParameters,
                            message: bytes, seed: Optional[int] = None
                            ) -> Tuple[CleanIQSignal, Dict]:
        """Generate signal from KernelParameters dataclass.

        Convenience wrapper around generate().

        Args:
            kernel_params: Kernel parameters object
            message: Message bytes
            seed: Random seed

        Returns:
            Tuple[CleanIQSignal, Dict]: Signal and metadata
        """
        return self.generate(
            kernel_params.pattern_id,
            kernel_params.frequency_triple,
            kernel_params.modulation,
            kernel_params.polar_rate,
            message,
            seed
        )
