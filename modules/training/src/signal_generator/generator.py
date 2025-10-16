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


def raised_cosine_filter(beta=0.20, span_symbols=16, samples_per_symbol=240):
    """Generate raised cosine filter for pulse shaping.

    Args:
        beta: Roll-off factor (0-1), controls bandwidth (default 0.20 for optimal spectral efficiency)
        span_symbols: Filter span in symbol periods (increased to 16 for better spectral containment)
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

    # Apply additional windowing for sidelobe suppression
    # Hamming window reduces sidelobes by 40+ dB
    window = np.hamming(len(h))
    h = h * window

    # Normalize for unit energy (after windowing)
    h /= np.sqrt(np.sum(h**2))
    return h


def apply_carrier_ramps(signal, ramp_duration_ms, sample_rate):
    """Apply smooth preamble/postamble ramps to prevent spectral splatter.

    Uses Tukey (tapered cosine) window for smooth transitions that allow
    filter transients to settle before full amplitude modulation.

    Args:
        signal: Complex signal to ramp
        ramp_duration_ms: Ramp duration in milliseconds
        sample_rate: Sample rate in Hz

    Returns:
        np.ndarray: Ramped signal with smooth start/stop transitions
    """
    ramp_samples = int(ramp_duration_ms * sample_rate / 1000)

    if len(signal) <= 2 * ramp_samples:
        return signal  # Too short to ramp

    # Use Tukey window (tapered cosine) for smoother spectral characteristics
    # alpha controls taper: 0 = rectangular, 1 = full Hann window
    # alpha = ramp / total gives us the ramp fraction
    alpha = (2 * ramp_samples) / len(signal)

    # Generate Tukey window
    from scipy import signal as scipy_signal
    window = scipy_signal.windows.tukey(len(signal), alpha=alpha)

    # Apply window
    ramped = signal * window

    return ramped


@dataclass
class KernelParameters:
    """Discrete portion of CASCADE kernel (TX parameters)."""
    pattern_id: int  # 0-3 (4 patterns with 3 tones each)
    frequency_triple: int  # 0-42 (3-FSK: 3 adjacent channels)
    modulation: str  # 'BPSK', 'QPSK', '8-PSK', '16-APSK'
    polar_rate: Tuple[int, int]  # (k, n) e.g., (2, 3)
    data_symbol_rate: int  # 75, 100, 125, 150, 175, 200, 250, 300 sym/s
    num_centers: int = 4  # Fixed 4 centers for optimal performance (30 users at -10 dB)
    use_always_on_center: bool = True  # Use new always-on center frequency design


@dataclass
class CleanIQSignal:
    """Clean CASCADE V2 signal with metadata."""
    iq_samples: np.ndarray  # Complex64, shape (num_samples,)
    sample_rate: int  # 48000 Hz
    pattern_symbol_rate: int  # 25 symbols/second (pattern layer)
    data_symbol_rate: int  # 75-300 symbols/second (data layer, variable)
    pattern_length: int  # Partial pattern length: 64, 128, 256, 512, or 1024 symbols (from 1024 master)
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
    PATTERN_LENGTH = 1024  # Fixed pattern length (always 1024 symbols)
    VALID_DATA_SYMBOL_RATES = [75, 100, 125, 150, 175, 200, 250, 300]  # sym/s (from kernel)
    MIN_FREQ = 500  # Hz (guard band to prevent sub-500 Hz)
    MAX_FREQ = 2600  # Hz (guard band to prevent >2600 Hz)
    TONE_SPACING = 50  # Hz (increased from 14 Hz, balanced for 2-center diversity)
    NUM_CHANNELS = 42  # 500-2600 Hz in 50 Hz steps (2100/50 = 42)
    NUM_FREQUENCY_TRIPLES = 14  # 42 channels / 3 = 14 triples for 3-FSK
    RAMP_DURATION_MS = 150  # Preamble/postamble ramp (150ms for spectral containment)

    def __init__(self, patterns_dir: Optional[Path] = None):
        """Initialize signal generator.

        Args:
            patterns_dir: Directory containing pattern .pkl files
        """
        self.pattern_loader = pattern_loader.PatternLoader(patterns_dir)

        # Pre-load all patterns for performance (silent mode)
        try:
            self.pattern_loader.load_all_patterns()
        except Exception as e:
            # Silently continue if patterns can't be pre-loaded
            pass

    @staticmethod
    def select_num_centers(snr_db: float, qrm_level: float = 0.0, multipath_severity: float = 0.0) -> int:
        """Select number of center frequencies - FIXED at 4 for frequency diversity.

        Using 4 centers provides:
        - Frequency diversity against selective fading
        - +6 dB SNR gain from coherent combining
        - Robust operation in poor conditions

        Trade-off:
        - Wider spectral footprint (~280-320 Hz vs ~100 Hz)
        - Amplitude beating creates vertical streaks in spectrogram (normal for multi-carrier)
        - Reduced network capacity (~8 users vs ~21)

        Args:
            snr_db: Estimated SNR in dB (unused)
            qrm_level: QRM interference level (unused)
            multipath_severity: Multipath severity (unused)

        Returns:
            int: Always returns 4
        """
        # FIXED 4-CENTER CONFIGURATION
        # Use 4 centers for frequency diversity (user requested)
        return 4

    def validate_parameters(self, pattern_id: int, frequency_triple: int,
                           modulation_scheme: str, polar_rate: Tuple[int, int],
                           data_symbol_rate: int) -> None:
        """Validate kernel parameters.

        Args:
            pattern_id: Pattern ID (0-3, 4 patterns with 3 tones each)
            frequency_triple: Frequency triple index (0-60 with 14 Hz spacing)
            modulation_scheme: Modulation type
            polar_rate: Polar code rate (k, n)
            data_symbol_rate: Data layer symbol rate (75-300 sym/s)

        Raises:
            ValueError: If any parameter is invalid
        """
        if not (0 <= pattern_id <= 3):
            raise ValueError(f"pattern_id must be 0-3 (4 patterns), got {pattern_id}")

        if not (0 <= frequency_triple <= 60):
            raise ValueError(f"frequency_triple must be 0-60, got {frequency_triple}")

        if modulation_scheme not in ['BPSK', 'QPSK', '8-PSK', '16-APSK']:
            raise ValueError(
                f"modulation must be BPSK/QPSK/8-PSK/16-APSK, got {modulation_scheme}"
            )

        if data_symbol_rate not in self.VALID_DATA_SYMBOL_RATES:
            raise ValueError(
                f"data_symbol_rate must be one of {self.VALID_DATA_SYMBOL_RATES}, got {data_symbol_rate}"
            )

        polar_codec.validate_rate(polar_rate[0], polar_rate[1])

    def get_tone_triple_frequencies(self, frequency_triple: int) -> Tuple[float, float, float]:
        """Calculate tone frequencies for a frequency triple (3-FSK).

        CASCADE V2 uses 183 channels at 14 Hz spacing (300-2856 Hz).
        Channels are organized into 61 frequency triples (183 / 3 = 61).
        Optimized for 30 users at -10 dB with moderate overlap.

        Args:
            frequency_triple: Triple index (0-60)

        Returns:
            Tuple[float, float, float]: (tone_a_hz, tone_b_hz, tone_c_hz)

        Example:
            >>> gen = SignalGenerator()
            >>> gen.get_tone_triple_frequencies(0)
            (300.0, 314.0, 328.0)
            >>> gen.get_tone_triple_frequencies(30)
            (1560.0, 1574.0, 1588.0)
            >>> gen.get_tone_triple_frequencies(60)
            (2820.0, 2834.0, 2848.0)
        """
        if not (0 <= frequency_triple <= 60):
            raise ValueError(f"frequency_triple must be 0-60, got {frequency_triple}")

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
        Uses partial patterns (extracts first N symbols from 1024-symbol master).

        Args:
            message_bits: Number of data bits to transmit
            polar_rate: Polar code rate (k, n)
            modulation_scheme: Modulation type

        Returns:
            int: Minimum pattern length in symbols (rounded up to power of 2 for efficiency)

        Raises:
            ValueError: If message too large for master pattern (1024 symbols)
        """
        # NOTE: This method is no longer used - pattern_length is determined
        # dynamically in generate() based on actual encoded data after zero-truncation
        # Keeping for backwards compatibility

        bits_per_symbol = modulation.get_bits_per_symbol(modulation_scheme)
        k, n = polar_rate
        encoded_bits = int(np.ceil(message_bits * n / k))

        # Return a reasonable estimate (will be recalculated in generate())
        pattern_length = 2 ** int(np.ceil(np.log2(max(64, int(np.ceil(encoded_bits / bits_per_symbol))))))

        return min(pattern_length, self.PATTERN_LENGTH)

    def generate(self, pattern_id: int, frequency_triple: int,
                modulation_scheme: str, polar_rate: Tuple[int, int],
                data_symbol_rate: int, message: bytes,
                seed: Optional[int] = None, num_centers: int = 4,
                use_always_on_center: bool = True) -> Tuple[CleanIQSignal, Dict]:
        """Generate clean CASCADE V2 signal with 3-FSK modulation.

        Args:
            pattern_id: Pattern ID (0-3, 4 patterns)
            frequency_triple: Frequency triple (0-42)
            modulation_scheme: 'BPSK', 'QPSK', '8-PSK', or '16-APSK'
            polar_rate: Polar code rate (k, n)
            data_symbol_rate: Data layer symbol rate (75, 100, 125, 150, 175, 200, 250, 300 sym/s)
            message: Message bytes to transmit
            seed: Random seed for deterministic generation
            num_centers: Number of center frequencies (default 4, fixed configuration)
            use_always_on_center: Use new always-on center frequency design (default True)

        Returns:
            Tuple[CleanIQSignal, Dict]: Signal and metadata dict

        Example:
            >>> gen = SignalGenerator()
            >>> signal, metadata = gen.generate(
            ...     pattern_id=3, frequency_triple=21, modulation_scheme='QPSK',
            ...     polar_rate=(2, 3), data_symbol_rate=150, message=b"Hello CASCADE", seed=42
            ... )
            >>> signal.iq_samples.shape
            (349440,)  # Pattern @ 25 sym/s + data @ 150 sym/s
            >>> signal.has_carrier_ramps
            True
        """
        if seed is not None:
            np.random.seed(seed)

        # Validate parameters
        self.validate_parameters(pattern_id, frequency_triple, modulation_scheme,
                                polar_rate, data_symbol_rate)

        # Convert message to bits
        message_bytes = np.frombuffer(message, dtype=np.uint8)
        message_bits = np.unpackbits(message_bytes)

        # Calculate bits per symbol for modulation
        bits_per_symbol = modulation.get_bits_per_symbol(modulation_scheme)

        # Calculate Polar block length (must be power of 2) based on message
        # Minimum 64 bits for small messages
        k, n = polar_rate
        encoded_bits = int(np.ceil(len(message_bits) * n / k))
        polar_block_length = 2 ** int(np.ceil(np.log2(max(encoded_bits, 64))))

        # Check against max capacity
        if polar_block_length > self.PATTERN_LENGTH * bits_per_symbol:
            max_capacity = self.estimate_message_capacity(self.PATTERN_LENGTH, polar_rate, modulation_scheme)
            raise ValueError(
                f"Message ({len(message_bits)} bits) exceeds maximum capacity "
                f"({max_capacity} bits) for master pattern length {self.PATTERN_LENGTH}"
            )

        # Encode message with Polar code (produces power-of-2 codeword)
        polar_codeword = polar_codec.encode(message_bits, polar_rate, polar_block_length)

        # OPTIMIZATION: Truncate trailing zeros to save transmission time!
        # Receiver will pad back to power-of-2 before decoding
        nonzero_indices = np.nonzero(polar_codeword)[0]
        if len(nonzero_indices) > 0:
            last_nonzero = nonzero_indices[-1] + 1
            # Round up to nearest symbol boundary
            last_nonzero_symbols = int(np.ceil(last_nonzero / bits_per_symbol))
            actual_bits_needed = last_nonzero_symbols * bits_per_symbol
            # Keep minimum 64 bits for reliable detection
            actual_bits_needed = max(actual_bits_needed, 64)
            polar_codeword_truncated = polar_codeword[:actual_bits_needed]
        else:
            # All zeros (empty message) - use minimum 64 bits
            polar_codeword_truncated = polar_codeword[:64]

        # Pad to ensure divisibility by bits_per_symbol (needed for 8-PSK = 3 bits/symbol)
        if len(polar_codeword_truncated) % bits_per_symbol != 0:
            pad_bits = bits_per_symbol - (len(polar_codeword_truncated) % bits_per_symbol)
            polar_codeword_truncated = np.pad(polar_codeword_truncated, (0, pad_bits), constant_values=0)

        # Pattern length = actual transmitted symbols (after zero-truncation)
        pattern_length = len(polar_codeword_truncated) // bits_per_symbol

        # Load partial pattern (first N symbols from 1024-symbol master)
        pattern_symbols = self.pattern_loader.load_pattern(pattern_id, pattern_length)

        # Modulate data with constellation
        data_symbols = modulation.map_to_constellation(polar_codeword_truncated, modulation_scheme)

        # Add preamble and postamble symbols for RC filter settling + NN channel learning
        # ENHANCED: Pilot tones + training sequence (150ms each @ data_symbol_rate)
        if len(data_symbols) > 0:
            preamble_duration_ms = 150  # Fixed 150ms
            total_pilot_symbols = int(preamble_duration_ms * data_symbol_rate / 1000)

            # PREAMBLE (150ms): Swept pilot (50ms) + Constant pilot (100ms)
            swept_pilot_symbols = int(50 * data_symbol_rate / 1000)  # 50ms swept
            constant_pilot_symbols = total_pilot_symbols - swept_pilot_symbols  # 100ms constant

            # Generate pilots for NN channel learning
            swept_pilots = modulation.generate_pilot_sequence(modulation_scheme, swept_pilot_symbols)
            constant_pilots = modulation.generate_constant_pilot(modulation_scheme, constant_pilot_symbols)
            preamble = np.concatenate([swept_pilots, constant_pilots])

            # POSTAMBLE (150ms): Constant pilot (100ms) + Ramp to last symbol (50ms)
            postamble_pilot_symbols = int(100 * data_symbol_rate / 1000)  # 100ms constant
            ramp_symbols = total_pilot_symbols - postamble_pilot_symbols  # 50ms ramp

            postamble_pilots = modulation.generate_constant_pilot(modulation_scheme, postamble_pilot_symbols)
            last_symbol = data_symbols[-1]
            ramp_to_last = np.full(ramp_symbols, last_symbol, dtype=np.complex64)
            postamble = np.concatenate([postamble_pilots, ramp_to_last])

            data_symbols = np.concatenate([preamble, data_symbols, postamble])

        # Generate GMSK 3-FSK for pattern layer (25 sym/s, optimized for low power)
        tone_a, tone_b, tone_c = self.get_tone_triple_frequencies(frequency_triple)

        # Calculate signal duration
        # Add extra duration to cover raised cosine filter tail (16 data symbols)
        base_samples = pattern_length * (self.SAMPLE_RATE // self.PATTERN_SYMBOL_RATE)
        rc_tail_samples = 16 * (self.SAMPLE_RATE // data_symbol_rate)  # Filter tail duration
        total_samples = base_samples + rc_tail_samples

        # Generate GMSK carrier for full duration (including filter tail)
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

        # Generate GMSK carrier with always-on center or traditional mode
        if use_always_on_center:
            # NEW: Always-on center frequency with alternating outers
            # Provides continuous sync reference and 3-5 dB effective SNR gain!
            gmsk_signal = gmsk.generate_gmsk_3fsk_always_on_center(
                pattern_symbols_extended, tone_a, tone_b, tone_c,
                self.SAMPLE_RATE, self.PATTERN_SYMBOL_RATE,
                num_centers=num_centers
            )
        else:
            # Traditional 3-FSK (all tones on/off together)
            gmsk_signal = gmsk.generate_gmsk_3fsk(
                pattern_symbols_extended, tone_a, tone_b, tone_c,
                self.SAMPLE_RATE, self.PATTERN_SYMBOL_RATE
            )

        # Apply raised cosine pulse shaping to data symbols (data layer at variable rate from kernel)
        samples_per_data_symbol = self.SAMPLE_RATE // data_symbol_rate

        # Pad data symbols to fill entire GMSK signal duration for clean boundaries
        # Calculate how many data symbols needed to fill the GMSK signal
        num_data_symbols_needed = int(np.ceil(len(gmsk_signal) / samples_per_data_symbol))

        # Pad by repeating last symbol to cover raised cosine filter tail
        # This prevents discontinuities at data end
        rc_filter_span = 16  # symbols
        num_data_symbols_with_tail = num_data_symbols_needed + rc_filter_span

        if len(data_symbols) < num_data_symbols_with_tail:
            padding_length = num_data_symbols_with_tail - len(data_symbols)
            # Repeat last data symbol to maintain smooth transition (no discontinuity)
            last_symbol = data_symbols[-1] if len(data_symbols) > 0 else complex(1, 0)
            padding = np.full(padding_length, last_symbol, dtype=np.complex64)
            data_symbols_padded = np.concatenate([data_symbols, padding])
        else:
            data_symbols_padded = data_symbols[:num_data_symbols_with_tail]

        # Upsample data symbols with raised cosine filtering
        # Create array large enough for all padded symbols
        upsampled_length = len(data_symbols_padded) * samples_per_data_symbol
        data_upsampled = np.zeros(upsampled_length, dtype=np.complex64)
        for i, sym in enumerate(data_symbols_padded):
            start_idx = i * samples_per_data_symbol
            if start_idx < len(data_upsampled):
                data_upsampled[start_idx] = sym

        # Apply raised cosine filter to I and Q separately
        # β=0.20: 11% narrower spectrum than β=0.35, with identical sidelobe suppression
        # 16-symbol span + Hamming window provides >38 dB sidelobe suppression
        rc_filter = raised_cosine_filter(beta=0.20, span_symbols=16, samples_per_symbol=samples_per_data_symbol)
        i_shaped = np.convolve(data_upsampled.real, rc_filter, mode='same')
        q_shaped = np.convolve(data_upsampled.imag, rc_filter, mode='same')
        data_shaped = i_shaped + 1j * q_shaped

        # Ensure same length
        min_len = min(len(gmsk_signal), len(data_shaped))
        gmsk_signal = gmsk_signal[:min_len]
        data_shaped = data_shaped[:min_len]

        # Combine layers: multiply GMSK carrier with pulse-shaped data
        iq_signal = gmsk_signal * data_shaped

        # Apply ONLY ramp-up at start (streaming dataset handles ramp-down at end)
        ramp_duration_ms = 150  # Fixed 150ms
        ramp_up_samples = int(ramp_duration_ms * self.SAMPLE_RATE / 1000)  # 7200 samples

        if len(iq_signal) > ramp_up_samples:
            # Create window with ramp-up only
            ramp_window = np.ones(len(iq_signal))

            # Ramp up at start only (raised cosine)
            ramp_up = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_up_samples) / ramp_up_samples))
            ramp_window[:ramp_up_samples] = ramp_up

            iq_signal = iq_signal * ramp_window

        # Create result objects
        kernel_params = KernelParameters(
            data_symbol_rate=data_symbol_rate,
            pattern_id=pattern_id,
            frequency_triple=frequency_triple,
            modulation=modulation_scheme,
            polar_rate=polar_rate,
            num_centers=num_centers,
            use_always_on_center=use_always_on_center
        )

        from datetime import datetime
        timestamp = datetime.utcnow().isoformat() + 'Z'

        clean_signal = CleanIQSignal(
            iq_samples=iq_signal.astype(np.complex64),
            sample_rate=self.SAMPLE_RATE,
            pattern_symbol_rate=self.PATTERN_SYMBOL_RATE,
            data_symbol_rate=data_symbol_rate,  # From kernel parameter
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
            kernel_params.data_symbol_rate,
            message,
            seed
        )
