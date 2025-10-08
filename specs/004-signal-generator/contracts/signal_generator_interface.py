"""
Signal Generator API Contract

This module defines the expected interface for the Core Signal Generator.
All implementations must satisfy this contract.

These are contract specifications, not implementations. Tests should be written
against this interface first (TDD), then implementations should satisfy the contract.
"""

from typing import Protocol, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class KernelParameters:
    """Discrete kernel parameters for signal generation."""
    pattern_id: int  # 0-7
    frequency_pair: int  # 0-66
    modulation: str  # 'BPSK', 'QPSK', '8-PSK', '16-APSK'
    polar_rate: Tuple[int, int]  # (k, n) e.g., (2, 3) for rate 2/3


@dataclass(frozen=True)
class MessageData:
    """Message to be transmitted."""
    content: str | bytes
    encoding: str = 'utf-8'  # 'utf-8' or 'binary'


@dataclass(frozen=True)
class CleanIQSignal:
    """Clean V2-compliant signal (output from Core Generator)."""
    iq_samples: np.ndarray  # complex64, shape (num_samples,)
    sample_rate: int  # 48000 Hz
    duration_seconds: float
    kernel_params: KernelParameters
    message_data: MessageData
    pattern_length: int  # 64, 128, 256, 512, 1024, or 2048
    tone_a_hz: float
    tone_b_hz: float
    polar_codeword: np.ndarray  # uint8, shape (pattern_length,)
    generation_timestamp: str  # ISO 8601
    generator_version: str


class SignalGeneratorInterface(Protocol):
    """
    Interface for CASCADE Core Signal Generator.

    Produces clean V2-compliant IQ signals using dual-layer modulation:
    - Layer 1: GMSK-modulated 2-FSK pattern (from kernel pattern_id)
    - Layer 2: BPSK/QPSK/8-PSK/16-APSK data (from kernel modulation)

    Contract Requirements:
    - FR-001 to FR-015 (see spec.md)
    - All outputs must pass V2 compliance validation
    - Signals must be deterministic given same inputs and seed
    """

    def generate(
        self,
        pattern_id: int,
        frequency_pair: int,
        modulation: str,
        polar_rate: Tuple[int, int],
        message: str | bytes,
        seed: Optional[int] = None
    ) -> Tuple[CleanIQSignal, dict]:
        """
        Generate clean CASCADE V2-compliant signal.

        Args:
            pattern_id: Which of 8 patterns to use (0-7)
            frequency_pair: Which of 67 frequency pairs (0-66)
            modulation: 'BPSK', 'QPSK', '8-PSK', or '16-APSK'
            polar_rate: (k, n) tuple for Polar code rate, e.g., (2, 3)
            message: Text string or binary bytes to transmit
            seed: Optional random seed for reproducibility

        Returns:
            Tuple of (CleanIQSignal, metadata_dict)
            - CleanIQSignal: Signal with IQ samples and parameters
            - metadata_dict: Additional generation metadata (timing, warnings, etc.)

        Raises:
            ValueError: If parameters are out of valid range
            ValueError: If message is too long for pattern capacity
            FileNotFoundError: If pattern file not found

        Contract:
            - Output sample rate must be 48000 Hz (FR-012)
            - Pattern length must be minimum to fit Polar-encoded message (FR-005, FR-010)
            - GMSK pulse shaping must use BT=0.3 (FR-006)
            - Constellation mapping must match CASCADE spec (FR-008)
            - Polar encoding must use specified rate (FR-009)
            - Tone frequencies must be on 135-channel grid (FR-011)
            - Output must be complex64 NumPy array (FR-014)
        """
        ...

    def generate_from_params(
        self,
        kernel_params: KernelParameters,
        message_data: MessageData,
        seed: Optional[int] = None
    ) -> Tuple[CleanIQSignal, dict]:
        """
        Generate signal from structured parameter objects.

        Args:
            kernel_params: KernelParameters instance
            message_data: MessageData instance
            seed: Optional random seed

        Returns:
            Tuple of (CleanIQSignal, metadata_dict)

        Raises:
            Same as generate()

        Contract:
            - Must be equivalent to generate() with unpacked parameters
            - Validation must occur on parameter objects
        """
        ...

    def validate_parameters(
        self,
        pattern_id: int,
        frequency_pair: int,
        modulation: str,
        polar_rate: Tuple[int, int]
    ) -> None:
        """
        Validate kernel parameters without generating signal.

        Args:
            pattern_id: Pattern ID to validate
            frequency_pair: Frequency pair to validate
            modulation: Modulation scheme to validate
            polar_rate: Polar code rate to validate

        Raises:
            ValueError: If any parameter is invalid with descriptive message

        Contract:
            - Must check all parameter ranges per spec
            - Must verify pattern file exists for pattern_id
            - Must validate polar_rate is (k, n) where k < n and n is power of 2
        """
        ...

    def estimate_message_capacity(
        self,
        pattern_length: int,
        polar_rate: Tuple[int, int]
    ) -> int:
        """
        Calculate maximum message size (bits) for pattern length and Polar rate.

        Args:
            pattern_length: Pattern length in symbols (64-2048)
            polar_rate: (k, n) Polar code rate

        Returns:
            Maximum data bits that fit in pattern after Polar encoding

        Contract:
            - capacity = pattern_length × (k / n)
            - Must account for Polar encoding overhead
        """
        ...

    def get_tone_frequencies(self, frequency_pair: int) -> Tuple[float, float]:
        """
        Get tone A and tone B frequencies for a frequency pair.

        Args:
            frequency_pair: Pair ID (0-66)

        Returns:
            Tuple of (tone_a_hz, tone_b_hz)

        Contract:
            - Must use 135-channel grid: 300-3000 Hz, 20 Hz spacing
            - tone_a = 300 + (2 × frequency_pair) × 20
            - tone_b = 300 + (2 × frequency_pair + 1) × 20
            - Both tones must be in [300, 3000] Hz
        """
        ...

    def get_required_pattern_length(
        self,
        message_bits: int,
        polar_rate: Tuple[int, int]
    ) -> int:
        """
        Determine minimum pattern length to fit message with Polar encoding.

        Args:
            message_bits: Number of data bits (before Polar encoding)
            polar_rate: (k, n) Polar code rate

        Returns:
            Minimum pattern length (64, 128, 256, 512, 1024, or 2048)

        Raises:
            ValueError: If message too long for 2048-symbol pattern

        Contract:
            - required_encoded = message_bits / (k / n)
            - Return next pattern length: min(L) where L >= required_encoded
            - L must be in {64, 128, 256, 512, 1024, 2048}
        """
        ...

    def load_pattern(self, pattern_id: int, length: int) -> np.ndarray:
        """
        Load pattern from genetic algorithm output file.

        Args:
            pattern_id: Pattern ID (0-7)
            length: Pattern length (64, 128, 256, 512, 1024, 2048)

        Returns:
            NumPy array of pattern bits (uint8, shape (length,))

        Raises:
            FileNotFoundError: If pattern file doesn't exist
            ValueError: If pattern data is invalid

        Contract:
            - Must load from modules/training/patterns/tournament/
            - Filename: pattern_{pattern_id}_len_{length}.pkl
            - Must validate pattern length matches file
            - Should cache patterns for performance (FR-015)
        """
        ...

    def verify_v2_compliance(self, signal: CleanIQSignal) -> dict:
        """
        Verify generated signal matches CASCADE V2 specification.

        Args:
            signal: Generated signal to validate

        Returns:
            Dictionary with compliance results:
            {
                'symbol_rate': (expected, actual, pass/fail),
                'gmsk_bandwidth': (expected, actual, pass/fail),
                'tone_spacing': (expected, actual, pass/fail),
                'sample_rate': (expected, actual, pass/fail),
                'pattern_orthogonality': (threshold, actual, pass/fail),
                'overall': True/False
            }

        Contract:
            - Symbol rate must be 200 ± 0.1 symbols/second
            - GMSK bandwidth must be < 30 Hz at -40 dB
            - Tone spacing must be 20 Hz ± 0.5 Hz
            - Sample rate must be exactly 48000 Hz
            - Pattern orthogonality must be < -20 dB
        """
        ...


class PatternLoaderInterface(Protocol):
    """Interface for pattern loading and caching."""

    def load_all_patterns(self) -> dict:
        """
        Load all 48 patterns (8 IDs × 6 lengths) into cache.

        Returns:
            Dictionary: {(pattern_id, length): bits_array}

        Contract:
            - Must load all 48 pattern files
            - Must validate each pattern on load
            - Should complete in < 1 second
        """
        ...

    def get_pattern(self, pattern_id: int, length: int) -> np.ndarray:
        """
        Get pattern from cache (load if not cached).

        Args:
            pattern_id: 0-7
            length: 64, 128, 256, 512, 1024, or 2048

        Returns:
            Pattern bits array

        Contract:
            - Must use cache if available
            - Must load from file if not cached
            - O(1) access time when cached
        """
        ...


class GMSKModulatorInterface(Protocol):
    """Interface for GMSK pulse shaping."""

    def generate_gmsk_fsk(
        self,
        pattern_bits: np.ndarray,
        frequency_pair: int,
        sample_rate: int = 48000
    ) -> np.ndarray:
        """
        Generate GMSK-modulated 2-FSK signal from pattern bits.

        Args:
            pattern_bits: Binary pattern (0s and 1s), shape (N,)
            frequency_pair: Which frequency pair to use (0-66)
            sample_rate: Output sample rate (default 48000 Hz)

        Returns:
            Complex IQ signal, shape (N × samples_per_symbol,)

        Contract:
            - Must use BT=0.3 (FR-006)
            - Symbol rate must be 200 symbols/second (FR-012)
            - Samples per symbol = sample_rate / 200 = 240
            - Must produce constant envelope: |I² + Q²| ≈ 1 (±1%)
            - Gaussian filter span should be 4-5 symbols
        """
        ...

    def generate_gaussian_filter(self, BT: float, span_symbols: int) -> np.ndarray:
        """
        Generate Gaussian filter for GMSK.

        Args:
            BT: Bandwidth-time product (0.3 for CASCADE)
            span_symbols: Filter length in symbols (typically 4-5)

        Returns:
            Gaussian filter coefficients

        Contract:
            - std = sqrt(ln(2)) / (2π × BT × T_symbol)
            - Filter must be normalized (sum = 1)
        """
        ...


class ConstellationMapperInterface(Protocol):
    """Interface for IQ constellation mapping."""

    def map_to_constellation(
        self,
        bits: np.ndarray,
        modulation: str
    ) -> np.ndarray:
        """
        Map bit groups to IQ constellation points.

        Args:
            bits: Data bits to modulate, shape (N,)
            modulation: 'BPSK', 'QPSK', '8-PSK', or '16-APSK'

        Returns:
            Complex constellation symbols, shape (N // bits_per_symbol,)

        Raises:
            ValueError: If modulation invalid or bits length incompatible

        Contract:
            - BPSK: 1 bit/symbol, points: [-1, +1]
            - QPSK: 2 bits/symbol, points: (±1±j)/√2, Gray coded
            - 8-PSK: 3 bits/symbol, unit circle, Gray coded
            - 16-APSK: 4 bits/symbol, 4+12 ring configuration
            - All constellations normalized to unit average power
        """
        ...

    def get_bits_per_symbol(self, modulation: str) -> int:
        """
        Get number of bits per symbol for modulation scheme.

        Args:
            modulation: 'BPSK', 'QPSK', '8-PSK', or '16-APSK'

        Returns:
            Bits per symbol (1, 2, 3, or 4)
        """
        ...


class PolarCodecInterface(Protocol):
    """Interface for Polar error correction encoding."""

    def encode(
        self,
        data_bits: np.ndarray,
        code_rate: Tuple[int, int],
        block_length: int
    ) -> np.ndarray:
        """
        Encode data bits with Polar code.

        Args:
            data_bits: Information bits, shape (K,)
            code_rate: (k, n) rate tuple
            block_length: Codeword length N (must be power of 2)

        Returns:
            Polar codeword, shape (block_length,)

        Contract:
            - block_length must be power of 2
            - K = int(block_length × (k / n))
            - Must use systematic encoding (data bits appear in codeword)
            - Must pad data_bits if len(data_bits) < K
        """
        ...

    def get_supported_rates(self) -> list:
        """
        Get list of supported Polar code rates.

        Returns:
            List of (k, n) tuples: [(1,2), (2,3), (3,4), (4,5), (5,6), (7,8)]
        """
        ...


# Type aliases for clarity
FrequencyHz = float
TimeSeconds = float
SampleRate = int
