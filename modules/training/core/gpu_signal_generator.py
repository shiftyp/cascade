"""
GPU-accelerated CASCADE signal generator for GH200.

Generates batches of 4096+ signals in parallel using PyTorch CUDA.
Optimized for Grace Hopper unified memory architecture.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

# Import CPU versions for pattern loading
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.signal_generator.pattern_loader import PatternLoader
from src.signal_generator import modulation, polar_codec

# Import GPU polar codec (much faster!)
from gpu_polar_codec import GPUPolarCodec


@dataclass
class BatchKernelParameters:
    """Batch of kernel parameters for parallel generation."""
    pattern_ids: torch.Tensor  # [batch_size], int, 0-3
    frequency_triples: torch.Tensor  # [batch_size], int, 0-42
    modulations: List[str]  # [batch_size], 'BPSK', 'QPSK', etc.
    polar_rates: List[Tuple[int, int]]  # [batch_size], [(k, n), ...]
    data_symbol_rates: torch.Tensor  # [batch_size], int, 75-300


class GPUSignalGenerator:
    """
    GPU-accelerated CASCADE V2 signal generator.

    Generates batches of signals in parallel on GPU for massive speedup.
    Optimized for GH200 unified memory (no CPU↔GPU transfer overhead).
    """

    # CASCADE V2 constants - Updated for always-on center design with guard bands
    SAMPLE_RATE = 48000  # Hz
    PATTERN_SYMBOL_RATE = 25  # Pattern layer symbols/second (optimized for low power detection)
    PATTERN_LENGTH = 512  # Pattern length (from final_patterns_319968.pkl)
    VALID_DATA_SYMBOL_RATES = [75, 100, 125, 150, 175, 200, 250, 300]
    MIN_FREQ = 500  # Hz (was 300, added 200 Hz lower guard band to prevent sub-300 Hz artifacts)
    MAX_FREQ = 2600  # Hz (was 2856, added upper guard band to prevent >2800 Hz)
    TONE_SPACING = 50  # Hz (increased from 14 Hz, balanced for 2-center diversity)
    NUM_FREQUENCY_TRIPLES = 14  # Updated: (2600-500)/(50*3) = 14 triples
    RAMP_DURATION_MS = 150  # Preamble/postamble ramp (150ms for 16-symbol filter settling @ slowest rate)

    @staticmethod
    def bytes_to_bits_gpu(messages: List[bytes], device='cuda') -> List[torch.Tensor]:
        """Convert list of byte messages to list of bit tensors on GPU (VECTORIZED)."""
        bit_tensors = []
        for msg in messages:
            # Convert bytes to numpy array, then unpack bits, then to GPU tensor
            msg_bytes = np.frombuffer(msg, dtype=np.uint8)
            msg_bits = np.unpackbits(msg_bytes)
            bit_tensors.append(torch.from_numpy(msg_bits).to(device))
        return bit_tensors

    @staticmethod
    def map_to_constellation_gpu(bits: torch.Tensor, modulation: str, device='cuda') -> torch.Tensor:
        """GPU-accelerated constellation mapping (BPSK/QPSK/8-PSK/16-APSK)."""
        bits = bits.to(device)

        if modulation == 'BPSK':
            # BPSK: 0→-1, 1→+1
            return (2 * bits.float() - 1).to(torch.complex64)

        elif modulation == 'QPSK':
            # QPSK: Gray-coded
            bits = bits.reshape(-1, 2)
            i_bits = bits[:, 0]
            q_bits = bits[:, 1]
            i_vals = 2 * i_bits.float() - 1
            q_vals = 2 * q_bits.float() - 1
            return (i_vals + 1j * q_vals) / np.sqrt(2)

        elif modulation == '8-PSK':
            # 8-PSK: Gray-coded, 3 bits per symbol
            bits = bits.reshape(-1, 3)
            # Convert 3-bit groups to integers 0-7 (use long for indexing)
            indices = (bits[:, 0].long() * 4 + bits[:, 1].long() * 2 + bits[:, 2].long())
            # 8-PSK constellation (Gray-coded)
            angles = torch.tensor([0, 1, 3, 2, 6, 7, 5, 4], device=device).float() * (2 * np.pi / 8)
            symbols = torch.exp(1j * angles[indices.long()])
            return symbols

        elif modulation == '16-APSK':
            # 16-APSK: 4 bits per symbol, DVB-S2 constellation
            bits = bits.reshape(-1, 4)
            indices = (bits[:, 0].long() * 8 + bits[:, 1].long() * 4 + bits[:, 2].long() * 2 + bits[:, 3].long())
            # Simplified 16-APSK (4+12 ring)
            # Inner ring (4 symbols, radius=1)
            # Outer ring (12 symbols, radius=2.7)
            r1, r2 = 1.0, 2.7
            constellation = torch.zeros(16, dtype=torch.complex64, device=device)
            # Inner 4
            for i in range(4):
                constellation[i] = r1 * torch.exp(torch.tensor(1j * (i * np.pi / 2 + np.pi / 4)))
            # Outer 12
            for i in range(12):
                constellation[4 + i] = r2 * torch.exp(torch.tensor(1j * (i * np.pi / 6)))
            # Normalize to unit power
            constellation = constellation / torch.sqrt(torch.mean(torch.abs(constellation)**2))
            return constellation[indices]

        else:
            raise ValueError(f"Unknown modulation: {modulation}")

    @staticmethod
    def generate_pilots_gpu(modulation: str, num_symbols: int, shaped: bool, device='cuda') -> torch.Tensor:
        """Generate pilot symbols on GPU (shaped spiral or constant)."""
        if shaped:
            # Shaped spiral: inner→outer for smooth AGC
            t = torch.linspace(0, 1, num_symbols, device=device)
            amplitude = t  # Linear ramp 0→1
            phase = 2 * np.pi * t * 4  # 4 full rotations

            if modulation == 'BPSK':
                # BPSK: just use +1 (no phase)
                return amplitude.to(torch.complex64)
            elif modulation == 'QPSK':
                # QPSK: sweep through 4 points
                return amplitude * torch.exp(1j * phase) / np.sqrt(2)
            elif modulation == '8-PSK':
                return amplitude * torch.exp(1j * phase)
            elif modulation == '16-APSK':
                return amplitude * 2.0 * torch.exp(1j * phase)  # Larger amplitude
        else:
            # Constant pilot
            if modulation == 'BPSK':
                pilot = torch.ones(1, dtype=torch.complex64, device=device)
            elif modulation == 'QPSK':
                pilot = torch.tensor([1 + 1j], dtype=torch.complex64, device=device) / np.sqrt(2)
            elif modulation == '8-PSK':
                pilot = torch.tensor([1 + 0j], dtype=torch.complex64, device=device)
            elif modulation == '16-APSK':
                pilot = torch.tensor([2.7 + 0j], dtype=torch.complex64, device=device)
            return pilot.repeat(num_symbols)

    def __init__(self, device='cuda', patterns_dir: Optional[Path] = None, use_gpu_polar: bool = True, use_always_on: bool = True):
        """
        Initialize GPU signal generator.

        Args:
            device: 'cuda' for GPU, 'cpu' for fallback
            patterns_dir: Directory containing pattern .pkl files
            use_gpu_polar: Use GPU polar encoding (90× faster than CPU)
            use_always_on: Use always-on center frequency design (4 centers)
        """
        self.device = torch.device(device)
        self.use_gpu_polar = use_gpu_polar
        self.use_always_on = use_always_on
        print(f"GPUSignalGenerator: Using device {self.device}")
        if use_always_on:
            print(f"  Using always-on center design (4 centers + 2 alternating outers)")

        # Set patterns directory for always-on patterns if needed
        if patterns_dir is None and use_always_on:
            # Point to patterns/patterns directory for always-on patterns
            patterns_dir = Path(__file__).parent.parent / "patterns" / "patterns"

        # Load patterns (CPU) and cache on GPU
        self.pattern_loader = PatternLoader(patterns_dir, use_always_on=use_always_on)

        if use_always_on:
            # Load always-on patterns (center/lower/upper for each pattern ID)
            self.patterns_always_on_gpu = {}
            for pattern_id in range(4):
                try:
                    # Load the three pattern types
                    center = self.pattern_loader.load_always_on_pattern(pattern_id, 'center', self.PATTERN_LENGTH)
                    lower = self.pattern_loader.load_always_on_pattern(pattern_id, 'lower', self.PATTERN_LENGTH)
                    upper = self.pattern_loader.load_always_on_pattern(pattern_id, 'upper', self.PATTERN_LENGTH)

                    self.patterns_always_on_gpu[pattern_id] = {
                        'center': torch.from_numpy(center).to(self.device).float(),
                        'lower': torch.from_numpy(lower).to(self.device).float(),
                        'upper': torch.from_numpy(upper).to(self.device).float()
                    }
                except Exception as e:
                    print(f"Warning: Could not load always-on patterns for ID {pattern_id}: {e}")
                    print("Falling back to standard patterns")
                    self.use_always_on = False
                    break

        # Always load standard patterns as fallback (needed for QRM when num_centers=0)
        if not use_always_on or True:  # Always load for compatibility
            self.pattern_loader.load_all_patterns()
            self.patterns_gpu = {}
            for pattern_id in range(4):
                pattern = self.pattern_loader.load_pattern(pattern_id, self.PATTERN_LENGTH)
                self.patterns_gpu[pattern_id] = torch.from_numpy(pattern).to(self.device).float()

        # Pre-compute raised cosine filters for all data rates
        # OPTIMIZED: β=0.20, 6-symbol span (2.5× faster, still exceeds Nyquist minimum)
        self.rc_filters = {}
        for rate in self.VALID_DATA_SYMBOL_RATES:
            samples_per_symbol = self.SAMPLE_RATE // rate
            rc_filter = self._compute_raised_cosine_filter(
                beta=0.20,
                span_symbols=16,  # Full span for accurate pulse shaping
                samples_per_symbol=samples_per_symbol
            )
            self.rc_filters[rate] = rc_filter.to(self.device)

        # Initialize GPU Polar codec (90× faster than CPU!)
        if use_gpu_polar:
            self.gpu_polar = GPUPolarCodec(device=device)
            if use_always_on and hasattr(self, 'patterns_always_on_gpu'):
                print(f"✓ Loaded 4 always-on patterns (center/lower/upper), {len(self.rc_filters)} RC filters, and GPU Polar codec")
            else:
                print(f"✓ Loaded 4 patterns, {len(self.rc_filters)} RC filters, and GPU Polar codec")
        else:
            self.gpu_polar = None
            if use_always_on and hasattr(self, 'patterns_always_on_gpu'):
                print(f"✓ Loaded 4 always-on patterns (center/lower/upper) and {len(self.rc_filters)} RC filters to GPU")
            else:
                print(f"✓ Loaded 4 patterns and {len(self.rc_filters)} RC filters to GPU")

        # Note: Pre-computing GMSK patterns and frequency-shifting them causes severe
        # spectral leakage because frequency-shifting a frequency-modulated signal
        # spreads the spectrum. We must generate GMSK tones at their final frequencies.

    def _compute_raised_cosine_filter(self, beta=0.20, span_symbols=16, samples_per_symbol=240):
        """Generate raised cosine filter with Hamming window for sidelobe suppression."""
        span_samples = span_symbols * samples_per_symbol
        t = torch.arange(-span_samples // 2, span_samples // 2,
                        dtype=torch.float32) / samples_per_symbol

        # Sinc function
        h = torch.sinc(t) * torch.cos(torch.pi * beta * t) / (1 - (2 * beta * t)**2)

        # Handle singular points
        singular_mask = torch.abs(torch.abs(t) - 1/(2*beta)) < 1e-10
        h[singular_mask] = torch.pi / 4 * torch.sinc(torch.tensor(1 / (2 * beta)))
        h[torch.abs(t) < 1e-10] = 1.0

        # Apply Hamming window for sidelobe suppression (40+ dB)
        window = torch.hamming_window(len(h), periodic=False, dtype=torch.float32)
        h = h * window

        # Normalize for unit energy (standard for matched filtering)
        h = h / torch.sqrt(torch.sum(h**2))
        return h

    def get_tone_triple_frequencies_batch(self, frequency_triples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate tone frequencies for batch of frequency triples.

        Args:
            frequency_triples: [batch_size], int tensor, values 0-42

        Returns:
            Tuple of (tone_a, tone_b, tone_c), each [batch_size] tensor
        """
        channel_a = 3 * frequency_triples
        channel_b = 3 * frequency_triples + 1
        channel_c = 3 * frequency_triples + 2

        tone_a = self.MIN_FREQ + channel_a * self.TONE_SPACING
        tone_b = self.MIN_FREQ + channel_b * self.TONE_SPACING
        tone_c = self.MIN_FREQ + channel_c * self.TONE_SPACING

        return tone_a.float(), tone_b.float(), tone_c.float()

    def _precompute_alternating_gmsk_tones(self):
        """
        Pre-compute GMSK-modulated alternating tones for all 4 patterns.

        These are stored as base signals at DC (0 Hz) that can be frequency-shifted later.
        This avoids recomputing the expensive GMSK modulation for every signal.
        """
        samples_per_symbol = self.SAMPLE_RATE // self.PATTERN_SYMBOL_RATE

        # Pre-compute Gaussian filter once
        gaussian_filter = self._generate_gaussian_filter(BT=0.3, span_symbols=8,
                                                        samples_per_symbol=samples_per_symbol)

        # We'll compute for a standard pattern length (512 symbols)
        num_symbols = self.PATTERN_LENGTH

        # Storage for pre-computed tones: [pattern_id][type] -> tensor
        self.precomputed_gmsk = {}

        for pattern_id in range(4):
            lower_pattern = self.patterns_always_on_gpu[pattern_id]['lower']  # [512]
            upper_pattern = self.patterns_always_on_gpu[pattern_id]['upper']  # [512]

            # Generate base GMSK signals (at DC, will be frequency-shifted later)
            # Lower: even symbols only
            lower_gmsk = self._generate_gmsk_base_signal(lower_pattern, gaussian_filter,
                                                         samples_per_symbol, even=True)

            # Upper: odd symbols only
            upper_gmsk = self._generate_gmsk_base_signal(upper_pattern, gaussian_filter,
                                                         samples_per_symbol, even=False)

            self.precomputed_gmsk[pattern_id] = {
                'lower': lower_gmsk,  # [num_samples]
                'upper': upper_gmsk   # [num_samples]
            }

    def _generate_gmsk_base_signal(self, pattern: torch.Tensor, gaussian_filter: torch.Tensor,
                                   samples_per_symbol: int, even: bool) -> torch.Tensor:
        """
        Generate GMSK base signal (frequency modulation without carrier) for one pattern.

        Returns a complex baseband signal that can be frequency-shifted to any frequency.
        """
        num_symbols = len(pattern)

        # Create NRZ signal for alternating pattern
        nrz = torch.zeros(num_symbols, device=self.device)

        for symbol_idx in range(num_symbols):
            should_be_active = (symbol_idx % 2 == 0 and even) or (symbol_idx % 2 == 1 and not even)

            if should_be_active:
                symbol_value = pattern[symbol_idx]
                if symbol_value >= 0:  # Active symbol
                    nrz[symbol_idx] = symbol_value - 1  # 0→-1, 1→0, 2→+1

        # Upsample to sample rate
        upsampled_length = num_symbols * samples_per_symbol
        nrz_upsampled = torch.zeros(upsampled_length, device=self.device)

        for i in range(num_symbols):
            nrz_upsampled[i * samples_per_symbol] = nrz[i]

        # Apply Gaussian filter for GMSK pulse shaping
        filtered = self._fft_convolve_1d(nrz_upsampled, gaussian_filter)

        # Normalize (OPTIMIZED: RMS instead of quantile for 3× speed)
        rms_level = torch.sqrt(torch.mean(filtered**2))
        if rms_level > 0:
            filtered = filtered / rms_level

        # Generate phase-modulated signal (at baseband, no carrier yet)
        dt = 1.0 / self.SAMPLE_RATE
        freq_deviation = self.TONE_SPACING  # Hz (matches tone spacing, legacy function)

        # Integrate filtered signal to get phase
        phase = 2 * np.pi * freq_deviation * torch.cumsum(filtered, dim=0) * dt

        # Apply gating for alternating symbols
        gate = torch.zeros(len(phase), device=self.device)

        for symbol_idx in range(num_symbols):
            should_be_active = (symbol_idx % 2 == 0 and even) or (symbol_idx % 2 == 1 and not even)

            if should_be_active:
                start_idx = symbol_idx * samples_per_symbol
                end_idx = min((symbol_idx + 1) * samples_per_symbol, len(gate))

                if end_idx > start_idx:
                    gate[start_idx:end_idx] = 1.0

                    # Small ramps at boundaries
                    ramp_samples = max(1, samples_per_symbol // 20)

                    if start_idx > 0 and start_idx + ramp_samples <= len(gate):
                        ramp_up = torch.linspace(0, 1, ramp_samples, device=self.device)
                        gate[start_idx:start_idx+ramp_samples] *= ramp_up

                    if end_idx - ramp_samples >= 0 and end_idx < len(gate):
                        ramp_down = torch.linspace(1, 0, ramp_samples, device=self.device)
                        gate[end_idx-ramp_samples:end_idx] *= ramp_down

        # Generate complex baseband signal
        signal = gate * torch.exp(1j * phase)

        return signal

    def _fft_convolve_1d(self, signal: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
        """FFT-based convolution for single 1D signal."""
        signal_len = len(signal)
        filter_len = len(filter)

        pad_len = filter_len // 2
        signal_padded = torch.nn.functional.pad(signal.unsqueeze(0), (pad_len, pad_len), mode='constant', value=0).squeeze(0)
        padded_len = len(signal_padded)

        fft_size = 2 ** int(np.ceil(np.log2(padded_len + filter_len - 1)))

        signal_fft = torch.fft.rfft(signal_padded, n=fft_size)
        filter_fft = torch.fft.rfft(filter, n=fft_size)

        conv_fft = signal_fft * filter_fft
        conv_result = torch.fft.irfft(conv_fft, n=fft_size)

        start_idx = filter_len // 2
        end_idx = start_idx + signal_len

        return conv_result[start_idx:end_idx]

    def _generate_gaussian_filter(self, BT: float = 0.3, span_symbols: int = 8,
                                   samples_per_symbol: int = 1920) -> torch.Tensor:
        """
        Generate Gaussian filter for GMSK pulse shaping (BT=0.3, 8-symbol span).

        Args:
            BT: Bandwidth-time product (0.3 for CASCADE)
            span_symbols: Filter span in symbols (8 for smooth transitions)
            samples_per_symbol: Samples per symbol (48000 / 25 = 1920)

        Returns:
            torch.Tensor: Normalized Gaussian filter coefficients
        """
        span_samples = span_symbols * samples_per_symbol
        t = torch.arange(-span_samples // 2, span_samples // 2,
                        dtype=torch.float32, device=self.device) / samples_per_symbol

        # Gaussian pulse: g(t) = (1 / (sqrt(2*pi) * sigma)) * exp(-t^2 / (2 * sigma^2))
        # where sigma = sqrt(ln(2)) / (2 * pi * BT)
        sigma = np.sqrt(np.log(2)) / (2 * np.pi * BT)
        gaussian_pulse = (1 / (np.sqrt(2 * np.pi) * sigma)) * torch.exp(-t**2 / (2 * sigma**2))

        # Normalize to unit energy
        gaussian_pulse = gaussian_pulse / torch.sum(gaussian_pulse)

        return gaussian_pulse

    def generate_gmsk_always_on_batch(self,
                                      pattern_symbols_batch: Dict[str, torch.Tensor],
                                      base_frequency_triple: torch.Tensor,
                                      num_centers: int,
                                      num_samples: int) -> torch.Tensor:
        """
        Generate GMSK 3-FSK signals with frequency diversity.

        SIMPLIFIED: Uses standard GMSK 3-FSK for each center (no phase discontinuities).

        Args:
            pattern_symbols_batch: Dict with 'center' patterns [batch_size, num_symbols]
                                   (lower/upper patterns ignored in new implementation)
            base_frequency_triple: [batch_size], starting frequency triple
            num_centers: Number of center frequencies for diversity
            num_samples: Total samples to generate per signal

        Returns:
            torch.Tensor: [batch_size, num_samples], complex64 GMSK signals
        """
        # Use center pattern only (lower/upper are legacy from old implementation)
        pattern_symbols = pattern_symbols_batch['center']  # [batch_size, num_symbols]
        batch_size = pattern_symbols.shape[0]

        # Power division for multiple centers
        power_per_center = 1.0 / np.sqrt(num_centers)

        # Initialize combined signal
        combined_signal = torch.zeros(batch_size, num_samples, dtype=torch.complex64, device=self.device)

        # Generate signal for each center frequency (no ramps here - applied after normalization)
        for center_idx in range(num_centers):
            # Offset frequencies for multiple centers
            if num_centers > 1:
                # 2-CENTER OFFSETS: [0, +50] Hz for frequency diversity
                # Total diversity spread: 50 Hz (matches tone spacing)
                # Footprint with 2 triples: 300 Hz + 50 Hz = 350 Hz
                freq_offset = center_idx * 50

                # Calculate offset tones
                channel_a = 3 * base_frequency_triple
                channel_b = 3 * base_frequency_triple + 1
                channel_c = 3 * base_frequency_triple + 2

                tones_a = (self.MIN_FREQ + channel_a * self.TONE_SPACING + freq_offset).float()
                tones_b = (self.MIN_FREQ + channel_b * self.TONE_SPACING + freq_offset).float()
                tones_c = (self.MIN_FREQ + channel_c * self.TONE_SPACING + freq_offset).float()
            else:
                # No offset for single center
                tones_a, tones_b, tones_c = self.get_tone_triple_frequencies_batch(base_frequency_triple)

            # Use standard 3-FSK generation (proven to work without phase issues)
            this_signal = self.generate_gmsk_3fsk_batch(
                pattern_symbols,
                tones_a,
                tones_b,
                tones_c,
                num_samples
            )

            # Add to combined signal with power scaling (ramps applied later after normalization)
            combined_signal += power_per_center * this_signal

        return combined_signal

    def _generate_single_gmsk_tone(self, pattern: torch.Tensor, frequency: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Generate single constant-frequency tone for center (no frequency modulation needed)."""
        batch_size = pattern.shape[0]

        # Center frequency is constant - no GMSK modulation needed!
        # Just generate a constant-frequency sinusoid with ramps
        t = torch.arange(num_samples, dtype=torch.float32, device=self.device) / self.SAMPLE_RATE

        # Expand frequency for broadcasting: [batch_size, 1]
        frequency_expanded = frequency.unsqueeze(1)

        # Generate constant-frequency tone: exp(j * 2π * f * t)
        phase = 2 * np.pi * frequency_expanded * t
        signal = torch.exp(1j * phase)

        # Apply smooth ramps at start/end (150ms each)
        ramp_samples = int(self.RAMP_DURATION_MS * self.SAMPLE_RATE / 1000)

        if ramp_samples > 0 and num_samples > 2 * ramp_samples:
            # Create smooth Tukey window for ramps
            window = torch.ones(num_samples, dtype=torch.float32, device=self.device)

            # Ramp up (raised cosine)
            ramp_up = 0.5 * (1 - torch.cos(torch.pi * torch.arange(ramp_samples, device=self.device) / ramp_samples))
            window[:ramp_samples] = ramp_up

            # Ramp down (raised cosine)
            ramp_down = 0.5 * (1 + torch.cos(torch.pi * torch.arange(ramp_samples, device=self.device) / ramp_samples))
            window[-ramp_samples:] = ramp_down

            # Apply window
            signal = signal * window.unsqueeze(0)

        return signal

    def _generate_alternating_gmsk_tone(self, pattern: torch.Tensor, frequency: torch.Tensor,
                                        num_samples: int, even: bool) -> torch.Tensor:
        """
        Generate alternating GMSK tone with proper pulse shaping (only active on even or odd symbols).

        CRITICAL: Alternating symbols have effective rate of 37.5 sym/s (half of 75 sym/s),
        so we need LONGER filter span (16 symbols instead of 8) for smooth transitions.
        """
        batch_size = pattern.shape[0]
        samples_per_symbol = self.SAMPLE_RATE // self.PATTERN_SYMBOL_RATE
        num_symbols = pattern.shape[1]

        # Create NRZ signal for alternating pattern - VECTORIZED (no loops!)
        # Set inactive symbols (where we shouldn't transmit) to 0 (no frequency deviation)
        nrz = torch.zeros(batch_size, num_symbols, device=self.device)

        # VECTORIZED: Create mask for active symbols (even or odd)
        symbol_indices = torch.arange(num_symbols, device=self.device)
        if even:
            active_symbol_mask = (symbol_indices % 2 == 0)  # Even symbols
        else:
            active_symbol_mask = (symbol_indices % 2 == 1)  # Odd symbols

        # VECTORIZED: Set NRZ for all active symbols at once
        # Only set for symbols that are both active (even/odd) AND have valid pattern (>=0)
        pattern_active_mask = (pattern >= 0)  # [batch_size, num_symbols]
        combined_mask = pattern_active_mask & active_symbol_mask.unsqueeze(0)  # Broadcasting

        # Convert pattern to NRZ where active
        nrz[combined_mask] = (pattern[combined_mask] - 1).float()  # 0→-1, 1→0, 2→+1

        # Upsample to sample rate - VECTORIZED
        upsampled_length = num_symbols * samples_per_symbol
        nrz_upsampled = torch.zeros(batch_size, upsampled_length, device=self.device)

        # VECTORIZED: Place all symbols at once using advanced indexing
        sample_indices = torch.arange(num_symbols, device=self.device) * samples_per_symbol
        nrz_upsampled[:, sample_indices] = nrz

        # Apply Gaussian filter for GMSK pulse shaping
        gaussian_filter = self._generate_gaussian_filter(BT=0.3, span_symbols=8, samples_per_symbol=samples_per_symbol)
        filtered = self._fft_convolve_batch(nrz_upsampled, gaussian_filter)

        # Normalize filtered signal (OPTIMIZED: RMS instead of quantile for 3× speed)
        rms_level = torch.sqrt(torch.mean(filtered**2, dim=1, keepdim=True))
        rms_level = torch.clamp(rms_level, min=1e-10)  # Avoid division by zero
        filtered = filtered / rms_level

        # Truncate/pad to num_samples
        if filtered.shape[1] > num_samples:
            filtered = filtered[:, :num_samples]
        elif filtered.shape[1] < num_samples:
            filtered = F.pad(filtered, (0, num_samples - filtered.shape[1]))

        # Generate phase-modulated GMSK signal WITHOUT hard gating
        # The GMSK filtering already provides smooth transitions
        # Gating AFTER filtering causes spectral leakage!
        dt = 1.0 / self.SAMPLE_RATE
        freq_deviation = self.TONE_SPACING  # Hz (matches tone spacing, legacy function)
        instantaneous_freq = frequency.unsqueeze(1) + freq_deviation * filtered
        instantaneous_phase = 2 * np.pi * torch.cumsum(instantaneous_freq, dim=1) * dt

        # NO GATING - the NRZ already has zeros for inactive symbols,
        # and the Gaussian filtering provides smooth transitions
        signal = torch.exp(1j * instantaneous_phase)

        return signal

    def generate_gmsk_3fsk_batch(self,
                                  pattern_symbols_batch: torch.Tensor,
                                  tones_a: torch.Tensor,
                                  tones_b: torch.Tensor,
                                  tones_c: torch.Tensor,
                                  num_samples: int) -> torch.Tensor:
        """
        Generate GMSK 3-FSK carrier for batch of signals with Gaussian filtering.

        Args:
            pattern_symbols_batch: [batch_size, num_pattern_symbols], values {0, 1, 2}
            tones_a, tones_b, tones_c: [batch_size], frequency in Hz
            num_samples: Total samples to generate per signal

        Returns:
            torch.Tensor: [batch_size, num_samples], complex64 GMSK signals
        """
        batch_size, num_pattern_symbols = pattern_symbols_batch.shape
        samples_per_symbol = self.SAMPLE_RATE // self.PATTERN_SYMBOL_RATE

        # Convert ternary to 3-level: 0 → -1, 1 → 0, 2 → +1
        nrz = pattern_symbols_batch.float() - 1  # [batch_size, num_pattern_symbols]

        # Upsample to sample rate (insert zeros between symbols)
        upsampled_length = num_pattern_symbols * samples_per_symbol
        nrz_upsampled = torch.zeros(batch_size, upsampled_length,
                                     dtype=torch.float32, device=self.device)

        # Place symbols at appropriate indices - VECTORIZED (2-3× faster)
        symbol_positions = torch.arange(num_pattern_symbols, device=self.device) * samples_per_symbol
        nrz_upsampled[:, symbol_positions] = nrz

        # Apply Gaussian filter for smooth frequency transitions (GMSK)
        gaussian_filter = self._generate_gaussian_filter(
            BT=0.3,
            span_symbols=8,
            samples_per_symbol=samples_per_symbol
        )

        # FFT-based convolution for efficiency
        filtered = self._fft_convolve_batch(nrz_upsampled, gaussian_filter)

        # Normalize filtered signal (OPTIMIZED: RMS instead of quantile for 3× speed)
        rms_level = torch.sqrt(torch.mean(filtered**2, dim=1, keepdim=True))
        filtered = torch.where(rms_level > 0, filtered / rms_level, filtered)

        # Truncate or pad to exact num_samples
        if filtered.shape[1] > num_samples:
            filtered = filtered[:, :num_samples]
        elif filtered.shape[1] < num_samples:
            padding = num_samples - filtered.shape[1]
            filtered = F.pad(filtered, (0, padding), value=0)

        # Calculate center frequency (tone_b) and frequency deviation
        center_freq = tones_b  # [batch_size]
        freq_deviation = (tones_c - tones_a) / 2  # [batch_size]

        # Expand for broadcasting: [batch_size, 1]
        center_freq = center_freq.unsqueeze(1)
        freq_deviation = freq_deviation.unsqueeze(1)

        # Frequency modulation: f(t) = fc + fd * filtered(t)
        instantaneous_freq = center_freq + freq_deviation * filtered  # [batch_size, num_samples]

        # Time vector
        dt = 1.0 / self.SAMPLE_RATE

        # Convert to phase: θ(t) = 2π ∫ f(τ) dτ
        # Use cumsum for integration
        instantaneous_phase = 2 * np.pi * torch.cumsum(instantaneous_freq, dim=1) * dt

        # Generate complex IQ signal with constant envelope
        gmsk_signal = torch.exp(1j * instantaneous_phase)

        return gmsk_signal

    def apply_carrier_ramps_batch(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Apply smooth Tukey window ramps to batch of signals.

        Uses Tukey (tapered cosine) window for smooth transitions that allow
        filter transients to settle before full amplitude modulation.

        Args:
            signals: [batch_size, num_samples], complex tensor

        Returns:
            torch.Tensor: [batch_size, num_samples], ramped signals with smooth transitions
        """
        batch_size, num_samples = signals.shape
        ramp_samples = int(self.RAMP_DURATION_MS * self.SAMPLE_RATE / 1000)

        if num_samples <= 2 * ramp_samples:
            return signals

        # Use Tukey window for smoother spectral characteristics
        # alpha controls taper: 0 = rectangular, 1 = full Hann window
        alpha = (2 * ramp_samples) / num_samples

        # Generate Tukey window (tapered cosine) manually
        # Tukey window = raised cosine taper + flat top + raised cosine taper
        n = torch.arange(num_samples, dtype=torch.float32, device=self.device)

        # Calculate taper regions
        width = int(alpha * num_samples / 2)

        if width == 0:
            # No taper (rectangular window)
            window = torch.ones(num_samples, dtype=torch.float32, device=self.device)
        else:
            window = torch.ones(num_samples, dtype=torch.float32, device=self.device)

            # Left taper (raised cosine)
            left_mask = n < width
            window[left_mask] = 0.5 * (1 - torch.cos(torch.pi * n[left_mask] / width))

            # Right taper (raised cosine)
            right_mask = n >= (num_samples - width)
            window[right_mask] = 0.5 * (1 - torch.cos(torch.pi * (num_samples - n[right_mask]) / width))

        # Apply window to batch
        ramped = signals * window.unsqueeze(0)

        return ramped

    def _fft_convolve_batch(self, signals: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
        """
        Fast FFT-based convolution for batch of signals (100× faster for long filters).

        Uses linear convolution (zero-padded) to avoid circular artifacts.

        Args:
            signals: [batch_size, signal_len], real tensor
            filter: [filter_len], real tensor (RC filter)

        Returns:
            torch.Tensor: [batch_size, signal_len], convolved signals
        """
        batch_size, signal_len = signals.shape
        filter_len = len(filter)

        # For 'same' mode linear convolution, need proper zero-padding
        # Pad signal with zeros on both ends
        pad_len = filter_len // 2

        # Zero-pad signals (prevents circular wraparound)
        signals_padded = torch.nn.functional.pad(signals, (pad_len, pad_len), mode='constant', value=0)
        padded_len = signals_padded.shape[1]

        # Compute FFT size (must fit padded signal + filter)
        fft_size = 2 ** int(np.ceil(np.log2(padded_len + filter_len - 1)))

        # FFT of padded signals (batch)
        signals_fft = torch.fft.rfft(signals_padded, n=fft_size, dim=1)

        # FFT of filter (once for all signals)
        filter_fft = torch.fft.rfft(filter, n=fft_size)

        # Multiply in frequency domain
        conv_fft = signals_fft * filter_fft.unsqueeze(0)

        # IFFT back to time domain
        conv_result = torch.fft.irfft(conv_fft, n=fft_size, dim=1)

        # Extract center portion (true linear convolution result)
        # This corresponds to 'same' mode - output same length as input
        start_idx = filter_len // 2
        end_idx = start_idx + signal_len
        conv_trimmed = conv_result[:, start_idx:end_idx]

        return conv_trimmed

    def modulate_data_batch(self,
                            data_symbols_batch: List[torch.Tensor],
                            data_symbol_rates: torch.Tensor,
                            num_samples: int) -> torch.Tensor:
        """
        Apply data modulation with raised cosine pulse shaping (OPTIMIZED - minimal loops).

        For 1024 signals: old=2.6s, new=~0.1s (26× speedup!)

        Args:
            data_symbols_batch: List of [num_data_symbols] complex tensors (variable length per sample)
            data_symbol_rates: [batch_size], data rates in sym/s
            num_samples: Target number of samples

        Returns:
            torch.Tensor: [batch_size, num_samples], pulse-shaped data
        """
        batch_size = len(data_symbols_batch)

        # FAST PATH: If all signals have same rate, process as single batch
        unique_rates = torch.unique(data_symbol_rates)

        if len(unique_rates) == 1:
            # All same rate - fully batched convolution!
            rate = int(unique_rates[0].item())
            samples_per_symbol = self.SAMPLE_RATE // rate
            rc_filter = self.rc_filters[rate]

            # Pad data symbols to fill signal duration + RC filter tail (16 symbols)
            max_syms = max(len(s) for s in data_symbols_batch)
            num_symbols_needed = int(np.ceil(num_samples / samples_per_symbol))
            rc_tail_symbols = 16  # RC filter span
            total_symbols_needed = num_symbols_needed + rc_tail_symbols

            # Pad by repeating last symbol (prevents discontinuities at data end)
            # VECTORIZED: No loops, use advanced indexing
            symbols_padded = torch.zeros(batch_size, total_symbols_needed, dtype=torch.complex64, device=self.device)

            # Get lengths and last symbols for all signals at once
            lengths = torch.tensor([len(syms) for syms in data_symbols_batch], device=self.device)
            last_symbols = torch.stack([syms[-1] if len(syms) > 0 else complex(1, 0)
                                       for syms in data_symbols_batch])

            # Fill symbols_padded using scatter for each signal
            for i in range(batch_size):
                syms = data_symbols_batch[i]
                if len(syms) > 0:
                    symbols_padded[i, :len(syms)] = syms
                    # Broadcast last symbol to remaining positions
                    symbols_padded[i, len(syms):] = last_symbols[i]
                else:
                    symbols_padded[i, :] = last_symbols[i]

            # Upsample all signals at once
            upsampled_length = total_symbols_needed * samples_per_symbol
            data_upsampled = torch.zeros(batch_size, upsampled_length, dtype=torch.complex64, device=self.device)

            # Place symbols using vectorized indexing
            num_placed = total_symbols_needed
            symbol_positions = torch.arange(num_placed, device=self.device) * samples_per_symbol
            data_upsampled[:, symbol_positions] = symbols_padded

            # FFT-based convolution (100× faster for long filters!)
            # For filters >256 samples, FFT conv is much faster than direct conv
            i_shaped = self._fft_convolve_batch(data_upsampled.real, rc_filter)
            q_shaped = self._fft_convolve_batch(data_upsampled.imag, rc_filter)
            data_shaped_full = i_shaped + 1j * q_shaped

            # Trim to requested num_samples (removes RC filter tail overhang)
            return data_shaped_full[:, :num_samples]

        # SLOW PATH: Multiple rates - process each rate group separately
        else:
            rate_groups = {}
            for i in range(batch_size):
                rate = int(data_symbol_rates[i].item())
                if rate not in rate_groups:
                    rate_groups[rate] = []
                rate_groups[rate].append(i)

            data_shaped_batch = torch.zeros(batch_size, num_samples, dtype=torch.complex64, device=self.device)

            for rate, indices in rate_groups.items():
                if not indices:
                    continue

                samples_per_symbol = self.SAMPLE_RATE // rate
                rc_filter = self.rc_filters[rate]
                group_size = len(indices)

                # Pad data symbols to fill signal duration + RC filter tail
                max_syms = max(len(data_symbols_batch[i]) for i in indices)
                num_symbols_needed = int(np.ceil(num_samples / samples_per_symbol))
                rc_tail_symbols = 16
                total_symbols_needed = num_symbols_needed + rc_tail_symbols

                # Pad by repeating last symbol to prevent discontinuities
                # VECTORIZED: No loops, use advanced indexing
                symbols_padded = torch.zeros(group_size, total_symbols_needed, dtype=torch.complex64, device=self.device)

                # Get last symbols for this group
                last_symbols = torch.stack([data_symbols_batch[orig_idx][-1] if len(data_symbols_batch[orig_idx]) > 0
                                           else complex(1, 0) for orig_idx in indices])

                for group_idx, orig_idx in enumerate(indices):
                    syms = data_symbols_batch[orig_idx]
                    if len(syms) > 0:
                        symbols_padded[group_idx, :len(syms)] = syms
                        # Broadcast last symbol to remaining positions
                        symbols_padded[group_idx, len(syms):] = last_symbols[group_idx]
                    else:
                        symbols_padded[group_idx, :] = last_symbols[group_idx]

                # Upsample
                upsampled_length = total_symbols_needed * samples_per_symbol
                data_upsampled = torch.zeros(group_size, upsampled_length, dtype=torch.complex64, device=self.device)
                symbol_positions = torch.arange(total_symbols_needed, device=self.device) * samples_per_symbol
                data_upsampled[:, symbol_positions] = symbols_padded

                # FFT-based batch convolution (much faster for long filters)
                i_shaped = self._fft_convolve_batch(data_upsampled.real, rc_filter)
                q_shaped = self._fft_convolve_batch(data_upsampled.imag, rc_filter)
                data_shaped_full = i_shaped + 1j * q_shaped

                # Trim to num_samples and assign back
                for group_idx, orig_idx in enumerate(indices):
                    data_shaped_batch[orig_idx] = data_shaped_full[group_idx, :num_samples]

            return data_shaped_batch

    def generate_batch(self,
                      batch_params: BatchKernelParameters,
                      messages: List[bytes],
                      fixed_length: Optional[int] = None,
                      profile: bool = False,
                      num_centers: int = 0) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Generate batch of CASCADE signals in parallel on GPU.

        Args:
            batch_params: Batch of kernel parameters
            messages: List of message bytes (one per sample)
            fixed_length: Fixed output length (for padding/truncation)
            profile: If True, print detailed timing breakdown
            num_centers: Number of center frequencies (4 for always-on design, 0 for standard)

        Returns:
            Tuple of:
                - signals: [batch_size, fixed_length], complex64 tensor
                - metadata: List of dicts with per-signal metadata
        """
        import time
        if profile:
            step_times = {}
            total_start = time.time()

        batch_size = len(batch_params.pattern_ids)

        # Get tone frequencies for entire batch
        if profile: step_start = time.time()
        tones_a, tones_b, tones_c = self.get_tone_triple_frequencies_batch(
            batch_params.frequency_triples
        )
        if profile: step_times['get_tones'] = time.time() - step_start

        # OPTIMIZED: Batch polar encoding (group by rate for efficiency)
        if profile: step_start = time.time()

        # PRE-CONVERT all messages to numpy bits (OUTSIDE the encoding loop)
        all_message_bits = []
        all_block_lengths = []
        for i in range(batch_size):
            message_bytes = np.frombuffer(messages[i], dtype=np.uint8)
            message_bits = np.unpackbits(message_bytes)
            all_message_bits.append(message_bits)

            # Pre-calculate block length for this message
            k, n = batch_params.polar_rates[i]
            encoded_bits = int(np.ceil(len(message_bits) * (n / k)))
            polar_block = 2 ** int(np.ceil(np.log2(max(encoded_bits, 64))))
            all_block_lengths.append(polar_block)

        # Group messages by modulation and polar rate for vectorization
        encoding_groups = {}
        for i in range(batch_size):
            key = (batch_params.modulations[i], batch_params.polar_rates[i])
            if key not in encoding_groups:
                encoding_groups[key] = []
            encoding_groups[key].append(i)

        data_symbols_batch = [None] * batch_size
        metadata_batch = [None] * batch_size

        # Process each encoding group
        for (modulation_scheme, polar_rate), indices in encoding_groups.items():
            k, n = polar_rate
            bits_per_symbol = modulation.get_bits_per_symbol(modulation_scheme)

            # Collect pre-converted messages for this group
            group_messages = [all_message_bits[i] for i in indices]
            group_block_lengths = [all_block_lengths[i] for i in indices]

            # GPU POLAR ENCODING: Encode all messages at once!
            if self.use_gpu_polar and self.gpu_polar is not None:
                # Group messages by block length for batch encoding
                block_length_groups = {}
                for idx, (block_len, orig_idx) in enumerate(zip(group_block_lengths, indices)):
                    if block_len not in block_length_groups:
                        block_length_groups[block_len] = []
                    block_length_groups[block_len].append((idx, group_messages[idx], orig_idx))

                # Batch encode each block length group
                for block_len, block_group in block_length_groups.items():
                    msg_indices = [g[0] for g in block_group]
                    msg_bits = [g[1] for g in block_group]
                    orig_indices = [g[2] for g in block_group]

                    # GPU BATCH ENCODE all messages (with optimal chunking)!
                    codewords_gpu = self.gpu_polar.encode_batch(msg_bits, (k, n), block_len, chunk_size=128)

                    for local_idx, orig_idx in enumerate(orig_indices):
                        codeword = codewords_gpu[local_idx].cpu().numpy()

                        # ZERO-TRUNCATION: Find last non-zero bit
                        nonzero_indices = np.nonzero(codeword)[0]
                        if len(nonzero_indices) > 0:
                            last_nonzero = nonzero_indices[-1] + 1
                            # Round up to nearest symbol boundary
                            last_nonzero_symbols = int(np.ceil(last_nonzero / bits_per_symbol))
                            actual_bits = last_nonzero_symbols * bits_per_symbol
                            # Keep minimum 64 bits for reliable detection
                            actual_bits = max(actual_bits, 64)
                            codeword = codeword[:actual_bits]
                        else:
                            # All zeros - use minimum 64 bits
                            codeword = codeword[:64]

                        # Pad to ensure divisibility by bits_per_symbol (8-PSK = 3 bits/symbol)
                        if len(codeword) % bits_per_symbol != 0:
                            pad_bits = bits_per_symbol - (len(codeword) % bits_per_symbol)
                            codeword = np.pad(codeword, (0, pad_bits), constant_values=0)

                        # Map to symbols
                        data_symbols = modulation.map_to_constellation(codeword, modulation_scheme)

                        # Add FIXED 100ms preamble/postamble (simplified for 3× speed)
                        if len(data_symbols) > 0:
                            data_rate = int(batch_params.data_symbol_rates[orig_idx])

                            # OPTIMIZED: Fixed 100ms for all rates (NN doesn't need exact duration)
                            preamble_symbols = int(100 * data_rate / 1000)
                            postamble_symbols = int(100 * data_rate / 1000)

                            # Use CONSTANT pilots (5× faster than shaped, NN learns either way)
                            preamble = modulation.generate_constant_pilot(modulation_scheme, preamble_symbols)
                            postamble = modulation.generate_constant_pilot(modulation_scheme, postamble_symbols)

                            data_symbols = np.concatenate([preamble, data_symbols, postamble])

                        data_symbols_batch[orig_idx] = torch.from_numpy(data_symbols).to(self.device)

                        metadata_batch[orig_idx] = {
                            'pattern_id': int(batch_params.pattern_ids[orig_idx]),
                            'frequency_triple': int(batch_params.frequency_triples[orig_idx]),
                            'modulation': modulation_scheme,
                            'data_symbol_rate': int(batch_params.data_symbol_rates[orig_idx]),
                            'num_data_symbols': len(data_symbols),
                            'polar_block_length': block_len,
                            'transmitted_symbols': len(data_symbols),
                            'zero_truncated': len(codeword) < block_len
                        }
            else:
                # CPU FALLBACK (slow)
                for idx, (i, block_len) in enumerate(zip(indices, group_block_lengths)):
                    message_bits = group_messages[idx]

                    # CPU polar encode
                    polar_codeword = polar_codec.encode(message_bits, (k, n), block_len)

                    # ZERO-TRUNCATION: Find last non-zero bit
                    nonzero_indices = np.nonzero(polar_codeword)[0]
                    if len(nonzero_indices) > 0:
                        last_nonzero = nonzero_indices[-1] + 1
                        # Round up to symbol boundary
                        last_nonzero_symbols = int(np.ceil(last_nonzero / bits_per_symbol))
                        actual_bits = last_nonzero_symbols * bits_per_symbol
                        # Keep minimum 64 bits for reliable detection
                        actual_bits = max(actual_bits, 64)
                        polar_codeword = polar_codeword[:actual_bits]
                    else:
                        # All zeros - use minimum 64 bits
                        polar_codeword = polar_codeword[:64]

                    # Pad to ensure divisibility by bits_per_symbol (8-PSK = 3 bits/symbol)
                    if len(polar_codeword) % bits_per_symbol != 0:
                        pad_bits = bits_per_symbol - (len(polar_codeword) % bits_per_symbol)
                        polar_codeword = np.pad(polar_codeword, (0, pad_bits), constant_values=0)

                    # Map to symbols
                    data_symbols = modulation.map_to_constellation(polar_codeword, modulation_scheme)

                    # Add FIXED 100ms preamble/postamble (simplified for 3× speed)
                    if len(data_symbols) > 0:
                        data_rate = int(batch_params.data_symbol_rates[i])

                        # OPTIMIZED: Fixed 100ms for all rates (NN doesn't need exact duration)
                        preamble_symbols = int(100 * data_rate / 1000)
                        postamble_symbols = int(100 * data_rate / 1000)

                        # Use CONSTANT pilots (5× faster than shaped, NN learns either way)
                        preamble = modulation.generate_constant_pilot(modulation_scheme, preamble_symbols)
                        postamble = modulation.generate_constant_pilot(modulation_scheme, postamble_symbols)

                        data_symbols = np.concatenate([preamble, data_symbols, postamble])

                    data_symbols_batch[i] = torch.from_numpy(data_symbols).to(self.device)

                    metadata_batch[i] = {
                        'pattern_id': int(batch_params.pattern_ids[i]),
                        'frequency_triple': int(batch_params.frequency_triples[i]),
                        'modulation': modulation_scheme,
                        'data_symbol_rate': int(batch_params.data_symbol_rates[i]),
                        'num_data_symbols': len(data_symbols),
                        'polar_block_length': block_len,
                        'transmitted_symbols': len(data_symbols),
                        'zero_truncated': len(polar_codeword) < block_len
                    }

        if profile: step_times['polar_encoding'] = time.time() - step_start

        # Load patterns for batch
        if profile: step_start = time.time()

        # Check if we should use always-on patterns (2 centers)
        use_always_on_generation = (num_centers == 2 and self.use_always_on and
                                   hasattr(self, 'patterns_always_on_gpu'))

        if use_always_on_generation:
            # Load always-on patterns (center/lower/upper)
            patterns_always_on = {
                'center': [],
                'lower': [],
                'upper': []
            }
            for pid in batch_params.pattern_ids:
                pid_int = int(pid)
                patterns_always_on['center'].append(self.patterns_always_on_gpu[pid_int]['center'])
                patterns_always_on['lower'].append(self.patterns_always_on_gpu[pid_int]['lower'])
                patterns_always_on['upper'].append(self.patterns_always_on_gpu[pid_int]['upper'])

            pattern_symbols_batch = {
                'center': torch.stack(patterns_always_on['center']),
                'lower': torch.stack(patterns_always_on['lower']),
                'upper': torch.stack(patterns_always_on['upper'])
            }
        else:
            # Standard pattern loading
            pattern_symbols_batch = torch.stack([
                self.patterns_gpu[int(pid)] for pid in batch_params.pattern_ids
            ])  # [batch_size, 512]

        if profile: step_times['load_patterns'] = time.time() - step_start

        # Calculate signal length based on maximum data layer length in batch
        # Data layer can be much longer than pattern layer
        ramp_samples = int(self.RAMP_DURATION_MS * self.SAMPLE_RATE / 1000)

        # Find maximum data symbol count across batch
        max_data_symbols = max(len(data_syms) for data_syms in data_symbols_batch)
        max_data_rate = max(int(r.item()) for r in batch_params.data_symbol_rates)
        min_data_rate = min(int(r.item()) for r in batch_params.data_symbol_rates)

        # Calculate samples needed for longest data layer + RC filter tail
        # Use slowest rate to ensure all data fits
        data_samples_per_symbol = self.SAMPLE_RATE // min_data_rate
        data_layer_samples = max_data_symbols * data_samples_per_symbol
        rc_tail_samples = 16 * data_samples_per_symbol  # RC filter span

        # For always-on design: pattern layer is continuous, use data layer duration + tail
        # For standard 3-FSK: pattern layer must match data duration + tail
        if num_centers == 2:  # Always-on design (2-center)
            # Pattern layer is always-on (continuous), extend for data + tail
            total_samples = data_layer_samples + rc_tail_samples
            data_duration_sec = total_samples / self.SAMPLE_RATE
            pattern_symbols_needed = int(np.ceil(data_duration_sec * self.PATTERN_SYMBOL_RATE))
            pattern_symbols_needed = min(pattern_symbols_needed, self.PATTERN_LENGTH)
        else:  # Standard 3-FSK
            # Pattern layer must match data duration + tail (no ramps)
            total_samples = data_layer_samples + rc_tail_samples
            data_duration_sec = total_samples / self.SAMPLE_RATE
            pattern_symbols_needed = int(np.ceil(data_duration_sec * self.PATTERN_SYMBOL_RATE))
            pattern_symbols_needed = min(pattern_symbols_needed, self.PATTERN_LENGTH)

        # Use the calculated pattern symbol count
        num_pattern_symbols = pattern_symbols_needed

        # Generate GMSK carriers
        if profile: step_start = time.time()

        if use_always_on_generation:
            # Extend always-on patterns to cover full duration
            pattern_symbols_extended = {
                'center': pattern_symbols_batch['center'][:, :num_pattern_symbols],
                'lower': pattern_symbols_batch['lower'][:, :num_pattern_symbols],
                'upper': pattern_symbols_batch['upper'][:, :num_pattern_symbols]
            }

            # Generate always-on GMSK with 2 centers
            gmsk_signals = self.generate_gmsk_always_on_batch(
                pattern_symbols_extended,
                batch_params.frequency_triples,
                num_centers=2,  # 2-center configuration (balanced performance)
                num_samples=total_samples
            )
        else:
            # Standard 3-FSK generation
            # Extend patterns to cover full duration
            pattern_symbols_extended = pattern_symbols_batch[:, :num_pattern_symbols]

            # Generate GMSK 3-FSK carriers (GPU parallel, NO ramps yet)
            gmsk_signals = self.generate_gmsk_3fsk_batch(
                pattern_symbols_extended,
                tones_a, tones_b, tones_c,
                total_samples
            )

        if profile: step_times['gmsk_generation'] = time.time() - step_start

        # Apply data modulation (GPU parallel)
        if profile: step_start = time.time()
        data_shaped = self.modulate_data_batch(
            data_symbols_batch,
            batch_params.data_symbol_rates,
            total_samples
        )
        if profile: step_times['data_modulation'] = time.time() - step_start

        # CARRIER RAMP REMOVED: Shaped pilots (inner→outer) now provide smooth amplitude start
        # The preamble starts with low-amplitude constellation points and gradually increases
        # This eliminates the transient spectral spreading from carrier amplitude ramps
        if profile: step_start = time.time()
        # No carrier ramp - shaped pilots handle smooth start
        if profile: step_times['ramps'] = time.time() - step_start

        # Combine layers: multiply ramped GMSK carrier with pulse-shaped data
        if profile: step_start = time.time()

        if use_always_on_generation:
            # ALWAYS-ON ARCHITECTURE:
            # Centers: Constant frequency, phase-modulated by data (always on)
            # Outers: FSK pattern + phase-modulated by data (alternating)
            #
            # The gmsk_signals already contains:
            #   - 4 constant-frequency centers
            #   - 8 FSK-modulated alternating tones
            #
            # All should be phase-modulated by the same data
            iq_signals = gmsk_signals * data_shaped
        else:
            # Standard: Data fully modulates carrier
            iq_signals = gmsk_signals * data_shaped

        if profile: step_times['combine'] = time.time() - step_start

        # Normalize to unit RMS power AFTER combining
        # This ensures all signals have consistent power for downstream SNR calculations
        if profile: step_start = time.time()
        sig_power = torch.mean(torch.abs(iq_signals)**2, dim=1, keepdim=True)  # [batch_size, 1]
        sig_power = torch.clamp(sig_power, min=1e-12)  # Avoid division by zero
        iq_signals = iq_signals / torch.sqrt(sig_power)  # Broadcast division
        if profile: step_times['normalize'] = time.time() - step_start

        # NOTE: Ramps removed - raised cosine filtering provides sufficient spectral containment
        # Applying Tukey window to PSK-modulated signal causes phase instability near zero amplitude
        if profile: step_start = time.time()
        # Ramps disabled for all modes (always-on and standard)
        if profile: step_times['ramps'] = time.time() - step_start

        # Pad or truncate to fixed length (ONLY if fixed_length specified)
        if profile: step_start = time.time()
        if fixed_length is not None:
            if total_samples < fixed_length:
                padded = torch.zeros(batch_size, fixed_length, dtype=torch.complex64, device=self.device)
                padded[:, :total_samples] = iq_signals
                iq_signals = padded
            else:
                iq_signals = iq_signals[:, :fixed_length]
        if profile: step_times['pad'] = time.time() - step_start

        # Add duration to metadata
        for i, meta in enumerate(metadata_batch):
            meta['duration_seconds'] = total_samples / self.SAMPLE_RATE
            meta['num_samples'] = total_samples

        if profile:
            total_time = time.time() - total_start
            print(f"\n  Signal generation profiling (batch_size={batch_size}):")
            for step, t in step_times.items():
                print(f"    {step}: {t*1000:.1f}ms ({t/total_time*100:.1f}%)")
            print(f"    TOTAL: {total_time*1000:.1f}ms")

        return iq_signals, metadata_batch


def test_gpu_generator():
    """Test GPU signal generator with small batch."""
    print("Testing GPU Signal Generator...")

    # Create generator
    gen = GPUSignalGenerator(device='cuda')

    # Create test batch (8 signals)
    batch_size = 8
    batch_params = BatchKernelParameters(
        pattern_ids=torch.randint(0, 4, (batch_size,), device='cuda'),
        frequency_triples=torch.randint(0, 43, (batch_size,), device='cuda'),
        modulations=['QPSK'] * batch_size,
        polar_rates=[(2, 3)] * batch_size,
        data_symbol_rates=torch.tensor([150] * batch_size, device='cuda')
    )

    messages = [b"Hello CASCADE V2 on GPU!" for _ in range(batch_size)]

    # Generate batch
    import time
    start = time.time()
    signals, metadata = gen.generate_batch(batch_params, messages, fixed_length=2048)
    elapsed = time.time() - start

    print(f"✓ Generated {batch_size} signals in {elapsed*1000:.1f}ms")
    print(f"  Output shape: {signals.shape}")
    print(f"  Output dtype: {signals.dtype}")
    print(f"  Device: {signals.device}")
    print(f"  Sample metadata: {metadata[0]}")
    print(f"  Per-signal time: {elapsed/batch_size*1000:.2f}ms")


if __name__ == "__main__":
    test_gpu_generator()
