"""
Channel Orchestrator API Contract

This module defines the expected interface for the Synthetic Data Orchestrator.
The orchestrator generates training data for CASCADE's expert-based neural network architecture.

KEY REQUIREMENT: Generate separate training datasets for each expert:
- QRN Expert: Pure atmospheric noise (NO signal)
- Signal Expert: Clean signals (NO interference/noise)
- Timing Expert: Collision scenarios (1-3 overlapping signals with time offsets)
- Channel Expert: Known channel models (multipath, Doppler)
- QRM Expert: Pure interference patterns (NO CASCADE signal)

These are contract specifications, not implementations. Tests should be written
against this interface first (TDD), then implementations should satisfy the contract.
"""

from typing import Protocol, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ChannelConditions:
    """HF channel conditions for simulation."""
    snr_db: float = 0.0  # -35 to +20 dB
    awgn_enabled: bool = True
    qrn_enabled: bool = False
    qrn_burst_rate: float = 2.5  # bursts/second
    qrn_intensity: float = 0.8  # relative to signal
    multipath_enabled: bool = False
    multipath_delay_spread_ms: float = 3.0  # 1-5ms
    multipath_taps: int = 3  # 2-5 taps
    qrm_enabled: bool = False
    qrm_interferer_count: int = 2  # 0-10
    qrm_strength_db: float = -10  # -20 to +10 dB


@dataclass(frozen=True)
class CollisionScenario:
    """Configuration for temporal collision generation."""
    num_signals: int  # 1-3 overlapping signals
    time_offsets_ms: List[float]  # Time offset for each signal (first is 0.0)
    frequency_pairs: List[int]  # Frequency pair for each signal (can be same or different)
    snr_db_list: List[float]  # SNR for each signal
    relative_powers: List[float]  # Relative power (1.0 = equal power)


@dataclass(frozen=True)
class RealisticIQSignal:
    """Signal with HF channel effects applied."""
    iq_samples: np.ndarray  # complex64, shape (num_samples,)
    sample_rate: int  # 48000 Hz
    clean_signal: 'CleanIQSignal'  # Reference to original
    channel_conditions: ChannelConditions
    measured_snr_db: float  # Actual SNR after effects
    generation_timestamp: str  # ISO 8601
    orchestrator_version: str


@dataclass(frozen=True)
class ExpertTrainingExample:
    """Single training example for expert network."""
    iq_samples: np.ndarray  # complex64, input to expert
    labels: dict  # Expert-specific ground truth labels
    metadata: dict  # Generation parameters
    expert_type: str  # 'qrn', 'signal', 'timing', 'channel', 'qrm'


class ChannelOrchestratorInterface(Protocol):
    """
    Interface for CASCADE Synthetic Data Orchestrator.

    Generates training data for expert-based neural network architecture.
    Each expert requires specialized training data:

    - QRN Expert: Pure atmospheric noise (NO signal)
    - Signal Expert: Clean CASCADE signals (NO interference)
    - Timing Expert: Collision scenarios (overlapping signals)
    - Channel Expert: Known channel models
    - QRM Expert: Pure interference (NO CASCADE signal)

    Contract Requirements:
    - FR-016 to FR-028 (see spec.md)
    - Support batch generation for dataset creation
    - Provide ground truth labels for all generated data
    - Enable reproducible generation via seeds
    """

    def add_channel_effects(
        self,
        clean_iq: np.ndarray,
        channel_conditions: ChannelConditions,
        seed: Optional[int] = None
    ) -> Tuple[RealisticIQSignal, dict]:
        """
        Apply HF channel effects to clean signal.

        Args:
            clean_iq: Clean IQ samples (from Core Generator)
            channel_conditions: Channel parameters to apply
            seed: Optional random seed for reproducibility

        Returns:
            Tuple of (RealisticIQSignal, metadata_dict)

        Raises:
            ValueError: If channel conditions invalid

        Contract:
            - Must apply effects in order: AWGN, QRN, multipath, QRM
            - Measured SNR must be within 1 dB of target (FR-019)
            - Must issue warning if SNR below Shannon limit (FR-019)
            - All noise sources must be reproducible with seed
        """
        ...

    def generate_qrn_expert_data(
        self,
        duration_seconds: float,
        qrn_type: str,  # 'crackling', 'static', 'lightning', 'power_line'
        intensity: float,
        sample_rate: int = 48000,
        seed: Optional[int] = None
    ) -> ExpertTrainingExample:
        """
        Generate pure atmospheric noise for QRN Expert training.

        **CRITICAL**: NO CASCADE signal included - pure noise only!

        Args:
            duration_seconds: Length of noise sample
            qrn_type: Type of atmospheric noise
            intensity: Noise intensity (0.1-5.0)
            sample_rate: Sample rate (default 48000 Hz)
            seed: Random seed

        Returns:
            ExpertTrainingExample with:
            - iq_samples: Pure QRN noise (no signal)
            - labels: {'qrn_type': str, 'intensity': float, 'burst_times': array}
            - expert_type: 'qrn'

        Contract:
            - Output must contain ONLY noise (no signal)
            - QRN types: crackling (Poisson bursts), static (1/f), lightning (impulse), power_line (50/60 Hz harmonics)
            - Labels must include exact burst timing for supervised learning
            - Duration must match requested time ±1ms
        """
        ...

    def generate_signal_expert_data(
        self,
        clean_iq: np.ndarray,
        seed: Optional[int] = None
    ) -> ExpertTrainingExample:
        """
        Generate clean CASCADE signal for Signal Expert training.

        **CRITICAL**: NO noise or interference - clean signal only!

        Args:
            clean_iq: Clean IQ from Core Generator
            seed: Random seed (unused, for API consistency)

        Returns:
            ExpertTrainingExample with:
            - iq_samples: Clean signal (no noise)
            - labels: {'pattern_id': int, 'frequency_pair': int, 'modulation': str, 'polar_codeword': array}
            - expert_type: 'signal'

        Contract:
            - Output must be bit-identical to input (no modifications)
            - Labels must include all kernel discrete parameters
            - Labels must include Polar codeword for supervised decoding
            - Signal must have perfect SNR (∞ dB)
        """
        ...

    def generate_timing_expert_data(
        self,
        collision_scenario: CollisionScenario,
        clean_signals: List[np.ndarray],
        base_noise_floor_db: float = -30,
        seed: Optional[int] = None
    ) -> ExpertTrainingExample:
        """
        Generate collision scenario for Timing Expert training.

        **CRITICAL**: 1-3 overlapping CASCADE signals with time offsets.

        Args:
            collision_scenario: Collision configuration
            clean_signals: List of clean IQ signals (1-3 signals)
            base_noise_floor_db: Background noise level
            seed: Random seed

        Returns:
            ExpertTrainingExample with:
            - iq_samples: Overlapped signals + noise
            - labels: {
                'num_signals': int,
                'time_offsets_ms': list,
                'signal_boundaries': list of (start, end) samples,
                'individual_signals': list of separated IQ arrays,
                'kernels': list of kernel parameters for each signal
              }
            - expert_type: 'timing'

        Contract:
            - Must support 1-3 simultaneous signals (FR-023)
            - Time offsets: 0-100ms typical, quantized to sample boundaries
            - Labels must include precise sample boundaries for each signal
            - Labels must include separated signals for supervised learning
            - Collision types: same frequency pair (hard) or different pairs (easier)
            - SNR per signal must be independently controllable
        """
        ...

    def generate_channel_expert_data(
        self,
        clean_iq: np.ndarray,
        channel_type: str,  # 'rayleigh_fading', 'rician_fading', 'multipath', 'doppler'
        channel_params: dict,
        kernel_parameters: dict,  # NEW: Required for embedding encoder training
        seed: Optional[int] = None
    ) -> ExpertTrainingExample:
        """
        Generate known channel model for Channel Expert training.

        **CRITICAL**: Clean signal with KNOWN channel distortion (no QRN/QRM).
        **UPDATED**: Now requires kernel_parameters per CLAUDE.md architecture update.

        Args:
            clean_iq: Clean IQ from Core Generator
            channel_type: Type of channel model
            channel_params: Model-specific parameters
            kernel_parameters: Kernel params from signal generation (pattern_id, frequency_pair, modulation, polar_rate, snr_estimate)
            seed: Random seed

        Returns:
            ExpertTrainingExample with:
            - iq_samples: Signal after channel (no noise)
            - labels: {
                'channel_type': str,
                'channel_params': dict,
                'impulse_response': array,
                'doppler_shift_hz': float,
                'delay_spread_ms': float,
                'kernel_parameters': dict  # NEW: pattern_id, frequency_pair, modulation, polar_rate, snr_estimate
              }
            - expert_type: 'channel'

        Contract:
            - Channel types: Rayleigh (NLOS), Rician (LOS), multipath (tapped delay), Doppler (motion)
            - Labels must include exact channel impulse response
            - Labels must include all model parameters
            - NO noise or interference (pure channel effects only)
            - **NEW**: Labels must include kernel_parameters (embedding encoder needs this context)
            - Rayleigh: Multiple taps with Rayleigh-distributed gains
            - Rician: K-factor specified (ratio of LOS to scattered)
            - Multipath: Delay spread 1-5ms, 2-5 taps
            - Doppler: Frequency shift ±5 Hz typical for HF
            - kernel_parameters format: {'pattern_id': int, 'frequency_pair': int, 'modulation': str, 'polar_rate': tuple, 'snr_estimate': float}
        """
        ...

    def generate_qrm_expert_data(
        self,
        duration_seconds: float,
        interference_type: str,  # 'cw', 'ssb', 'ft8', 'digital', 'radar', 'power_line'
        frequency_offset_hz: float,
        strength_db: float,
        sample_rate: int = 48000,
        seed: Optional[int] = None
    ) -> ExpertTrainingExample:
        """
        Generate pure interference for QRM Expert training.

        **CRITICAL**: NO CASCADE signal - pure interference only!

        Args:
            duration_seconds: Length of interference sample
            interference_type: Type of interferer
            frequency_offset_hz: Offset from CASCADE signal (±200 Hz typical)
            strength_db: Interference power (dB relative to typical signal)
            sample_rate: Sample rate (default 48000 Hz)
            seed: Random seed

        Returns:
            ExpertTrainingExample with:
            - iq_samples: Pure interference (no CASCADE signal)
            - labels: {
                'interference_type': str,
                'frequency_offset_hz': float,
                'strength_db': float,
                'bandwidth_hz': float,
                'modulation_params': dict
              }
            - expert_type: 'qrm'

        Contract:
            - Output must contain ONLY interference (no CASCADE signal)
            - Interference types:
              - CW: Single tone (amateur CW, carrier)
              - SSB: Voice spectrum (300-3000 Hz, amplitude modulation)
              - FT8: 50 Hz wide, GFSK, 15s cycles
              - Digital: PSK31, RTTY, etc.
              - Radar: Pulsed wideband
              - Power line: 50/60 Hz harmonics with sidebands
            - Labels must include all modulation parameters
            - Frequency offset must be accurate to ±0.1 Hz
        """
        ...

    def generate_batch(
        self,
        expert_type: str,
        num_examples: int,
        config: dict,
        seed: Optional[int] = None
    ) -> List[ExpertTrainingExample]:
        """
        Generate batch of training examples for specified expert.

        Args:
            expert_type: 'qrn', 'signal', 'timing', 'channel', or 'qrm'
            num_examples: Number of examples to generate
            config: Expert-specific configuration dictionary
            seed: Random seed (incremented per example)

        Returns:
            List of ExpertTrainingExample instances

        Raises:
            ValueError: If expert_type invalid or config incomplete

        Contract:
            - Must support all 5 expert types
            - Examples must be deterministic given seed
            - Must support batch sizes 1-10,000
            - Must complete 100 examples in <30s (FR-023 performance)
            - Config must specify ranges for random parameter sampling
        """
        ...

    def generate_snr_sweep(
        self,
        clean_iq: np.ndarray,
        snr_start_db: float,
        snr_stop_db: float,
        snr_step_db: float,
        seed: Optional[int] = None
    ) -> List[RealisticIQSignal]:
        """
        Generate series of signals at different SNR levels.

        Args:
            clean_iq: Clean IQ from Core Generator
            snr_start_db: Starting SNR
            snr_stop_db: Ending SNR
            snr_step_db: SNR increment
            seed: Random seed

        Returns:
            List of RealisticIQSignal instances at each SNR level

        Contract:
            - Must generate (stop - start) / step + 1 signals
            - Each signal must have different noise realization
            - SNR must be accurate to ±0.5 dB
            - Useful for decoder BER vs SNR curves
        """
        ...

    def save_expert_dataset(
        self,
        examples: List[ExpertTrainingExample],
        output_dir: str,
        dataset_name: str,
        format: str = 'npz'  # 'npz', 'hdf5', 'zarr'
    ) -> dict:
        """
        Save expert training dataset to disk.

        Args:
            examples: List of training examples
            output_dir: Directory to save files
            dataset_name: Dataset name (e.g., 'qrn_expert_train_v1')
            format: Storage format

        Returns:
            Dictionary with file paths and dataset statistics

        Contract:
            - Must save IQ samples and labels separately
            - Must include metadata JSON with generation parameters
            - Must be loadable by PyTorch/TensorFlow data loaders
            - Format options:
              - npz: NumPy compressed (good for small datasets)
              - hdf5: Hierarchical (good for large datasets, streaming)
              - zarr: Chunked (good for very large datasets, cloud)
            - File naming: {dataset_name}_{expert_type}_{split}.{format}
        """
        ...


class AWGNGeneratorInterface(Protocol):
    """Interface for Additive White Gaussian Noise."""

    def add_awgn(
        self,
        signal: np.ndarray,
        snr_db: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Add AWGN at specified SNR.

        Args:
            signal: Clean IQ signal
            snr_db: Target signal-to-noise ratio
            seed: Random seed

        Returns:
            Signal with AWGN added

        Contract:
            - SNR must be accurate to ±0.5 dB
            - Noise must be white (flat spectrum)
            - Noise must be Gaussian (normal distribution)
            - Complex noise (both I and Q)
        """
        ...


class QRNGeneratorInterface(Protocol):
    """Interface for atmospheric noise (QRN) generation."""

    def generate_crackling_noise(
        self,
        duration_seconds: float,
        burst_rate: float,
        intensity: float,
        sample_rate: int = 48000,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate crackling atmospheric noise (Poisson bursts).

        Args:
            duration_seconds: Duration of noise
            burst_rate: Bursts per second (0.1-10)
            intensity: Peak amplitude relative to signal
            sample_rate: Sample rate
            seed: Random seed

        Returns:
            Tuple of (noise_iq, labels_dict)
            labels_dict: {'burst_times': array, 'burst_amplitudes': array}

        Contract:
            - Burst timing: Poisson process (random intervals)
            - Burst shape: Exponential decay, 5-20ms duration
            - Complex noise (I and Q decorrelated)
            - Labels must include exact burst sample indices
        """
        ...

    def generate_static_noise(
        self,
        duration_seconds: float,
        intensity: float,
        sample_rate: int = 48000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate static noise (1/f spectrum).

        Args:
            duration_seconds: Duration
            intensity: RMS amplitude
            sample_rate: Sample rate
            seed: Random seed

        Returns:
            Complex IQ noise with 1/f spectrum

        Contract:
            - Spectrum: Power ~ 1/f (pink noise)
            - Broadband (covers all frequencies)
            - Continuous (no bursts)
        """
        ...

    def generate_lightning_noise(
        self,
        duration_seconds: float,
        strike_rate: float,
        intensity: float,
        sample_rate: int = 48000,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate lightning impulse noise.

        Args:
            duration_seconds: Duration
            strike_rate: Strikes per second (0.01-1.0)
            intensity: Peak amplitude
            sample_rate: Sample rate
            seed: Random seed

        Returns:
            Tuple of (noise_iq, labels_dict)
            labels_dict: {'strike_times': array}

        Contract:
            - Impulse shape: Very short (<1ms), broadband
            - Strike timing: Poisson process
            - Exponential decay tail
        """
        ...


class MultipathGeneratorInterface(Protocol):
    """Interface for multipath fading simulation."""

    def apply_multipath(
        self,
        signal: np.ndarray,
        delay_spread_ms: float,
        num_taps: int,
        fading_type: str,  # 'rayleigh' or 'rician'
        sample_rate: int = 48000,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply multipath fading to signal.

        Args:
            signal: Clean IQ signal
            delay_spread_ms: RMS delay spread (1-5ms)
            num_taps: Number of multipath taps (2-5)
            fading_type: 'rayleigh' (NLOS) or 'rician' (LOS)
            sample_rate: Sample rate
            seed: Random seed

        Returns:
            Tuple of (faded_signal, channel_params)
            channel_params: {'tap_delays': array, 'tap_gains': array, 'impulse_response': array}

        Contract:
            - Tap delays: Exponentially distributed
            - Rayleigh: Tap gains ~ Rayleigh distribution
            - Rician: First tap has K-factor advantage
            - Complex channel (phase shifts per tap)
            - Labels must include exact impulse response
        """
        ...


class QRMGeneratorInterface(Protocol):
    """Interface for interference (QRM) generation."""

    def generate_cw_interference(
        self,
        duration_seconds: float,
        frequency_offset_hz: float,
        strength_db: float,
        sample_rate: int = 48000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate CW (continuous wave) interference.

        Args:
            duration_seconds: Duration
            frequency_offset_hz: Offset from signal frequency
            strength_db: Interference power (dB)
            sample_rate: Sample rate
            seed: Random seed

        Returns:
            Complex IQ interference signal

        Contract:
            - Single tone at specified offset
            - Constant amplitude (no modulation)
            - May include slow drift (±0.1 Hz/s typical)
        """
        ...

    def generate_ft8_interference(
        self,
        duration_seconds: float,
        frequency_offset_hz: float,
        strength_db: float,
        sample_rate: int = 48000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate FT8 digital mode interference.

        Args:
            duration_seconds: Duration
            frequency_offset_hz: Offset from signal frequency
            strength_db: Interference power
            sample_rate: Sample rate
            seed: Random seed

        Returns:
            Complex IQ FT8 interference

        Contract:
            - 50 Hz bandwidth (8-FSK GFSK)
            - 15-second transmission cycles
            - Realistic FT8 message patterns
        """
        ...

    def generate_ssb_interference(
        self,
        duration_seconds: float,
        frequency_offset_hz: float,
        strength_db: float,
        sample_rate: int = 48000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate SSB voice interference.

        Args:
            duration_seconds: Duration
            frequency_offset_hz: Offset from signal frequency
            strength_db: Interference power
            sample_rate: Sample rate
            seed: Random seed

        Returns:
            Complex IQ SSB interference

        Contract:
            - Voice spectrum: 300-3000 Hz
            - Amplitude modulation (AM envelope)
            - Realistic speech patterns (pauses, phonemes)
        """
        ...


class CollisionGeneratorInterface(Protocol):
    """Interface for temporal collision scenario generation."""

    def generate_collision_scenario(
        self,
        clean_signals: List[np.ndarray],
        time_offsets_ms: List[float],
        snr_db_list: List[float],
        relative_powers: List[float],
        base_noise_floor_db: float = -30,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate temporal collision of multiple CASCADE signals.

        Args:
            clean_signals: List of 1-3 clean IQ signals
            time_offsets_ms: Time offset for each signal (first = 0.0)
            snr_db_list: SNR for each signal
            relative_powers: Relative power scaling (1.0 = equal)
            base_noise_floor_db: Background noise level
            seed: Random seed

        Returns:
            Tuple of (combined_iq, labels_dict)
            labels_dict: {
                'num_signals': int,
                'time_offsets_samples': list,
                'signal_boundaries': list of (start, end),
                'individual_signals': list of separated IQ,
                'kernels': list of kernel params
            }

        Contract:
            - Must support 1-3 signals (FR-023)
            - Time offsets quantized to sample boundaries
            - Signals overlap in time (partial or complete)
            - Labels include sample-accurate boundaries
            - Labels include separated signals (ground truth)
            - Combined signal = sum of time-shifted signals + noise
            - Each signal independently scaled by SNR and relative power
        """
        ...
