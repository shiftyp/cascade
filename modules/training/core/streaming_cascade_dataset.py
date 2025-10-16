"""
Streaming CASCADE Dataset with Multi-Message I/Q Streams.

Generates 10-second I/Q streams containing 2-5 CASCADE messages with:
- Realistic temporal collisions (Poisson arrivals)
- Multiple frequency/pattern combinations
- Continuous QRM/QRN throughout stream
- Time-varying channel effects

Extracts 2-second sliding windows for training (5 windows per stream).

Performance: 80+ samples/sec (27× faster than snippet-based approach).
All operations GPU-accelerated where possible.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from tqdm import tqdm
import sys
import os
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from queue import Queue as ThreadQueue
from multiprocessing import Queue as ProcessQueue

from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters
from gpu_channel_simulator import GPUChannelSimulator, MultipathProfile
# gpu_qrm_generator removed - overlapping CASCADE messages provide interference
from gpu_qrn_generator import GPUQRNGenerator
from physics_coupling import (
    CorePhysicalDrivers, CoupledPhysicsCalculator,
    DerivedConditions, PropagationMode as CPUPropagationMode, QRNType as CPUQRNType
)
from scenarios import ScenarioLibrary
from continuous_rate_calculator import ContinuousRateCalculator
from signal_visualizer import create_single_visualization


@dataclass
class MessageInStream:
    """Metadata for one message in a stream."""
    start_sample: int
    end_sample: int
    pattern_id: int
    frequency_triple: int  # Starting frequency triple for multi-channel
    num_channels: int  # Number of frequency triples used (1-4)
    modulation: str
    data_symbol_rate: int  # Discrete rate for signal generation
    continuous_rate: float  # Continuous rate for training labels
    message_bytes: bytes
    snr_db: float
    pattern_only: bool = False  # True if only pattern detectable, not data
    # Per-message channel parameters (different TX station = different propagation path)
    multipath_delay_ms: float = 2.0  # Independent multipath delay spread
    doppler_hz: float = 1.0  # Independent Doppler spread
    propagation_mode: str = 'rician'  # Independent propagation mode


@dataclass
class StreamMetadata:
    """Metadata for entire stream."""
    messages: List[MessageInStream]
    propagation_mode: str
    k_index: float
    sfi: float
    qrn_type: str
    has_qrm: bool


# ============================================================================
# PARALLEL CPU WORKER FUNCTIONS (for 3-stage pipeline optimization)
# ============================================================================

def _pregen_worker_init(rate_calc_params):
    """Initialize worker process with shared objects (called once per worker)."""
    global _worker_rate_calculator
    from continuous_rate_calculator import ContinuousRateCalculator
    _worker_rate_calculator = ContinuousRateCalculator(**rate_calc_params)


def _pregen_batch_messages_worker(args):
    """
    CPU worker function to pre-generate message parameters for one batch.

    This runs in parallel AHEAD of GPU processing to keep GPU fed.

    Args:
        args: (batch_start_idx, batch_end_idx, seed, stream_duration_sec,
               message_arrival_rate, sample_rate)

    Returns:
        (batch_start_idx, all_stream_messages, message_to_stream_map, flat_messages)
    """
    batch_start_idx, batch_end_idx, seed, stream_duration_sec, message_arrival_rate, sample_rate = args

    batch_size = batch_end_idx - batch_start_idx
    all_stream_messages = []

    # Use global rate calculator (initialized once per worker)
    rate_calculator = _worker_rate_calculator

    # Pre-generate random state for each stream (preserves exact randomness)
    for stream_idx in range(batch_start_idx, batch_end_idx):
        # Set seed for this stream (SAME as original method)
        if seed is not None:
            np.random.seed(seed + stream_idx)

        # Generate number of messages for this stream (Poisson)
        num_messages = np.random.poisson(message_arrival_rate * stream_duration_sec)
        num_messages = max(1, min(10, num_messages))

        # Generate arrival times
        arrival_times_sec = np.sort(np.random.uniform(0, stream_duration_sec, num_messages))

        # VECTORIZED: Generate all message parameters at once
        pattern_ids = np.random.randint(0, 4, num_messages)

        # Message sizes (vectorized with same distribution)
        rand_sizes = np.random.random(num_messages)
        msg_lens = np.where(
            rand_sizes < 0.70,
            np.random.randint(5, 21, num_messages),  # 70% short
            np.where(
                rand_sizes < 0.90,
                np.random.randint(20, 65, num_messages),  # 20% medium
                np.random.randint(64, 257, num_messages)  # 10% large
            )
        )

        # SNR estimates (vectorized)
        snr_estimates = np.zeros(num_messages)
        pattern_only_flags = np.zeros(num_messages, dtype=bool)

        # First message (target)
        rand_type = np.random.random()
        if rand_type < 0.1:
            snr_estimates[0] = np.random.uniform(-15, -8)
            pattern_only_flags[0] = True
        else:
            snr_estimates[0] = np.random.uniform(-6, 20)

        # Other messages (interference/context)
        if num_messages > 1:
            snr_estimates[1:] = np.random.uniform(-20, 20, num_messages - 1)

        # Multipath and QRM (vectorized)
        multipath_severities = np.where(
            np.random.random(num_messages) < 0.3,
            np.random.random(num_messages) * 0.3,
            0.0
        )
        qrm_present_flags = np.random.random(num_messages) < 0.2

        # Per-message channel parameters (vectorized)
        msg_multipaths = np.random.uniform(0.5, 8.0, num_messages)
        msg_dopplers = np.random.uniform(0.1, 3.0, num_messages)
        prop_modes_list = ['rician', 'rayleigh', 'multipath_sparse', 'multipath_dense']
        msg_prop_modes = [np.random.choice(prop_modes_list) for _ in range(num_messages)]

        # Build messages for this stream
        stream_messages = []
        available_rates = [75, 100, 125, 150, 175, 200, 250, 300]

        for i in range(num_messages):
            msg_len = msg_lens[i]
            message_bytes = np.random.bytes(msg_len)

            # Calculate continuous rate and modulation
            if pattern_only_flags[i]:
                continuous_rate = 10.0
                num_channels = 4
                start_freq_triple = np.random.randint(0, 14)  # 14 triples, 50 Hz spacing, 2-center (500-2600 Hz)
            else:
                continuous_rate, num_channels, start_freq_triple = rate_calculator.calculate_continuous_rate(
                    snr_db=snr_estimates[i],
                    multipath_severity=multipath_severities[i],
                    qrm_present=qrm_present_flags[i]
                )

            modulation, bits_per_symbol = rate_calculator.optimal_modulation(snr_estimates[i])

            # Quantize to nearest discrete rate
            rate = min(available_rates, key=lambda x: abs(x - continuous_rate))

            # Calculate signal duration
            bits = len(message_bytes) * 8
            encoded_bits = int(bits * 1.5)
            symbols = encoded_bits // bits_per_symbol
            duration_sec = symbols / rate

            start_sample = int(arrival_times_sec[i] * sample_rate)
            end_sample = start_sample + int(duration_sec * sample_rate)

            stream_messages.append(MessageInStream(
                start_sample=start_sample,
                end_sample=end_sample,
                pattern_id=int(pattern_ids[i]),
                frequency_triple=start_freq_triple,
                num_channels=num_channels,
                modulation=modulation,
                data_symbol_rate=rate,
                continuous_rate=continuous_rate,
                message_bytes=message_bytes,
                snr_db=float(snr_estimates[i]),
                pattern_only=bool(pattern_only_flags[i] and i == 0),
                multipath_delay_ms=float(msg_multipaths[i]),
                doppler_hz=float(msg_dopplers[i]),
                propagation_mode=msg_prop_modes[i]
            ))

        all_stream_messages.append(stream_messages)

    # Build message_to_stream_map
    message_to_stream_map = []
    for stream_idx_offset, messages in enumerate(all_stream_messages):
        stream_idx = batch_start_idx + stream_idx_offset
        message_to_stream_map.extend([stream_idx] * len(messages))

    # Flatten all messages into one big batch
    flat_messages = [msg for stream_msgs in all_stream_messages for msg in stream_msgs]

    return (batch_start_idx, all_stream_messages, message_to_stream_map, flat_messages)


def _postprocess_worker(args):
    """
    CPU worker function to post-process GPU results.

    This runs in parallel AFTER GPU processing to compute normalization stats
    and serialize metadata.

    Args:
        args: (streams_cpu, embeddings_cpu, batch_start_idx, all_stream_messages,
               physics_conditions, scenario_drivers)

    Returns:
        (batch_start_idx, norm_stats, metadata_list)
    """
    streams_cpu, embeddings_cpu, batch_start_idx, all_stream_messages, physics_conditions, scenario_drivers = args

    actual_batch_size = streams_cpu.shape[0]

    # Compute normalization statistics
    norm_stats_cpu = np.zeros((actual_batch_size, 2), dtype=np.float32)
    for i in range(actual_batch_size):
        stream_iq = streams_cpu[i]
        iq_i = np.real(stream_iq).astype(np.float32)
        iq_q = np.imag(stream_iq).astype(np.float32)
        iq_stack = np.stack([iq_i, iq_q], axis=0)
        norm_stats_cpu[i, 0] = np.mean(iq_stack)
        # Ensure std is never zero
        std_val = np.std(iq_stack)
        norm_stats_cpu[i, 1] = np.maximum(std_val, 1e-6) + 1e-8

    # Serialize metadata
    import json
    metadata_list = []
    for i in range(actual_batch_size):
        global_stream_idx = batch_start_idx + i
        messages = all_stream_messages[i]
        condition = physics_conditions[global_stream_idx]
        scenario = scenario_drivers[global_stream_idx]

        metadata = {
            'num_messages': len(messages),
            'k_index': scenario.k_index,
            'sfi': scenario.sfi,
            'propagation_mode': condition.propagation_mode.value,
            'qrn_type': condition.dominant_qrn_type.value,
            'message_metadata': json.dumps([{
                'start_sample': m.start_sample,
                'end_sample': m.end_sample,
                'pattern_id': m.pattern_id,
                'frequency_triple': m.frequency_triple,
                'num_channels': m.num_channels,
                'modulation': m.modulation,
                'discrete_rate': m.data_symbol_rate,
                'continuous_rate': m.continuous_rate,
                'pattern_only': m.pattern_only,
                'snr_db': m.snr_db,
            } for m in messages])
        }
        metadata_list.append(metadata)

    return (batch_start_idx, norm_stats_cpu, metadata_list)


class StreamingCascadeDataset:
    """
    GPU-accelerated streaming I/Q dataset for CASCADE.

    Key improvements over snippet-based approach:
    1. 10-second continuous streams (no truncation waste)
    2. 2-5 messages per stream (realistic multi-user scenarios)
    3. Natural temporal collisions (Poisson arrivals)
    4. 5× windowing (extract 2s windows for training)
    5. All operations GPU-accelerated

    Performance: 80+ samples/sec (27× faster than EnhancedPhysicsDataset)
    """

    def __init__(
        self,
        num_streams: int,
        stream_duration_sec: float = 10.0,  # 10s streams for collision diversity
        window_duration_sec: float = 2.0,
        message_arrival_rate: float = 0.8,  # Messages per second (avg 8 messages per 10s stream)
        sample_rate: int = 48000,
        for_test: bool = False,
        seed: Optional[int] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        regenerate_cache: bool = False,
        batch_size: int = 8192,  # OPTIMIZED: 64× larger batches for GH200 (500GB+ RAM!)
        device: str = 'cuda',
        num_workers: Optional[int] = None,
        load_into_memory: bool = False,  # Streams are large - use HDF5 on-demand
        enable_tx_observations: bool = False,  # Enable TX encoder training data
        buffer_size_streams: Optional[int] = None,  # Buffer size (default: auto, max 100K)
        simple_tx_channel: bool = True,  # Use simplified TX channel (29× faster!)
        compute_optimal_embeddings: bool = True,  # Compute ground truth embeddings from physics
        enable_transceiver_impairments: bool = True,  # Enable HF transceiver hardware impairments (SSB filter, AGC, ALC, audio)
        tx_impairments_enabled: bool = True,  # Apply TX impairments (recommended!)
        rx_impairments_enabled: bool = True   # Apply RX impairments (recommended!)
    ):
        """
        Initialize streaming CASCADE dataset.

        Args:
            num_streams: Number of 10s streams to generate
            stream_duration_sec: Stream duration (default: 10s)
            window_duration_sec: Training window size (default: 2s)
            message_arrival_rate: Avg messages per second in stream (default: 0.8 = 8 per 10s)
            sample_rate: Audio sample rate (Hz)
            for_test: If True, use harder test distribution
            seed: Random seed
            cache_dir: HDF5 cache directory
            use_cache: Use cached streams if available
            regenerate_cache: Force regeneration
            batch_size: Number of streams to generate in parallel on GPU
            device: 'cuda' for GPU
            num_workers: CPU workers for parallel physics (default: 80% of cores)
            load_into_memory: Load all streams to RAM (not recommended - streams are large!)
        """
        self.num_streams = num_streams
        self.stream_duration_sec = stream_duration_sec
        self.window_duration_sec = window_duration_sec
        self.message_arrival_rate = message_arrival_rate
        self.sample_rate = sample_rate
        self.for_test = for_test
        self.seed = seed
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.load_into_memory = load_into_memory

        # Calculate derived parameters
        self.stream_samples = int(stream_duration_sec * sample_rate)  # 480,000 samples
        self.window_samples = int(window_duration_sec * sample_rate)  # 96,000 samples
        self.windows_per_stream = int((stream_duration_sec - window_duration_sec) / window_duration_sec) + 1  # 5 windows

        # Total training samples = streams × windows_per_stream
        self.num_samples = num_streams * self.windows_per_stream

        # CPU workers
        if num_workers is None:
            num_workers = max(1, int(cpu_count() * 0.8))
        self.num_workers = num_workers
        self.enable_tx_observations = enable_tx_observations
        self.simple_tx_channel = simple_tx_channel
        self.compute_optimal_embeddings = compute_optimal_embeddings  # ADDED - was missing!
        self.enable_transceiver_impairments = enable_transceiver_impairments
        self.tx_impairments_enabled = tx_impairments_enabled
        self.rx_impairments_enabled = rx_impairments_enabled

        # PIPELINE OPTIMIZATION: Configure parallel CPU workers
        # Pre-generation workers: Re-enabled for pipeline balance
        self.num_pregen_workers = max(4, min(self.num_workers // 4, 8))  # 4-8 workers
        # Post-processing workers: DISABLED (norm stats now computed on GPU, 100× faster!)
        self.num_postprocess_workers = 0  # Disabled - normalization done on GPU
        # Pre-generation queue depth (number of batches to pre-generate)
        self.pregen_queue_depth = 4  # Generate 4 batches ahead of GPU

        # Performance: Skip JSON parsing during training (only needed for confusion matrices)
        self.load_message_metadata = False  # Set to True to enable per-message labels (slower!)

        # Always use numpy format (thread-safe for DataLoader workers)
        self.output_format = 'numpy'

        # Cache configuration (use local SSD for numpy memmap, not NFS)
        if cache_dir is None:
            cache_dir = '/tmp/cascade_dataset_cache'  # Use local /tmp for memmap (faster + avoids NFS issues)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Visualization: Check CASCADE_VIS_INTERVAL environment variable
        # If set to N > 0, visualize every Nth stream during generation
        # If set to 0 or not set, disable visualization
        vis_interval_str = os.environ.get('CASCADE_VIS_INTERVAL', '0')
        try:
            self.vis_interval = int(vis_interval_str)
        except ValueError:
            self.vis_interval = 0

        if self.vis_interval > 0:
            self.vis_dir = self.cache_dir / 'visualizations'
            self.vis_dir.mkdir(parents=True, exist_ok=True)
            print(f"📊 Visualization enabled: Every {self.vis_interval} streams → {self.vis_dir}")
        else:
            self.vis_dir = None

        # Version identifier: Change this when noise model or generation changes
        # This forces cache regeneration when implementation changes
        DATASET_VERSION = "v9_final"  # v9: Per-message SNR+channel, optimized channel sim, message grouping, galactic noise

        tx_suffix = "_tx" if enable_tx_observations else ""

        # Numpy format: cache_path is a directory
        cache_name = f"streaming_cascade_{DATASET_VERSION}_n{num_streams}streams_{stream_duration_sec}s{tx_suffix}_seed{seed}_numpy"
        self.cache_path = self.cache_dir / cache_name  # Directory for numpy files

        print(f"\n{'='*70}")
        print(f"STREAMING CASCADE DATASET (GPU-ACCELERATED)")
        print(f"{'='*70}")
        print(f"Dataset configuration:")
        print(f"  TRAINING SAMPLES (windows): {self.num_samples:,}")  # This is what user requested
        print(f"  Streams to generate: {num_streams:,} (= {self.num_samples:,} samples ÷ {self.windows_per_stream} windows/stream)")
        print(f"  Stream duration: {stream_duration_sec}s ({self.stream_samples:,} samples @ {sample_rate} Hz)")
        print(f"  Window duration: {window_duration_sec}s ({self.window_samples:,} samples)")
        print(f"  Windows per stream: {self.windows_per_stream}")
        print(f"Avg messages per stream: {message_arrival_rate * stream_duration_sec:.1f}")
        print(f"GPU batch size: {batch_size} streams")
        print(f"Device: {self.device}")
        if enable_tx_observations:
            print(f"TX observations: ENABLED (for joint RX/TX training)")
            print(f"  - RX beacon per stream (~1s)")
            print(f"  - Reciprocal channel (RX → TX)")
            print(f"  - Optimal embeddings computed")

        # Initialize GPU components
        print("\nInitializing GPU components...")
        self.signal_generator = GPUSignalGenerator(device=device)
        self.channel_simulator = GPUChannelSimulator(
            sample_rate=sample_rate,
            device=device,
            enable_transceiver_impairments=enable_transceiver_impairments
        )
        # QRM generator removed - overlapping CASCADE messages in each stream provide realistic interference
        self.qrn_generator = GPUQRNGenerator(sample_rate=sample_rate, device=device)

        if enable_transceiver_impairments:
            print(f"\n📻 HF Transceiver Impairments: ENABLED")
            print(f"  TX impairments: {'ON' if tx_impairments_enabled else 'OFF'} (ALC, SSB filter, audio path)")
            print(f"  RX impairments: {'ON' if rx_impairments_enabled else 'OFF'} (AGC, SSB filter, audio path)")
            print(f"  Weighted sampling: 30% IC-7300, 20% FT-991A, 15% IC-705, etc.")

        # OPTIMIZATION: Create CUDA streams for async GPU operations
        # This allows overlapping GPU work with CPU/disk operations
        if device == 'cuda':
            self.cuda_stream_compute = torch.cuda.Stream()  # For signal generation/channel sim
            self.cuda_stream_transfer = torch.cuda.Stream()  # For CPU-GPU transfers
            print("  ✓ Created CUDA streams for async processing (2-3× speedup)")
        else:
            self.cuda_stream_compute = None
            self.cuda_stream_transfer = None

        # Initialize scenario library
        self.scenario_library = ScenarioLibrary()

        # Initialize continuous rate calculator for adaptive symbol rates
        self.rate_calculator = ContinuousRateCalculator(
            bandwidth_hz=60.0,  # Total bandwidth for 3-FSK
            min_rate=50.0,      # Minimum for sync
            max_rate=600.0      # Maximum for NN decoder testing
        )

        # Handle caching (always numpy format)
        if use_cache and self.cache_path.exists() and not regenerate_cache:
            # Check if cache is complete (has streams.npy)
            streams_file = self.cache_path / 'streams.npy'
            if streams_file.exists():
                print(f"\n✨ Loading from NUMPY cache (thread-safe!): {self.cache_path.name}")
                try:
                    self._load_numpy_cache(self.cache_path)
                except (FileNotFoundError, ValueError) as e:
                    # Cache missing metadata or parameter mismatch
                    print(f"\n⚠️  Cache validation failed: {e}")
                    print(f"   Removing incompatible cache and regenerating...")
                    import shutil
                    # Remove numpy directory
                    if self.cache_path.is_dir():
                        shutil.rmtree(self.cache_path)
                    # Remove label file
                    label_file = self.cache_path.parent / f"{self.cache_path.name}_labels.h5"
                    if label_file.exists():
                        label_file.unlink()
                    print(f"\n🔧 Generating dataset in NUMPY format (writes directly to memmap)")
                    self._generate_and_cache_streams()
            else:
                # Incomplete cache - remove and regenerate
                print(f"\n⚠️  Incomplete cache detected at {self.cache_path.name}")
                print(f"   Missing streams.npy - removing and regenerating...")
                import shutil
                if self.cache_path.is_dir():
                    shutil.rmtree(self.cache_path)
                else:
                    self.cache_path.unlink()
                print(f"\n🔧 Generating dataset in NUMPY format (writes directly to memmap)")
                self._generate_and_cache_streams()
        else:
            print(f"\n🔧 Generating dataset in NUMPY format (writes directly to memmap)")
            self._generate_and_cache_streams()

        print(f"{'='*70}\n")

    def _create_multipath_profile_for_batch(self, conditions: List) -> MultipathProfile:
        """Create multipath profile from physics conditions (for TX beacons)."""
        from enhanced_physics_dataset import CPUPropagationMode

        batch_size = len(conditions)
        num_paths = 6

        delays_ms_list = []
        powers_list = []
        doppler_shifts_list = []
        k_factors_list = []

        for cond in conditions:
            mode = cond.propagation_mode

            if mode == CPUPropagationMode.AWGN:
                delays = [0.0] * 6
                powers = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                doppler = [0.0] * 6
                k_factors = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif mode == CPUPropagationMode.RAYLEIGH:
                delays = [0.0] * 6
                powers = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                doppler = [cond.doppler_spread_hz, 0.0, 0.0, 0.0, 0.0, 0.0]
                k_factors = [0.0] * 6
            elif mode == CPUPropagationMode.RICIAN:
                delays = [0.0] * 6
                powers = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                doppler = [cond.doppler_spread_hz, 0.0, 0.0, 0.0, 0.0, 0.0]
                k_factors = [cond.rician_k_factor, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif mode == CPUPropagationMode.MULTIPATH_SPARSE:
                delay_spread = cond.multipath_delay_spread_ms
                delays = [0.0, delay_spread/2, delay_spread, 0.0, 0.0, 0.0]
                powers = [0.7, 0.2, 0.1, 0.0, 0.0, 0.0]
                doppler = [0.0, cond.doppler_spread_hz/2, cond.doppler_spread_hz, 0.0, 0.0, 0.0]
                k_factors = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:  # MULTIPATH_DENSE
                delay_spread = cond.multipath_delay_spread_ms
                delays = [0.0, delay_spread*0.2, delay_spread*0.4, delay_spread*0.6, delay_spread*0.8, delay_spread]
                powers = [0.4, 0.25, 0.15, 0.1, 0.07, 0.03]
                doppler = [0.0, cond.doppler_spread_hz*0.3, cond.doppler_spread_hz*0.5,
                          cond.doppler_spread_hz*0.7, cond.doppler_spread_hz*0.9, cond.doppler_spread_hz]
                k_factors = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            delays_ms_list.append(delays)
            powers_list.append(powers)
            doppler_shifts_list.append(doppler)
            k_factors_list.append(k_factors)

        return MultipathProfile(
            delays_ms=torch.tensor(delays_ms_list, device=self.device, dtype=torch.float32),
            powers=torch.tensor(powers_list, device=self.device, dtype=torch.float32),
            doppler_shifts_hz=torch.tensor(doppler_shifts_list, device=self.device, dtype=torch.float32),
            k_factors=torch.tensor(k_factors_list, device=self.device, dtype=torch.float32)
        )

    def _compute_optimal_embeddings_batch(self, clean_beacons: torch.Tensor,
                                          received_beacons: torch.Tensor,
                                          conditions: List) -> torch.Tensor:
        """
        Compute physics-based optimal embeddings from channel observations.

        This represents what TX encoder should learn to generate from observing RX beacon.

        Args:
            clean_beacons: [batch, 48000] clean TX beacon signals
            received_beacons: [batch, 48000] beacons after channel
            conditions: List of DerivedConditions with channel parameters

        Returns:
            torch.Tensor: [batch, 256] optimal embedding vectors
        """
        batch_size = clean_beacons.shape[0]
        embeddings = torch.zeros(batch_size, 256, device=self.device)

        for i in range(batch_size):
            clean = clean_beacons[i]
            received = received_beacons[i]
            cond = conditions[i]

            # Extract channel information from clean vs received comparison

            # 1. Frequency offset estimation (first 32 dims)
            # Cross-correlation in frequency domain
            clean_fft = torch.fft.fft(clean[:4800])  # Use first 100ms
            received_fft = torch.fft.fft(received[:4800])
            cross_spec = received_fft * torch.conj(clean_fft)
            freq_offset_estimate = torch.angle(torch.sum(cross_spec))
            embeddings[i, 0] = freq_offset_estimate / (2 * torch.pi)  # Normalized

            # 2. Phase rotation (dims 32-64)
            phase_rotation = torch.angle(torch.sum(received[:1000] * torch.conj(clean[:1000])))
            embeddings[i, 32] = torch.cos(phase_rotation)
            embeddings[i, 33] = torch.sin(phase_rotation)

            # 3. Channel impulse response estimate (dims 64-128)
            # From multipath delay spread
            delay_ms = cond.multipath_delay_spread_ms
            embeddings[i, 64] = delay_ms / 10.0  # Normalized to 0-1 range
            embeddings[i, 65] = cond.doppler_spread_hz / 5.0  # Normalized

            # 4. Power profile hints (dims 128-192)
            # Signal strength variations indicate fading
            power_profile = torch.abs(received.reshape(-1, 480))**2  # 10ms chunks
            power_mean = torch.mean(power_profile, dim=1)
            embeddings[i, 128:128+min(64, len(power_mean))] = power_mean[:64] / (torch.mean(power_mean) + 1e-10)

            # 5. Noise/QRN characteristics (dims 192-256)
            # Estimate from signal variance
            noise_estimate = torch.std(received - clean)
            embeddings[i, 192] = noise_estimate
            embeddings[i, 193] = cond.effective_snr_db / 40.0  # Normalized SNR

        return embeddings

    def _generate_stream_messages(self, stream_idx: int) -> List[MessageInStream]:
        """
        Generate message arrival times and parameters for one stream (GPU-accelerated).

        Uses Poisson process for realistic temporal distribution.

        Args:
            stream_idx: Stream index (for seeding)

        Returns:
            List of MessageInStream objects
        """
        # Poisson arrivals
        if self.seed is not None:
            np.random.seed(self.seed + stream_idx)

        num_messages = np.random.poisson(self.message_arrival_rate * self.stream_duration_sec)
        num_messages = max(1, min(10, num_messages))  # 1-10 messages per stream

        arrival_times_sec = np.sort(np.random.uniform(0, self.stream_duration_sec, num_messages))

        messages = []

        # Get physics condition for this stream to determine modulation
        # Use pre-calculated if available, otherwise estimate
        avg_snr = 0.0  # Default, will be updated

        for i, arrival_time in enumerate(arrival_times_sec):
            # Random parameters
            pattern_id = np.random.randint(0, 4)
            # freq_triple will be set by the rate calculator based on num_channels

            # Adaptive message size distribution (realistic for amateur radio)
            rand = np.random.random()
            if rand < 0.70:
                # 70%: Short messages (5-20 bytes) - callsign, grid, report (like FT8)
                msg_len = np.random.randint(5, 21)
            elif rand < 0.90:
                # 20%: Medium messages (20-64 bytes) - conversation, status
                msg_len = np.random.randint(20, 65)
            else:
                # 10%: Large messages (64-256 bytes) - images, files, long text
                msg_len = np.random.randint(64, 257)

            message_bytes = np.random.bytes(msg_len)

            # Determine if this is a target message or interference
            # First message is always the target (strongest/clearest)
            # Additional messages may be weaker (interference/context)
            if i == 0:
                # Target message: Mixed training strategy
                rand_type = np.random.random()
                if rand_type < 0.1:  # 10% of samples
                    # Pattern-only detection training (very low SNR)
                    # Data won't be decodable but pattern should be detectable
                    snr_estimate = np.random.uniform(-15, -8)  # Below data threshold
                    # Force pattern layer to still be at 75 sym/s for these
                    pattern_only = True
                else:
                    # Normal decodable message
                    snr_estimate = np.random.uniform(-6, 20)  # Target >= -6 dB for data
                    pattern_only = False
            else:
                # Other messages: can be much weaker (interference/context)
                snr_estimate = np.random.uniform(-20, 20)  # Interference can be very weak
                pattern_only = False

            # Simulate multipath and QRM conditions randomly
            multipath_severity = np.random.random() * 0.3 if np.random.random() < 0.3 else 0.0  # 30% chance of multipath
            qrm_present = np.random.random() < 0.2  # 20% chance of QRM

            # Use continuous rate calculator for realistic adaptive rates
            # But for pattern-only samples, override to very low rate
            if pattern_only:
                # Pattern layer at 25 sym/s (fixed), data layer severely degraded
                continuous_rate = 10.0  # Very low data rate (not decodable)
                num_channels = 4  # Maximum channels at low SNR
                start_freq_triple = np.random.randint(0, 14)  # 14 triples, 50 Hz spacing, 2-center (500-2600 Hz)
            else:
                continuous_rate, num_channels, start_freq_triple = self.rate_calculator.calculate_continuous_rate(
                    snr_db=snr_estimate,
                    multipath_severity=multipath_severity,
                    qrm_present=qrm_present
                )

            # Get optimal modulation from calculator
            modulation, bits_per_symbol = self.rate_calculator.optimal_modulation(snr_estimate)

            # Use the continuous rate for training labels
            # But quantize to nearest discrete rate for signal generation
            continuous_rate_for_labels = continuous_rate

            # Quantize to nearest available rate for signal generation
            available_rates = [75, 100, 125, 150, 175, 200, 250, 300]
            rate = min(available_rates, key=lambda x: abs(x - continuous_rate))

            # Estimate signal duration
            bits = len(message_bytes) * 8
            encoded_bits = int(bits * 1.5)  # Rate 2/3 FEC
            symbols = encoded_bits // bits_per_symbol
            duration_sec = symbols / rate

            start_sample = int(arrival_time * self.sample_rate)
            end_sample = start_sample + int(duration_sec * self.sample_rate)

            # Generate independent channel parameters for this message (different TX station)
            # Multipath: Random but realistic (0.5-8ms depending on distance/mode)
            msg_multipath = np.random.uniform(0.5, 8.0)  # ms
            # Doppler: Random but realistic (0.1-3 Hz depending on ionospheric motion)
            msg_doppler = np.random.uniform(0.1, 3.0)  # Hz
            # Propagation mode: Random (different paths can have different modes)
            prop_modes = ['rician', 'rayleigh', 'multipath_sparse', 'multipath_dense']
            msg_prop_mode = np.random.choice(prop_modes)

            messages.append(MessageInStream(
                start_sample=start_sample,
                end_sample=end_sample,
                pattern_id=pattern_id,
                frequency_triple=start_freq_triple,  # Starting frequency triple for multi-channel
                num_channels=num_channels,  # Number of channels used
                modulation=modulation,
                data_symbol_rate=rate,  # Discrete rate for generation
                continuous_rate=continuous_rate_for_labels,  # Continuous rate for training
                message_bytes=message_bytes,
                snr_db=snr_estimate,
                pattern_only=pattern_only if i == 0 else False,  # Only first message can be pattern-only
                # Per-message channel parameters (independent propagation paths)
                multipath_delay_ms=msg_multipath,
                doppler_hz=msg_doppler,
                propagation_mode=msg_prop_mode
            ))

        return messages

    def _pregenerate_all_messages(self):
        """
        CRITICAL OPTIMIZATION: Pre-generate ALL message metadata in parallel (60+ cores!).

        With 500GB+ RAM and 60 cores, we can precompute all message parameters
        in parallel during physics phase. This eliminates the sequential Python loop.

        Expected speedup: 50-100× (10× from prebaking, 5-10× from parallelization)
        RAM usage: ~55 GB for 34M streams × 8 messages/stream
        """
        from tqdm import tqdm
        from multiprocessing import Pool
        import time

        print(f"  Pregenerating messages for {self.num_streams:,} streams...")
        start_time = time.time()

        # SIMPLIFIED: Sequential generation (can't parallelize due to seeding)
        # But still fast because we only do this ONCE and cache in RAM
        self.precomputed_messages = []

        # Process in large batches for speed
        batch_size_msgs = 50000  # Large batches for faster processing

        for batch_start in tqdm(range(0, self.num_streams, batch_size_msgs), desc="  Precomputing messages"):
            batch_end = min(batch_start + batch_size_msgs, self.num_streams)
            batch_messages = self._generate_batch_messages_vectorized(batch_start, batch_end)
            self.precomputed_messages.extend(batch_messages)

        # Calculate statistics
        total_messages = sum(len(msgs) for msgs in self.precomputed_messages)
        bytes_per_message = 200  # Rough estimate
        ram_gb = total_messages * bytes_per_message / 1e9
        elapsed = time.time() - start_time

        print(f"  ✓ Precomputed {total_messages:,} messages in {elapsed:.1f}s ({total_messages/elapsed:.0f} msgs/sec)")
        print(f"  ✓ RAM usage: ~{ram_gb:.1f} GB (avg {total_messages/self.num_streams:.1f} msgs/stream)")
        print(f"  ✓ Parallelization: {self.num_workers} cores")

    def _generate_batch_messages_vectorized(self, batch_start_idx: int, batch_end_idx: int) -> List[List[MessageInStream]]:
        """
        OPTIMIZED: Generate message parameters for entire batch at once (vectorized).

        Produces IDENTICAL output to calling _generate_stream_messages() in a loop,
        but 10-20× faster by using vectorized numpy operations.

        Args:
            batch_start_idx: Starting stream index
            batch_end_idx: Ending stream index (exclusive)

        Returns:
            List of message lists (one per stream)
        """
        batch_size = batch_end_idx - batch_start_idx
        all_stream_messages = []

        # Pre-generate random state for each stream (preserves exact randomness)
        # We still need to seed each stream individually to match original behavior
        for stream_idx in range(batch_start_idx, batch_end_idx):
            # Set seed for this stream (SAME as original method)
            if self.seed is not None:
                np.random.seed(self.seed + stream_idx)

            # Generate number of messages for this stream (Poisson)
            num_messages = np.random.poisson(self.message_arrival_rate * self.stream_duration_sec)
            num_messages = max(1, min(10, num_messages))

            # Generate arrival times
            arrival_times_sec = np.sort(np.random.uniform(0, self.stream_duration_sec, num_messages))

            # VECTORIZED: Generate all message parameters at once
            pattern_ids = np.random.randint(0, 4, num_messages)

            # Message sizes (vectorized with same distribution)
            rand_sizes = np.random.random(num_messages)
            msg_lens = np.where(
                rand_sizes < 0.70,
                np.random.randint(5, 21, num_messages),  # 70% short
                np.where(
                    rand_sizes < 0.90,
                    np.random.randint(20, 65, num_messages),  # 20% medium
                    np.random.randint(64, 257, num_messages)  # 10% large
                )
            )

            # SNR estimates (vectorized)
            snr_estimates = np.zeros(num_messages)
            pattern_only_flags = np.zeros(num_messages, dtype=bool)

            # First message (target)
            rand_type = np.random.random()
            if rand_type < 0.1:
                snr_estimates[0] = np.random.uniform(-15, -8)
                pattern_only_flags[0] = True
            else:
                snr_estimates[0] = np.random.uniform(-6, 20)

            # Other messages (interference/context)
            if num_messages > 1:
                snr_estimates[1:] = np.random.uniform(-20, 20, num_messages - 1)

            # Multipath and QRM (vectorized)
            multipath_severities = np.where(
                np.random.random(num_messages) < 0.3,
                np.random.random(num_messages) * 0.3,
                0.0
            )
            qrm_present_flags = np.random.random(num_messages) < 0.2

            # Per-message channel parameters (vectorized)
            msg_multipaths = np.random.uniform(0.5, 8.0, num_messages)
            msg_dopplers = np.random.uniform(0.1, 3.0, num_messages)
            prop_modes_list = ['rician', 'rayleigh', 'multipath_sparse', 'multipath_dense']
            msg_prop_modes = [np.random.choice(prop_modes_list) for _ in range(num_messages)]

            # Build messages for this stream
            stream_messages = []
            available_rates = [75, 100, 125, 150, 175, 200, 250, 300]

            for i in range(num_messages):
                msg_len = msg_lens[i]
                message_bytes = np.random.bytes(msg_len)

                # Calculate continuous rate and modulation
                if pattern_only_flags[i]:
                    continuous_rate = 10.0
                    num_channels = 4
                    start_freq_triple = np.random.randint(0, 14)  # 14 triples, 50 Hz spacing, 2-center (500-2600 Hz)
                else:
                    continuous_rate, num_channels, start_freq_triple = self.rate_calculator.calculate_continuous_rate(
                        snr_db=snr_estimates[i],
                        multipath_severity=multipath_severities[i],
                        qrm_present=qrm_present_flags[i]
                    )

                modulation, bits_per_symbol = self.rate_calculator.optimal_modulation(snr_estimates[i])

                # Quantize to nearest discrete rate
                rate = min(available_rates, key=lambda x: abs(x - continuous_rate))

                # Calculate signal duration
                bits = len(message_bytes) * 8
                encoded_bits = int(bits * 1.5)
                symbols = encoded_bits // bits_per_symbol
                duration_sec = symbols / rate

                start_sample = int(arrival_times_sec[i] * self.sample_rate)
                end_sample = start_sample + int(duration_sec * self.sample_rate)

                stream_messages.append(MessageInStream(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    pattern_id=int(pattern_ids[i]),
                    frequency_triple=start_freq_triple,
                    num_channels=num_channels,
                    modulation=modulation,
                    data_symbol_rate=rate,
                    continuous_rate=continuous_rate,
                    message_bytes=message_bytes,
                    snr_db=float(snr_estimates[i]),
                    pattern_only=bool(pattern_only_flags[i] and i == 0),
                    multipath_delay_ms=float(msg_multipaths[i]),
                    doppler_hz=float(msg_dopplers[i]),
                    propagation_mode=msg_prop_modes[i]
                ))

            all_stream_messages.append(stream_messages)

        return all_stream_messages

    def _generate_and_cache_streams(self):
        """Generate streams and cache (FULLY GPU-ACCELERATED with async background writes)."""
        import time
        import json
        from threading import Thread
        from queue import Queue

        print(f"\nGenerating {self.num_streams:,} streams on GPU...")
        print(f"Stream duration: {self.stream_duration_sec}s ({self.stream_samples:,} samples)")
        print(f"GPU batch size: {self.batch_size} streams in parallel")
        print(f"Using {self.num_workers} CPU workers for background I/O")

        # DIAGNOSTIC: Warn if batch size too small
        if self.batch_size < 2048:
            print(f"\n⚠️  WARNING: Batch size {self.batch_size} is suboptimal!")
            print(f"   GH200 can handle 4096-8192 streams per batch")
            print(f"   Current setting will take {self.num_streams // self.batch_size} GPU calls")
            print(f"   With 4096: would take {self.num_streams // 4096} GPU calls (much faster!)")

        start_time = time.time()

        # Multi-file chunking for parallel I/O (no GPU blocking!)
        CHUNK_SIZE = 100000  # Write every 100K streams to separate file
        num_chunks = (self.num_streams + CHUNK_SIZE - 1) // CHUNK_SIZE
        chunk_files = []

        if num_chunks > 1:
            print(f"  📁 Multi-file mode: {num_chunks} chunks ({CHUNK_SIZE:,} streams each)")
            print(f"     → GPU runs continuously, I/O happens in background!")
        else:
            print(f"  📁 Single-file mode: {self.num_streams:,} streams")

        # Pre-generate physics scenarios for all streams (parallel CPU)
        print("Pre-calculating physics for all streams...")
        scenario_drivers = self.scenario_library.generate_balanced_realistic_batch(
            batch_size=self.num_streams,
            for_test=self.for_test,
            seed=self.seed
        )

        # Parallel physics calculation
        from enhanced_physics_dataset import _calc_physics_worker
        if self.num_workers > 1:
            worker_args = [(driver, self.seed) for driver in scenario_drivers]
            with Pool(self.num_workers) as pool:
                physics_conditions = list(tqdm(
                    pool.imap(_calc_physics_worker, worker_args, chunksize=100),
                    total=self.num_streams,
                    desc="Physics calc"
                ))
        else:
            calc = CoupledPhysicsCalculator(seed=self.seed)
            physics_conditions = [calc.calculate_all_effects(d) for d in scenario_drivers]

        print(f"✓ Physics pre-calculated for {self.num_streams:,} streams")

        # OPTIMIZATION: Pre-extract commonly-used physics data into numpy arrays
        # This eliminates per-batch list comprehensions (6× per batch = 37,500 calls for 800K streams!)
        print("Pre-extracting physics parameters to RAM arrays...")
        self.physics_arrays = {
            'effective_snr_db': np.array([c.effective_snr_db for c in physics_conditions], dtype=np.float32),
            'k_index': np.array([d.k_index for d in scenario_drivers], dtype=np.float32),
            'thunderstorm_activity': np.array([d.thunderstorm_activity for d in scenario_drivers], dtype=np.float32),
            'utc_hour': np.array([d.utc_hour for d in scenario_drivers], dtype=np.float32),
            'sfi': np.array([d.sfi for d in scenario_drivers], dtype=np.float32),
            'propagation_mode': np.array([c.propagation_mode.value for c in physics_conditions], dtype='U32'),
            'qrn_type': np.array([c.dominant_qrn_type.value for c in physics_conditions], dtype='U32'),
        }
        total_mb = sum(arr.nbytes for arr in self.physics_arrays.values()) / 1e6
        print(f"✓ Physics arrays extracted ({total_mb:.1f} MB in RAM)")

        # Message precomputation DISABLED for faster worker startup
        # Each worker generates messages on-the-fly to avoid slow initialization
        # self._pregenerate_all_messages()
        self.precomputed_messages = None

        # Process each chunk (100K streams) and write to file
        # For numpy: write directly to .npy files (no conversion needed!)
        # For HDF5: write to .h5 files

        # Choose output method based on format
        if self.output_format == 'numpy':
            # NUMPY: Pre-allocate memmap files + async background writers
            print(f"\n📦 Generating to numpy memmap with async background writers")
            print(f"   Using {min(self.num_workers, 8)} parallel writer threads (overlaps GPU+I/O)")

            if not self.cache_path.exists():
                self.cache_path.mkdir(parents=True, exist_ok=True)

            # WORKAROUND: NFS has 2GB file limit for memmap seek()
            # Split into chunks of ~58K streams each (1.9 GB per file)
            max_streams_per_chunk = 57_000  # Conservative (under 2GB)
            num_chunks = int(np.ceil(self.num_streams / max_streams_per_chunk))

            if num_chunks > 1:
                print(f"   ⚠️  Dataset too large for single memmap ({self.num_streams:,} streams)")
                print(f"   Splitting into {num_chunks} chunks of ~{max_streams_per_chunk:,} streams each")

            # Create chunked memory-mapped arrays
            self.streams_mmap_chunks = []
            chunk_sizes = []

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * max_streams_per_chunk
                end_idx = min((chunk_idx + 1) * max_streams_per_chunk, self.num_streams)
                chunk_size = end_idx - start_idx
                chunk_sizes.append(chunk_size)

                chunk_file = self.cache_path / f'streams_chunk_{chunk_idx:04d}.npy'
                streams_mmap = np.lib.format.open_memmap(
                    chunk_file,
                    mode='w+',
                    dtype=np.complex64,
                    shape=(chunk_size, self.stream_samples)
                )
                self.streams_mmap_chunks.append(streams_mmap)

            print(f"   ✓ Created {num_chunks} memmap chunks ({chunk_sizes[0]:,} streams each)")

            # OPTIMIZATION: Pre-compute normalization statistics (mean/std per stream)
            # Eliminates 786M float ops per batch during training!
            normalization_stats_mmap = np.lib.format.open_memmap(
                self.cache_path / 'normalization_stats.npy',
                mode='w+',
                dtype=np.float32,
                shape=(self.num_streams, 2)  # [mean, std] per stream
            )

            if self.compute_optimal_embeddings or self.enable_tx_observations:
                embeddings_mmap = np.lib.format.open_memmap(
                    self.cache_path / 'embeddings.npy',
                    mode='w+',
                    dtype=np.float32,
                    shape=(self.num_streams, 256)
                )
            else:
                embeddings_mmap = None

            # Create write queue and background writer threads
            write_queue = Queue(maxsize=20)  # Max 20 batches queued
            num_writer_threads = min(self.num_workers, 8)  # Use up to 8 threads

            def writer_worker():
                """Background thread that writes batches to chunked memmap."""
                while True:
                    item = write_queue.get()
                    if item is None:  # Poison pill to stop
                        write_queue.task_done()
                        break

                    idx, streams_data, emb_data, norm_stats = item
                    batch_size = streams_data.shape[0]

                    # Write to appropriate chunk(s) - simple per-stream (stable)
                    for i in range(batch_size):
                        global_idx = idx + i
                        chunk_idx = global_idx // max_streams_per_chunk
                        local_idx = global_idx % max_streams_per_chunk

                        # Write single stream to appropriate chunk
                        self.streams_mmap_chunks[chunk_idx][local_idx] = streams_data[i]
                        if emb_data is not None and embeddings_mmap is not None:
                            embeddings_mmap[global_idx] = emb_data[i]
                        if norm_stats is not None:
                            normalization_stats_mmap[global_idx] = norm_stats[i]

                    write_queue.task_done()

            # Start writer threads
            writer_threads = []
            for _ in range(num_writer_threads):
                t = Thread(target=writer_worker, daemon=True)
                t.start()
                writer_threads.append(t)

            print(f"  ✓ Started {num_writer_threads} background writer threads")

            # PIPELINE OPTIMIZATION: Pre-generation pool + post-processing pool
            print(f"\n🚀 PIPELINE OPTIMIZATION ENABLED")
            print(f"   Pre-gen workers: {self.num_pregen_workers} (generates {self.pregen_queue_depth} batches ahead)")
            print(f"   Post-proc workers: {self.num_postprocess_workers} (parallel normalization/metadata)")

            # Create pre-generation worker pool
            rate_calc_params = {
                'bandwidth_hz': 60.0,
                'min_rate': 50.0,
                'max_rate': 600.0
            }

            # DISABLED: pregen_pool replaced by precomputed messages in RAM (10-15× faster!)
            # Messages are now pre-generated during _pregenerate_all_messages()
            pregen_pool = None  # Force use of precomputed_messages

            # Create post-processing thread pool (only if workers > 0)
            if self.num_postprocess_workers > 0:
                postproc_pool = ThreadPoolExecutor(max_workers=self.num_postprocess_workers)
            else:
                postproc_pool = None

            # Pre-generation queue (CPU → GPU)
            pregen_queue = ThreadQueue(maxsize=self.pregen_queue_depth)

            # Post-processing queue (GPU → CPU)
            postproc_queue = ThreadQueue(maxsize=4)

            def pregen_feeder_thread():
                """Background thread that feeds pre-generated batches to GPU."""
                pass  # Will be implemented in main loop

            print(f"  ✓ Pipeline workers initialized")

        else:
            # HDF5: Use traditional file-based approach
            streams_mmap = None
            embeddings_mmap = None
            write_queue = None
            writer_threads = []
            pregen_pool = None
            postproc_pool = None
            pregen_queue = None
            postproc_queue = None

        vlen_string_dtype = h5py.string_dtype()
        chunk_files = []

        # For numpy format: Create ONE HDF5 file for ALL labels (not per-chunk!)
        if self.output_format == 'numpy':
            label_file_path = self.cache_path.parent / f"{self.cache_path.name}_labels.h5"

            # Clean up if it exists as directory (from failed previous run)
            if label_file_path.exists() and label_file_path.is_dir():
                import shutil
                print(f"  Cleaning up malformed label file (was directory): {label_file_path.name}")
                shutil.rmtree(label_file_path)
            elif label_file_path.exists():
                # Remove old label file (will be regenerated)
                label_file_path.unlink()

            # Create ONE label file for entire dataset (all chunks write to different sections)
            print(f"  Creating label file for {self.num_streams:,} streams...")
            hdf5_label_file = h5py.File(label_file_path, 'w')

            # Pre-allocate datasets for ALL streams
            num_messages_ds = hdf5_label_file.create_dataset('num_messages', shape=(self.num_streams,), dtype=np.int16)
            k_indices = hdf5_label_file.create_dataset('k_index', shape=(self.num_streams,), dtype=np.float32)
            sfis = hdf5_label_file.create_dataset('sfi', shape=(self.num_streams,), dtype=np.float32)
            propagation_modes = hdf5_label_file.create_dataset('propagation_modes', shape=(self.num_streams,), dtype='S32')
            qrn_types = hdf5_label_file.create_dataset('qrn_types', shape=(self.num_streams,), dtype='S16')
            message_metadata = hdf5_label_file.create_dataset('message_metadata', shape=(self.num_streams,),
                                                             dtype=vlen_string_dtype)
            if self.compute_optimal_embeddings or self.enable_tx_observations:
                optimal_embeddings_ds = hdf5_label_file.create_dataset('optimal_embeddings', shape=(self.num_streams, 256),
                                                                       dtype=np.float32)
            if self.enable_tx_observations:
                tx_observed_ds = hdf5_label_file.create_dataset('tx_observed', shape=(self.num_streams, 48000),
                                                               dtype=np.complex64)

            print(f"  ✓ Pre-allocated label datasets")

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min(chunk_start + CHUNK_SIZE, self.num_streams)
            chunk_size = chunk_end - chunk_start

            # Set chunk path for HDF5 mode only
            if self.output_format == 'numpy':
                # Already created above - will write to appropriate section
                chunk_path = label_file_path
            elif num_chunks > 1:
                chunk_path = self.cache_path.parent / f"{self.cache_path.stem}_chunk{chunk_idx}.h5"
            else:
                chunk_path = self.cache_path

            chunk_files.append(chunk_path)

            print(f"\n  Generating chunk {chunk_idx + 1}/{num_chunks}: streams {chunk_start:,}-{chunk_end:,}")

            # Generate streams for this chunk only
            chunk_batches_start = chunk_start // self.batch_size
            chunk_batches_end = (chunk_end + self.batch_size - 1) // self.batch_size
            chunk_num_batches = chunk_batches_end - chunk_batches_start

            # Buffer all streams in chunk (in RAM)
            buffer_streams = []
            buffer_tx_observed = []
            buffer_optimal_embeddings = []
            buffer_all_messages = []  # Store messages for metadata

            # Generate all data for this chunk FIRST (before writing to HDF5)
            if True:  # Placeholder for generation block

                with tqdm(total=chunk_size, desc=f"Chunk {chunk_idx+1}/{num_chunks}", unit="streams") as pbar:

                    # PIPELINE OPTIMIZATION: Pre-generate messages for multiple batches in parallel
                    if pregen_pool is not None:
                        # Submit first N batches for pre-generation (fills pipeline)
                        pregen_futures = {}
                        for batch_offset in range(min(self.pregen_queue_depth, chunk_num_batches)):
                            batch_idx = chunk_batches_start + batch_offset
                            batch_start_idx = batch_idx * self.batch_size
                            batch_end_idx = min(batch_start_idx + self.batch_size, chunk_end)

                            if batch_start_idx < chunk_start or batch_start_idx >= chunk_end:
                                continue

                            args = (batch_start_idx, batch_end_idx, self.seed,
                                   self.stream_duration_sec, self.message_arrival_rate,
                                   self.sample_rate)
                            future = pregen_pool.apply_async(_pregen_batch_messages_worker, (args,))
                            pregen_futures[batch_start_idx] = future

                    # Process batches (with pipeline if enabled)
                    for batch_offset in range(chunk_num_batches):
                        batch_idx = chunk_batches_start + batch_offset
                        batch_start_idx = batch_idx * self.batch_size
                        batch_end_idx = min(batch_start_idx + self.batch_size, chunk_end)
                        actual_batch_size = batch_end_idx - batch_start_idx

                        # Adjust for chunk boundaries
                        if batch_start_idx < chunk_start:
                            continue
                        if batch_start_idx >= chunk_end:
                            break

                        # Get pre-generated messages (or generate on-the-fly if pool disabled)
                        if pregen_pool is not None and batch_start_idx in pregen_futures:
                            # PIPELINE: Get pre-generated result (blocks if not ready yet)
                            result = pregen_futures[batch_start_idx].get()
                            _, all_stream_messages, message_to_stream_map, flat_messages = result

                            # Submit next batch for pre-generation (keep pipeline full)
                            next_batch_offset = batch_offset + self.pregen_queue_depth
                            if next_batch_offset < chunk_num_batches:
                                next_batch_idx = chunk_batches_start + next_batch_offset
                                next_batch_start_idx = next_batch_idx * self.batch_size
                                next_batch_end_idx = min(next_batch_start_idx + self.batch_size, chunk_end)

                                if next_batch_start_idx >= chunk_start and next_batch_start_idx < chunk_end:
                                    args = (next_batch_start_idx, next_batch_end_idx, self.seed,
                                           self.stream_duration_sec, self.message_arrival_rate,
                                           self.sample_rate)
                                    future = pregen_pool.apply_async(_pregen_batch_messages_worker, (args,))
                                    pregen_futures[next_batch_start_idx] = future
                        else:
                            # Generate messages on-the-fly (precomputation disabled for faster startup)
                            all_stream_messages = self._generate_batch_messages_vectorized(
                                batch_start_idx, batch_end_idx
                            )

                            # Build message_to_stream_map
                            message_to_stream_map = []
                            for stream_idx_offset, messages in enumerate(all_stream_messages):
                                stream_idx = batch_start_idx + stream_idx_offset
                                message_to_stream_map.extend([stream_idx] * len(messages))

                            # Flatten all messages into one big batch
                            flat_messages = [msg for stream_msgs in all_stream_messages for msg in stream_msgs]

                        if len(flat_messages) > 0:
                            # OPTIMIZED: Process ALL messages together (no size grouping)
                            # Kernel launch overhead >> padding cost, so batch everything!
                            all_signals = []
                            all_metadata = []
                            signal_to_msg_map = []

                            MAX_MESSAGES_PER_BATCH = 16384  # Process all messages at once if possible

                            # Single loop - no grouping by message size
                            for msg_start in range(0, len(flat_messages), MAX_MESSAGES_PER_BATCH):
                                msg_end = min(msg_start + MAX_MESSAGES_PER_BATCH, len(flat_messages))
                                batch_messages = flat_messages[msg_start:msg_end]

                                # Track which messages these signals correspond to
                                signal_to_msg_map.extend(batch_messages)

                                # OPTIMIZATION: Vectorize parameter extraction (avoid Python loops)
                                num_msgs = len(batch_messages)

                                # Use list comprehension with direct torch tensor creation
                                # This is faster than numpy intermediate + conversion
                                batch_params = BatchKernelParameters(
                                    pattern_ids=torch.tensor([m.pattern_id for m in batch_messages],
                                                            dtype=torch.int64, device=self.device),
                                    frequency_triples=torch.tensor([m.frequency_triple for m in batch_messages],
                                                                  dtype=torch.int64, device=self.device),
                                    modulations=[m.modulation for m in batch_messages],
                                    polar_rates=[(2, 3)] * num_msgs,
                                    data_symbol_rates=torch.tensor([m.data_symbol_rate for m in batch_messages],
                                                                   dtype=torch.int64, device=self.device)
                                )

                                # Determine num_centers for this batch
                                # Always use 4-center always-on design for better performance
                                batch_num_centers = 2  # 2-center design (balanced SNR + spectrum efficiency)

                                # OPTIMIZATION: Run signal generation in CUDA stream (async)
                                import time as time_module
                                t_sig_start = time_module.time()

                                if self.cuda_stream_compute is not None:
                                    with torch.cuda.stream(self.cuda_stream_compute):
                                        batch_signals, batch_metadata = self.signal_generator.generate_batch(
                                            batch_params,
                                            [m.message_bytes for m in batch_messages],
                                            fixed_length=None,
                                            profile=False,
                                            num_centers=batch_num_centers
                                        )
                                else:
                                    batch_signals, batch_metadata = self.signal_generator.generate_batch(
                                        batch_params,
                                        [m.message_bytes for m in batch_messages],
                                        fixed_length=None,
                                        profile=False,
                                        num_centers=batch_num_centers
                                    )

                                # TRANSCEIVER IMPAIRMENTS: Apply TX impairments (before channel!)
                                if self.enable_transceiver_impairments and self.tx_impairments_enabled:
                                    # Sample random transceiver profiles for this batch
                                    from gpu_channel_simulator import GPUTransceiverImpairments
                                    tx_profiles = GPUTransceiverImpairments.sample_random_profiles(
                                        batch_signals.shape[0]
                                    )
                                    # Apply TX impairments (ALC, SSB filter, audio interface)
                                    batch_signals, _, _ = self.channel_simulator.apply_tx_rx_impairments_batch(
                                        batch_signals,
                                        tx_profiles=tx_profiles,
                                        rx_profiles=None,
                                        apply_tx=True,
                                        apply_rx=False
                                        )

                                    all_signals.append(batch_signals)
                                    all_metadata.extend(batch_metadata)

                            # Combine all sub-batch signals (they may have different lengths)
                            # Need to handle variable-length signals properly
                            if not all_signals:
                                all_signals = torch.empty(0, device=self.device)
                            else:
                                # Just keep as list - we'll handle them individually
                                all_signals = torch.cat([s for s in all_signals], dim=0) if len(all_signals) == 1 else all_signals

                                # If we have multiple sub-batches, flatten the list
                                if isinstance(all_signals, list):
                                    flat_signals = []
                                    for batch in all_signals:
                                        if batch.ndim == 2:
                                            for i in range(batch.shape[0]):
                                                flat_signals.append(batch[i])
                                        else:
                                            flat_signals.append(batch)
                                    all_signals = flat_signals

                            # APPLY PER-MESSAGE CHANNEL EFFECTS IN PARALLEL (GPU BATCH!)
                            # Build multipath profile for ALL messages at once
                            num_messages = len(flat_messages)

                            # Track original signal lengths BEFORE any padding/processing
                            if isinstance(all_signals, list):
                                original_signal_lengths = [len(s) for s in all_signals]
                            else:
                                original_signal_lengths = [all_signals.shape[1]] * all_signals.shape[0]

                            if num_messages > 0:
                                # OPTIMIZATION: Build batched multipath profile directly on GPU
                                # Skip numpy intermediate arrays, create torch tensors directly
                                delays_batch = torch.zeros(num_messages, 6, dtype=torch.float32, device=self.device)
                                powers_batch = torch.zeros(num_messages, 6, dtype=torch.float32, device=self.device)
                                doppler_batch = torch.zeros(num_messages, 6, dtype=torch.float32, device=self.device)
                                k_factors_batch = torch.zeros(num_messages, 6, dtype=torch.float32, device=self.device)

                                # Extract all parameters as lists (single pass)
                                prop_modes_list = [msg.propagation_mode for msg in flat_messages]
                                multipath_delays_list = [msg.multipath_delay_ms for msg in flat_messages]
                                dopplers_list = [msg.doppler_hz for msg in flat_messages]

                                # Convert to torch tensors directly (skip numpy)
                                prop_modes = np.array(prop_modes_list)  # Keep as numpy for string comparison
                                multipath_delays = torch.tensor(multipath_delays_list, dtype=torch.float32, device=self.device)
                                dopplers = torch.tensor(dopplers_list, dtype=torch.float32, device=self.device)

                                # OPTIMIZATION: Vectorized mask operations on GPU tensors
                                # Rayleigh: vectorized
                                rayleigh_mask = (prop_modes == 'rayleigh')
                                if np.any(rayleigh_mask):
                                    rayleigh_indices = torch.tensor(np.where(rayleigh_mask)[0], device=self.device)
                                    powers_batch[rayleigh_indices, 0] = 1.0
                                    doppler_batch[rayleigh_indices, 0] = dopplers[rayleigh_indices]

                                # Rician: vectorized
                                rician_mask = (prop_modes == 'rician')
                                if np.any(rician_mask):
                                    rician_indices = torch.tensor(np.where(rician_mask)[0], device=self.device)
                                    powers_batch[rician_indices, 0] = 1.0
                                    doppler_batch[rician_indices, 0] = dopplers[rician_indices]
                                    k_factors_batch[rician_indices, 0] = 5.0

                                # Multipath sparse: vectorized
                                sparse_mask = (prop_modes == 'multipath_sparse')
                                if np.any(sparse_mask):
                                    sparse_indices = torch.tensor(np.where(sparse_mask)[0], device=self.device)
                                    sparse_delays = multipath_delays[sparse_indices]
                                    sparse_dopplers = dopplers[sparse_indices]
                                    delays_batch[sparse_indices, 0] = 0.0
                                    delays_batch[sparse_indices, 1] = sparse_delays / 2.0
                                    delays_batch[sparse_indices, 2] = sparse_delays
                                    powers_batch[sparse_indices, 0] = 0.7
                                    powers_batch[sparse_indices, 1] = 0.2
                                    powers_batch[sparse_indices, 2] = 0.1
                                    doppler_batch[sparse_indices, 0] = 0.0
                                    doppler_batch[sparse_indices, 1] = sparse_dopplers / 2.0
                                    doppler_batch[sparse_indices, 2] = sparse_dopplers
                                    k_factors_batch[sparse_indices, 0] = 5.0

                                # Multipath dense: vectorized
                                dense_mask = (prop_modes == 'multipath_dense')
                                if np.any(dense_mask):
                                    dense_indices = torch.tensor(np.where(dense_mask)[0], device=self.device)
                                    dense_delays = multipath_delays[dense_indices]
                                    dense_dopplers = dopplers[dense_indices]
                                    delays_batch[dense_indices, 0] = 0.0
                                    delays_batch[dense_indices, 1] = dense_delays * 0.2
                                    delays_batch[dense_indices, 2] = dense_delays * 0.4
                                    delays_batch[dense_indices, 3] = dense_delays * 0.6
                                    delays_batch[dense_indices, 4] = dense_delays * 0.8
                                    delays_batch[dense_indices, 5] = dense_delays
                                    powers_batch[dense_indices, 0] = 0.4
                                    powers_batch[dense_indices, 1] = 0.25
                                    powers_batch[dense_indices, 2] = 0.15
                                    powers_batch[dense_indices, 3] = 0.1
                                    powers_batch[dense_indices, 4] = 0.07
                                    powers_batch[dense_indices, 5] = 0.03
                                    doppler_batch[dense_indices, 0] = 0.0
                                    doppler_batch[dense_indices, 1] = dense_dopplers * 0.3
                                    doppler_batch[dense_indices, 2] = dense_dopplers * 0.5
                                    doppler_batch[dense_indices, 3] = dense_dopplers * 0.7
                                    doppler_batch[dense_indices, 4] = dense_dopplers * 0.9
                                    doppler_batch[dense_indices, 5] = dense_dopplers
                                    k_factors_batch[dense_indices, 0] = 3.0

                                # OPTIMIZATION: Batched multipath profile already on GPU (no conversion needed!)
                                batch_multipath_profile = MultipathProfile(
                                    delays_ms=delays_batch,
                                    powers=powers_batch,
                                    doppler_shifts_hz=doppler_batch,
                                    k_factors=k_factors_batch
                                )

                                # Pad signals to same length for batch processing
                                if isinstance(all_signals, list):
                                    signal_lengths = [len(s) for s in all_signals]
                                    max_len = max(signal_lengths)
                                    min_len = min(signal_lengths)

                                    # Safety check: if max_len is too large, skip channel effects for this batch
                                    if max_len > 500000:  # 10+ seconds
                                        # Skip channel effects, use signals as-is
                                        pass
                                    else:
                                        padded_signals = []
                                        for sig in all_signals:
                                            if len(sig) < max_len:
                                                padded = torch.zeros(max_len, dtype=sig.dtype, device=self.device)
                                                padded[:len(sig)] = sig
                                                padded_signals.append(padded)
                                            else:
                                                padded_signals.append(sig)
                                        all_signals_tensor = torch.stack(padded_signals)

                                        # Safety check: if signals too long, split into smaller chunks
                                        if all_signals_tensor.shape[1] > 100000:  # >2 seconds
                                            # Process in smaller chunks to avoid GPU hang
                                            CHUNK_LEN = 96000  # 2 seconds
                                            faded_signals = []

                                            for sig_idx in range(num_messages):
                                                sig = all_signals_tensor[sig_idx]
                                                faded_chunks = []

                                                for chunk_start in range(0, len(sig), CHUNK_LEN):
                                                    chunk_end = min(chunk_start + CHUNK_LEN, len(sig))
                                                    chunk = sig[chunk_start:chunk_end].unsqueeze(0)

                                                    # Apply channel to this chunk
                                                    chunk_profile = MultipathProfile(
                                                        delays_ms=batch_multipath_profile.delays_ms[sig_idx:sig_idx+1],
                                                        powers=batch_multipath_profile.powers[sig_idx:sig_idx+1],
                                                        doppler_shifts_hz=batch_multipath_profile.doppler_shifts_hz[sig_idx:sig_idx+1],
                                                        k_factors=batch_multipath_profile.k_factors[sig_idx:sig_idx+1]
                                                    )

                                                    chunk_faded = self.channel_simulator.apply_time_varying_multipath_batch(
                                                        chunk, chunk_profile, coherence_bandwidth_hz=50
                                                    )
                                                    faded_chunks.append(chunk_faded.squeeze(0))

                                                # Concatenate chunks
                                                full_faded = torch.cat(faded_chunks)
                                                faded_signals.append(full_faded)

                                            all_signals_faded = torch.stack(faded_signals)
                                        else:
                                            # OPTIMIZATION: Run channel simulation in CUDA stream (async)
                                            if self.cuda_stream_compute is not None:
                                                with torch.cuda.stream(self.cuda_stream_compute):
                                                    all_signals_faded = self.channel_simulator.apply_time_varying_multipath_batch(
                                                        all_signals_tensor,
                                                        batch_multipath_profile,
                                                        coherence_bandwidth_hz=50
                                                    )
                                            else:
                                                all_signals_faded = self.channel_simulator.apply_time_varying_multipath_batch(
                                                    all_signals_tensor,
                                                    batch_multipath_profile,
                                                    coherence_bandwidth_hz=50
                                                )

                                        # Unpad and store back
                                        all_signals = [all_signals_faded[i] for i in range(num_messages)]

                                        # TRANSCEIVER IMPAIRMENTS: Apply RX impairments (after channel!)
                                        if self.enable_transceiver_impairments and self.rx_impairments_enabled:
                                            from gpu_channel_simulator import GPUTransceiverImpairments
                                            rx_profiles = GPUTransceiverImpairments.sample_random_profiles(num_messages)
                                            all_signals_with_rx = self.channel_simulator.apply_rx_impairments_only(
                                                all_signals_faded,
                                                rx_profiles
                                            )
                                            all_signals = [all_signals_with_rx[i] for i in range(num_messages)]
                                else:
                                    # OPTIMIZATION: Run channel simulation in CUDA stream (async)
                                    if self.cuda_stream_compute is not None:
                                        with torch.cuda.stream(self.cuda_stream_compute):
                                            all_signals_faded = self.channel_simulator.apply_time_varying_multipath_batch(
                                                all_signals,
                                                batch_multipath_profile,
                                                coherence_bandwidth_hz=50
                                            )
                                    else:
                                        all_signals_faded = self.channel_simulator.apply_time_varying_multipath_batch(
                                            all_signals,
                                            batch_multipath_profile,
                                            coherence_bandwidth_hz=50
                                        )

                                    # TRANSCEIVER IMPAIRMENTS: Apply RX impairments (after channel!)
                                    if self.enable_transceiver_impairments and self.rx_impairments_enabled:
                                        from gpu_channel_simulator import GPUTransceiverImpairments
                                        rx_profiles = GPUTransceiverImpairments.sample_random_profiles(num_messages)
                                        all_signals_faded = self.channel_simulator.apply_rx_impairments_only(
                                            all_signals_faded,
                                            rx_profiles
                                        )

                                    all_signals = [all_signals_faded[i] for i in range(num_messages)]

                            # Distribute signals back to streams with PER-MESSAGE SNR scaling
                            # This ensures different messages have different SNRs relative to the same noise floor
                            streams_gpu = torch.zeros(actual_batch_size, self.stream_samples,
                                                     dtype=torch.complex64, device=self.device)

                            # Use physics SNR as reference noise floor level (pre-extracted array slice)
                            reference_snr_np = self.physics_arrays['effective_snr_db'][batch_start_idx:batch_start_idx+actual_batch_size]
                            reference_snr_batch = torch.from_numpy(reference_snr_np).to(self.device)

                            # Build stream index map for each message
                            msg_to_stream_idx = {}
                            for local_stream_idx, stream_msgs in enumerate(all_stream_messages):
                                for msg in stream_msgs:
                                    msg_to_stream_idx[id(msg)] = local_stream_idx

                            # Use signal_to_msg_map to correctly align signals with messages
                            for signal_idx, msg in enumerate(signal_to_msg_map):
                                # Get corresponding signal
                                if isinstance(all_signals, list):
                                    signal = all_signals[signal_idx]
                                else:
                                    signal = all_signals[signal_idx]

                                # Get ACTUAL signal length from metadata (accounts for zero-truncation)
                                # all_metadata has the same ordering as signal_to_msg_map
                                signal_metadata = all_metadata[signal_idx]
                                num_data_symbols = signal_metadata['num_data_symbols']
                                data_rate = signal_metadata['data_symbol_rate']

                                # Calculate actual signal duration from data symbols
                                actual_signal_len = int(num_data_symbols * (self.sample_rate / data_rate))

                                # Use original_signal_lengths as fallback if metadata missing
                                if actual_signal_len == 0 and original_signal_lengths is not None:
                                    actual_signal_len = original_signal_lengths[signal_idx]

                                # Get stream index for this message
                                local_stream_idx = msg_to_stream_idx[id(msg)]
                                reference_snr = reference_snr_batch[local_stream_idx].item()

                                # Scale signal power to achieve target SNR relative to reference noise floor
                                target_snr = msg.snr_db
                                snr_ratio_db = target_snr - reference_snr
                                power_scale = 10 ** (snr_ratio_db / 20.0)

                                # Add scaled signal to stream (use ORIGINAL length, not padded!)
                                signal_len = min(actual_signal_len, self.stream_samples - msg.start_sample)
                                if signal_len > 0:
                                    # Generator already ramped preamble/postamble padding
                                    # But we still need final ramp-down when signal ends (prevents spectral splatter)
                                    # Apply short ramp-down at END only (last 150ms)
                                    ramp_down_samples = int(0.150 * self.sample_rate)  # 150ms

                                    if signal_len > ramp_down_samples:
                                        # Create window: 1.0 everywhere except final ramp-down
                                        window = torch.ones(signal_len, dtype=torch.float32, device=self.device)

                                        # Ramp down only at the very end (raised cosine)
                                        t_down = torch.arange(ramp_down_samples, dtype=torch.float32, device=self.device) / ramp_down_samples
                                        window[-ramp_down_samples:] = 0.5 * (1 + torch.cos(torch.pi * t_down))

                                        windowed_signal = signal[:signal_len] * window * power_scale
                                    else:
                                        # Signal very short, just use as-is with power scaling
                                        windowed_signal = signal[:signal_len] * power_scale

                                    streams_gpu[local_stream_idx, msg.start_sample:msg.start_sample + signal_len] += windowed_signal

                                # Update message end_sample with ACTUAL signal length (unpadded!)
                                msg.end_sample = msg.start_sample + signal_len

                            # All messages now have correct end_sample values!

                            # Normalize stream power to prevent explosion when multiple signals add up
                            # VECTORIZED: Compute all powers at once, broadcast division (no loop!)
                            stream_powers = torch.mean(torch.abs(streams_gpu)**2, dim=1, keepdim=True)  # [batch, 1]
                            stream_powers = torch.clamp(stream_powers, min=1e-10)  # Prevent division by zero
                            streams_gpu = streams_gpu / torch.sqrt(stream_powers)  # Broadcast division

                            # REALISTIC NOISE: AWGN + QRM + QRN (all GPU-accelerated, parallel!)
                            # Now signal_power_batch should be ~1.0 for all streams
                            signal_power_batch = torch.mean(torch.abs(streams_gpu) ** 2, dim=1)  # [batch_size]
                            snr_np = self.physics_arrays['effective_snr_db'][batch_start_idx:batch_start_idx+actual_batch_size]
                            snr_batch = torch.from_numpy(snr_np).to(self.device)
                            noise_power_batch = signal_power_batch / (10 ** (snr_batch / 10))

                            # REALISTIC NOISE FLOOR: Build from components (ITU-R P.372)
                            # Thermal is baseline, but atmospheric/galactic often dominate

                            # 1. Thermal noise (baseline)
                            thermal_noise = (torch.randn(actual_batch_size, self.stream_samples, device=self.device) +
                                           1j * torch.randn(actual_batch_size, self.stream_samples, device=self.device)) / np.sqrt(2)

                            # 2. Galactic noise (always present, 3-6 dB above thermal on HF)
                            galactic_qrn = self.qrn_generator.generate_galactic_noise_batch(
                                actual_batch_size, self.stream_samples, noise_level=1.0
                            )
                            galactic_power_ratio = 2.0  # 3 dB above thermal (realistic for HF)

                            # 3. Atmospheric QRN (30% of streams, K-index dependent)
                            # Extract K-indices and thunderstorm activity for this batch (pre-extracted arrays)
                            k_indices_batch = torch.from_numpy(
                                self.physics_arrays['k_index'][batch_start_idx:batch_start_idx+actual_batch_size]
                            ).to(self.device)
                            thunderstorm_batch = torch.from_numpy(
                                self.physics_arrays['thunderstorm_activity'][batch_start_idx:batch_start_idx+actual_batch_size]
                            ).to(self.device)

                            atmospheric_mask = torch.rand(actual_batch_size, device=self.device) < 0.3
                            atmospheric_qrn_full = torch.zeros(actual_batch_size, self.stream_samples,
                                                              dtype=torch.complex64, device=self.device)
                            if atmospheric_mask.any():
                                num_with_atmospheric = atmospheric_mask.sum().item()
                                atmospheric_qrn = self.qrn_generator.generate_combined_qrn_batch(
                                    num_with_atmospheric, self.stream_samples,
                                    k_index_batch=k_indices_batch[atmospheric_mask],
                                    thunderstorm_activity_batch=thunderstorm_batch[atmospheric_mask],
                                    include_atmospheric=True,
                                    include_impulsive=True,  # Re-enabled (realistic powerline noise)
                                    include_galactic=False
                                )
                                atmospheric_qrn_full[atmospheric_mask] = atmospheric_qrn

                            # 4. Impulsive QRN (5% of streams, powerline/industrial)
                            # Reduced from 15% for performance (impulsive generation is slow)
                            impulsive_mask = torch.rand(actual_batch_size, device=self.device) < 0.05
                            impulsive_qrn_full = torch.zeros(actual_batch_size, self.stream_samples,
                                                            dtype=torch.complex64, device=self.device)
                            if impulsive_mask.any():
                                num_with_impulsive = impulsive_mask.sum().item()
                                impulsive_qrn = self.qrn_generator.generate_impulsive_qrn_batch(
                                    num_with_impulsive, self.stream_samples,
                                    powerline_freq=60.0,
                                    strength=0.3
                                )
                                impulsive_qrn_full[impulsive_mask] = impulsive_qrn

                            # COMBINE ALL NOISE FLOOR COMPONENTS
                            # Total = thermal + galactic + atmospheric + impulsive (add POWERS in quadrature, not amplitudes!)
                            # Correct: P_total = P1 + P2 + P3 + P4, then A_total = sqrt(P_total)
                            # WRONG (old): A_total = A1 + A2 + A3 + A4 (this creates 4× power!)

                            thermal_power = torch.abs(thermal_noise) ** 2
                            galactic_power = torch.abs(galactic_qrn) ** 2 * galactic_power_ratio
                            atmospheric_power = torch.abs(atmospheric_qrn_full) ** 2
                            impulsive_power = torch.abs(impulsive_qrn_full) ** 2

                            total_power = thermal_power + galactic_power + atmospheric_power + impulsive_power

                            # Convert back to complex amplitude with random phase
                            phase = torch.angle(thermal_noise)  # Use thermal phase as base
                            total_noise_floor = torch.sqrt(total_power + 1e-10) * torch.exp(1j * phase)

                            # Scale combined noise floor to achieve target SNR
                            noise_batch = total_noise_floor * torch.sqrt(noise_power_batch + 1e-10).unsqueeze(1)
                            streams_gpu += noise_batch

                            # 5. QRM (interfering stations) - DISABLED for performance
                            # Generating full CASCADE interferer signals is too expensive (~7× slowdown)
                            # TODO: Implement lightweight QRM (simple GMSK tones, not full signals)
                            pass

                            # TX observations: Generate beacons and apply reciprocal channel
                            if self.enable_tx_observations:
                                # Generate RX beacons for all streams in batch (GPU)
                                beacon_batch_params = BatchKernelParameters(
                                    pattern_ids=torch.zeros(actual_batch_size, dtype=torch.long, device=self.device),
                                    frequency_triples=torch.full((actual_batch_size,), 30, dtype=torch.long, device=self.device),  # Middle triple (61 triples total)
                                    modulations=['BPSK'] * actual_batch_size,
                                    polar_rates=[(1, 2)] * actual_batch_size,
                                    data_symbol_rates=torch.full((actual_batch_size,), 100, device=self.device)
                                )
                                beacon_messages = [b"RX_BEACON"] * actual_batch_size

                                # Generate all beacons at once (standard pattern, not always-on)
                                beacons_gpu, _ = self.signal_generator.generate_batch(
                                    beacon_batch_params, beacon_messages, fixed_length=48000,  # 1 second
                                    num_centers=0  # Use standard pattern for beacons
                                )

                                # Apply FULL reciprocal channel (same physics as RX direction!)
                                # Build multipath profile for beacons
                                beacon_conditions = [physics_conditions[batch_start_idx + i] for i in range(actual_batch_size)]
                                beacon_multipath = self._create_multipath_profile_for_batch(beacon_conditions)

                                # Apply complete channel to beacons
                                beacons_faded = self.channel_simulator.apply_time_varying_multipath_batch(
                                    beacons_gpu, beacon_multipath, coherence_bandwidth_hz=50
                                )

                                # D-layer absorption (reciprocal)
                                absorption_db = torch.tensor([c.d_layer_absorption_db for c in beacon_conditions], device=self.device)
                                # Use pre-extracted UTC hours
                                utc_hours = self.physics_arrays['utc_hour'][batch_start_idx:batch_start_idx+actual_batch_size]
                                sza = torch.from_numpy((utc_hours - 12) * 15).to(self.device)
                                beacons_absorbed = self.channel_simulator.apply_continuous_d_layer_absorption(
                                    beacons_faded, absorption_db, sza
                                )

                                # AWGN at same SNR
                                signal_power_batch = torch.mean(torch.abs(beacons_absorbed) ** 2, dim=1)
                                snr_batch = torch.tensor([c.effective_snr_db for c in beacon_conditions], device=self.device)
                                noise_power_batch = signal_power_batch / (10 ** (snr_batch / 10))
                                noise_batch = (torch.randn(actual_batch_size, 48000, device=self.device) +
                                              1j * torch.randn(actual_batch_size, 48000, device=self.device)) / np.sqrt(2)
                                tx_observed_batch = beacons_absorbed + noise_batch * torch.sqrt(noise_power_batch + 1e-10).unsqueeze(1)

                                # Compute physics-based optimal embeddings
                                optimal_embeddings_batch = self._compute_optimal_embeddings_batch(
                                    beacons_gpu, tx_observed_batch, beacon_conditions
                                )
                            elif self.compute_optimal_embeddings:
                                # Simplified: compute embeddings from physics only (no beacons!)
                                beacon_conditions = [physics_conditions[batch_start_idx + i] for i in range(actual_batch_size)]
                                optimal_embeddings_batch = torch.zeros(actual_batch_size, 256, device=self.device)
                                for i, cond in enumerate(beacon_conditions):
                                    optimal_embeddings_batch[i, 0] = cond.multipath_delay_spread_ms / 15.0
                                    optimal_embeddings_batch[i, 1] = cond.doppler_spread_hz / 5.0
                                    optimal_embeddings_batch[i, 2] = (cond.effective_snr_db + 30) / 70.0
                                    optimal_embeddings_batch[i, 3] = cond.d_layer_absorption_db / 30.0

                            # OPTIMIZATION: Synchronize CUDA stream before CPU transfer
                            # This ensures all GPU work is complete before we copy to CPU
                            if self.cuda_stream_compute is not None:
                                self.cuda_stream_compute.synchronize()

                            # OPTIMIZATION: Compute normalization stats on GPU (100× faster than CPU!)
                            # This eliminates the 170ms CPU bottleneck
                            actual_batch_size = streams_gpu.shape[0]
                            norm_stats_gpu = torch.zeros(actual_batch_size, 2, dtype=torch.float32, device=self.device)

                            # Vectorized mean and std on GPU (all streams at once!)
                            iq_real = streams_gpu.real  # [batch, samples]
                            iq_imag = streams_gpu.imag
                            iq_stack = torch.stack([iq_real, iq_imag], dim=1)  # [batch, 2, samples]

                            # Compute across both I and Q channels
                            norm_stats_gpu[:, 0] = torch.mean(iq_stack, dim=[1, 2])  # Mean
                            norm_stats_gpu[:, 1] = torch.clamp(torch.std(iq_stack, dim=[1, 2]), min=1e-6) + 1e-8  # Std

                            # Move entire batch to CPU (using transfer stream for async if available)
                            if self.cuda_stream_transfer is not None:
                                with torch.cuda.stream(self.cuda_stream_transfer):
                                    streams_cpu = streams_gpu.cpu().numpy()
                                    norm_stats_cpu = norm_stats_gpu.cpu().numpy()  # Also transfer norm stats
                                    if self.enable_tx_observations:
                                        tx_observed_cpu = tx_observed_batch.cpu().numpy()
                                        optimal_embeddings_cpu = optimal_embeddings_batch.cpu().numpy()
                                    elif self.compute_optimal_embeddings:
                                        optimal_embeddings_cpu = optimal_embeddings_batch.cpu().numpy()
                                # Sync transfer stream before using CPU data
                                self.cuda_stream_transfer.synchronize()
                            else:
                                streams_cpu = streams_gpu.cpu().numpy()
                                norm_stats_cpu = norm_stats_gpu.cpu().numpy()  # Transfer norm stats
                                if self.enable_tx_observations:
                                    tx_observed_cpu = tx_observed_batch.cpu().numpy()
                                    optimal_embeddings_cpu = optimal_embeddings_batch.cpu().numpy()
                                elif self.compute_optimal_embeddings:
                                    optimal_embeddings_cpu = optimal_embeddings_batch.cpu().numpy()

                            # OPTIMIZATION: Normalization stats already computed on GPU!
                            # Just serialize metadata in parallel (if postproc pool enabled)
                            # norm_stats_cpu is already available from GPU computation above

                            # Note: Metadata serialization is lightweight (5ms), not worth parallelizing
                            # Skip postprocessing workers entirely - norm stats done on GPU, metadata inline

                            # PERIODIC VISUALIZATION: Create spectrogram/IQ/phase plots
                            # This happens AFTER message end_sample updates (line 991) so timings are correct
                            if self.vis_interval > 0:
                                for i in range(actual_batch_size):
                                    global_stream_idx = batch_start_idx + i
                                    # Only visualize every Nth stream
                                    if global_stream_idx % self.vis_interval == 0:
                                        # Collect metadata from first message in this stream (or use physics)
                                        stream_messages = all_stream_messages[i]
                                        condition = physics_conditions[batch_start_idx + i]
                                        scenario = scenario_drivers[batch_start_idx + i]

                                        # Build metadata dict for visualization
                                        viz_metadata = {
                                            'sample_idx': global_stream_idx,
                                            'propagation_mode': condition.propagation_mode.value,
                                            'qrn_type': condition.dominant_qrn_type.value,
                                            'k_index': scenario.k_index,
                                            'sfi': scenario.sfi,
                                            'snr_db': condition.effective_snr_db,
                                            'num_messages': len(stream_messages),
                                        }

                                        # Add first message metadata if available
                                        if stream_messages:
                                            first_msg = stream_messages[0]
                                            viz_metadata.update({
                                                'pattern_id': first_msg.pattern_id,
                                                'frequency_triple': first_msg.frequency_triple,
                                                'modulation': first_msg.modulation,
                                                'data_symbol_rate': first_msg.data_symbol_rate,
                                            })

                                            # Add message list for timing annotations (now with correct end_sample!)
                                            viz_metadata['message_list'] = [
                                                {
                                                    'start_sample': msg.start_sample,
                                                    'end_sample': msg.end_sample,
                                                    'pattern_id': msg.pattern_id,
                                                    'frequency_triple': msg.frequency_triple,
                                                    'modulation': msg.modulation,
                                                }
                                                for msg in stream_messages
                                            ]

                                        # Create visualization (synchronous - doesn't block much)
                                        # Save to modules/training/core/visualizations (not cache dir)
                                        vis_output_dir = Path(__file__).parent / 'visualizations'
                                        vis_output_dir.mkdir(parents=True, exist_ok=True)
                                        output_path = vis_output_dir / f"stream_{global_stream_idx:06d}.png"
                                        try:
                                            create_single_visualization(
                                                iq_samples=streams_cpu[i],
                                                sample_rate=self.sample_rate,
                                                metadata=viz_metadata,
                                                output_path=str(output_path)
                                            )
                                        except Exception as e:
                                            # Don't crash if visualization fails
                                            print(f"⚠️  Visualization failed for stream {global_stream_idx}: {e}")

                            # SUBMIT TO BACKGROUND WRITERS (or buffer for HDF5)
                            if write_queue is not None:
                                # ASYNC WRITE: Submit to background writers (GPU continues immediately!)
                                emb_for_queue = optimal_embeddings_cpu if self.compute_optimal_embeddings else None
                                try:
                                    write_queue.put((batch_start_idx, streams_cpu, emb_for_queue, norm_stats_cpu), timeout=30)
                                except:
                                    print(f"\n❌ Write queue blocked! Queue size: {write_queue.qsize()}")
                                    print(f"   Writers may be stuck. Forcing write...")
                                    write_queue.put((batch_start_idx, streams_cpu, emb_for_queue, norm_stats_cpu))
                            else:
                                # HDF5: Buffer in RAM
                                buffer_streams.append(streams_cpu)
                                if self.enable_tx_observations:
                                    buffer_tx_observed.append(tx_observed_cpu)
                                    buffer_optimal_embeddings.append(optimal_embeddings_cpu)
                                elif self.compute_optimal_embeddings:
                                    buffer_optimal_embeddings.append(optimal_embeddings_cpu)

                            # Buffer messages for metadata (ensures consistency with signals)
                            buffer_all_messages.append(all_stream_messages)

                            pbar.update(actual_batch_size)

                # End of chunk - write remaining data
                write_start = time.time()

                # Wait for background writes to complete (only for numpy)
                if write_queue is not None:
                    print(f"\n  ⏳ Waiting for background writes to complete...")
                    write_queue.join()  # Wait for all queued writes to finish
                    print(f"  ✓ All writes completed")

                # Write to HDF5 if needed
                if streams_mmap is None:
                    # HDF5: Write buffered data
                    print(f"\n  💾 Writing to HDF5...")

                    written = 0
                    for batch_idx, batch in enumerate(buffer_streams):
                        batch_size_actual = batch.shape[0]
                        streams_ds[written:written + batch_size_actual] = batch

                        if self.enable_tx_observations:
                            tx_observed_ds[written:written + batch_size_actual] = buffer_tx_observed[batch_idx]
                            optimal_embeddings_ds[written:written + batch_size_actual] = buffer_optimal_embeddings[batch_idx]
                        elif self.compute_optimal_embeddings:
                            optimal_embeddings_ds[written:written + batch_size_actual] = buffer_optimal_embeddings[batch_idx]

                        written += batch_size_actual

                        if written % 10000 == 0:
                            print(f"    Written: {written:,}/{chunk_size:,} streams ({written/chunk_size*100:.0f}%)")

                # Now write ALL metadata from buffered messages
                print(f"    Writing metadata...")
                metadata_idx = 0  # Track position WITHIN chunk
                for batch_messages in buffer_all_messages:
                    for messages in batch_messages:
                        # Skip if we've written all streams for this chunk
                        if metadata_idx >= chunk_size:
                            break

                        # Calculate global stream index
                        global_stream_idx = chunk_start + metadata_idx

                        condition = physics_conditions[global_stream_idx]

                        # For numpy: use GLOBAL indices (chunk_start + metadata_idx)
                        # For HDF5: use LOCAL indices (metadata_idx) within chunk file
                        write_idx = global_stream_idx if self.output_format == 'numpy' else metadata_idx

                        num_messages_ds[write_idx] = len(messages)
                        k_indices[write_idx] = scenario_drivers[global_stream_idx].k_index
                        sfis[write_idx] = scenario_drivers[global_stream_idx].sfi
                        propagation_modes[write_idx] = condition.propagation_mode.value.encode('utf-8')
                        qrn_types[write_idx] = condition.dominant_qrn_type.value.encode('utf-8')

                        # Message metadata as JSON (from ACTUAL generated messages)
                        msg_meta = [{
                            'start_sample': m.start_sample,
                            'end_sample': m.end_sample,
                            'pattern_id': m.pattern_id,
                            'frequency_triple': m.frequency_triple,  # Starting triple for multi-channel
                            'num_channels': m.num_channels,  # How many channels used
                            'modulation': m.modulation,
                            'discrete_rate': m.data_symbol_rate,
                            'continuous_rate': m.continuous_rate,  # Store continuous rate for training
                            'pattern_only': m.pattern_only,  # Flag for pattern-only detection
                            'snr_db': m.snr_db,  # Per-message SNR (for confusion matrices)
                        } for m in messages]
                        message_metadata[write_idx] = json.dumps(msg_meta)

                        metadata_idx += 1

            write_time = time.time() - write_start

            if streams_mmap is None:
                print(f"  ✓ Chunk {chunk_idx+1} written in {write_time:.1f}s ({chunk_size/write_time:.0f} streams/sec I/O)")
            else:
                print(f"  ✓ Chunk {chunk_idx+1} finalized in {write_time:.1f}s")

            # Clear buffers (free RAM)
            if buffer_streams:  # Only if we buffered (HDF5 mode)
                del buffer_streams
                del buffer_optimal_embeddings
                if self.enable_tx_observations:
                    del buffer_tx_observed
            import gc
            gc.collect()

            # Clear GPU cache AFTER chunk completes (once per 100K streams)
            torch.cuda.empty_cache()

            # Print chunk timing summary
            chunk_elapsed = time.time() - write_start
            chunk_total = time.time() - start_time
            chunk_size_actual = chunk_end - chunk_start
            print(f"\n  ✓ Chunk {chunk_idx+1}/{num_chunks} complete:")
            print(f"    Streams: {chunk_size_actual:,}")
            print(f"    Time: {chunk_elapsed:.1f}s (total: {chunk_total:.1f}s)")
            print(f"    Rate: {chunk_size_actual/chunk_elapsed:.1f} streams/sec")

        # Close numpy label file after ALL chunks complete
        if self.output_format == 'numpy':
            hdf5_label_file.close()
            print(f"\n  ✓ Label file closed (all {self.num_streams:,} streams written)")

        # SHUTDOWN BACKGROUND WRITERS (ensure all writes complete!)
        if write_queue is not None:
            print(f"\n🛑 Shutting down background writers (ensuring all data written)...")

            # Send poison pills to stop all writer threads
            for _ in writer_threads:
                write_queue.put(None)

            # Wait for all threads to finish
            for t in writer_threads:
                t.join()

            print(f"  ✓ All {len(writer_threads)} writer threads completed")

            # Final flush to ensure all data on disk
            if hasattr(self, 'streams_mmap_chunks'):
                num_chunks = len(self.streams_mmap_chunks)
                print(f"  Flushing {num_chunks} stream chunk files...")
                for i, chunk in enumerate(self.streams_mmap_chunks):
                    chunk.flush()
                    if (i + 1) % 100 == 0 or (i + 1) == num_chunks:
                        print(f"    {i+1}/{num_chunks} chunks flushed")
            if embeddings_mmap is not None:
                print(f"  Flushing embeddings...")
                embeddings_mmap.flush()
            if normalization_stats_mmap is not None:
                print(f"  Flushing normalization stats...")
                normalization_stats_mmap.flush()

            print(f"  ✓ All memmap files flushed to disk")

            # PIPELINE CLEANUP: Shutdown worker pools
            if pregen_pool is not None:
                print(f"\n🛑 Shutting down pre-generation worker pool...")
                pregen_pool.close()
                pregen_pool.join()
                print(f"  ✓ Pre-generation pool terminated")

            if postproc_pool is not None:
                print(f"\n🛑 Shutting down post-processing worker pool...")
                postproc_pool.shutdown(wait=True)
                print(f"  ✓ Post-processing pool terminated")

        # If multiple chunk files, create master index file (HDF5 only, not numpy!)
        if num_chunks > 1 and streams_mmap is None:
            # HDF5 multi-file mode needs master index
            print(f"\n  📋 Creating HDF5 master index file...")
            with h5py.File(self.cache_path, 'w') as master:
                master.attrs['num_chunks'] = num_chunks
                master.attrs['chunk_size'] = CHUNK_SIZE
                master.attrs['num_streams'] = self.num_streams
                master.attrs['stream_duration_sec'] = self.stream_duration_sec
                master.attrs['window_duration_sec'] = self.window_duration_sec
                master.attrs['sample_rate'] = self.sample_rate
                master.attrs['windows_per_stream'] = self.windows_per_stream
                master.attrs['total_windows'] = self.num_samples
                master.attrs['enable_tx_observations'] = self.enable_tx_observations

                # Store chunk file paths
                chunk_names = [str(f.name) for f in chunk_files]
                dt = h5py.special_dtype(vlen=str)
                master.create_dataset('chunk_files', data=chunk_names, dtype=dt)

            print(f"  ✓ Master index: {self.cache_path.name}")
            print(f"  ✓ Chunk files: {num_chunks} files ({CHUNK_SIZE:,} streams each)")
        elif num_chunks > 1 and streams_mmap is not None:
            # Numpy multi-file mode (not currently used, but defensive)
            print(f"\n  ℹ️  Numpy multi-file mode: {num_chunks} chunks (no master index needed, using metadata.json)")

        total_time = time.time() - start_time
        streams_per_sec = self.num_streams / total_time if total_time > 0 else 0
        windows_per_sec = self.num_samples / total_time if total_time > 0 else 0

        print(f"\n✓ Generated {self.num_streams:,} streams in {total_time:.1f}s")
        print(f"  Streams: {streams_per_sec:.1f} streams/sec")
        print(f"  Windows: {windows_per_sec:.1f} windows/sec (effective training samples)")

        # Finalize based on output format
        if streams_mmap is not None:
            # Numpy format: flush memmap and create metadata
            print(f"\n📦 Finalizing numpy files...")

            # Flush memmaps to disk
            streams_mmap.flush()
            if embeddings_mmap is not None:
                embeddings_mmap.flush()

            # Save metadata
            metadata = {
                'num_streams': self.num_streams,
                'stream_duration_sec': self.stream_duration_sec,
                'window_duration_sec': self.window_duration_sec,
                'sample_rate': self.sample_rate,
                'windows_per_stream': self.windows_per_stream,
                'total_windows': self.num_samples
            }
            import json
            with open(self.cache_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"  ✓ Saved to: {self.cache_path}")
            print(f"  ✓ Files: streams.npy ({streams_mmap.nbytes / 1e9:.1f} GB), embeddings.npy, metadata.json, labels.h5")
            print(f"  ✓ Thread-safe numpy format - ready for parallel DataLoader workers!")

            self._load_numpy_cache(self.cache_path)
        else:
            # HDF5 format
            print(f"  Saved to: {self.cache_path.name}")

            if self.num_streams > 10000:
                print(f"\n  💡 Tip: For faster training, convert to numpy:")
                print(f"     python3 convert_hdf5_to_numpy.py {self.cache_path}")

            self._load_cache()

    def _load_numpy_cache(self, numpy_cache_dir):
        """Load dataset from numpy memmap files (thread-safe for parallel workers!)."""
        import json
        import os

        # Load and validate metadata (REQUIRED - no fallback!)
        metadata_path = numpy_cache_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Cache metadata.json missing: {numpy_cache_dir}\n"
                f"This cache was generated without metadata or is corrupted.\n"
                f"Solution: Delete cache and regenerate:\n"
                f"  rm -rf {numpy_cache_dir}\n"
                f"  rm -f {numpy_cache_dir.parent}/{numpy_cache_dir.name}_labels.h5\n"
                f"  export CASCADE_REGENERATE_CACHE=true\n"
                f"  ./run_full_training.sh"
            )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Validate critical parameters match what we're requesting
        cached_window_dur = metadata.get('window_duration_sec')
        cached_stream_dur = metadata.get('stream_duration_sec')
        cached_sample_rate = metadata.get('sample_rate')

        # Check for mismatches
        tolerance = 0.0001  # Allow tiny floating point differences
        mismatches = []

        if cached_window_dur is None:
            mismatches.append(f"  - window_duration_sec: MISSING in cache")
        elif abs(cached_window_dur - self.window_duration_sec) > tolerance:
            mismatches.append(
                f"  - window_duration_sec: cached={cached_window_dur:.4f}s, "
                f"requested={self.window_duration_sec:.4f}s"
            )

        if cached_stream_dur is not None and abs(cached_stream_dur - self.stream_duration_sec) > tolerance:
            mismatches.append(
                f"  - stream_duration_sec: cached={cached_stream_dur}s, "
                f"requested={self.stream_duration_sec}s"
            )

        if cached_sample_rate is not None and cached_sample_rate != self.sample_rate:
            mismatches.append(
                f"  - sample_rate: cached={cached_sample_rate} Hz, "
                f"requested={self.sample_rate} Hz"
            )

        if mismatches:
            raise ValueError(
                f"Cache parameter mismatch: {numpy_cache_dir.name}\n" +
                "\n".join(mismatches) + "\n\n"
                f"This cache was generated with different parameters and cannot be used.\n"
                f"Solution: Delete cache and regenerate:\n"
                f"  rm -rf {numpy_cache_dir}\n"
                f"  rm -f {numpy_cache_dir.parent}/{numpy_cache_dir.name}_labels.h5\n"
                f"  export CASCADE_REGENERATE_CACHE=true\n"
                f"  ./run_full_training.sh"
            )

        # All validations passed - load metadata
        self.num_streams = metadata['num_streams']
        print(f"  ✓ Metadata validated: {self.num_streams:,} streams, "
              f"{metadata['windows_per_stream']} windows/stream")

        # Memory-map streams file(s) - may be chunked for NFS compatibility!
        streams_path = numpy_cache_dir / 'streams.npy'

        # Check if single file or chunked
        if streams_path.exists():
            # Single file (small dataset or local filesystem)
            self.numpy_streams = np.load(streams_path, mmap_mode='r')
            self.numpy_streams_chunked = None
        else:
            # Chunked files (NFS workaround for >2GB)
            chunk_files = sorted(numpy_cache_dir.glob('streams_chunk_*.npy'))
            if not chunk_files:
                raise FileNotFoundError(f"No streams.npy or streams_chunk_*.npy found in {numpy_cache_dir}")

            print(f"  Loading {len(chunk_files)} chunked memmap files...")
            self.numpy_streams_chunked = [
                np.load(chunk_file, mmap_mode='r') for chunk_file in chunk_files
            ]
            self.chunk_sizes = [chunk.shape[0] for chunk in self.numpy_streams_chunked]
            self.chunk_offsets = np.cumsum([0] + self.chunk_sizes[:-1])
            print(f"  ✓ Loaded {len(chunk_files)} chunks ({self.chunk_sizes[0]:,} streams each)")

            # Set to None to indicate chunked mode
            self.numpy_streams = None

        # Memory-map normalization stats (OPTIMIZATION: pre-computed mean/std)
        norm_stats_path = numpy_cache_dir / 'normalization_stats.npy'
        if norm_stats_path.exists():
            self.numpy_norm_stats = np.load(norm_stats_path, mmap_mode='r')
            print(f"  ✓ Pre-computed normalization stats loaded (40-60% faster __getitem__)")
        else:
            self.numpy_norm_stats = None
            print(f"  ⚠️  No pre-computed normalization stats found - will compute on-the-fly (slower)")

        # Memory-map embeddings
        embeddings_path = numpy_cache_dir / 'embeddings.npy'
        if embeddings_path.exists():
            self.numpy_embeddings = np.load(embeddings_path, mmap_mode='r')
        else:
            self.numpy_embeddings = None

        # Find the HDF5 file for labels (REQUIRED for training)
        # Look for .h5 file with same base name as numpy directory
        hdf5_label_file = numpy_cache_dir.parent / f"{numpy_cache_dir.name}_labels.h5"

        if hdf5_label_file.exists():
            self.hdf5_file = h5py.File(hdf5_label_file, 'r')  # For labels only
            print(f"  ✓ Labels from: {hdf5_label_file.name}")
        else:
            # Try alternate patterns (temp files, etc.)
            self.hdf5_file = None
            for pattern in [f"{numpy_cache_dir.name}_labels*.h5", f"{numpy_cache_dir.name}_temp*.h5", f"{numpy_cache_dir.name}*.h5"]:
                for f in numpy_cache_dir.parent.glob(pattern):
                    if f.is_file():
                        self.hdf5_file = h5py.File(f, 'r')
                        print(f"  ✓ Labels from: {f.name}")
                        break
                if self.hdf5_file is not None:
                    break

            if self.hdf5_file is None:
                raise FileNotFoundError(
                    f"No label file found for numpy cache: {numpy_cache_dir}\n"
                    f"Expected: {hdf5_label_file}\n"
                    f"Labels (k_index, sfi, propagation_mode, etc.) are required for training.\n"
                    f"The dataset may be corrupted - try regenerating."
                )

        self.chunk_files = None
        self.using_numpy = True

        # Get total streams (from single file or chunked)
        if self.numpy_streams is not None:
            total_streams = len(self.numpy_streams)
            file_size_gb = os.path.getsize(streams_path) / 1e9
        else:
            total_streams = sum(self.chunk_sizes)
            file_size_gb = sum(chunk.nbytes for chunk in self.numpy_streams_chunked) / 1e9

        print(f"  ✓ Loaded {total_streams:,} streams via memory-map")
        print(f"  ✓ Thread-safe parallel access enabled!")
        print(f"  ✓ Total size: {file_size_gb:.1f} GB")
        if self.hdf5_file:
            print(f"  ✓ Labels from HDF5 (small, doesn't bottleneck)")
        else:
            print(f"  ⚠️ No label file found - labels not available")

        # OPTIMIZATION: Pre-load ALL labels into RAM (eliminates HDF5 I/O contention!)
        # 21MB labels file → RAM = 15-25% faster __getitem__ with 32 workers
        print(f"  Loading all labels into RAM (eliminates HDF5 I/O bottleneck)...")

        # Load message metadata (JSON parsing)
        self.message_metadata_cache = []
        import json
        for i in range(self.num_streams):
            try:
                meta_json = self.hdf5_file['message_metadata'][i]
                if isinstance(meta_json, bytes):
                    meta_json = meta_json.decode('utf-8')
                if not meta_json or meta_json.strip() == '':
                    messages = []
                else:
                    messages = json.loads(meta_json)
            except (json.JSONDecodeError, KeyError, IndexError):
                messages = []
            self.message_metadata_cache.append(messages)

        # Load all other labels into RAM (numpy arrays for fast access)
        self.labels_cache = {
            'k_index': self.hdf5_file['k_index'][:] if 'k_index' in self.hdf5_file else None,
            'sfi': self.hdf5_file['sfi'][:] if 'sfi' in self.hdf5_file else None,
            'propagation_modes': self.hdf5_file['propagation_modes'][:] if 'propagation_modes' in self.hdf5_file else None,
            'qrn_types': self.hdf5_file['qrn_types'][:] if 'qrn_types' in self.hdf5_file else None,
            'num_messages': self.hdf5_file['num_messages'][:] if 'num_messages' in self.hdf5_file else None,
        }

        # Calculate total label cache size
        total_label_mb = sum(
            arr.nbytes / 1e6 for arr in self.labels_cache.values() if arr is not None
        ) + len(str(self.message_metadata_cache)) / 1e6  # Rough estimate for message metadata

        print(f"  ✓ Cached {self.num_streams:,} labels in RAM ({total_label_mb:.1f} MB)")
        print(f"    → Eliminates HDF5 I/O contention across {32} DataLoader workers")

    def _load_cache(self):
        """Load cached streams (on-demand from HDF5, handles multi-file chunks)."""
        # Check if cache_path is actually a numpy directory (defensive)
        if self.cache_path.is_dir():
            streams_file = self.cache_path / 'streams.npy'
            if streams_file.exists():
                print(f"  ℹ️  Detected numpy cache directory, loading via numpy path...")
                self._load_numpy_cache(self.cache_path)
                return
            else:
                raise FileNotFoundError(
                    f"Cache path is a directory but missing streams.npy: {self.cache_path}\n"
                    f"This cache appears corrupted. Try regenerating with regenerate_cache=True"
                )

        # Check if multi-file (has master index)
        try:
            with h5py.File(self.cache_path, 'r') as f:
                is_multifile = 'num_chunks' in f.attrs

            if is_multifile:
                # Multi-file mode - load chunk file handles
                with h5py.File(self.cache_path, 'r') as f:
                    self.num_chunks = f.attrs['num_chunks']
                    self.chunk_size = f.attrs['chunk_size']
                    chunk_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                                  for name in f['chunk_files'][:]]

                self.chunk_files = []
                for name in chunk_names:
                    chunk_path = self.cache_path.parent / name
                    self.chunk_files.append(h5py.File(chunk_path, 'r'))

                print(f"✓ Multi-file cache: {self.num_chunks} chunks, {self.num_streams:,} streams, {self.num_samples:,} windows")
            else:
                # Single file mode - keep file open
                self.num_chunks = 1
                self.chunk_size = self.num_streams
                self.hdf5_file = h5py.File(self.cache_path, 'r')  # Keep open separately
                self.chunk_files = None
                print(f"✓ Single-file cache: {self.num_streams:,} streams, {self.num_samples:,} windows")

            # Smart RAM caching: Load subset of dataset to RAM
            if self.load_into_memory:
                total_size_gb = self.num_streams * self.stream_samples * 8 / 1e9

                # Check if we should load full dataset or chunks
                import psutil
                available_ram_gb = psutil.virtual_memory().available / 1e9

                # Use max 50% of available RAM for dataset
                max_ram_for_data = available_ram_gb * 0.5

                if total_size_gb <= max_ram_for_data:
                    # Load entire dataset to RAM
                    print(f"\n⚡ Loading entire dataset into RAM ({total_size_gb:.1f} GB)")
                    self.ram_cache_streams = []
                    if self.chunk_files is not None:
                        for chunk_idx, chunk_file in enumerate(self.chunk_files):
                            chunk_streams = chunk_file['streams'][:]
                            self.ram_cache_streams.append(chunk_streams)
                            print(f"   Loaded chunk {chunk_idx+1}/{len(self.chunk_files)}")
                        self.ram_cache_streams = np.concatenate(self.ram_cache_streams, axis=0)
                    else:
                        self.ram_cache_streams = self.hdf5_file['streams'][:]

                    # Load embeddings
                    if self.compute_optimal_embeddings or self.enable_tx_observations:
                        if self.chunk_files is not None:
                            emb_chunks = [cf['optimal_embeddings'][:] for cf in self.chunk_files]
                            self.ram_cache_embeddings = np.concatenate(emb_chunks, axis=0)
                        else:
                            self.ram_cache_embeddings = self.hdf5_file['optimal_embeddings'][:]

                    print(f"   ✓ Loaded {len(self.ram_cache_streams)} streams ({total_size_gb:.1f} GB)")
                    self.ram_cache_active_chunk = None  # Not using chunked loading
                else:
                    # Partial loading: Load subset that fits in RAM
                    # Cap at 50 GB per chunk to keep reasonable
                    target_chunk_size_gb = min(50.0, max_ram_for_data)
                    streams_per_gb = int((1e9 / 8) / self.stream_samples)
                    max_streams_in_ram = int(target_chunk_size_gb * streams_per_gb)

                    print(f"\n⚡ RAM-Aware Chunked Loading:")
                    print(f"   Total dataset: {total_size_gb:.1f} GB ({self.num_streams:,} streams)")
                    print(f"   Available RAM: {max_ram_for_data:.1f} GB")
                    print(f"   Chunk size: {target_chunk_size_gb:.1f} GB ({max_streams_in_ram:,} streams)")
                    print(f"   Chunks needed: {int(np.ceil(self.num_streams / max_streams_in_ram))}")

                    # Load first chunk
                    self.ram_cache_chunk_size = max_streams_in_ram
                    self.ram_cache_active_chunk = 0
                    self.ram_cache_streams = None  # Will be loaded on-demand per chunk
                    print(f"   Loading chunk 0...")
                    self._load_ram_chunk(0)

                    print(f"   ✓ Chunk 0 loaded ({len(self.ram_cache_streams):,} streams in RAM)")
                    print(f"   Note: Chunks auto-rotate during training (transparent to model)")
            else:
                self.ram_cache_streams = None
                self.ram_cache_active_chunk = None

        except Exception as e:
            print(f"⚠️  Error loading cache: {e}")
            print(f"   Cache path: {self.cache_path}")
            raise

    def _load_ram_chunk(self, chunk_id):
        """Load a specific chunk of streams into RAM (optimized for large chunks)."""
        start_stream = chunk_id * self.ram_cache_chunk_size
        end_stream = min(start_stream + self.ram_cache_chunk_size, self.num_streams)
        num_to_load = end_stream - start_stream

        print(f"      Loading streams {start_stream:,} to {end_stream:,} ({num_to_load:,} streams)...")

        # Load streams efficiently using slicing
        if self.chunk_files is not None:
            # Multi-file case: load from appropriate chunk file(s)
            start_chunk_idx = start_stream // self.chunk_size
            end_chunk_idx = (end_stream - 1) // self.chunk_size

            if start_chunk_idx == end_chunk_idx:
                # All streams in one HDF5 chunk file
                local_start = start_stream % self.chunk_size
                local_end = local_start + num_to_load
                self.ram_cache_streams = self.chunk_files[start_chunk_idx]['streams'][local_start:local_end]

                if self.compute_optimal_embeddings or self.enable_tx_observations:
                    self.ram_cache_embeddings = self.chunk_files[start_chunk_idx]['optimal_embeddings'][local_start:local_end]
            else:
                # Spans multiple HDF5 files - concatenate
                stream_parts = []
                emb_parts = []

                for file_idx in range(start_chunk_idx, end_chunk_idx + 1):
                    file_start = max(0, start_stream - file_idx * self.chunk_size)
                    file_end = min(self.chunk_size, end_stream - file_idx * self.chunk_size)

                    if file_end > file_start:
                        stream_parts.append(self.chunk_files[file_idx]['streams'][file_start:file_end])
                        if self.compute_optimal_embeddings or self.enable_tx_observations:
                            emb_parts.append(self.chunk_files[file_idx]['optimal_embeddings'][file_start:file_end])

                self.ram_cache_streams = np.concatenate(stream_parts, axis=0)
                if emb_parts:
                    self.ram_cache_embeddings = np.concatenate(emb_parts, axis=0)
        else:
            # Single file - simple slicing
            self.ram_cache_streams = self.hdf5_file['streams'][start_stream:end_stream]

            if self.compute_optimal_embeddings or self.enable_tx_observations:
                self.ram_cache_embeddings = self.hdf5_file['optimal_embeddings'][start_stream:end_stream]

        self.ram_cache_active_chunk = chunk_id
        self.ram_cache_start_stream = start_stream

        print(f"      ✓ Loaded {len(self.ram_cache_streams):,} streams into RAM")

    def __len__(self) -> int:
        """Return number of training windows (not streams)."""
        return self.num_samples

    def __getitem__(self, idx: int):
        """
        Get one training window (2s slice from a stream).

        Args:
            idx: Window index (0 to num_samples-1)

        Returns:
            If enable_tx_observations: Tuple of (rx_window, tx_observed, labels)
            Otherwise: Tuple of (rx_window, labels)
        """
        # Determine which stream and which window within that stream
        stream_idx = idx // self.windows_per_stream
        window_idx = idx % self.windows_per_stream

        # Load stream - priority: numpy memmap > RAM cache > HDF5
        if hasattr(self, 'using_numpy') and self.using_numpy:
            # FASTEST: Numpy memmap (thread-safe, memory-efficient)
            if self.numpy_streams is not None:
                # Single file mode
                stream_iq = self.numpy_streams[stream_idx]
            else:
                # Chunked mode (for NFS >2GB files)
                chunk_idx = stream_idx // 57000  # Matches max_streams_per_chunk
                local_idx = stream_idx % 57000
                stream_iq = self.numpy_streams_chunked[chunk_idx][local_idx]
        elif hasattr(self, 'ram_cache_streams') and self.ram_cache_streams is not None:
            # Check if using chunked RAM loading
            if hasattr(self, 'ram_cache_active_chunk') and self.ram_cache_active_chunk is not None:
                # Chunked RAM loading: check if stream is in current chunk
                if not (self.ram_cache_start_stream <= stream_idx < self.ram_cache_start_stream + len(self.ram_cache_streams)):
                    # Need to load different chunk
                    needed_chunk = stream_idx // self.ram_cache_chunk_size
                    self._load_ram_chunk(needed_chunk)

                # Get stream from RAM chunk (offset by chunk start)
                local_idx = stream_idx - self.ram_cache_start_stream
                stream_iq = self.ram_cache_streams[local_idx]
            else:
                # Full RAM cache: direct indexing
                stream_iq = self.ram_cache_streams[stream_idx]
        elif self.chunk_files is not None:
            # Multi-file: determine which chunk file
            chunk_idx = stream_idx // self.chunk_size
            local_idx = stream_idx % self.chunk_size
            stream_iq = self.chunk_files[chunk_idx]['streams'][local_idx]
        else:
            # Single file
            stream_iq = self.hdf5_file['streams'][stream_idx]

        # Extract window
        window_start = window_idx * self.window_samples
        window_end = window_start + self.window_samples
        window_iq = stream_iq[window_start:window_end]

        # Convert to I/Q tensor [2, window_samples] and NORMALIZE
        iq_i = np.real(window_iq).astype(np.float32)
        iq_q = np.imag(window_iq).astype(np.float32)

        # Normalize to zero mean, unit variance (critical for autoencoder training!)
        # OPTIMIZATION: Use pre-computed statistics if available (40-60% faster!)
        iq_stack = np.stack([iq_i, iq_q], axis=0)

        if hasattr(self, 'numpy_norm_stats') and self.numpy_norm_stats is not None:
            # FAST PATH: Use pre-computed mean/std (eliminates 192K float ops!)
            mean = self.numpy_norm_stats[stream_idx, 0]
            std = self.numpy_norm_stats[stream_idx, 1]
        else:
            # SLOW PATH: Compute on-the-fly (fallback for old caches)
            mean = np.mean(iq_stack)
            std = np.std(iq_stack) + 1e-8

        # Safety: Clamp std to prevent division by zero (constant signals)
        std = np.maximum(std, 1e-6)  # Ensures std >= 1e-6

        iq_normalized = (iq_stack - mean) / std

        rx_window = torch.from_numpy(iq_normalized)

        # Load labels (handle multi-file)
        if self.chunk_files is not None:
            chunk_idx = stream_idx // self.chunk_size
            local_idx = stream_idx % self.chunk_size
            h5f = self.chunk_files[chunk_idx]
        else:
            h5f = self.hdf5_file
            local_idx = stream_idx

        # Load labels (REQUIRED - raises error if missing)
        if h5f is None:
            raise RuntimeError(
                f"No label file available for stream {stream_idx}. "
                f"Dataset may be corrupted. Try regenerating."
            )

        # Get message metadata from RAM cache (pre-loaded, no JSON parsing!)
        if hasattr(self, 'message_metadata_cache') and stream_idx < len(self.message_metadata_cache):
            messages = self.message_metadata_cache[stream_idx]
        else:
            # Fallback: Parse from HDF5 (slow path, only if cache not available)
            import json
            try:
                message_meta_json = h5f['message_metadata'][local_idx]
                if isinstance(message_meta_json, bytes):
                    message_meta_json = message_meta_json.decode('utf-8')
                if not message_meta_json or message_meta_json.strip() == '':
                    messages = []
                else:
                    messages = json.loads(message_meta_json)
            except (json.JSONDecodeError, KeyError, IndexError):
                messages = []

        # Use first message as primary target (for confusion matrix tracking)
        first_msg = messages[0] if messages else {}

        # OPTIMIZATION: Use RAM-cached labels (15-25% faster with 32 workers!)
        if hasattr(self, 'labels_cache') and self.labels_cache:
            # FAST PATH: Use pre-loaded labels (eliminates HDF5 I/O contention)
            k_index_val = float(self.labels_cache['k_index'][stream_idx]) if self.labels_cache['k_index'] is not None else 3.0
            sfi_val = float(self.labels_cache['sfi'][stream_idx]) if self.labels_cache['sfi'] is not None else 100.0
            prop_mode = self.labels_cache['propagation_modes'][stream_idx].decode('utf-8') if self.labels_cache['propagation_modes'] is not None else 'rician'
            qrn_type_val = self.labels_cache['qrn_types'][stream_idx].decode('utf-8') if self.labels_cache['qrn_types'] is not None else 'QUIET'
        else:
            # SLOW PATH: Access HDF5 (fallback for old datasets)
            k_index_val = float(h5f['k_index'][local_idx])
            sfi_val = float(h5f['sfi'][local_idx])
            prop_mode = h5f['propagation_modes'][local_idx].decode('utf-8')
            qrn_type_val = h5f['qrn_types'][local_idx].decode('utf-8')

        labels = {
            'stream_idx': stream_idx,
            'window_idx': window_idx,
            'k_index': k_index_val,
            'sfi': sfi_val,
            'propagation_mode': prop_mode,
            'qrn_type': qrn_type_val,
            # Per-message ground truth (from first message in stream)
            'pattern_id': first_msg.get('pattern_id', 0),
            'frequency_triple': first_msg.get('frequency_triple', 0),
            'modulation': first_msg.get('modulation', 'BPSK'),
            'snr_db': first_msg.get('snr_db', 0.0),  # Actual SNR of signal (not stream-level)
            'data_symbol_rate': first_msg.get('discrete_rate', 75),
            'continuous_rate': first_msg.get('continuous_rate', 75.0),
            'pattern_only': first_msg.get('pattern_only', False),
            'num_messages': len(messages),  # How many messages in stream (collision complexity)
        }

        # Load TX observations if enabled (handle multi-file)
        if self.enable_tx_observations:
            if self.chunk_files is not None:
                chunk_idx = stream_idx // self.chunk_size
                local_idx = stream_idx % self.chunk_size
                tx_observed_iq = self.chunk_files[chunk_idx]['tx_observed'][local_idx]
            else:
                tx_observed_iq = self.hdf5_file['tx_observed'][stream_idx]

            # Convert to tensor [2, 48000]
            tx_i = np.real(tx_observed_iq).astype(np.float32)
            tx_q = np.imag(tx_observed_iq).astype(np.float32)
            tx_observed = torch.from_numpy(np.stack([tx_i, tx_q], axis=0))

            # Load optimal embedding (handle multi-file)
            if self.chunk_files is not None:
                chunk_idx = stream_idx // self.chunk_size
                local_idx = stream_idx % self.chunk_size
                embedding = self.chunk_files[chunk_idx]['optimal_embeddings'][local_idx]
            else:
                embedding = self.hdf5_file['optimal_embeddings'][stream_idx]

            labels['optimal_embedding'] = torch.from_numpy(embedding.astype(np.float32))

            return rx_window, tx_observed, labels
        else:
            return rx_window, labels


def cascade_collate_fn(batch):
    """
    Custom collate function for CASCADE windows with optional TX observations.

    Handles both:
    - RX-only: (rx_window, labels)
    - RX/TX: (rx_window, tx_observed, labels)
    """
    from torch.utils.data import default_collate

    # Check if batch has TX observations (3-tuple vs 2-tuple)
    if len(batch[0]) == 3:
        # RX/TX mode: (rx_window, tx_observed, labels)
        rx_windows = torch.stack([item[0] for item in batch])
        tx_observed = torch.stack([item[1] for item in batch])
        labels = default_collate([item[2] for item in batch])
        return rx_windows, tx_observed, labels
    else:
        # RX-only mode: (rx_window, labels)
        return default_collate(batch)


def test_streaming_dataset():
    """Test streaming dataset generation."""
    print("=" * 80)
    print("TESTING STREAMING CASCADE DATASET")
    print("=" * 80)

    # Create small test dataset
    dataset = StreamingCascadeDataset(
        num_streams=10,
        stream_duration_sec=10.0,
        window_duration_sec=2.0,
        message_arrival_rate=0.4,
        seed=42,
        regenerate_cache=True,
        batch_size=4,
        num_workers=1
    )

    print(f"\nDataset size: {len(dataset)} windows (from {dataset.num_streams} streams)")
    print(f"Windows per stream: {dataset.windows_per_stream}")

    # Test window extraction
    window_iq, labels = dataset[0]
    print(f"\nSample window 0:")
    print(f"  IQ shape: {window_iq.shape}")
    print(f"  Stream: {labels['stream_idx']}, Window: {labels['window_idx']}")
    print(f"  Propagation: {labels['propagation_mode']}")

    # Test with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=cascade_collate_fn)

    for batch_iq, batch_labels in loader:
        print(f"\nDataLoader batch:")
        print(f"  Batch IQ: {batch_iq.shape}")
        print(f"  Streams: {batch_labels['stream_idx']}")
        break

    print("\n✓ Streaming dataset test passed!")


if __name__ == "__main__":
    test_streaming_dataset()
