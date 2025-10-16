"""
Physics-constrained dataset for CASCADE training.

Replaces random QRN/propagation generation with physics-based scenarios.
All parameters are coupled and sampled continuously to prevent overfitting.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import h5py
import os
import sys
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from physics_coupling import (
    CorePhysicalDrivers, CoupledPhysicsCalculator,
    DerivedConditions, PropagationMode, QRNType
)
from scenarios import ScenarioLibrary, ScenarioType
from modules.training.src.signal_generator.generator import SignalGenerator
from modules.training.src.channel_simulator.multipath import apply_multipath_fading, MultipathProfile
from modules.training.src.channel_simulator.awgn import generate_awgn
from signal_visualizer import SignalVisualizer


@dataclass
class PhysicsBasedSample:
    """A single training sample with physics-coupled parameters."""
    # Input signal
    clean_signal: np.ndarray  # Clean transmitted signal
    received_signal: np.ndarray  # After channel + noise

    # Core physical state
    drivers: CorePhysicalDrivers

    # Derived channel conditions
    conditions: DerivedConditions

    # Ground truth labels
    message_bits: np.ndarray  # Transmitted bits
    pattern_id: int  # Ternary orthogonal pattern (0-3)
    frequency_triple: int  # Which frequency triple (0-42)
    snr_db: float  # Actual SNR

    # Metadata
    scenario_name: str
    scenario_type: ScenarioType


class PhysicsConstrainedDataset(Dataset):
    """
    Dataset that generates signals with physics-coupled channel effects.

    Key features:
    1. **Physics coupling**: QRN, propagation, absorption all from same physical state
    2. **Continuous variation**: Every sample unique (prevents overfitting)
    3. **Balanced-realistic weighting**: Rare conditions oversampled
    4. **Scenario-based**: 9 fundamental scenarios → many variations
    """

    def __init__(
        self,
        num_samples: int,
        signal_generator: SignalGenerator,
        sample_rate: int = 48000,
        for_test: bool = False,
        seed: Optional[int] = None,
        enable_real_signal_augmentation: bool = False,
        real_signal_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        regenerate_cache: bool = False,
        enable_visualization: bool = False,
        visualization_dir: Optional[str] = None,
        visualization_sample_rate: float = 0.01
    ):
        """
        Initialize physics-constrained dataset.

        Args:
            num_samples: Number of samples in dataset
            signal_generator: SignalGenerator instance for creating signals
            sample_rate: Audio sample rate (Hz)
            for_test: If True, use harder test distribution
            seed: Random seed for reproducibility
            enable_real_signal_augmentation: Apply physics to real HF recordings
            real_signal_path: Path to real signal recordings (.npz or .wav)
            cache_dir: Directory to store cached signals (default: ./dataset_cache)
            use_cache: If True, use cached signals if available
            regenerate_cache: If True, regenerate cache even if it exists
            enable_visualization: If True, create spectrograms during generation
            visualization_dir: Directory to save visualizations (default: ./visualizations)
            visualization_sample_rate: Fraction of signals to visualize (0.01 = 1%)
        """
        self.num_samples = num_samples
        self.signal_generator = signal_generator
        self.sample_rate = sample_rate
        self.for_test = for_test
        self.seed = seed
        self.enable_real_signal_augmentation = enable_real_signal_augmentation
        self.real_signal_path = real_signal_path
        self.use_cache = use_cache
        self.enable_visualization = enable_visualization
        self.visualization_sample_rate = visualization_sample_rate

        # Cache configuration
        if cache_dir is None:
            cache_dir = './dataset_cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Visualization configuration
        if visualization_dir is None:
            visualization_dir = './visualizations'
        self.visualization_dir = Path(visualization_dir)
        if self.enable_visualization:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)

        cache_name = f"physics_dataset_n{num_samples}_sr{sample_rate}_{'test' if for_test else 'train'}_seed{seed}.h5"
        self.cache_path = self.cache_dir / cache_name

        # Initialize scenario library and physics calculator (suppress verbose output)
        self.scenario_library = ScenarioLibrary()
        self.physics_calc = CoupledPhysicsCalculator(seed=seed)

        # Pre-generate scenario parameters
        print(f"\n{'='*60}")
        print(f"PHYSICS-CONSTRAINED DATASET INITIALIZATION")
        print(f"{'='*60}")
        print(f"Samples: {num_samples:,} ({'test' if for_test else 'train'} distribution)")
        print(f"Generating physics-coupled scenarios...")

        self.scenario_drivers = self.scenario_library.generate_balanced_realistic_batch(
            batch_size=num_samples,
            for_test=for_test,
            seed=seed
        )
        print(f"✓ Scenarios generated")

        # Load real signals if augmentation enabled
        self.real_signals = None
        if enable_real_signal_augmentation and real_signal_path:
            self.real_signals = self._load_real_signals(real_signal_path)

        # Handle caching
        self.cached_data = None
        if use_cache:
            if self.cache_path.exists() and not regenerate_cache:
                print(f"\nLoading from cache: {self.cache_path.name}")
                self._load_cache()
            else:
                self._generate_and_cache()
        else:
            print("\n⚠ Cache disabled - signals will be generated on-the-fly (SLOW)")

        self._print_distribution_stats()

    def _load_real_signals(self, path: str) -> List[np.ndarray]:
        """Load real HF signal recordings."""
        try:
            data = np.load(path)
            if 'signals' in data:
                signals = data['signals']
            else:
                # Assume first array is signals
                signals = list(data.values())[0]

            print(f"✓ Loaded {len(signals)} real signal recordings")
            return signals
        except Exception as e:
            print(f"⚠ Could not load real signals from {path}: {e}")
            return None

    def _print_distribution_stats(self):
        """Print statistics about scenario distribution."""
        # Calculate derived conditions for all scenarios
        all_conditions = [
            self.physics_calc.calculate_all_effects(drivers)
            for drivers in self.scenario_drivers[:100]  # Sample first 100
        ]

        # Count scenario types
        snr_ranges = {'<-10 dB': 0, '-10 to 0 dB': 0, '0 to 10 dB': 0, '10 to 20 dB': 0, '>20 dB': 0}
        prop_modes = {}
        qrn_types = {}

        for cond in all_conditions:
            # SNR distribution
            if cond.effective_snr_db < -10:
                snr_ranges['<-10 dB'] += 1
            elif cond.effective_snr_db < 0:
                snr_ranges['-10 to 0 dB'] += 1
            elif cond.effective_snr_db < 10:
                snr_ranges['0 to 10 dB'] += 1
            elif cond.effective_snr_db < 20:
                snr_ranges['10 to 20 dB'] += 1
            else:
                snr_ranges['>20 dB'] += 1

            # Propagation mode distribution
            mode = cond.propagation_mode.value
            prop_modes[mode] = prop_modes.get(mode, 0) + 1

            # QRN type distribution
            qrn = cond.dominant_qrn_type.value
            qrn_types[qrn] = qrn_types.get(qrn, 0) + 1

        print("\n" + "="*60)
        print("DATASET DISTRIBUTION (sampled from first 100):")
        print("="*60)
        print("\nSNR Distribution:")
        for range_name, count in snr_ranges.items():
            print(f"  {range_name}: {count}%")

        print("\nPropagation Modes:")
        for mode, count in sorted(prop_modes.items(), key=lambda x: -x[1]):
            print(f"  {mode}: {count}%")

        print("\nDominant QRN Types:")
        for qrn, count in sorted(qrn_types.items(), key=lambda x: -x[1]):
            print(f"  {qrn}: {count}%")
        print("="*60 + "\n")

    def _generate_and_cache(self):
        """Generate all signals and cache to disk with optimizations."""
        # Use fixed signal length (matches IQ encoder input size)
        FIXED_SIGNAL_LEN = 2048
        BATCH_SIZE = 1000  # Write in batches

        # Determine number of worker processes (leave 1 core free)
        num_workers = max(1, cpu_count() - 1)

        print(f"\nGenerating {self.num_samples:,} signals with {num_workers} parallel workers...")
        print(f"Cache: {self.cache_path.name}")

        # Initialize visualizer if enabled
        visualizer = None
        if self.enable_visualization:
            visualizer = SignalVisualizer(
                output_dir=str(self.visualization_dir),
                max_queue_size=50
            )
            visualizer.start()
            num_to_visualize = max(1, int(self.num_samples * self.visualization_sample_rate))
            print(f"Visualization: enabled ({num_to_visualize} samples @ {self.visualization_sample_rate*100:.1f}%)")

        # Create HDF5 file
        with h5py.File(self.cache_path, 'w') as f:
            # Create datasets with fixed length and LZF compression (10-100x faster than gzip)
            signals_ds = f.create_dataset('signals', shape=(self.num_samples, FIXED_SIGNAL_LEN),
                                         dtype=np.complex64, compression='lzf')
            clean_signals_ds = f.create_dataset('clean_signals', shape=(self.num_samples, FIXED_SIGNAL_LEN),
                                               dtype=np.complex64, compression='lzf')
            snr_ds = f.create_dataset('snr_db', shape=(self.num_samples,), dtype=np.float32)
            pattern_ids = f.create_dataset('pattern_ids', shape=(self.num_samples,), dtype=np.int16)
            freq_triples = f.create_dataset('frequency_triples', shape=(self.num_samples,), dtype=np.int16)
            qrn_types = f.create_dataset('qrn_types', shape=(self.num_samples,), dtype='S16')  # String dtype for QRN type
            modulations = f.create_dataset('modulations', shape=(self.num_samples,), dtype='S16')  # BPSK, QPSK, etc.
            data_symbol_rates = f.create_dataset('data_symbol_rates', shape=(self.num_samples,), dtype=np.int16)  # 75-300
            duration_windows_ds = f.create_dataset('duration_windows', shape=(self.num_samples,), dtype=np.uint8)  # 0-255
            propagation_modes = f.create_dataset('propagation_modes', shape=(self.num_samples,), dtype='S32')  # Propagation mode strings

            # Batch processing with multiprocessing
            num_batches = (self.num_samples + BATCH_SIZE - 1) // BATCH_SIZE

            # Initialize pool with worker init function (loads patterns once per worker)
            with Pool(processes=num_workers, initializer=self._init_worker) as pool:
                # Single unified progress bar tracking total samples processed
                with tqdm(total=self.num_samples, desc="Generating & caching", unit="samples",
                         ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                         mininterval=0.5, file=sys.stdout) as pbar:

                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * BATCH_SIZE
                        end_idx = min(start_idx + BATCH_SIZE, self.num_samples)
                        batch_size = end_idx - start_idx

                        # Prepare arguments for parallel processing
                        args_list = [
                            (idx, self.scenario_drivers[idx], FIXED_SIGNAL_LEN)
                            for idx in range(start_idx, end_idx)
                        ]

                        # Generate batch in parallel
                        batch_results = pool.starmap(self._generate_single_sample_static, args_list)

                        # Extract results
                        batch_signals = np.zeros((batch_size, FIXED_SIGNAL_LEN), dtype=np.complex64)
                        batch_clean_signals = np.zeros((batch_size, FIXED_SIGNAL_LEN), dtype=np.complex64)
                        batch_snrs = np.zeros(batch_size, dtype=np.float32)
                        batch_pattern_ids = np.zeros(batch_size, dtype=np.int16)
                        batch_freq_triples = np.zeros(batch_size, dtype=np.int16)
                        batch_qrn_types = []
                        batch_modulations = []
                        batch_data_symbol_rates = np.zeros(batch_size, dtype=np.int16)
                        batch_duration_windows = np.zeros(batch_size, dtype=np.uint8)
                        batch_propagation_modes = []

                        for i, result in enumerate(batch_results):
                            (received_signal, clean_signal, snr_db, pattern_id, frequency_triple,
                             qrn_type, modulation, data_symbol_rate, duration_windows, propagation_mode) = result
                            batch_signals[i] = received_signal
                            batch_clean_signals[i] = clean_signal
                            batch_snrs[i] = snr_db
                            batch_pattern_ids[i] = pattern_id
                            batch_freq_triples[i] = frequency_triple
                            batch_qrn_types.append(qrn_type.encode('utf-8'))
                            batch_modulations.append(modulation.encode('utf-8'))
                            batch_data_symbol_rates[i] = data_symbol_rate
                            batch_duration_windows[i] = duration_windows
                            batch_propagation_modes.append(propagation_mode.encode('utf-8'))

                        # Write entire batch at once (much faster than individual writes)
                        signals_ds[start_idx:end_idx] = batch_signals
                        clean_signals_ds[start_idx:end_idx] = batch_clean_signals
                        snr_ds[start_idx:end_idx] = batch_snrs
                        pattern_ids[start_idx:end_idx] = batch_pattern_ids
                        freq_triples[start_idx:end_idx] = batch_freq_triples
                        qrn_types[start_idx:end_idx] = batch_qrn_types
                        modulations[start_idx:end_idx] = batch_modulations
                        data_symbol_rates[start_idx:end_idx] = batch_data_symbol_rates
                        duration_windows_ds[start_idx:end_idx] = batch_duration_windows
                        propagation_modes[start_idx:end_idx] = batch_propagation_modes

                        # Randomly sample signals for visualization
                        if visualizer is not None:
                            for i in range(batch_size):
                                # Random sampling based on visualization_sample_rate
                                if np.random.rand() < self.visualization_sample_rate:
                                    sample_idx = start_idx + i

                                    # Create metadata dictionary
                                    metadata = {
                                        'pattern_id': int(batch_pattern_ids[i]),
                                        'frequency_triple': int(batch_freq_triples[i]),
                                        'snr_db': float(batch_snrs[i]),
                                        'qrn_type': batch_qrn_types[i].decode('utf-8') if isinstance(batch_qrn_types[i], bytes) else str(batch_qrn_types[i]),
                                        'modulation': batch_modulations[i].decode('utf-8') if isinstance(batch_modulations[i], bytes) else str(batch_modulations[i]),
                                        'data_symbol_rate': int(batch_data_symbol_rates[i]),
                                        'propagation_mode': batch_propagation_modes[i].decode('utf-8') if isinstance(batch_propagation_modes[i], bytes) else str(batch_propagation_modes[i]),
                                        'sample_idx': sample_idx
                                    }

                                    # Send to visualizer (non-blocking)
                                    visualizer.add_signal(
                                        iq_samples=batch_signals[i],
                                        sample_rate=self.sample_rate,
                                        metadata=metadata,
                                        sample_idx=sample_idx,
                                        blocking=False
                                    )

                        # Update progress bar
                        pbar.update(batch_size)
                        pbar.refresh()  # Force immediate update
                        sys.stdout.flush()  # Force output flush

            # Store metadata
            f.attrs['num_samples'] = self.num_samples
            f.attrs['sample_rate'] = self.sample_rate
            f.attrs['signal_length'] = FIXED_SIGNAL_LEN
            f.attrs['for_test'] = self.for_test
            f.attrs['seed'] = self.seed if self.seed is not None else -1

        # Stop visualizer and wait for completion
        if visualizer is not None:
            print("Completing visualizations...")
            visualizer.stop()
            print(f"✓ Visualizations saved to {self.visualization_dir}")

        # Load the cache into memory
        self._load_cache()

        print(f"✓ Cache generated: {self.num_samples:,} samples saved to {self.cache_path.name}")

    # Worker-level cache (shared across all samples in a worker process)
    _worker_signal_gen = None
    _worker_physics_calc = None

    @staticmethod
    def _init_worker():
        """Initialize worker process with cached objects (called once per worker)."""
        from modules.training.src.signal_generator.generator import SignalGenerator

        # Create objects once per worker and cache them
        PhysicsConstrainedDataset._worker_physics_calc = CoupledPhysicsCalculator(seed=None)
        PhysicsConstrainedDataset._worker_signal_gen = SignalGenerator()

    @staticmethod
    def _generate_single_sample_static(idx: int, drivers: CorePhysicalDrivers, fixed_len: int) -> Tuple:
        """
        Static method for parallel signal generation (must be picklable).

        This is a wrapper that uses cached objects from worker initialization.
        Returns: (received_signal, clean_signal, snr_db, pattern_id, frequency_triple, qrn_type,
                  modulation, data_symbol_rate, duration_windows, propagation_mode)
        """
        # Use worker-level cached instances (initialized once per worker)
        physics_calc = PhysicsConstrainedDataset._worker_physics_calc
        signal_gen = PhysicsConstrainedDataset._worker_signal_gen

        # Calculate conditions
        conditions = physics_calc.calculate_all_effects(drivers)

        # Generate clean signal
        message_len = np.random.randint(50, 150)
        message_bytes = np.random.bytes(message_len)
        pattern_id = np.random.randint(0, 4)
        frequency_triple = int((drivers.frequency_mhz - 0.3) / 0.02) // 3
        frequency_triple = np.clip(frequency_triple, 0, 42)

        modulation_scheme = 'QPSK'
        polar_rate = (2, 3)
        data_symbol_rate = 150

        clean_iq_signal, metadata = signal_gen.generate(
            pattern_id=pattern_id,
            frequency_triple=frequency_triple,
            modulation_scheme=modulation_scheme,
            polar_rate=polar_rate,
            data_symbol_rate=data_symbol_rate,
            message=message_bytes,
            seed=None
        )

        clean_signal_original = clean_iq_signal.iq_samples.copy()

        # Apply channel effects
        received_signal = PhysicsConstrainedDataset._apply_channel_effects_static(
            clean_signal_original, drivers, conditions, 48000
        )

        # Pad or truncate BOTH signals to fixed length
        clean_signal = PhysicsConstrainedDataset._pad_or_truncate(clean_signal_original, fixed_len)
        received_signal = PhysicsConstrainedDataset._pad_or_truncate(received_signal, fixed_len)

        # Get dominant QRN type
        qrn_type = conditions.dominant_qrn_type.value

        # Get propagation mode
        propagation_mode = conditions.propagation_mode.value

        # Calculate duration in 341ms windows
        # duration_seconds is in metadata, convert to windows
        duration_seconds = metadata.get('duration_seconds', 1.0)
        duration_windows = int(np.ceil(duration_seconds / 0.341))
        duration_windows = np.clip(duration_windows, 0, 255)

        return (received_signal, clean_signal, conditions.effective_snr_db, pattern_id, frequency_triple,
                qrn_type, modulation_scheme, data_symbol_rate, duration_windows, propagation_mode)

    @staticmethod
    def _pad_or_truncate(signal: np.ndarray, fixed_len: int) -> np.ndarray:
        """Pad or truncate signal to fixed length."""
        if len(signal) < fixed_len:
            padded = np.zeros(fixed_len, dtype=np.complex64)
            padded[:len(signal)] = signal
            return padded
        else:
            return signal[:fixed_len]

    @staticmethod
    def _apply_channel_effects_static(
        clean_signal: np.ndarray,
        drivers: CorePhysicalDrivers,
        conditions: DerivedConditions,
        sample_rate: int
    ) -> np.ndarray:
        """
        Static method to apply channel effects (for parallel processing).

        Simplified version that doesn't require instance methods.
        """
        signal = clean_signal.copy()

        # 1. Apply propagation mode
        mode = conditions.propagation_mode

        if mode == PropagationMode.RAYLEIGH:
            from modules.training.src.channel_simulator.multipath import apply_multipath_fading, MultipathProfile
            profile = MultipathProfile(
                delays=[0.0],
                powers=[1.0],
                doppler_shifts=[conditions.doppler_spread_hz],
                k_factors=[0.0]
            )
            signal = apply_multipath_fading(signal, profile, sample_rate)

        elif mode == PropagationMode.RICIAN:
            from modules.training.src.channel_simulator.multipath import apply_multipath_fading, MultipathProfile
            profile = MultipathProfile(
                delays=[0.0],
                powers=[1.0],
                doppler_shifts=[conditions.doppler_spread_hz],
                k_factors=[conditions.rician_k_factor]
            )
            signal = apply_multipath_fading(signal, profile, sample_rate)

        elif mode == PropagationMode.MULTIPATH_SPARSE:
            from modules.training.src.channel_simulator.multipath import watterson_hf_profile, apply_multipath_fading
            profile = watterson_hf_profile(
                delay_spread_ms=conditions.multipath_delay_spread_ms,
                doppler_spread_hz=conditions.doppler_spread_hz
            )
            signal = apply_multipath_fading(signal, profile, sample_rate)

        elif mode == PropagationMode.MULTIPATH_DENSE:
            from modules.training.src.channel_simulator.multipath import severe_multipath_profile, apply_multipath_fading
            profile = severe_multipath_profile(
                num_paths=6,
                max_delay_ms=conditions.multipath_delay_spread_ms,
                max_doppler_hz=conditions.doppler_spread_hz
            )
            signal = apply_multipath_fading(signal, profile, sample_rate)

        # 2. Apply D-layer absorption
        absorption_linear = 10 ** (-conditions.d_layer_absorption_db / 20)
        signal = signal * absorption_linear

        # 3. Apply simplified QRN (basic atmospheric noise)
        qrn_power_db = conditions.atmospheric_noise_db
        qrn_power_linear = 10 ** (qrn_power_db / 20)
        qrn_signal = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * qrn_power_linear
        signal = signal + qrn_signal

        # 4. Apply AWGN to reach target SNR
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (conditions.effective_snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power / 2), len(signal)) + \
                1j * np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
        signal = signal + noise

        return signal

    def _load_cache(self):
        """Load cached signals into memory."""
        with h5py.File(self.cache_path, 'r') as f:
            # Load all data into memory for fast access
            self.cached_data = {
                'signals': f['signals'][:],
                'clean_signals': f['clean_signals'][:] if 'clean_signals' in f else None,
                'snr_db': f['snr_db'][:],
                'pattern_ids': f['pattern_ids'][:],
                'frequency_triples': f['frequency_triples'][:],
                'qrn_types': f['qrn_types'][:] if 'qrn_types' in f else None,
                'modulations': f['modulations'][:] if 'modulations' in f else None,
                'data_symbol_rates': f['data_symbol_rates'][:] if 'data_symbol_rates' in f else None,
                'duration_windows': f['duration_windows'][:] if 'duration_windows' in f else None,
                'propagation_modes': f['propagation_modes'][:] if 'propagation_modes' in f else None
            }

        print(f"✓ Loaded {len(self.cached_data['signals']):,} samples into memory")
        extra_features = []
        if self.cached_data['clean_signals'] is not None:
            extra_features.append("clean signals")
        if self.cached_data['modulations'] is not None:
            extra_features.append("modulation/rate/duration")
        if extra_features:
            print(f"  (includes {', '.join(extra_features)})")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a single training sample.

        Returns:
            received_iq: Torch tensor of received I/Q samples [2, num_samples]
            labels: Dictionary with ground truth and metadata
        """
        # Use cached data if available
        if self.cached_data is not None:
            # Load from cache (FAST)
            received_signal = self.cached_data['signals'][idx]
            snr_db = float(self.cached_data['snr_db'][idx])
            pattern_id = int(self.cached_data['pattern_ids'][idx])
            frequency_triple = int(self.cached_data['frequency_triples'][idx])

            # Get scenario parameters
            drivers = self.scenario_drivers[idx]
            conditions = self.physics_calc.calculate_all_effects(drivers)

            # Convert complex signal to I/Q tensor shape (2, num_samples)
            iq_i = np.real(received_signal).astype(np.float32)
            iq_q = np.imag(received_signal).astype(np.float32)
            received_iq = torch.from_numpy(np.stack([iq_i, iq_q], axis=0))

            labels = {
                'pattern_id': pattern_id,
                'frequency_triple': frequency_triple,
                'snr_db': snr_db,
                'sfi': drivers.sfi,
                'k_index': drivers.k_index,
                'frequency_mhz': drivers.frequency_mhz,
                'sample_idx': idx
            }

            # Add clean signal and QRN type if available (for QRN Expert training)
            if self.cached_data['clean_signals'] is not None:
                clean_signal = self.cached_data['clean_signals'][idx]
                clean_iq_i = np.real(clean_signal).astype(np.float32)
                clean_iq_q = np.imag(clean_signal).astype(np.float32)
                labels['clean_iq'] = torch.from_numpy(np.stack([clean_iq_i, clean_iq_q], axis=0))

            if self.cached_data['qrn_types'] is not None:
                qrn_type_bytes = self.cached_data['qrn_types'][idx]
                labels['qrn_type'] = qrn_type_bytes.decode('utf-8') if isinstance(qrn_type_bytes, bytes) else str(qrn_type_bytes)

            # Add modulation, data rate, duration (for Signal Expert and Decoder training)
            if self.cached_data['modulations'] is not None:
                modulation_bytes = self.cached_data['modulations'][idx]
                labels['modulation'] = modulation_bytes.decode('utf-8') if isinstance(modulation_bytes, bytes) else str(modulation_bytes)

            if self.cached_data['data_symbol_rates'] is not None:
                labels['data_symbol_rate'] = int(self.cached_data['data_symbol_rates'][idx])

            if self.cached_data['duration_windows'] is not None:
                labels['duration_windows'] = int(self.cached_data['duration_windows'][idx])

            # Add propagation mode (for Channel Expert training)
            if self.cached_data['propagation_modes'] is not None:
                prop_mode_bytes = self.cached_data['propagation_modes'][idx]
                labels['propagation_mode'] = prop_mode_bytes.decode('utf-8') if isinstance(prop_mode_bytes, bytes) else str(prop_mode_bytes)

        else:
            # Generate on-the-fly (SLOW - only if cache disabled)
            drivers = self.scenario_drivers[idx]
            conditions = self.physics_calc.calculate_all_effects(drivers)

            if self.enable_real_signal_augmentation and self.real_signals is not None:
                clean_signal = self._get_random_real_signal()
                message_bits = None
                pattern_id = None
                frequency_triple = None
            else:
                message_bits, clean_signal, pattern_id, frequency_triple = self._generate_clean_signal(
                    drivers.frequency_mhz
                )

            received_signal = self._apply_channel_effects(clean_signal, drivers, conditions)

            # Convert to I/Q tensor
            iq_i = np.real(received_signal).astype(np.float32)
            iq_q = np.imag(received_signal).astype(np.float32)
            received_iq = torch.from_numpy(np.stack([iq_i, iq_q], axis=0))

            labels = {
                'message_bits': message_bits,
                'pattern_id': pattern_id,
                'frequency_triple': frequency_triple,
                'snr_db': conditions.effective_snr_db,
                'sfi': drivers.sfi,
                'k_index': drivers.k_index,
                'frequency_mhz': drivers.frequency_mhz,
                'utc_hour': drivers.utc_hour,
                'latitude': drivers.latitude,
                'thunderstorm_activity': drivers.thunderstorm_activity,
                'propagation_mode': conditions.propagation_mode.value,
                'dominant_qrn_type': conditions.dominant_qrn_type.value,
                'muf_mhz': conditions.muf_mhz,
                'd_layer_absorption_db': conditions.d_layer_absorption_db,
                'delay_spread_ms': conditions.multipath_delay_spread_ms,
                'doppler_spread_hz': conditions.doppler_spread_hz,
                'is_real_signal': self.enable_real_signal_augmentation and self.real_signals is not None,
                'sample_idx': idx
            }

        return received_iq, labels

    def _generate_clean_signal(self, frequency_mhz: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Generate a clean synthetic CASCADE signal."""
        # Random message (50-150 bytes)
        message_len = np.random.randint(50, 150)
        message_bytes = np.random.bytes(message_len)

        # Convert to bits for labels
        message_bits = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))

        # Random pattern ID (0-3 for 4 patterns with 3-FSK)
        pattern_id = np.random.randint(0, 4)

        # Convert frequency to frequency triple (0-42)
        # 129 channels / 3 = 43 triples (0-42)
        # Simplified mapping from frequency_mhz to triple
        frequency_triple = int((frequency_mhz - 0.3) / 0.02) // 3
        frequency_triple = np.clip(frequency_triple, 0, 42)

        # Select modulation and symbol rate based on expected SNR
        # (simplified - in real system would come from kernel)
        modulation_scheme = 'QPSK'  # Default
        polar_rate = (2, 3)  # Rate 2/3
        data_symbol_rate = 150  # Default 150 sym/s

        # Generate signal using correct SignalGenerator API
        clean_iq_signal, metadata = self.signal_generator.generate(
            pattern_id=pattern_id,
            frequency_triple=frequency_triple,
            modulation_scheme=modulation_scheme,
            polar_rate=polar_rate,
            data_symbol_rate=data_symbol_rate,
            message=message_bytes,
            seed=None
        )

        # Extract IQ samples from CleanIQSignal dataclass
        clean_signal = clean_iq_signal.iq_samples

        return message_bits, clean_signal, pattern_id, frequency_triple

    def _get_random_real_signal(self) -> np.ndarray:
        """Get a random real signal from loaded recordings."""
        idx = np.random.randint(0, len(self.real_signals))
        signal = self.real_signals[idx]

        # Random crop if signal is long
        if len(signal) > self.sample_rate * 10:  # Longer than 10 seconds
            start = np.random.randint(0, len(signal) - self.sample_rate * 10)
            signal = signal[start:start + self.sample_rate * 10]

        return signal

    def _apply_channel_effects(
        self,
        clean_signal: np.ndarray,
        drivers: CorePhysicalDrivers,
        conditions: DerivedConditions
    ) -> np.ndarray:
        """
        Apply physics-coupled channel effects to signal.

        All effects (QRN, propagation, absorption) are derived from same physical state.
        """
        signal = clean_signal.copy()

        # 1. Apply propagation mode
        signal = self._apply_propagation(signal, conditions)

        # 2. Apply QRN (coupled to same physics)
        signal = self._apply_qrn(signal, drivers, conditions)

        # 3. Apply D-layer absorption
        absorption_linear = 10 ** (-conditions.d_layer_absorption_db / 20)
        signal = signal * absorption_linear

        # 4. Apply AWGN to reach target SNR
        signal = self._add_awgn_for_target_snr(signal, conditions.effective_snr_db)

        return signal

    def _apply_propagation(self, signal: np.ndarray, conditions: DerivedConditions) -> np.ndarray:
        """Apply propagation mode (multipath, fading, Doppler)."""
        mode = conditions.propagation_mode

        if mode == PropagationMode.AWGN:
            # No multipath
            return signal

        elif mode == PropagationMode.RAYLEIGH:
            # Rayleigh fading (single path with fading)
            profile = MultipathProfile(
                delays=[0.0],
                powers=[1.0],
                doppler_shifts=[conditions.doppler_spread_hz],
                k_factors=[0.0]  # Pure Rayleigh (no LOS)
            )
            return apply_multipath_fading(signal, profile, self.sample_rate)

        elif mode == PropagationMode.RICIAN:
            # Rician fading (dominant LOS path)
            # Use single path with Rician K-factor
            profile = MultipathProfile(
                delays=[0.0],
                powers=[1.0],
                doppler_shifts=[conditions.doppler_spread_hz],
                k_factors=[conditions.rician_k_factor]  # Rician K-factor in dB
            )
            return apply_multipath_fading(signal, profile, self.sample_rate)

        elif mode == PropagationMode.MULTIPATH_SPARSE:
            # Sparse multipath (2-3 paths)
            # Create simple 2-path Watterson model
            from modules.training.src.channel_simulator.multipath import watterson_hf_profile
            profile = watterson_hf_profile(
                delay_spread_ms=conditions.multipath_delay_spread_ms,
                doppler_spread_hz=conditions.doppler_spread_hz
            )
            return apply_multipath_fading(signal, profile, self.sample_rate)

        elif mode == PropagationMode.MULTIPATH_DENSE:
            # Dense multipath (4+ paths)
            from modules.training.src.channel_simulator.multipath import severe_multipath_profile
            profile = severe_multipath_profile(
                num_paths=6,
                max_delay_ms=conditions.multipath_delay_spread_ms,
                max_doppler_hz=conditions.doppler_spread_hz
            )
            return apply_multipath_fading(signal, profile, self.sample_rate)

        return signal

    def _apply_qrn(
        self,
        signal: np.ndarray,
        drivers: CorePhysicalDrivers,
        conditions: DerivedConditions
    ) -> np.ndarray:
        """
        Apply QRN based on physics-coupled QRN characteristics.

        QRN type and level are derived from same drivers as propagation.
        """
        # Mix of QRN components
        qrn_signal = np.zeros_like(signal)

        for qrn_type, weight in conditions.qrn_components.items():
            if weight < 0.01:
                continue

            # Generate QRN component
            if qrn_type == QRNType.STATIC:
                noise = self._generate_atmospheric_static(len(signal))
            elif qrn_type == QRNType.CRACKLING:
                noise = self._generate_crackling_noise(len(signal), drivers.thunderstorm_activity)
            elif qrn_type == QRNType.POPCORN:
                noise = self._generate_popcorn_noise(len(signal), drivers.thunderstorm_activity)
            elif qrn_type == QRNType.THUNDERSTORM:
                noise = self._generate_thunderstorm_crashes(len(signal), drivers.thunderstorm_activity)
            elif qrn_type == QRNType.HISS:
                noise = self._generate_auroral_hiss(len(signal), drivers.k_index)
            elif qrn_type == QRNType.AURORAL:
                noise = self._generate_auroral_noise(len(signal), drivers.k_index)
            elif qrn_type == QRNType.FLUTTERY:
                noise = self._generate_fluttery_noise(len(signal), drivers.k_index)
            else:  # QUIET
                noise = np.zeros(len(signal))

            qrn_signal += weight * noise

        # Scale QRN to target atmospheric noise level
        qrn_power_db = conditions.atmospheric_noise_db
        qrn_power_linear = 10 ** (qrn_power_db / 20)

        # Normalize and scale
        if np.std(qrn_signal) > 0:
            qrn_signal = qrn_signal / np.std(qrn_signal) * qrn_power_linear

        return signal + qrn_signal

    def _add_awgn_for_target_snr(self, signal: np.ndarray, target_snr_db: float) -> np.ndarray:
        """Add AWGN to achieve target SNR."""
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (target_snr_db / 10))

        noise = np.random.normal(0, np.sqrt(noise_power / 2), len(signal)) + \
                1j * np.random.normal(0, np.sqrt(noise_power / 2), len(signal))

        return signal + noise

    # QRN generation methods (simplified - real versions in channel_simulator)

    def _generate_atmospheric_static(self, length: int) -> np.ndarray:
        """Continuous atmospheric static."""
        # Pink noise (1/f spectrum)
        white = np.random.randn(length) + 1j * np.random.randn(length)
        # Apply 1/f filter in frequency domain
        freqs = np.fft.fftfreq(length)
        H = 1 / (1 + np.abs(freqs) * 100)
        static = np.fft.ifft(np.fft.fft(white) * H)
        return static

    def _generate_crackling_noise(self, length: int, intensity: float) -> np.ndarray:
        """Lightning crashes and crackles."""
        noise = np.zeros(length, dtype=complex)
        # Impulse rate depends on intensity
        impulse_rate = intensity * 100  # Impulses per second
        num_impulses = int(impulse_rate * length / self.sample_rate)

        for _ in range(num_impulses):
            idx = np.random.randint(0, length)
            amplitude = np.random.exponential(5.0)
            duration = np.random.randint(10, 100)
            noise[idx:idx+duration] += amplitude * (np.random.randn(min(duration, length-idx)) +
                                                    1j * np.random.randn(min(duration, length-idx)))

        return noise

    def _generate_popcorn_noise(self, length: int, intensity: float) -> np.ndarray:
        """Sporadic popcorn impulses."""
        noise = np.zeros(length, dtype=complex)
        num_pops = int(intensity * 50 * length / self.sample_rate)

        for _ in range(num_pops):
            idx = np.random.randint(0, length)
            amplitude = np.random.exponential(3.0)
            duration = np.random.randint(5, 30)
            noise[idx:idx+duration] += amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi))

        return noise

    def _generate_thunderstorm_crashes(self, length: int, intensity: float) -> np.ndarray:
        """Severe thunderstorm crashes."""
        noise = self._generate_crackling_noise(length, intensity * 2)
        # Add low-frequency rumble
        rumble = np.random.randn(length) + 1j * np.random.randn(length)
        freqs = np.fft.fftfreq(length, 1/self.sample_rate)
        H = 1 / (1 + (freqs / 100) ** 2)  # Low-pass filter
        rumble = np.fft.ifft(np.fft.fft(rumble) * H)
        return noise + rumble * intensity

    def _generate_auroral_hiss(self, length: int, k_index: float) -> np.ndarray:
        """Auroral hiss (broadband noise)."""
        hiss = np.random.randn(length) + 1j * np.random.randn(length)
        # Modulate amplitude slowly
        mod_freq = 0.5 + k_index * 0.1  # Hz
        t = np.arange(length) / self.sample_rate
        envelope = 1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        return hiss * envelope

    def _generate_auroral_noise(self, length: int, k_index: float) -> np.ndarray:
        """Auroral noise (mix of hiss and flutter)."""
        hiss = self._generate_auroral_hiss(length, k_index)
        flutter = self._generate_fluttery_noise(length, k_index)
        return 0.7 * hiss + 0.3 * flutter

    def _generate_fluttery_noise(self, length: int, k_index: float) -> np.ndarray:
        """Rapid fading/modulation noise."""
        noise = np.random.randn(length) + 1j * np.random.randn(length)
        # Rapid amplitude modulation
        flutter_rate = 5 + k_index * 2  # Hz
        t = np.arange(length) / self.sample_rate
        envelope = 1 + 0.8 * np.sin(2 * np.pi * flutter_rate * t + np.random.uniform(0, 2*np.pi))
        return noise * envelope


def demo_physics_constrained_dataset():
    """Demonstrate physics-constrained dataset generation."""
    from modules.training.src.signal_generator.generator import SignalGenerator

    print("=" * 80)
    print("PHYSICS-CONSTRAINED DATASET DEMONSTRATION")
    print("=" * 80)

    # Create signal generator
    signal_gen = SignalGenerator()

    # Create dataset
    dataset = PhysicsConstrainedDataset(
        num_samples=1000,
        signal_generator=signal_gen,
        sample_rate=48000,
        for_test=False,  # Training distribution
        seed=42
    )

    print(f"\nDataset size: {len(dataset)} samples")

    # Get a few samples
    print("\n### Sample 1 (should vary each time without seed) ###")
    iq, labels = dataset[0]
    print(f"IQ shape: {iq.shape}")
    print(f"SNR: {labels['snr_db']:.1f} dB")
    print(f"SFI: {labels['sfi']:.1f}, K-index: {labels['k_index']:.2f}")
    print(f"Propagation: {labels['propagation_mode']}, QRN: {labels['dominant_qrn_type']}")
    print(f"MUF: {labels['muf_mhz']:.1f} MHz, Absorption: {labels['d_layer_absorption_db']:.1f} dB")

    print("\n### Sample 2 (different physics!) ###")
    iq, labels = dataset[1]
    print(f"SNR: {labels['snr_db']:.1f} dB")
    print(f"SFI: {labels['sfi']:.1f}, K-index: {labels['k_index']:.2f}")
    print(f"Propagation: {labels['propagation_mode']}, QRN: {labels['dominant_qrn_type']}")

    print("\n" + "=" * 80)
    print("KEY FEATURES DEMONSTRATED:")
    print("1. Every sample has unique, continuously-varying parameters")
    print("2. All effects (SNR, propagation, QRN) coupled through physics")
    print("3. Distribution statistics match balanced-realistic weighting")
    print("4. Ready for integration into cascade.ipynb training loop")
    print("=" * 80)


if __name__ == "__main__":
    demo_physics_constrained_dataset()
