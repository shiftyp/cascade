"""Channel Orchestrator for CASCADE V2 synthetic training data generation.

Combines signal generation with channel effects to create expert datasets:
- Clean (high SNR)
- AWGN noise
- QRN (atmospheric noise)
- Multipath fading
- QRM (interference)
- Combined realistic scenarios

Source: CASCADE V2 training data specification
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from .awgn import generate_awgn
from .qrn import generate_mixed_qrn
from .multipath import apply_multipath_fading, watterson_hf_profile, MultipathProfile
from .qrm import generate_mixed_qrm
from .collisions import create_collision_scenarios, CollisionSignal


@dataclass
class ExpertConfig:
    """Configuration for an expert dataset type.

    Attributes:
        name: Expert type name ('clean', 'awgn', 'qrn', 'multipath', 'combined')
        awgn_enabled: Enable AWGN noise
        awgn_snr_range_db: SNR range for AWGN (min, max)
        qrn_enabled: Enable atmospheric noise
        qrn_power: QRN power level
        multipath_enabled: Enable multipath fading
        multipath_profile: Multipath profile to use
        qrm_enabled: Enable interference
        qrm_power: QRM power level
        collision_enabled: Enable collision scenarios
        collision_probability: Probability of collisions
    """
    name: str
    awgn_enabled: bool = False
    awgn_snr_range_db: Tuple[float, float] = (-20.0, 20.0)
    qrn_enabled: bool = False
    qrn_power: float = 0.1
    multipath_enabled: bool = False
    multipath_profile: Optional[MultipathProfile] = None
    qrm_enabled: bool = False
    qrm_power: float = 0.1
    collision_enabled: bool = False
    collision_probability: float = 0.1


def get_expert_configs() -> Dict[str, ExpertConfig]:
    """Get predefined expert configurations.

    Returns:
        Dict mapping expert names to ExpertConfig objects
    """
    configs = {}

    # Expert 1: Clean / High SNR
    configs['clean'] = ExpertConfig(
        name='clean',
        awgn_enabled=True,
        awgn_snr_range_db=(15.0, 30.0),  # High SNR only
    )

    # Expert 2: AWGN only
    configs['awgn'] = ExpertConfig(
        name='awgn',
        awgn_enabled=True,
        awgn_snr_range_db=(-20.0, 20.0),  # Full range
    )

    # Expert 3: QRN (atmospheric noise)
    configs['qrn'] = ExpertConfig(
        name='qrn',
        awgn_enabled=True,
        awgn_snr_range_db=(0.0, 10.0),  # Moderate AWGN
        qrn_enabled=True,
        qrn_power=0.5,  # Strong atmospheric noise
    )

    # Expert 4: Multipath fading
    configs['multipath'] = ExpertConfig(
        name='multipath',
        awgn_enabled=True,
        awgn_snr_range_db=(0.0, 10.0),
        multipath_enabled=True,
        multipath_profile=watterson_hf_profile(delay_spread_ms=2.0, doppler_spread_hz=0.5),
    )

    # Expert 5: Combined (realistic HF)
    configs['combined'] = ExpertConfig(
        name='combined',
        awgn_enabled=True,
        awgn_snr_range_db=(-10.0, 15.0),
        qrn_enabled=True,
        qrn_power=0.3,
        multipath_enabled=True,
        multipath_profile=watterson_hf_profile(delay_spread_ms=2.0, doppler_spread_hz=0.5),
        qrm_enabled=True,
        qrm_power=0.2,
        collision_enabled=False,  # Collisions handled separately
    )

    return configs


class ChannelOrchestrator:
    """Orchestrates signal generation and channel effects for training data.

    Generates expert datasets by applying various channel impairments to
    clean CASCADE V2 signals.
    """

    def __init__(self, sample_rate: int = 48000, seed: Optional[int] = None):
        """Initialize orchestrator.

        Args:
            sample_rate: Sample rate in Hz (default: 48000)
            seed: Random seed for reproducibility
        """
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)
        self.expert_configs = get_expert_configs()

    def apply_channel_effects(self, clean_signal: np.ndarray,
                             expert_config: ExpertConfig,
                             seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Apply channel effects to clean signal based on expert configuration.

        Args:
            clean_signal: Clean CASCADE signal (complex64)
            expert_config: Expert configuration specifying effects
            seed: Random seed for this application

        Returns:
            Tuple of (noisy_signal, metadata_dict)
        """
        rng = np.random.default_rng(seed)
        signal = clean_signal.copy()
        metadata = {'expert_type': expert_config.name, 'effects_applied': []}

        # Apply multipath fading first (before noise)
        if expert_config.multipath_enabled and expert_config.multipath_profile:
            signal = apply_multipath_fading(
                signal, expert_config.multipath_profile,
                self.sample_rate, seed=rng.integers(0, 2**31)
            )
            metadata['effects_applied'].append('multipath')
            metadata['multipath_profile'] = {
                'delays': expert_config.multipath_profile.delays,
                'powers': expert_config.multipath_profile.powers,
            }

        # Add QRN (atmospheric noise)
        if expert_config.qrn_enabled:
            qrn = generate_mixed_qrn(
                len(signal), self.sample_rate,
                static_power=expert_config.qrn_power * 0.6,
                crackling_rate=3.0,
                lightning_rate=0.2,
                powerline_power=expert_config.qrn_power * 0.2,
                seed=rng.integers(0, 2**31)
            )
            signal = signal + qrn
            metadata['effects_applied'].append('qrn')
            metadata['qrn_power'] = expert_config.qrn_power

        # Add QRM (man-made interference)
        if expert_config.qrm_enabled:
            qrm = generate_mixed_qrm(
                len(signal), self.sample_rate,
                interference_types=['cw', 'ssb', 'ft8'],
                freq_range=(300, 3000),
                power=expert_config.qrm_power,
                seed=rng.integers(0, 2**31)
            )
            signal = signal + qrm
            metadata['effects_applied'].append('qrm')
            metadata['qrm_power'] = expert_config.qrm_power

        # Add AWGN (always last)
        if expert_config.awgn_enabled:
            snr_db = rng.uniform(*expert_config.awgn_snr_range_db)
            signal = generate_awgn(signal, snr_db, seed=rng.integers(0, 2**31))
            metadata['effects_applied'].append('awgn')
            metadata['snr_db'] = float(snr_db)

        # Calculate final signal statistics
        metadata['signal_power'] = float(np.mean(np.abs(signal) ** 2))
        metadata['peak_amplitude'] = float(np.max(np.abs(signal)))

        return signal.astype(np.complex64), metadata

    def generate_expert_dataset(self, clean_signals: List[np.ndarray],
                               expert_type: str, seed: Optional[int] = None) -> List[Tuple[np.ndarray, Dict]]:
        """Generate expert dataset from clean signals.

        Args:
            clean_signals: List of clean CASCADE signals
            expert_type: Expert type name ('clean', 'awgn', 'qrn', 'multipath', 'combined')
            seed: Random seed

        Returns:
            List of (noisy_signal, metadata) tuples
        """
        if expert_type not in self.expert_configs:
            raise ValueError(f"Unknown expert type: {expert_type}. "
                           f"Available: {list(self.expert_configs.keys())}")

        config = self.expert_configs[expert_type]
        rng = np.random.default_rng(seed)

        dataset = []
        for sig_idx, clean_signal in enumerate(clean_signals):
            noisy_signal, metadata = self.apply_channel_effects(
                clean_signal, config, seed=rng.integers(0, 2**31)
            )
            metadata['signal_index'] = sig_idx
            dataset.append((noisy_signal, metadata))

        return dataset

    def generate_all_experts(self, clean_signals: List[np.ndarray],
                            num_per_expert: int = 100,
                            seed: Optional[int] = None) -> Dict[str, List[Tuple[np.ndarray, Dict]]]:
        """Generate datasets for all expert types.

        Args:
            clean_signals: Pool of clean CASCADE signals
            num_per_expert: Number of examples per expert (default: 100)
            seed: Random seed

        Returns:
            Dict mapping expert names to datasets
        """
        rng = np.random.default_rng(seed)

        all_datasets = {}

        for expert_name in self.expert_configs.keys():
            # Sample clean signals for this expert
            if len(clean_signals) < num_per_expert:
                # Repeat signals if not enough unique ones
                indices = rng.choice(len(clean_signals), num_per_expert, replace=True)
            else:
                indices = rng.choice(len(clean_signals), num_per_expert, replace=False)

            expert_clean_signals = [clean_signals[i] for i in indices]

            # Generate expert dataset
            dataset = self.generate_expert_dataset(
                expert_clean_signals, expert_name,
                seed=rng.integers(0, 2**31)
            )

            all_datasets[expert_name] = dataset

            print(f"Generated {len(dataset)} examples for expert '{expert_name}'")

        return all_datasets

    def save_dataset(self, dataset: List[Tuple[np.ndarray, Dict]],
                    output_path: Path, format: str = 'npz'):
        """Save dataset to disk.

        Args:
            dataset: List of (signal, metadata) tuples
            output_path: Output file path
            format: Save format - 'npz' (numpy), 'hdf5', or 'zarr'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'npz':
            # Save as numpy .npz
            signals = np.array([sig for sig, _ in dataset])
            metadata_list = [meta for _, meta in dataset]

            # Save signals
            np.savez_compressed(output_path, signals=signals)

            # Save metadata as JSON
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata_list, f, indent=2)

            print(f"Saved dataset: {output_path}")
            print(f"Saved metadata: {metadata_path}")

        elif format == 'hdf5':
            try:
                import h5py

                with h5py.File(output_path, 'w') as f:
                    # Save signals
                    signals = np.array([sig for sig, _ in dataset])
                    f.create_dataset('signals', data=signals, compression='gzip')

                    # Save metadata as JSON string
                    metadata_list = [meta for _, meta in dataset]
                    f.attrs['metadata'] = json.dumps(metadata_list)

                print(f"Saved dataset: {output_path}")

            except ImportError:
                raise ImportError("h5py not installed. Install with: pip install h5py")

        elif format == 'zarr':
            try:
                import zarr

                store = zarr.DirectoryStore(output_path)
                root = zarr.group(store=store, overwrite=True)

                # Save signals
                signals = np.array([sig for sig, _ in dataset])
                root.create_dataset('signals', data=signals, chunks=(1, -1))

                # Save metadata
                metadata_list = [meta for _, meta in dataset]
                root.attrs['metadata'] = json.dumps(metadata_list)

                print(f"Saved dataset: {output_path}")

            except ImportError:
                raise ImportError("zarr not installed. Install with: pip install zarr")

        else:
            raise ValueError(f"Unknown format: {format}. Use 'npz', 'hdf5', or 'zarr'")

    def load_dataset(self, dataset_path: Path, format: str = 'npz') -> List[Tuple[np.ndarray, Dict]]:
        """Load dataset from disk.

        Args:
            dataset_path: Path to dataset file
            format: Load format - 'npz', 'hdf5', or 'zarr'

        Returns:
            List of (signal, metadata) tuples
        """
        dataset_path = Path(dataset_path)

        if format == 'npz':
            # Load signals
            data = np.load(dataset_path)
            signals = data['signals']

            # Load metadata
            metadata_path = dataset_path.with_suffix('.json')
            with open(metadata_path, 'r') as f:
                metadata_list = json.load(f)

            dataset = list(zip(signals, metadata_list))

            print(f"Loaded dataset: {dataset_path} ({len(dataset)} examples)")
            return dataset

        elif format == 'hdf5':
            try:
                import h5py

                with h5py.File(dataset_path, 'r') as f:
                    signals = f['signals'][:]
                    metadata_list = json.loads(f.attrs['metadata'])

                dataset = list(zip(signals, metadata_list))

                print(f"Loaded dataset: {dataset_path} ({len(dataset)} examples)")
                return dataset

            except ImportError:
                raise ImportError("h5py not installed. Install with: pip install h5py")

        elif format == 'zarr':
            try:
                import zarr

                root = zarr.open(dataset_path, mode='r')
                signals = root['signals'][:]
                metadata_list = json.loads(root.attrs['metadata'])

                dataset = list(zip(signals, metadata_list))

                print(f"Loaded dataset: {dataset_path} ({len(dataset)} examples)")
                return dataset

            except ImportError:
                raise ImportError("zarr not installed. Install with: pip install zarr")

        else:
            raise ValueError(f"Unknown format: {format}. Use 'npz', 'hdf5', or 'zarr'")
