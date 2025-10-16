"""
Reciprocal Channel Dataset for CASCADE TX Encoder Training.

Generates paired TX/RX observations through the same channel instance:
- RX receives TX signal through channel (for RX decoder training)
- TX observes RX beacon through SAME channel (for TX encoder training)

This enables joint RX/TX training where TX learns to generate embeddings
that help RX demodulate, validated through the 113-bit compression bottleneck.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import h5py
from tqdm.auto import tqdm

# Import existing components
import sys
cascade_root = Path(__file__).parent.parent.parent.parent
if str(cascade_root) not in sys.path:
    sys.path.insert(0, str(cascade_root))

from modules.training.core.scenarios import ScenarioLibrary
from modules.training.core.gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters
from modules.training.core.gpu_channel_simulator import GPUChannelSimulator


class ReciprocalChannelDataset(torch.utils.data.Dataset):
    """
    Dataset with paired TX and RX observations through reciprocal channel.

    For each sample:
    - Generate TX signal
    - Apply channel (TX → RX)
    - RX receives degraded signal
    - Generate RX beacon
    - Apply SAME channel (RX → TX, reciprocal)
    - TX observes degraded beacon
    - Compute optimal demodulation parameters

    This provides:
    - rx_iq: What RX receives [2, 1024] complex
    - tx_observed_iq: What TX observes [2, 1024] complex
    - optimal_embedding: Ground truth demod params [256] floats
    - labels: Kernel parameters (pattern, frequency, etc.)
    """

    def __init__(self,
                 num_samples: int = 10000,
                 sample_rate: int = 48000,
                 batch_size: int = 4096,
                 collision_probability: float = 0.3,
                 qrm_probability: float = 0.2,
                 seed: Optional[int] = None,
                 cache_path: Optional[Path] = None,
                 regenerate_cache: bool = False,
                 device: str = 'cuda'):
        """
        Initialize reciprocal channel dataset.

        Args:
            num_samples: Total number of paired samples to generate
            sample_rate: Audio sample rate (Hz)
            batch_size: GPU batch size for parallel generation
            collision_probability: Fraction of samples with collisions
            qrm_probability: Fraction of samples with QRM
            seed: Random seed for reproducibility
            cache_path: Path to HDF5 cache file
            regenerate_cache: Force regeneration even if cache exists
            device: 'cuda' or 'cpu'
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.collision_probability = collision_probability
        self.qrm_probability = qrm_probability
        self.seed = seed
        self.device = device

        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Setup cache path
        if cache_path is None:
            cache_path = Path(__file__).parent / 'cache' / f'reciprocal_dataset_{num_samples}_seed{seed}.h5'
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Check cache
        if self.cache_path.exists() and not regenerate_cache:
            print(f"✓ Loading cached reciprocal dataset from {self.cache_path}")
            self._load_cache()
        else:
            print(f"Generating {num_samples:,} reciprocal channel samples...")
            self._generate_dataset()
            self._save_cache()

    def _generate_dataset(self):
        """Generate paired TX/RX observations."""
        print(f"\n{'='*80}")
        print("RECIPROCAL CHANNEL DATASET GENERATION")
        print(f"{'='*80}\n")

        # Initialize generators
        print("Initializing generators...")
        self.scenario_lib = ScenarioLibrary()
        self.signal_gen = GPUSignalGenerator(device=self.device)
        self.channel_sim = GPUChannelSimulator(
            sample_rate=self.sample_rate,
            device=self.device
        )
        print("✓ Generators initialized\n")

        # Storage for generated data
        self.rx_iq_samples = np.zeros((self.num_samples, 2, 1024), dtype=np.complex64)
        self.tx_observed_iq = np.zeros((self.num_samples, 2, 1024), dtype=np.complex64)
        self.optimal_embeddings = np.zeros((self.num_samples, 256), dtype=np.float32)

        # Labels (same structure as EnhancedPhysicsDataset)
        self.pattern_ids = np.zeros(self.num_samples, dtype=np.int64)
        self.frequency_triples = np.zeros(self.num_samples, dtype=np.int64)
        self.modulations = []
        self.data_symbol_rates = np.zeros(self.num_samples, dtype=np.int64)
        self.propagation_modes = []
        self.k_indices = np.zeros(self.num_samples, dtype=np.float32)
        self.sfis = np.zeros(self.num_samples, dtype=np.float32)
        self.qrn_types = []
        self.has_collisions = np.zeros(self.num_samples, dtype=np.bool_)
        self.has_qrm = np.zeros(self.num_samples, dtype=np.bool_)

        # Generate in batches
        num_batches = int(np.ceil(self.num_samples / self.batch_size))

        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            current_batch_size = end_idx - start_idx

            # Generate batch
            batch_data = self._generate_batch(current_batch_size)

            # Store data
            self.rx_iq_samples[start_idx:end_idx] = batch_data['rx_iq']
            self.tx_observed_iq[start_idx:end_idx] = batch_data['tx_observed_iq']
            self.optimal_embeddings[start_idx:end_idx] = batch_data['optimal_embedding']

            # Store labels
            self.pattern_ids[start_idx:end_idx] = batch_data['pattern_ids']
            self.frequency_triples[start_idx:end_idx] = batch_data['frequency_triples']
            self.modulations.extend(batch_data['modulations'])
            self.data_symbol_rates[start_idx:end_idx] = batch_data['data_symbol_rates']
            self.propagation_modes.extend(batch_data['propagation_modes'])
            self.k_indices[start_idx:end_idx] = batch_data['k_indices']
            self.sfis[start_idx:end_idx] = batch_data['sfis']
            self.qrn_types.extend(batch_data['qrn_types'])
            self.has_collisions[start_idx:end_idx] = batch_data['has_collisions']
            self.has_qrm[start_idx:end_idx] = batch_data['has_qrm']

        print(f"\n✓ Generated {self.num_samples:,} paired TX/RX samples")

    def _generate_batch(self, batch_size: int) -> Dict:
        """Generate one batch of paired TX/RX samples."""
        # Generate scenarios
        scenarios = self.scenario_lib.generate_balanced_realistic_batch(
            batch_size,
            seed=None,  # Use current numpy RNG state
            use_parallel=False  # Already in parallel GPU generation
        )

        # Generate random kernel parameters (scenarios only have physics, not CASCADE params)
        pattern_ids = torch.randint(0, 4, (batch_size,), device=self.device)
        frequency_triples = torch.randint(0, 43, (batch_size,), device=self.device)

        # Select modulation based on SNR (need to compute SNR from scenarios first)
        # For now, use QPSK for all
        modulations = ['QPSK'] * batch_size
        polar_rates = [(2, 3)] * batch_size  # Rate 2/3
        data_symbol_rates = torch.full((batch_size,), 150, device=self.device)

        # Generate random messages
        messages = [np.random.bytes(np.random.randint(20, 257)) for _ in range(batch_size)]

        # Create batch parameters
        batch_params = BatchKernelParameters(
            pattern_ids=pattern_ids,
            frequency_triples=frequency_triples,
            modulations=modulations,
            polar_rates=polar_rates,
            data_symbol_rates=data_symbol_rates
        )

        # Step 1: Generate TX signals (clean)
        tx_signals_clean, tx_metadata = self.signal_gen.generate_batch(
            batch_params,
            messages,
            fixed_length=2048
        )

        # Step 2: Generate RX beacons (for TX to observe)
        # Use fixed simple beacon pattern for consistency
        beacon_pattern_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        beacon_frequency = torch.ones(batch_size, dtype=torch.long, device=self.device) * 21  # Middle freq
        beacon_params = BatchKernelParameters(
            pattern_ids=beacon_pattern_ids,
            frequency_triples=beacon_frequency,
            modulations=['BPSK'] * batch_size,
            polar_rates=[(1, 2)] * batch_size,  # Rate 1/2 for robustness
            data_symbol_rates=torch.full((batch_size,), 100, device=self.device)  # Slow rate
        )
        beacon_messages = [b"RX_BEACON" for _ in range(batch_size)]
        rx_beacons_clean, _ = self.signal_gen.generate_batch(
            beacon_params,
            beacon_messages,
            fixed_length=2048
        )

        # Step 3: Apply channel to both directions (SAME channel instance, reciprocal)
        # This is the key: same propagation effects in both directions

        # Direction 1: TX → RX (for RX training)
        rx_received_iq = self.channel_sim.apply_batch(
            tx_signals_clean,
            scenarios,
            add_collisions=True,
            collision_probability=self.collision_probability,
            add_qrm=True,
            qrm_probability=self.qrm_probability
        )

        # Direction 2: RX → TX (reciprocal channel, for TX training)
        # Use SAME scenarios (reciprocal propagation)
        tx_observed_iq = self.channel_sim.apply_batch(
            rx_beacons_clean,
            scenarios,  # SAME scenarios = reciprocal channel
            add_collisions=False,  # Beacons typically don't collide
            add_qrm=False  # Beacons are clean
        )

        # Step 4: Compute optimal demodulation parameters (ground truth for embedding)
        optimal_embeddings = self._compute_optimal_embeddings(
            clean_signals=tx_signals_clean,
            received_signals=rx_received_iq,
            scenarios=scenarios
        )

        # Convert to I/Q format [batch, 2, 1024]
        rx_iq_formatted = torch.stack([rx_received_iq.real, rx_received_iq.imag], dim=1)[:, :, :1024]
        tx_iq_formatted = torch.stack([tx_observed_iq.real, tx_observed_iq.imag], dim=1)[:, :, :1024]

        # Move to CPU and convert to numpy
        return {
            'rx_iq': rx_iq_formatted.cpu().numpy(),
            'tx_observed_iq': tx_iq_formatted.cpu().numpy(),
            'optimal_embedding': optimal_embeddings.cpu().numpy(),
            'pattern_ids': pattern_ids.cpu().numpy(),
            'frequency_triples': frequency_triples.cpu().numpy(),
            'modulations': modulations,
            'data_symbol_rates': data_symbol_rates.cpu().numpy(),
            'propagation_modes': ['rician'] * batch_size,  # Scenarios are CorePhysicalDrivers, compute later
            'k_indices': np.array([s.k_index for s in scenarios]),
            'sfis': np.array([s.sfi for s in scenarios]),
            'qrn_types': ['static'] * batch_size,  # Placeholder
            'has_collisions': np.random.rand(batch_size) < self.collision_probability,
            'has_qrm': np.random.rand(batch_size) < self.qrm_probability
        }

    def _compute_optimal_embeddings(self,
                                    clean_signals: torch.Tensor,
                                    received_signals: torch.Tensor,
                                    scenarios: List) -> torch.Tensor:
        """
        Compute optimal demodulation parameters (ground truth for TX embedding).

        This represents what the TX encoder should learn to generate:
        - Frequency offset correction
        - Phase rotation
        - Equalization taps
        - Timing adjustments
        - Interference mitigation hints

        Args:
            clean_signals: [batch, signal_len] clean TX signals
            received_signals: [batch, signal_len] received signals after channel
            scenarios: List of scenario objects with channel params

        Returns:
            torch.Tensor: [batch, 256] optimal embedding vectors
        """
        batch_size = clean_signals.shape[0]
        embeddings = torch.zeros(batch_size, 256, device=self.device)

        for i in range(batch_size):
            # Extract first 256 samples for analysis
            clean = clean_signals[i, :256]
            received = received_signals[i, :256]
            scenario = scenarios[i]

            # Frequency offset (estimated from cross-correlation)
            # In real deployment, TX estimates this from observing RX beacon
            freq_offset = 0.0  # Placeholder - would compute from FFT analysis

            # Phase rotation (from channel phase shift)
            phase_rotation = torch.angle(torch.sum(received * torch.conj(clean)))

            # Channel impulse response estimate
            # For multipath: capture delay taps
            cir_length = 32
            cir = torch.zeros(cir_length, dtype=torch.complex64, device=self.device)
            if hasattr(scenario, 'multipath_delays') and scenario.multipath_delays is not None:
                # Encode multipath structure
                for delay_idx, delay in enumerate(scenario.multipath_delays[:cir_length]):
                    cir[delay_idx] = torch.tensor(delay, device=self.device)

            # Encode into embedding vector [256]
            # Layout:
            #   [0:32]: Frequency/timing adjustments
            #   [32:64]: Phase correction parameters
            #   [64:128]: Equalization taps (3-tone diversity)
            #   [128:192]: Channel impulse response
            #   [192:256]: Interference mitigation hints

            # Frequency/timing (32 dims)
            embeddings[i, 0] = freq_offset
            embeddings[i, 1] = phase_rotation.real
            embeddings[i, 2] = phase_rotation.imag
            # Rest filled with noise for now (placeholder)

            # Phase correction (32 dims)
            embeddings[i, 32] = torch.cos(phase_rotation)
            embeddings[i, 33] = torch.sin(phase_rotation)

            # Equalization taps (64 dims) - simplified placeholder
            embeddings[i, 64:96] = torch.randn(32, device=self.device) * 0.3

            # Channel impulse response (64 dims)
            embeddings[i, 128:128+cir_length] = torch.view_as_real(cir).flatten()[:64]

            # Interference hints (64 dims) - placeholder
            embeddings[i, 192:224] = torch.randn(32, device=self.device) * 0.2

        return embeddings

    def _save_cache(self):
        """Save dataset to HDF5 cache."""
        print(f"\nSaving dataset to {self.cache_path}...")

        with h5py.File(self.cache_path, 'w') as f:
            # Save I/Q data
            f.create_dataset('rx_iq', data=self.rx_iq_samples, compression='gzip', compression_opts=4)
            f.create_dataset('tx_observed_iq', data=self.tx_observed_iq, compression='gzip', compression_opts=4)
            f.create_dataset('optimal_embeddings', data=self.optimal_embeddings, compression='gzip', compression_opts=4)

            # Save labels
            f.create_dataset('pattern_ids', data=self.pattern_ids)
            f.create_dataset('frequency_triples', data=self.frequency_triples)
            f.create_dataset('data_symbol_rates', data=self.data_symbol_rates)
            f.create_dataset('k_indices', data=self.k_indices)
            f.create_dataset('sfis', data=self.sfis)
            f.create_dataset('has_collisions', data=self.has_collisions)
            f.create_dataset('has_qrm', data=self.has_qrm)

            # Save string arrays
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('modulations', data=self.modulations, dtype=dt)
            f.create_dataset('propagation_modes', data=self.propagation_modes, dtype=dt)
            f.create_dataset('qrn_types', data=self.qrn_types, dtype=dt)

            # Save metadata
            f.attrs['num_samples'] = self.num_samples
            f.attrs['sample_rate'] = self.sample_rate
            f.attrs['seed'] = self.seed if self.seed is not None else -1

        print(f"✓ Cached {self.num_samples:,} samples ({self.cache_path.stat().st_size / 1024**2:.1f} MB)")

    def _load_cache(self):
        """Load dataset from HDF5 cache."""
        with h5py.File(self.cache_path, 'r') as f:
            # Load I/Q data
            self.rx_iq_samples = f['rx_iq'][:]
            self.tx_observed_iq = f['tx_observed_iq'][:]
            self.optimal_embeddings = f['optimal_embeddings'][:]

            # Load labels
            self.pattern_ids = f['pattern_ids'][:]
            self.frequency_triples = f['frequency_triples'][:]
            self.data_symbol_rates = f['data_symbol_rates'][:]
            self.k_indices = f['k_indices'][:]
            self.sfis = f['sfis'][:]
            self.has_collisions = f['has_collisions'][:]
            self.has_qrm = f['has_qrm'][:]

            # Load string arrays
            self.modulations = f['modulations'][:].tolist()
            self.modulations = [m.decode('utf-8') if isinstance(m, bytes) else m for m in self.modulations]
            self.propagation_modes = f['propagation_modes'][:].tolist()
            self.propagation_modes = [p.decode('utf-8') if isinstance(p, bytes) else p for p in self.propagation_modes]
            self.qrn_types = f['qrn_types'][:].tolist()
            self.qrn_types = [q.decode('utf-8') if isinstance(q, bytes) else q for q in self.qrn_types]

        print(f"✓ Loaded {len(self.rx_iq_samples):,} cached samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get one paired sample.

        Returns:
            Tuple of:
                - rx_iq: [2, 1024] complex I/Q (what RX receives)
                - tx_observed_iq: [2, 1024] complex I/Q (what TX observes)
                - labels: Dict with all labels + optimal_embedding
        """
        # Convert complex to real format [2, 2, 1024] (I and Q channels, real/imag)
        rx_iq_real = np.stack([self.rx_iq_samples[idx].real, self.rx_iq_samples[idx].imag], axis=0)
        tx_iq_real = np.stack([self.tx_observed_iq[idx].real, self.tx_observed_iq[idx].imag], axis=0)

        labels = {
            'pattern_id': self.pattern_ids[idx],
            'frequency_triple': self.frequency_triples[idx],
            'modulation': self.modulations[idx],
            'data_symbol_rate': self.data_symbol_rates[idx],
            'propagation_mode': self.propagation_modes[idx],
            'k_index': self.k_indices[idx],
            'sfi': self.sfis[idx],
            'qrn_type': self.qrn_types[idx],
            'has_collision': self.has_collisions[idx],
            'has_qrm': self.has_qrm[idx],
            'optimal_embedding': self.optimal_embeddings[idx]
        }

        return torch.from_numpy(rx_iq_real).float(), torch.from_numpy(tx_iq_real).float(), labels


def test_reciprocal_dataset():
    """Test reciprocal channel dataset generation."""
    print("Testing Reciprocal Channel Dataset...")

    dataset = ReciprocalChannelDataset(
        num_samples=100,
        batch_size=32,
        seed=42,
        device='cuda',
        regenerate_cache=True
    )

    # Test __getitem__
    rx_iq, tx_iq, labels = dataset[0]

    print(f"\n✓ Dataset test passed!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  RX I/Q shape: {rx_iq.shape}")
    print(f"  TX I/Q shape: {tx_iq.shape}")
    print(f"  Optimal embedding shape: {labels['optimal_embedding'].shape}")
    print(f"  Sample labels: {list(labels.keys())}")


if __name__ == "__main__":
    test_reciprocal_dataset()
