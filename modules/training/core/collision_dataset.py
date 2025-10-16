"""
Collision-aware dataset for CASCADE training.

Extends PhysicsConstrainedDataset to include:
- Overlapping CASCADE signals (collision scenarios)
- QRM interference (SSB, CW, digital modes)
- Context signal tracking (8 recent transmissions for decoder)

This enables proper training of:
- Timing Expert (temporal collision separation)
- QRM Expert (interference detection)
- Integration Decoder (context-aware decoding)
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
import random

from physics_constrained_dataset import PhysicsConstrainedDataset
from modules.training.src.signal_generator.generator import SignalGenerator


@dataclass
class ContextSignal:
    """Metadata for a context signal (recently decoded transmission)."""
    pattern_id: int  # 0-7 (8 patterns)
    frequency_triple: int  # 0-42 (43 triples)
    modulation: int  # 0-3 (BPSK, QPSK, 8-PSK, 16-APSK)
    data_symbol_rate: int  # 0-7 (75, 100, 125, 150, 175, 200, 250, 300)
    polar_rate_idx: int  # 0-4 (1/2, 2/3, 3/4, 5/6, 7/8)
    duration_windows: int  # 0-255 (transmission duration)
    time_offset_ms: float  # Time offset relative to current signal
    snr_db: float  # SNR of this context signal


@dataclass
class CollisionScenario:
    """Complete collision scenario for a training sample."""
    # Primary signal (target to decode)
    primary_signal: np.ndarray
    primary_labels: Dict

    # Colliding signals (0-3 overlapping CASCADE transmissions)
    num_collisions: int = 0
    collision_signals: List[np.ndarray] = field(default_factory=list)
    collision_labels: List[Dict] = field(default_factory=list)
    collision_offsets_ms: List[float] = field(default_factory=list)

    # QRM interference
    has_qrm: bool = False
    qrm_type: Optional[str] = None
    qrm_signal: Optional[np.ndarray] = None

    # Context signals (up to 8 recent transmissions)
    context_signals: List[ContextSignal] = field(default_factory=list)


class CollisionAwareDataset(PhysicsConstrainedDataset):
    """
    Dataset that generates CASCADE signals with collisions, QRM, and context.

    Key features beyond PhysicsConstrainedDataset:
    1. **Collision scenarios**: 0-3 overlapping CASCADE signals
    2. **QRM interference**: SSB, CW, digital modes (20% of samples)
    3. **Context tracking**: 8 recent signal metadata for decoder
    4. **Timing offsets**: Random delays for collision separation training
    """

    def __init__(
        self,
        num_samples: int,
        signal_generator: SignalGenerator,
        sample_rate: int = 48000,
        for_test: bool = False,
        seed: Optional[int] = None,
        collision_probability: float = 0.3,
        qrm_probability: float = 0.2,
        max_collisions: int = 3,
        **kwargs
    ):
        """
        Initialize collision-aware dataset.

        Args:
            collision_probability: Probability of having colliding signals (default: 0.3)
            qrm_probability: Probability of QRM interference (default: 0.2)
            max_collisions: Maximum number of colliding signals (default: 3)
            **kwargs: Passed to PhysicsConstrainedDataset
        """
        super().__init__(
            num_samples=num_samples,
            signal_generator=signal_generator,
            sample_rate=sample_rate,
            for_test=for_test,
            seed=seed,
            **kwargs
        )

        self.collision_probability = collision_probability
        self.qrm_probability = qrm_probability
        self.max_collisions = max_collisions

        # Context signal history (circular buffer, 8 most recent per frequency region)
        # Key: frequency_region (0-14, each covers ~3 triples)
        # Value: List of recent ContextSignal objects
        self.context_history: Dict[int, List[ContextSignal]] = {i: [] for i in range(15)}

        print(f"✓ CollisionAwareDataset initialized:")
        print(f"  Collision probability: {collision_probability:.0%}")
        print(f"  QRM probability: {qrm_probability:.0%}")
        print(f"  Max collisions: {max_collisions}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a training sample with potential collisions, QRM, and context.

        Returns:
            received_iq: Torch tensor [2, 2048] with all signals mixed
            labels: Dict with primary signal labels + collision/context metadata
        """
        # 1. Get primary signal from parent class
        primary_iq, primary_labels = super().__getitem__(idx)

        # Convert to numpy for signal mixing
        primary_signal = primary_iq.numpy()  # [2, 2048]
        primary_complex = primary_signal[0] + 1j * primary_signal[1]  # Complex I/Q

        # 2. Initialize collision scenario
        scenario = CollisionScenario(
            primary_signal=primary_complex,
            primary_labels=primary_labels
        )

        # 3. Maybe add colliding CASCADE signals
        if np.random.random() < self.collision_probability:
            scenario = self._add_colliding_signals(scenario, idx)

        # 4. Maybe add QRM interference
        if np.random.random() < self.qrm_probability:
            scenario = self._add_qrm_interference(scenario)

        # 5. Get context signals for this frequency region
        freq_triple = primary_labels.get('frequency_triple', 21)
        freq_region = freq_triple // 3  # 43 triples / 3 = ~14 regions
        scenario.context_signals = self._get_context_signals(freq_region, scenario)

        # 6. Mix all signals
        mixed_signal = self._mix_signals(scenario)

        # 7. Convert back to PyTorch tensor [2, 2048]
        iq_i = np.real(mixed_signal).astype(np.float32)
        iq_q = np.imag(mixed_signal).astype(np.float32)
        mixed_iq = torch.from_numpy(np.stack([iq_i, iq_q], axis=0))

        # 8. Add collision/context metadata to labels
        enhanced_labels = self._create_enhanced_labels(primary_labels, scenario)

        # 9. Update context history with this transmission
        self._update_context_history(freq_region, scenario)

        return mixed_iq, enhanced_labels

    def _add_colliding_signals(self, scenario: CollisionScenario, base_idx: int) -> CollisionScenario:
        """Add 1-3 colliding CASCADE signals with random time offsets."""
        num_collisions = np.random.randint(1, self.max_collisions + 1)
        scenario.num_collisions = num_collisions

        for i in range(num_collisions):
            # Generate collision signal from a different sample
            collision_idx = (base_idx + np.random.randint(1, self.num_samples)) % self.num_samples
            collision_iq, collision_labels = super().__getitem__(collision_idx)

            # Convert to complex
            collision_signal = collision_iq.numpy()
            collision_complex = collision_signal[0] + 1j * collision_signal[1]

            # Random time offset (-500ms to +500ms)
            max_offset_samples = int(0.5 * self.sample_rate)  # 500ms
            time_offset_samples = np.random.randint(-max_offset_samples, max_offset_samples)
            time_offset_ms = (time_offset_samples / self.sample_rate) * 1000

            # Apply time offset (circular shift)
            collision_shifted = np.roll(collision_complex, time_offset_samples)

            # Random SNR offset (-10 to +5 dB relative to primary)
            snr_offset_db = np.random.uniform(-10, 5)
            primary_snr = scenario.primary_labels.get('snr_db', 10)
            collision_snr = primary_snr + snr_offset_db

            # Scale collision signal to target SNR
            power_ratio = 10 ** (snr_offset_db / 20)
            collision_scaled = collision_shifted * power_ratio

            # Store
            scenario.collision_signals.append(collision_scaled)
            scenario.collision_labels.append(collision_labels)
            scenario.collision_offsets_ms.append(time_offset_ms)

        return scenario

    def _add_qrm_interference(self, scenario: CollisionScenario) -> CollisionScenario:
        """Add QRM interference (SSB, CW, or digital mode)."""
        qrm_types = ['ssb', 'cw', 'psk31', 'rtty']
        qrm_type = np.random.choice(qrm_types)

        signal_len = len(scenario.primary_signal)

        if qrm_type == 'ssb':
            qrm_signal = self._generate_ssb_voice(signal_len)
        elif qrm_type == 'cw':
            qrm_signal = self._generate_cw_morse(signal_len)
        elif qrm_type == 'psk31':
            qrm_signal = self._generate_psk31(signal_len)
        else:  # rtty
            qrm_signal = self._generate_rtty(signal_len)

        # Random QRM power (-20 to +10 dB relative to primary)
        qrm_power_db = np.random.uniform(-20, 10)
        qrm_power_linear = 10 ** (qrm_power_db / 20)

        # Normalize and scale
        if np.std(qrm_signal) > 0:
            qrm_signal = qrm_signal / np.std(qrm_signal) * qrm_power_linear

        scenario.has_qrm = True
        scenario.qrm_type = qrm_type
        scenario.qrm_signal = qrm_signal

        return scenario

    def _get_context_signals(self, freq_region: int, scenario: CollisionScenario) -> List[ContextSignal]:
        """
        Get up to 8 context signals from recent transmissions in this frequency region.

        Returns signals from:
        - Same frequency region (most relevant)
        - Adjacent regions (±1)
        """
        context_signals = []

        # Get signals from this region and adjacent regions
        regions_to_check = [freq_region]
        if freq_region > 0:
            regions_to_check.append(freq_region - 1)
        if freq_region < 14:
            regions_to_check.append(freq_region + 1)

        for region in regions_to_check:
            if region in self.context_history:
                context_signals.extend(self.context_history[region])

        # Sort by time (most recent first) and take top 8
        context_signals.sort(key=lambda x: abs(x.time_offset_ms))

        return context_signals[:8]

    def _update_context_history(self, freq_region: int, scenario: CollisionScenario):
        """Add current transmission to context history."""
        # Create ContextSignal from primary signal
        context = ContextSignal(
            pattern_id=scenario.primary_labels.get('pattern_id', 0),
            frequency_triple=scenario.primary_labels.get('frequency_triple', 21),
            modulation=0,  # Default BPSK (would be from kernel)
            data_symbol_rate=2,  # Default 125 sym/s (index 2)
            polar_rate_idx=1,  # Default rate 2/3
            duration_windows=5,  # Estimated
            time_offset_ms=0.0,  # Current signal
            snr_db=scenario.primary_labels.get('snr_db', 10)
        )

        # Add to history
        if freq_region not in self.context_history:
            self.context_history[freq_region] = []

        self.context_history[freq_region].append(context)

        # Keep only last 20 signals per region (memory limit)
        if len(self.context_history[freq_region]) > 20:
            self.context_history[freq_region] = self.context_history[freq_region][-20:]

        # Update time offsets for all signals in history (aging)
        for signal in self.context_history[freq_region]:
            signal.time_offset_ms += 1000  # Age by 1 second (rough estimate)

    def _mix_signals(self, scenario: CollisionScenario) -> np.ndarray:
        """Mix primary signal + collisions + QRM."""
        mixed = scenario.primary_signal.copy()

        # Add colliding signals
        for collision_signal in scenario.collision_signals:
            mixed += collision_signal

        # Add QRM
        if scenario.has_qrm and scenario.qrm_signal is not None:
            mixed += scenario.qrm_signal

        return mixed

    def _create_enhanced_labels(self, primary_labels: Dict, scenario: CollisionScenario) -> Dict:
        """Create labels with collision and context metadata."""
        enhanced = primary_labels.copy()

        # Collision metadata
        enhanced['num_collisions'] = scenario.num_collisions
        enhanced['has_collisions'] = scenario.num_collisions > 0

        if scenario.num_collisions > 0:
            enhanced['collision_patterns'] = [l.get('pattern_id', 0) for l in scenario.collision_labels]
            enhanced['collision_offsets_ms'] = scenario.collision_offsets_ms

        # QRM metadata
        enhanced['has_qrm'] = scenario.has_qrm
        enhanced['qrm_type'] = scenario.qrm_type if scenario.has_qrm else None

        # Context signals (encode as tensor for neural network)
        if len(scenario.context_signals) > 0:
            # Create context kernel tensor [num_context, 16]
            context_kernels = []
            for ctx in scenario.context_signals[:8]:  # Max 8
                kernel = [
                    ctx.pattern_id / 7,  # Normalize to 0-1
                    ctx.frequency_triple / 42,
                    ctx.modulation / 3,
                    ctx.data_symbol_rate / 7,
                    ctx.polar_rate_idx / 4,
                    ctx.duration_windows / 255,
                    ctx.time_offset_ms / 10000,  # Normalize to ~0-1
                    (ctx.snr_db + 15) / 40,  # SNR range -15 to 25
                    # Pad to 16 dims
                    0, 0, 0, 0, 0, 0, 0, 0
                ]
                context_kernels.append(kernel)

            # Pad to 8 contexts
            while len(context_kernels) < 8:
                context_kernels.append([0] * 16)

            enhanced['context_kernels'] = torch.tensor(context_kernels, dtype=torch.float32)
            enhanced['context_mask'] = torch.tensor(
                [1.0 if i < len(scenario.context_signals) else 0.0 for i in range(8)],
                dtype=torch.float32
            )
        else:
            enhanced['context_kernels'] = torch.zeros(8, 16, dtype=torch.float32)
            enhanced['context_mask'] = torch.zeros(8, dtype=torch.float32)

        return enhanced

    # ========================================================================
    # QRM Signal Generators
    # ========================================================================

    def _generate_ssb_voice(self, length: int) -> np.ndarray:
        """Generate SSB (Single Sideband) voice signal."""
        # Simulate voice with formants
        t = np.arange(length) / self.sample_rate

        # Voice fundamental (100-300 Hz male voice)
        f0 = np.random.uniform(100, 300)
        voice = np.sin(2 * np.pi * f0 * t)

        # Add harmonics (formants)
        for harmonic in [2, 3, 4, 5]:
            amplitude = 1 / harmonic
            voice += amplitude * np.sin(2 * np.pi * f0 * harmonic * t + np.random.uniform(0, 2*np.pi))

        # Amplitude modulation (speech envelope)
        speech_rate = np.random.uniform(2, 5)  # Hz
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * speech_rate * t)
        voice *= envelope

        # SSB modulation (suppress carrier, single sideband)
        carrier_freq = np.random.uniform(500, 2500)  # Hz
        ssb = voice * np.exp(1j * 2 * np.pi * carrier_freq * t)

        return ssb

    def _generate_cw_morse(self, length: int) -> np.ndarray:
        """Generate CW (Morse code) signal."""
        t = np.arange(length) / self.sample_rate

        # CW tone (400-800 Hz)
        cw_freq = np.random.uniform(400, 800)
        carrier = np.exp(1j * 2 * np.pi * cw_freq * t)

        # Morse keying (random dots and dashes)
        wpm = np.random.uniform(15, 30)  # Words per minute
        dit_duration = 1.2 / wpm  # seconds

        keying = np.zeros(length)
        pos = 0
        while pos < length:
            # Random dit or dah
            if np.random.random() < 0.7:  # Dit (dot)
                duration_samples = int(dit_duration * self.sample_rate)
            else:  # Dah (dash)
                duration_samples = int(3 * dit_duration * self.sample_rate)

            # Key on
            end = min(pos + duration_samples, length)
            keying[pos:end] = 1.0
            pos = end

            # Inter-element space
            space_samples = int(dit_duration * self.sample_rate)
            pos += space_samples

        # Apply keying with smooth rise/fall
        keyed_signal = carrier * keying

        return keyed_signal

    def _generate_psk31(self, length: int) -> np.ndarray:
        """Generate PSK31 digital mode signal."""
        t = np.arange(length) / self.sample_rate

        # PSK31 parameters
        baud_rate = 31.25  # baud
        carrier_freq = np.random.uniform(800, 2000)  # Hz

        # Random data bits
        num_bits = int(length / self.sample_rate * baud_rate)
        data_bits = np.random.randint(0, 2, num_bits)

        # BPSK modulation
        symbols_per_sample = baud_rate / self.sample_rate
        phase = np.zeros(length)
        for i, bit in enumerate(data_bits):
            start_sample = int(i / symbols_per_sample)
            end_sample = int((i + 1) / symbols_per_sample)
            if end_sample > length:
                break
            phase[start_sample:end_sample] = np.pi * bit

        # Generate PSK31 signal
        psk31 = np.exp(1j * (2 * np.pi * carrier_freq * t + phase))

        return psk31

    def _generate_rtty(self, length: int) -> np.ndarray:
        """Generate RTTY (Radio Teletype) signal."""
        t = np.arange(length) / self.sample_rate

        # RTTY parameters (FSK)
        baud_rate = 45.45  # baud
        mark_freq = np.random.uniform(1000, 1500)  # Hz
        space_freq = mark_freq + 170  # 170 Hz shift

        # Random data bits
        num_bits = int(length / self.sample_rate * baud_rate)
        data_bits = np.random.randint(0, 2, num_bits)

        # FSK modulation
        freq = np.zeros(length)
        symbols_per_sample = baud_rate / self.sample_rate
        for i, bit in enumerate(data_bits):
            start_sample = int(i / symbols_per_sample)
            end_sample = int((i + 1) / symbols_per_sample)
            if end_sample > length:
                break
            freq[start_sample:end_sample] = mark_freq if bit else space_freq

        # Generate RTTY signal
        phase = np.cumsum(2 * np.pi * freq / self.sample_rate)
        rtty = np.exp(1j * phase)

        return rtty


# Demo function
def demo_collision_dataset():
    """Demonstrate collision-aware dataset generation."""
    from modules.training.src.signal_generator.generator import SignalGenerator

    print("=" * 80)
    print("COLLISION-AWARE DATASET DEMONSTRATION")
    print("=" * 80)

    # Create signal generator
    signal_gen = SignalGenerator()

    # Create collision-aware dataset
    dataset = CollisionAwareDataset(
        num_samples=100,
        signal_generator=signal_gen,
        sample_rate=48000,
        for_test=False,
        collision_probability=0.5,  # 50% for demo
        qrm_probability=0.3,  # 30% for demo
        seed=42,
        use_cache=False  # Disable cache for demo
    )

    print(f"\nDataset size: {len(dataset)} samples")

    # Test a few samples
    print("\n### Sample 1 ###")
    iq, labels = dataset[0]
    print(f"IQ shape: {iq.shape}")
    print(f"Pattern ID: {labels['pattern_id']}")
    print(f"Collisions: {labels['num_collisions']}")
    if labels['num_collisions'] > 0:
        print(f"  Collision offsets: {labels['collision_offsets_ms']} ms")
    print(f"QRM present: {labels['has_qrm']}")
    if labels['has_qrm']:
        print(f"  QRM type: {labels['qrm_type']}")
    print(f"Context signals: {labels['context_mask'].sum().item():.0f} available")

    print("\n### Sample 2 ###")
    iq, labels = dataset[1]
    print(f"Collisions: {labels['num_collisions']}")
    print(f"QRM present: {labels['has_qrm']}")
    print(f"Context signals: {labels['context_mask'].sum().item():.0f} available")

    print("\n" + "=" * 80)
    print("KEY FEATURES DEMONSTRATED:")
    print("1. Multiple overlapping CASCADE signals (collision scenarios)")
    print("2. QRM interference from other modes (SSB, CW, PSK31, RTTY)")
    print("3. Context signal tracking (up to 8 recent transmissions)")
    print("4. Ready for training Timing Expert, QRM Expert, and Integration Decoder")
    print("=" * 80)


if __name__ == "__main__":
    demo_collision_dataset()
