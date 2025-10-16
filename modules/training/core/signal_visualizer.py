"""
Signal visualization module for CASCADE dataset generation.

Creates spectrograms, IQ constellation plots, and phase space plots
for randomly sampled signals during dataset generation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for parallel processing
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
import multiprocessing as mp
from queue import Empty
from dataclasses import dataclass


@dataclass
class SignalVisualization:
    """Container for signal data to visualize."""
    iq_samples: np.ndarray  # Complex IQ samples
    sample_rate: int
    metadata: Dict  # Pattern ID, SNR, frequency, etc.
    sample_idx: int
    output_path: Path


class SignalVisualizer:
    """
    Parallel signal visualizer for dataset generation.

    Creates spectrograms and phase space plots asynchronously during
    dataset generation to avoid blocking the main process.
    """

    def __init__(self, output_dir: str = "./visualizations", max_queue_size: int = 100):
        """
        Initialize visualizer with parallel processing.

        Args:
            output_dir: Directory to save visualization PNGs
            max_queue_size: Maximum number of pending visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Queue for passing signals to visualization worker
        self.viz_queue = mp.Queue(maxsize=max_queue_size)

        # Worker process for visualization
        self.worker_process = None
        self.running = False

    def start(self):
        """Start the background visualization worker."""
        if not self.running:
            self.running = True
            self.worker_process = mp.Process(
                target=self._visualization_worker,
                args=(self.viz_queue, self.output_dir)
            )
            self.worker_process.daemon = True
            self.worker_process.start()

    def stop(self):
        """Stop the background visualization worker."""
        if self.running:
            # Send sentinel to stop worker
            self.viz_queue.put(None)
            self.worker_process.join(timeout=5)
            self.running = False

    def add_signal(self, iq_samples: np.ndarray, sample_rate: int,
                   metadata: Dict, sample_idx: int, blocking: bool = False):
        """
        Add signal to visualization queue.

        Args:
            iq_samples: Complex IQ samples
            sample_rate: Sample rate (Hz)
            metadata: Dictionary with pattern_id, snr_db, frequency_triple, etc.
            sample_idx: Sample index in dataset
            blocking: If True, wait for queue space; if False, skip if queue full
        """
        if not self.running:
            return

        # Create output filename
        pattern_id = metadata.get('pattern_id', -1)
        snr_db = metadata.get('snr_db', 0.0)
        freq_triple = metadata.get('frequency_triple', -1)

        filename = f"sample_{sample_idx:06d}_pat{pattern_id}_snr{snr_db:.0f}dB_freq{freq_triple:02d}.png"
        output_path = self.output_dir / filename

        viz_data = SignalVisualization(
            iq_samples=iq_samples.copy(),
            sample_rate=sample_rate,
            metadata=metadata.copy(),
            sample_idx=sample_idx,
            output_path=output_path
        )

        try:
            if blocking:
                self.viz_queue.put(viz_data)
            else:
                # Non-blocking: skip if queue full
                self.viz_queue.put_nowait(viz_data)
        except:
            # Queue full or other error - skip this visualization
            pass

    @staticmethod
    def _visualization_worker(queue: mp.Queue, output_dir: Path):
        """
        Background worker that creates visualizations.

        Runs in separate process to avoid blocking dataset generation.
        """
        while True:
            try:
                # Get next visualization task
                viz_data = queue.get(timeout=1.0)

                # Sentinel value to stop worker
                if viz_data is None:
                    break

                # Create visualization
                SignalVisualizer._create_visualization(viz_data)

            except Empty:
                continue
            except Exception as e:
                # Log error but continue processing
                print(f"Visualization error: {e}")
                continue

    @staticmethod
    def _create_visualization(viz_data: SignalVisualization):
        """
        Create high-resolution streaming-style visualization.

        Shows:
        - Time domain I/Q plot
        - High-res spectrogram (4096-point FFT, blackmanharris window, jet colormap)
        - Power envelope
        """
        iq_samples = viz_data.iq_samples
        sample_rate = viz_data.sample_rate
        metadata = viz_data.metadata

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        # Time array
        duration = len(iq_samples) / sample_rate
        t = np.arange(len(iq_samples)) / sample_rate

        # --- Subplot 1: Time Domain (I and Q) ---
        ax1 = axes[0]

        # Decimate if too many samples for plotting
        decimate_factor = max(1, len(iq_samples) // 50000)
        t_plot = t[::decimate_factor]
        i_plot = np.real(iq_samples[::decimate_factor])
        q_plot = np.imag(iq_samples[::decimate_factor])

        ax1.plot(t_plot, i_plot, 'b-', alpha=0.7, linewidth=0.5, label='I')
        ax1.plot(t_plot, q_plot, 'r-', alpha=0.7, linewidth=0.5, label='Q')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_xlim([0, duration])  # Align with other plots

        # Build title
        pattern_id = metadata.get('pattern_id', 'N/A')
        snr_db = metadata.get('snr_db', 0.0)
        freq_triple = metadata.get('frequency_triple', 'N/A')
        modulation = metadata.get('modulation', 'N/A')
        data_rate = metadata.get('data_symbol_rate', 'N/A')
        prop_mode = metadata.get('propagation_mode', 'N/A')
        qrn_type = metadata.get('qrn_type', 'N/A')
        num_messages = metadata.get('num_messages', 1)
        k_index = metadata.get('k_index', 0.0)

        ax1.set_title(
            f"Stream {viz_data.sample_idx} | {duration:.1f}s | {num_messages} messages | "
            f"Pattern {pattern_id} | Freq Triple {freq_triple} | SNR {snr_db:.1f} dB | "
            f"{modulation} @ {data_rate} sym/s",
            fontsize=10
        )

        # Mark message boundaries with actual timings (updated after signal generation)
        message_list = metadata.get('message_list', [])
        for i, msg in enumerate(message_list):
            start_time = msg.get('start_sample', 0) / sample_rate
            end_time = msg.get('end_sample', 0) / sample_rate
            if end_time > start_time:
                ax1.axvspan(start_time, end_time, alpha=0.15, color='green')
                # Add label at start of message
                msg_label = f"P{msg.get('pattern_id', '?')} F{msg.get('frequency_triple', '?')}"
                ax1.text(start_time + 0.05, ax1.get_ylim()[1]*0.85,
                        msg_label, fontsize=7, verticalalignment='top')

        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Subplot 2: High-Resolution Spectrogram ---
        ax2 = axes[1]

        # Use high-resolution spectrogram (same as visualize_streaming.py)
        from scipy.signal import get_window

        NFFT = 4096  # High frequency resolution
        noverlap = int(NFFT * 0.9)  # 90% overlap for smooth waterfall
        window = get_window('blackmanharris', NFFT)  # Excellent sidelobe suppression

        # Create spectrogram
        Sxx, freqs, bins, im = ax2.specgram(
            iq_samples,
            Fs=sample_rate,
            NFFT=NFFT,
            noverlap=noverlap,
            window=window,
            cmap='jet',  # Classic waterfall colors
            scale='dB',
            vmin=-80,
            vmax=-20
        )

        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (s)')
        ax2.set_title(f'Spectrogram | Propagation: {prop_mode} | QRN: {qrn_type} | K-index: {k_index:.1f}')
        ax2.set_ylim([0, 3500])  # Show full CASCADE bandwidth
        ax2.set_xlim([0, duration])  # Align with other plots

        fig.colorbar(im, ax=ax2, label='Power (dB)')

        # --- Subplot 3: Power Envelope ---
        ax3 = axes[2]

        power = np.abs(iq_samples)**2
        # Smooth with 10ms window
        window_size = int(0.01 * sample_rate)  # 10ms
        power_smooth = np.convolve(power, np.ones(window_size)/window_size, mode='same')

        ax3.plot(t, 10*np.log10(power_smooth + 1e-10), 'g-', linewidth=1)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Power (dB)')
        ax3.set_title(f'Power Envelope')
        ax3.set_xlim([0, duration])  # Align with other plots
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        plt.savefig(viz_data.output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def create_single_visualization(iq_samples: np.ndarray, sample_rate: int,
                               metadata: Dict, output_path: str):
    """
    Convenience function to create a single visualization synchronously.

    Args:
        iq_samples: Complex IQ samples
        sample_rate: Sample rate (Hz)
        metadata: Dictionary with pattern_id, snr_db, etc.
        output_path: Path to save PNG
    """
    viz_data = SignalVisualization(
        iq_samples=iq_samples,
        sample_rate=sample_rate,
        metadata=metadata,
        sample_idx=metadata.get('sample_idx', 0),
        output_path=Path(output_path)
    )

    SignalVisualizer._create_visualization(viz_data)


if __name__ == "__main__":
    """Test visualization with synthetic signal."""
    print("Testing signal visualizer...")

    # Create test signal (3-tone FSK-like)
    sample_rate = 48000
    duration = 2.0  # seconds
    t = np.arange(0, duration, 1/sample_rate)

    # Simulate 3-FSK with BPSK data modulation
    f1, f2, f3 = 1560, 1580, 1600  # Hz (frequency triple 21)

    # Pattern layer (3-FSK)
    pattern_symbols = np.random.choice([0, 1, 2], size=50)
    pattern_signal = np.zeros(len(t), dtype=complex)

    symbol_duration = len(t) // len(pattern_symbols)
    for i, symbol in enumerate(pattern_symbols):
        start_idx = i * symbol_duration
        end_idx = min((i + 1) * symbol_duration, len(t))
        t_sym = t[start_idx:end_idx]

        if symbol == 0:
            freq = f1
        elif symbol == 1:
            freq = f2
        else:
            freq = f3

        pattern_signal[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t_sym)

    # Data layer (BPSK)
    data_symbols = np.random.choice([-1, 1], size=300)
    data_signal = np.zeros(len(t), dtype=complex)

    data_symbol_duration = len(t) // len(data_symbols)
    for i, symbol in enumerate(data_symbols):
        start_idx = i * data_symbol_duration
        end_idx = min((i + 1) * data_symbol_duration, len(t))
        data_signal[start_idx:end_idx] = symbol

    # Combine layers
    iq_samples = pattern_signal * data_signal

    # Add some noise
    noise = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.3
    iq_samples += noise

    # Create visualization
    metadata = {
        'pattern_id': 2,
        'frequency_triple': 21,
        'snr_db': 12.5,
        'modulation': 'QPSK',
        'data_symbol_rate': 150,
        'propagation_mode': 'multipath_sparse',
        'qrn_type': 'static',
        'sample_idx': 0
    }

    output_dir = Path("./test_visualizations")
    output_dir.mkdir(exist_ok=True)

    create_single_visualization(
        iq_samples=iq_samples,
        sample_rate=sample_rate,
        metadata=metadata,
        output_path=output_dir / "test_signal.png"
    )

    print(f"✓ Test visualization saved to {output_dir / 'test_signal.png'}")
