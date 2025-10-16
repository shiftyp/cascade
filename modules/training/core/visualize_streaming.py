#!/usr/bin/env python3
"""
Visualize CASCADE streaming dataset.

Shows:
- Full 10s stream with multiple messages
- Individual 2048-sample windows
- Spectrograms
- Message timing
"""

import sys
sys.path.insert(0, '/lambda/nfs/cascade-rf/cascade/modules/training/core')

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import json

def visualize_stream(cache_path, stream_idx=0, output_dir='./visualizations'):
    """
    Visualize a complete 10s stream with multiple messages.

    Args:
        cache_path: Path to streaming dataset (numpy directory or HDF5 file)
        stream_idx: Which stream to visualize
        output_dir: Where to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    cache_path = Path(cache_path)
    print(f"Visualizing stream {stream_idx} from {cache_path}")

    # Detect format: directory = numpy, file = HDF5
    if cache_path.is_dir():
        # NUMPY FORMAT
        print(f"  Format: Numpy memmap (thread-safe)")

        # Load stream from memmap
        streams = np.load(cache_path / 'streams.npy', mmap_mode='r')
        stream = streams[stream_idx]

        # Load labels from HDF5 file
        label_file = None
        for f in cache_path.parent.glob(f"{cache_path.name}_labels*.h5"):
            label_file = f
            break

        if label_file:
            with h5py.File(label_file, 'r') as f:
                k_index = f['k_index'][stream_idx]
                sfi = f['sfi'][stream_idx]
                prop_mode = f['propagation_modes'][stream_idx].decode('utf-8')
                qrn_type = f['qrn_types'][stream_idx].decode('utf-8')

                try:
                    msg_meta_str = f['message_metadata'][stream_idx]
                    if isinstance(msg_meta_str, bytes):
                        msg_meta_str = msg_meta_str.decode('utf-8')
                    msg_meta = json.loads(msg_meta_str) if msg_meta_str else []
                except:
                    msg_meta = []
        else:
            print(f"  ⚠️  No label file found, using defaults")
            k_index = 0.0
            sfi = 100.0
            prop_mode = 'unknown'
            qrn_type = 'unknown'
            msg_meta = []

    elif cache_path.is_file():
        # HDF5 FORMAT (original)
        print(f"  Format: HDF5")
        with h5py.File(cache_path, 'r') as f:
            # Check if multi-file
                if 'num_chunks' in f.attrs:
                    # Load chunk file
                    chunk_idx = stream_idx // f.attrs['chunk_size']
                    local_idx = stream_idx % f.attrs['chunk_size']
                    chunk_name = f['chunk_files'][chunk_idx]
                    if isinstance(chunk_name, bytes):
                        chunk_name = chunk_name.decode('utf-8')

                    chunk_path = Path(cache_path).parent / chunk_name
                    print(f"  Loading from chunk file: {chunk_path.name}")

                    with h5py.File(chunk_path, 'r') as cf:
                        stream = cf['streams'][local_idx]
                        k_index = cf['k_index'][local_idx]
                        sfi = cf['sfi'][local_idx]
                        prop_mode = cf['propagation_modes'][local_idx].decode('utf-8')
                        qrn_type = cf['qrn_types'][local_idx].decode('utf-8')

                        try:
                            msg_meta_str = cf['message_metadata'][local_idx]
                            if isinstance(msg_meta_str, bytes):
                                msg_meta_str = msg_meta_str.decode('utf-8')
                            msg_meta = json.loads(msg_meta_str) if msg_meta_str else []
                        except:
                            msg_meta = []
                else:
                    # Single file
                    stream = f['streams'][stream_idx]
                    k_index = f['k_index'][stream_idx]
                    sfi = f['sfi'][stream_idx]
                    prop_mode = f['propagation_modes'][stream_idx].decode('utf-8')
                    qrn_type = f['qrn_types'][stream_idx].decode('utf-8')

                    try:
                        msg_meta_str = f['message_metadata'][stream_idx]
                        if isinstance(msg_meta_str, bytes):
                            msg_meta_str = msg_meta_str.decode('utf-8')
                        msg_meta = json.loads(msg_meta_str) if msg_meta_str else []
                    except:
                        msg_meta = []
    else:
        print(f"  ⚠️  Cache path not found: {cache_path}")
        return
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Time array
    sample_rate = 48000
    duration = len(stream) / sample_rate
    t = np.arange(len(stream)) / sample_rate
    
    # Plot 1: Time domain (I and Q)
    ax = axes[0]
    ax.plot(t, np.real(stream), 'b-', alpha=0.7, linewidth=0.5, label='I')
    ax.plot(t, np.imag(stream), 'r-', alpha=0.7, linewidth=0.5, label='Q')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Stream {stream_idx}: {duration:.1f}s | {len(msg_meta)} messages | {prop_mode} | K={k_index:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark message boundaries
    for msg in msg_meta:
        start_time = msg['start_sample'] / sample_rate
        end_time = msg['end_sample'] / sample_rate
        ax.axvspan(start_time, end_time, alpha=0.2, color='green')
        ax.text(start_time, ax.get_ylim()[1]*0.9, f"P{msg['pattern_id']}", fontsize=8)
    
    # Plot 2: Spectrogram (high resolution with waterfall colors)
    ax = axes[1]

    # Use larger FFT for better frequency resolution
    # And window function to reduce sidelobes
    from scipy.signal import get_window

    NFFT = 4096  # Higher resolution (was 1024)
    noverlap = int(NFFT * 0.9)  # 90% overlap for smoother time resolution
    window = get_window('blackmanharris', NFFT)  # Excellent sidelobe suppression

    # Compute spectrogram with windowing
    Sxx, freqs, bins, im = ax.specgram(
        stream,
        Fs=sample_rate,
        NFFT=NFFT,
        noverlap=noverlap,
        window=window,
        cmap='jet',  # Classic waterfall colors
        scale='dB',
        vmin=-80,  # Dynamic range
        vmax=-20
    )

    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Spectrogram | QRN: {qrn_type}')
    ax.set_ylim([0, 3500])  # Show full CASCADE bandwidth + guard

    # Add colorbar
    fig.colorbar(im, ax=ax, label='Power (dB)')
    
    # Plot 3: Power envelope
    ax = axes[2]
    power = np.abs(stream)**2
    window_size = 480  # 10ms windows
    power_smooth = np.convolve(power, np.ones(window_size)/window_size, mode='same')
    ax.plot(t, 10*np.log10(power_smooth+1e-10), 'g-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(f'Power Envelope | SFI={sfi:.0f}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f'stream_{stream_idx}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # Print message details
    print(f"\n  Messages in stream:")
    for i, msg in enumerate(msg_meta):
        start_time = msg['start_sample'] / sample_rate
        duration_msg = (msg['end_sample'] - msg['start_sample']) / sample_rate
        print(f"    Msg {i+1}: t={start_time:.2f}s dur={duration_msg*1000:.0f}ms "
              f"P{msg['pattern_id']} F{msg['frequency_triple']} {msg['modulation']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize CASCADE streaming dataset')
    parser.add_argument('--cache', type=str, required=True, help='Path to HDF5 cache file')
    parser.add_argument('--stream', type=int, default=0, help='Stream index to visualize')
    parser.add_argument('--output', type=str, default='./visualizations', help='Output directory')
    parser.add_argument('--num', type=int, default=5, help='Number of streams to visualize')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CASCADE STREAMING DATASET VISUALIZATION")
    print("=" * 80)
    
    for i in range(args.num):
        stream_idx = args.stream + i
        try:
            visualize_stream(args.cache, stream_idx, args.output)
        except Exception as e:
            print(f"  ⚠️  Error visualizing stream {stream_idx}: {e}")
    
    print(f"\n✓ Visualized {args.num} streams")
    print(f"  Output: {args.output}/")
