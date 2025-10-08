#!/usr/bin/env python3
"""
Convert audio WAV files (from WebSDR) to complex I/Q samples.

Usage:
    python convert_wav_to_iq.py input.wav output.npy
"""

import argparse
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path


def wav_to_iq(wav_path, output_path, method='hilbert'):
    """
    Convert stereo WAV to complex I/Q samples.

    Args:
        wav_path: Path to input WAV file
        output_path: Path to output .npy file
        method: 'stereo' (I=left, Q=right) or 'hilbert' (analytic signal from mono)
    """
    print(f"Reading: {wav_path}")

    # Read WAV file
    sample_rate, audio_data = wavfile.read(wav_path)

    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: {audio_data.shape[1] if audio_data.ndim > 1 else 1}")
    print(f"  Samples: {len(audio_data)}")
    print(f"  Duration: {len(audio_data) / sample_rate:.1f} seconds")

    # Normalize to [-1, 1]
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)

    if method == 'stereo' and audio_data.ndim > 1:
        # Stereo: Left=I, Right=Q
        print("  Method: Stereo (L=I, R=Q)")
        i_samples = audio_data[:, 0]
        q_samples = audio_data[:, 1]
        iq_samples = i_samples + 1j * q_samples

    elif method == 'hilbert' or audio_data.ndim == 1:
        # Mono: Use Hilbert transform to create analytic signal
        print("  Method: Hilbert transform (mono to I/Q)")

        if audio_data.ndim > 1:
            # If stereo, average to mono
            audio_mono = audio_data.mean(axis=1)
        else:
            audio_mono = audio_data

        # Apply Hilbert transform to get analytic signal
        from scipy.signal import hilbert
        iq_samples = hilbert(audio_mono)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Remove DC bias
    iq_samples = iq_samples - np.mean(iq_samples)

    # Normalize
    max_amplitude = np.max(np.abs(iq_samples))
    if max_amplitude > 0:
        iq_samples = iq_samples / max_amplitude

    print(f"\nConverted to I/Q:")
    print(f"  Complex samples: {len(iq_samples)}")
    print(f"  Mean power: {np.mean(np.abs(iq_samples)**2):.6f}")
    print(f"  Peak amplitude: {np.max(np.abs(iq_samples)):.6f}")

    # Save as numpy array
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, iq_samples)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")

    return iq_samples


def analyze_iq(iq_samples, sample_rate):
    """Analyze I/Q samples and show statistics."""
    import matplotlib.pyplot as plt

    print("\n" + "="*70)
    print("I/Q Analysis")
    print("="*70)

    # Power spectrum
    from scipy.signal import welch
    f, psd = welch(iq_samples, fs=sample_rate, nperseg=1024)

    # Statistics
    print(f"Mean: {np.mean(iq_samples):.6f}")
    print(f"Std dev: {np.std(iq_samples):.6f}")
    print(f"Power: {np.mean(np.abs(iq_samples)**2):.6f}")
    print(f"Peak/RMS: {np.max(np.abs(iq_samples)) / np.sqrt(np.mean(np.abs(iq_samples)**2)):.2f}")

    # Dominant frequency
    peak_idx = np.argmax(psd)
    peak_freq = f[peak_idx]
    print(f"Dominant frequency: {peak_freq:.1f} Hz (offset from center)")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Time domain
    t = np.arange(min(10000, len(iq_samples))) / sample_rate
    axes[0, 0].plot(t, iq_samples[:len(t)].real, alpha=0.7, label='I')
    axes[0, 0].plot(t, iq_samples[:len(t)].imag, alpha=0.7, label='Q')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Time Domain (first 10k samples)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Spectrum
    axes[0, 1].semilogy(f, psd)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('PSD')
    axes[0, 1].set_title('Power Spectral Density')
    axes[0, 1].grid(True, alpha=0.3)

    # Constellation
    decimation = max(1, len(iq_samples) // 5000)
    axes[1, 0].scatter(iq_samples[::decimation].real,
                      iq_samples[::decimation].imag,
                      alpha=0.3, s=1)
    axes[1, 0].set_xlabel('I')
    axes[1, 0].set_ylabel('Q')
    axes[1, 0].set_title('Constellation Diagram')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')

    # Histogram
    axes[1, 1].hist(np.abs(iq_samples), bins=50, alpha=0.7)
    axes[1, 1].set_xlabel('Amplitude')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Amplitude Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('iq_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: iq_analysis.png")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Convert WAV to I/Q samples')
    parser.add_argument('input', type=str, help='Input WAV file')
    parser.add_argument('output', type=str, help='Output .npy file')
    parser.add_argument('--method', type=str, default='stereo',
                       choices=['stereo', 'hilbert'],
                       help='Conversion method')
    parser.add_argument('--analyze', action='store_true',
                       help='Show analysis plots')
    parser.add_argument('--sample-rate', type=int, default=None,
                       help='Override sample rate (Hz)')

    args = parser.parse_args()

    # Convert
    iq_samples = wav_to_iq(args.input, args.output, method=args.method)

    # Analyze if requested
    if args.analyze:
        # Get sample rate from WAV file
        sample_rate_wav, _ = wavfile.read(args.input)
        sample_rate = args.sample_rate if args.sample_rate else sample_rate_wav
        analyze_iq(iq_samples, sample_rate)

    print("\n✓ Conversion complete!")


if __name__ == '__main__':
    main()
