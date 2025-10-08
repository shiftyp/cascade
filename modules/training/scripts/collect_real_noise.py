#!/usr/bin/env python3
"""
Collect real HF noise from WebSDR/KiwiSDR for CASCADE training augmentation.

Usage:
    python collect_real_noise.py --duration 600 --output data/real_noise/
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import time


def collect_from_kiwisdr(host, port, freq_khz, duration_sec, output_dir):
    """
    Collect real HF noise from KiwiSDR.

    Requires: pip install kiwiclient
    """
    try:
        from kiwiclient import KiwiSDRStream
    except ImportError:
        print("ERROR: kiwiclient not installed")
        print("Install with: pip install kiwiclient")
        return None

    print(f"Connecting to KiwiSDR: {host}:{port}")

    try:
        # Connect to KiwiSDR
        sdr = KiwiSDRStream(host=host, port=port)
        sdr.connect()

        # Set frequency and mode
        sdr.set_freq(freq_khz)
        sdr.set_mod('iq')  # Get I/Q samples
        sdr.set_agc(False)  # Disable AGC for consistent noise floor

        print(f"Recording {duration_sec} seconds at {freq_khz} kHz...")

        # Record
        samples = sdr.record(duration_sec)

        # Disconnect
        sdr.disconnect()

        print(f"Recorded {len(samples)} samples")

        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"kiwisdr_{host}_{freq_khz}khz_{timestamp}.npy"
        output_path = Path(output_dir) / filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, samples)

        print(f"Saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")

        return samples

    except Exception as e:
        print(f"Error: {e}")
        return None


def collect_from_websdr_manual():
    """
    Instructions for manually collecting from WebSDR.
    (WebSDR doesn't have a direct API, requires browser interaction)
    """
    print("\n" + "="*70)
    print("Manual WebSDR Collection Instructions")
    print("="*70)
    print("""
1. Open browser to: http://websdr.ewi.utwente.nl:8901/

2. Configure receiver:
   - Set frequency: 14.100 MHz (or another quiet frequency)
   - Set mode: USB or LSB
   - Set bandwidth: 2.7 kHz (or wider for full band noise)
   - Volume: ~50%

3. Start recording:
   - Use browser's built-in audio recorder, OR
   - Use Audacity (Input: Stereo Mix/System Audio):
     a. Click Record in Audacity
     b. Let run for 5-10 minutes
     c. Export as WAV (File → Export → WAV)

4. Convert WAV to numpy:
   - Run: python convert_wav_to_iq.py input.wav output.npy

5. Place in data/real_noise/ directory
    """)
    print("="*70 + "\n")


def collect_from_rtl_sdr(freq_hz, duration_sec, sample_rate, output_dir):
    """
    Collect using local RTL-SDR dongle.

    Requires: rtl-sdr tools installed
    """
    import subprocess

    print(f"Recording from RTL-SDR at {freq_hz/1e6:.3f} MHz")

    # Calculate number of samples
    num_samples = int(duration_sec * sample_rate)

    # Temporary raw file
    temp_file = Path(output_dir) / 'temp_rtlsdr.iq'

    # Run rtl_sdr command
    cmd = [
        'rtl_sdr',
        '-f', str(freq_hz),
        '-s', str(sample_rate),
        '-n', str(num_samples * 2),  # I and Q
        str(temp_file)
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)

        # Read raw I/Q data (unsigned 8-bit)
        raw_data = np.fromfile(temp_file, dtype=np.uint8)

        # Convert to complex float
        iq_data = (raw_data.astype(np.float32) - 127.5) / 127.5
        i_samples = iq_data[0::2]
        q_samples = iq_data[1::2]
        complex_samples = i_samples + 1j * q_samples

        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"rtlsdr_{freq_hz}hz_{timestamp}.npy"
        output_path = Path(output_dir) / filename

        np.save(output_path, complex_samples)

        # Clean up temp file
        temp_file.unlink()

        print(f"Saved to: {output_path}")
        print(f"Samples: {len(complex_samples)}")

        return complex_samples

    except subprocess.CalledProcessError as e:
        print(f"Error running rtl_sdr: {e}")
        print("Make sure rtl-sdr tools are installed:")
        print("  Ubuntu/Debian: sudo apt install rtl-sdr")
        print("  macOS: brew install rtl-sdr")
        return None


def collect_multiple_locations(freq_khz, duration_sec, output_dir):
    """
    Collect from multiple KiwiSDR locations for diversity.
    """
    # Popular public KiwiSDRs (check http://kiwisdr.com/public/ for more)
    kiwisdrs = [
        ('websdr.ewi.utwente.nl', 8073, 'Netherlands'),
        ('kiwisdr.dk6nl.de', 8073, 'Germany'),
        ('kiwisdr.sk3w.se', 8073, 'Sweden'),
        # Add more from http://kiwisdr.com/public/
    ]

    print(f"\nCollecting from {len(kiwisdrs)} locations...")

    for host, port, location in kiwisdrs:
        print(f"\n{'='*70}")
        print(f"Location: {location} ({host})")
        print(f"{'='*70}")

        samples = collect_from_kiwisdr(host, port, freq_khz, duration_sec, output_dir)

        if samples is not None:
            print(f"✓ Successfully recorded from {location}")
        else:
            print(f"✗ Failed to record from {location}")

        # Be polite - wait between requests
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description='Collect real HF noise for CASCADE training')
    parser.add_argument('--source', type=str, default='kiwisdr',
                       choices=['kiwisdr', 'websdr', 'rtlsdr', 'multi'],
                       help='Data source')
    parser.add_argument('--host', type=str, default='websdr.ewi.utwente.nl',
                       help='KiwiSDR hostname')
    parser.add_argument('--port', type=int, default=8073,
                       help='KiwiSDR port')
    parser.add_argument('--freq', type=float, default=14.100,
                       help='Frequency in MHz')
    parser.add_argument('--duration', type=int, default=600,
                       help='Recording duration in seconds')
    parser.add_argument('--sample-rate', type=int, default=2400000,
                       help='Sample rate (for RTL-SDR)')
    parser.add_argument('--output', type=str, default='data/real_noise',
                       help='Output directory')

    args = parser.parse_args()

    # Convert frequency to kHz for KiwiSDR
    freq_khz = int(args.freq * 1000)
    freq_hz = int(args.freq * 1e6)

    print("\n" + "="*70)
    print("CASCADE Real Noise Collection")
    print("="*70)
    print(f"Source: {args.source}")
    print(f"Frequency: {args.freq:.3f} MHz")
    print(f"Duration: {args.duration} seconds")
    print(f"Output: {args.output}")
    print("="*70 + "\n")

    if args.source == 'kiwisdr':
        collect_from_kiwisdr(args.host, args.port, freq_khz, args.duration, args.output)

    elif args.source == 'websdr':
        collect_from_websdr_manual()

    elif args.source == 'rtlsdr':
        collect_from_rtl_sdr(freq_hz, args.duration, args.sample_rate, args.output)

    elif args.source == 'multi':
        collect_multiple_locations(freq_khz, args.duration, args.output)

    print("\n" + "="*70)
    print("Collection complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Check collected files in: {args.output}/")
    print(f"2. Use HybridCascadeDataset with use_real_noise=True")
    print(f"3. Train CASCADE model with augmented data")


if __name__ == '__main__':
    main()
