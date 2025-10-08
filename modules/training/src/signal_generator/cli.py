"""Command-line interface for CASCADE V2 signal generator."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

from .generator import SignalGenerator, KernelParameters


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='CASCADE V2 Signal Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a signal
  cascade-signal generate --pattern-id 3 --freq-pair 25 --modulation QPSK \\
                         --polar-rate 2/3 --message "Hello CASCADE" --output signal.npy

  # Verify V2 compliance
  cascade-signal verify signal.npy
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate a signal')
    gen_parser.add_argument('--pattern-id', type=int, required=True,
                           help='Pattern ID (0-7)')
    gen_parser.add_argument('--freq-pair', type=int, required=True,
                           help='Frequency pair (0-66)')
    gen_parser.add_argument('--modulation', type=str, required=True,
                           choices=['BPSK', 'QPSK', '8-PSK', '16-APSK'],
                           help='Modulation scheme')
    gen_parser.add_argument('--polar-rate', type=str, required=True,
                           help='Polar code rate (e.g., 2/3)')
    gen_parser.add_argument('--message', type=str, required=True,
                           help='Message to transmit')
    gen_parser.add_argument('--output', type=str, required=True,
                           help='Output file path (.npy for IQ, .json for metadata)')
    gen_parser.add_argument('--seed', type=int, default=None,
                           help='Random seed for deterministic generation')
    gen_parser.add_argument('--patterns-dir', type=str, default=None,
                           help='Directory containing pattern files')

    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify V2 compliance')
    verify_parser.add_argument('signal_file', type=str,
                              help='Signal file to verify (.npy)')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_command(args)
    elif args.command == 'verify':
        verify_command(args)
    else:
        parser.print_help()
        sys.exit(1)


def generate_command(args):
    """Handle generate command."""
    try:
        # Parse polar rate
        if '/' in args.polar_rate:
            k, n = map(int, args.polar_rate.split('/'))
        else:
            print(f"Error: Polar rate must be in format k/n (e.g., 2/3)")
            sys.exit(1)

        # Initialize generator
        patterns_dir = Path(args.patterns_dir) if args.patterns_dir else None
        gen = SignalGenerator(patterns_dir=patterns_dir)

        # Generate signal
        print(f"Generating signal...")
        print(f"  Pattern: {args.pattern_id}")
        print(f"  Frequency pair: {args.freq_pair}")
        print(f"  Modulation: {args.modulation}")
        print(f"  Polar rate: {k}/{n}")
        print(f"  Message: {args.message}")

        signal, metadata = gen.generate(
            pattern_id=args.pattern_id,
            frequency_pair=args.freq_pair,
            modulation_scheme=args.modulation,
            polar_rate=(k, n),
            message=args.message.encode('utf-8'),
            seed=args.seed
        )

        # Save IQ samples
        output_path = Path(args.output)
        np.save(output_path, signal.iq_samples)
        print(f"\nSaved IQ samples: {output_path}")

        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        metadata['tone_a_hz'] = signal.tone_a_hz
        metadata['tone_b_hz'] = signal.tone_b_hz
        metadata['generation_timestamp'] = signal.generation_timestamp

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")

        # Print summary
        print(f"\nSignal summary:")
        print(f"  Duration: {metadata['duration_seconds']:.2f}s")
        print(f"  Samples: {metadata['num_samples']}")
        print(f"  Pattern length: {metadata['pattern_length']}")
        print(f"  Tone A: {signal.tone_a_hz:.1f} Hz")
        print(f"  Tone B: {signal.tone_b_hz:.1f} Hz")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def verify_command(args):
    """Handle verify command."""
    try:
        # Load signal
        signal_path = Path(args.signal_file)
        if not signal_path.exists():
            print(f"Error: File not found: {signal_path}")
            sys.exit(1)

        iq_samples = np.load(signal_path)
        print(f"Loaded signal: {signal_path}")
        print(f"  Shape: {iq_samples.shape}")
        print(f"  Dtype: {iq_samples.dtype}")

        # Basic checks
        print(f"\nV2 Compliance Checks:")

        # Check dtype
        if iq_samples.dtype == np.complex64:
            print(f"  ✓ Dtype: complex64")
        else:
            print(f"  ✗ Dtype: {iq_samples.dtype} (expected complex64)")

        # Check sample rate (infer from duration)
        # This is a basic check - full compliance needs more information
        print(f"  ℹ Sample rate: Assumed 48000 Hz (cannot verify from IQ alone)")

        # Check power normalization
        avg_power = np.mean(np.abs(iq_samples) ** 2)
        print(f"  ℹ Average power: {avg_power:.3f}")

        if 0.5 < avg_power < 2.0:
            print(f"  ✓ Power level reasonable")
        else:
            print(f"  ⚠ Power level unusual")

        # Check for NaN/Inf
        if np.any(np.isnan(iq_samples)) or np.any(np.isinf(iq_samples)):
            print(f"  ✗ Contains NaN or Inf values")
        else:
            print(f"  ✓ No NaN/Inf values")

        print(f"\nNote: Full V2 compliance verification requires metadata file.")
        print(f"      Run with metadata: cascade-signal verify {signal_path} --metadata {signal_path.with_suffix('.json')}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
