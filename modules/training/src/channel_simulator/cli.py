"""Command-line interface for CASCADE V2 Channel Orchestrator."""

import argparse
import sys
from pathlib import Path
import numpy as np

from .orchestrator import ChannelOrchestrator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='CASCADE V2 Channel Orchestrator - Synthetic Training Data Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 examples for each expert type
  cascade-orchestrator generate --input clean_signals.npz --output datasets/ \\
                               --num-per-expert 100

  # Generate specific expert types only
  cascade-orchestrator generate --input clean_signals.npz --output datasets/ \\
                               --experts clean awgn qrn --num-per-expert 50

  # List available expert types
  cascade-orchestrator list-experts
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate expert datasets')
    gen_parser.add_argument('--input', type=str, required=True,
                           help='Input NPZ file with clean signals')
    gen_parser.add_argument('--output', type=str, required=True,
                           help='Output directory for expert datasets')
    gen_parser.add_argument('--num-per-expert', type=int, default=100,
                           help='Number of examples per expert (default: 100)')
    gen_parser.add_argument('--experts', type=str, nargs='*', default=None,
                           help='Expert types to generate (default: all)')
    gen_parser.add_argument('--format', type=str, default='npz',
                           choices=['npz', 'hdf5', 'zarr'],
                           help='Output format (default: npz)')
    gen_parser.add_argument('--seed', type=int, default=None,
                           help='Random seed for reproducibility')
    gen_parser.add_argument('--sample-rate', type=int, default=48000,
                           help='Sample rate in Hz (default: 48000)')

    # List experts command
    list_parser = subparsers.add_parser('list-experts', help='List available expert types')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show dataset information')
    info_parser.add_argument('dataset', type=str, help='Path to dataset file')
    info_parser.add_argument('--format', type=str, default='npz',
                            choices=['npz', 'hdf5', 'zarr'],
                            help='Dataset format (default: npz)')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_command(args)
    elif args.command == 'list-experts':
        list_experts_command(args)
    elif args.command == 'info':
        info_command(args)
    else:
        parser.print_help()
        sys.exit(1)


def generate_command(args):
    """Handle generate command."""
    try:
        # Load clean signals
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        print(f"Loading clean signals from: {input_path}")
        data = np.load(input_path)

        if 'signals' in data:
            clean_signals = list(data['signals'])
        else:
            # Assume first array is signals
            clean_signals = list(data[list(data.keys())[0]])

        print(f"  Loaded {len(clean_signals)} clean signals")
        print(f"  Signal shape: {clean_signals[0].shape}")
        print()

        # Initialize orchestrator
        orchestrator = ChannelOrchestrator(
            sample_rate=args.sample_rate,
            seed=args.seed
        )

        # Determine which experts to generate
        if args.experts:
            expert_names = args.experts
            # Validate expert names
            available = list(orchestrator.expert_configs.keys())
            invalid = [e for e in expert_names if e not in available]
            if invalid:
                print(f"Error: Invalid expert types: {invalid}")
                print(f"Available: {available}")
                sys.exit(1)
        else:
            expert_names = list(orchestrator.expert_configs.keys())

        print(f"Generating datasets for {len(expert_names)} experts:")
        for name in expert_names:
            print(f"  - {name}")
        print()

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate datasets
        rng = np.random.default_rng(args.seed)

        for expert_name in expert_names:
            print(f"Generating expert: {expert_name}")

            # Sample clean signals
            if len(clean_signals) < args.num_per_expert:
                indices = rng.choice(len(clean_signals), args.num_per_expert, replace=True)
            else:
                indices = rng.choice(len(clean_signals), args.num_per_expert, replace=False)

            expert_clean_signals = [clean_signals[i] for i in indices]

            # Generate dataset
            dataset = orchestrator.generate_expert_dataset(
                expert_clean_signals, expert_name,
                seed=rng.integers(0, 2**31)
            )

            # Save dataset
            if args.format == 'npz':
                output_path = output_dir / f"{expert_name}_dataset.npz"
            elif args.format == 'hdf5':
                output_path = output_dir / f"{expert_name}_dataset.h5"
            elif args.format == 'zarr':
                output_path = output_dir / f"{expert_name}_dataset.zarr"

            orchestrator.save_dataset(dataset, output_path, format=args.format)
            print()

        print("=" * 60)
        print(f"✓ Generated {len(expert_names)} expert datasets")
        print(f"  Output directory: {output_dir}")
        print(f"  Total examples: {len(expert_names) * args.num_per_expert}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def list_experts_command(args):
    """Handle list-experts command."""
    orchestrator = ChannelOrchestrator()

    print("Available Expert Types:")
    print("=" * 60)

    for name, config in orchestrator.expert_configs.items():
        print(f"\n{name.upper()}:")
        print(f"  Effects:")
        if config.awgn_enabled:
            print(f"    - AWGN: SNR {config.awgn_snr_range_db[0]:.1f} to {config.awgn_snr_range_db[1]:.1f} dB")
        if config.qrn_enabled:
            print(f"    - QRN: power {config.qrn_power:.2f}")
        if config.multipath_enabled:
            print(f"    - Multipath: Watterson HF profile")
        if config.qrm_enabled:
            print(f"    - QRM: power {config.qrm_power:.2f}")
        if config.collision_enabled:
            print(f"    - Collisions: probability {config.collision_probability:.2f}")

        if not any([config.awgn_enabled, config.qrn_enabled, config.multipath_enabled,
                   config.qrm_enabled, config.collision_enabled]):
            print(f"    (no effects)")


def info_command(args):
    """Handle info command."""
    try:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {dataset_path}")
            sys.exit(1)

        orchestrator = ChannelOrchestrator()
        dataset = orchestrator.load_dataset(dataset_path, format=args.format)

        print("Dataset Information:")
        print("=" * 60)
        print(f"Path: {dataset_path}")
        print(f"Format: {args.format}")
        print(f"Number of examples: {len(dataset)}")

        if dataset:
            signal, metadata = dataset[0]
            print(f"\nSignal shape: {signal.shape}")
            print(f"Signal dtype: {signal.dtype}")

            print(f"\nExpert type: {metadata.get('expert_type', 'Unknown')}")
            print(f"Effects applied: {', '.join(metadata.get('effects_applied', []))}")

            if 'snr_db' in metadata:
                snrs = [meta['snr_db'] for _, meta in dataset if 'snr_db' in meta]
                print(f"\nSNR range: {min(snrs):.1f} to {max(snrs):.1f} dB")
                print(f"Mean SNR: {np.mean(snrs):.1f} dB")

            powers = [meta['signal_power'] for _, meta in dataset]
            print(f"\nPower range: {min(powers):.3f} to {max(powers):.3f}")
            print(f"Mean power: {np.mean(powers):.3f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
