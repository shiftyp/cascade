"""CASCADE Pattern Generation CLI

Usage:
    python -m modules.training.patterns.generator --count 128 --seed 42
    python -m modules.training.patterns.validator cascade_patterns_128.bin
"""

import argparse
import sys
from pathlib import Path

from .generator import generate_pattern_set
from .multi_trial import generate_multi_trial, save_best_patterns
from .binary_format import save_pattern_file
from .validator import generate_validation_report
from .platform_detect import print_platform_info, get_platform_config


def main():
    parser = argparse.ArgumentParser(
        description='CASCADE Pattern Generation - Generate orthogonal 4D pattern sets'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate pattern set')
    gen_parser.add_argument(
        '--count',
        type=int,
        choices=[64, 128],
        default=128,
        help='Number of patterns to generate (default: 128)'
    )
    gen_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: modules/training/data/cascade_patterns_{count}.bin)'
    )
    gen_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    gen_parser.add_argument(
        '--trials',
        type=int,
        default=None,
        help='Number of trials (default: 8 for local depth strategy, 32 for cloud breadth)'
    )
    gen_parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Iterations per pattern (default: 400K local depth, 100K cloud breadth)'
    )
    gen_parser.add_argument(
        '--auto-tune',
        action='store_true',
        default=True,
        help='Auto-detect CPU architecture and optimize (default: enabled)'
    )
    gen_parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    gen_parser.add_argument(
        '--distributed',
        action='store_true',
        help='Use Fly.io distributed execution'
    )
    gen_parser.add_argument(
        '--workers',
        type=int,
        default=32,
        help='Number of Fly.io workers for distributed mode (default: 32)'
    )
    gen_parser.add_argument(
        '--region',
        type=str,
        default='iad',
        help='Fly.io region for workers (default: iad)'
    )

    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate pattern file')
    val_parser.add_argument(
        'pattern_file',
        type=str,
        help='Pattern file to validate'
    )
    val_parser.add_argument(
        '--output-report',
        type=str,
        default=None,
        help='Save validation report to file'
    )

    # Platform info command
    subparsers.add_parser('platform', help='Show platform detection info')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_command(args)
    elif args.command == 'validate':
        validate_command(args)
    elif args.command == 'platform':
        platform_command(args)
    else:
        parser.print_help()
        sys.exit(1)


def generate_command(args):
    """Execute pattern generation"""
    print("=" * 60)
    print("CASCADE Pattern Generation")
    print("=" * 60)

    # Determine output file
    if args.output is None:
        output_dir = Path(__file__).parent.parent / 'data'
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f'cascade_patterns_{args.count}.bin')

    print(f"Output file: {args.output}")
    print(f"Pattern count: {args.count}")
    print(f"Base seed: {args.seed}")

    # Distributed execution mode
    if args.distributed:
        print(f"\n🌐 Distributed mode: {args.workers} Fly.io workers")
        print(f"Region: {args.region}")

        # Call coordinator
        import subprocess
        coordinator_script = Path(__file__).parent.parent / 'fly-pattern-worker' / 'coordinator.py'

        cmd = [
            sys.executable,
            str(coordinator_script),
            '--workers', str(args.workers),
            '--count', str(args.count),
            '--seed', str(args.seed),
            '--region', args.region,
            '--output', args.output
        ]

        subprocess.run(cmd, check=True)
        return

    # Use multi-trial generation if trials specified or auto-tune enabled
    elif args.trials is not None or args.auto_tune:
        print(f"💻 Local multi-trial mode enabled")

        best_result = generate_multi_trial(
            count=args.count,
            num_trials=args.trials,
            seed_base=args.seed,
            auto_tune=args.auto_tune,
            max_iterations=args.iterations,
            visualize=not args.no_viz
        )

        # Save best patterns
        save_best_patterns(best_result, args.output)

    else:
        # Single trial generation
        print("Single trial mode\n")

        patterns = generate_pattern_set(
            count=args.count,
            seed=args.seed
        )

        save_pattern_file(patterns, args.output)

        file_size = Path(args.output).stat().st_size
        print(f"✓ Saved {len(patterns)} patterns ({file_size:,} bytes)")

    print(f"\n✓ Pattern generation complete!")


def validate_command(args):
    """Execute pattern validation"""
    print("=" * 60)
    print("CASCADE Pattern Validation")
    print("=" * 60)
    print(f"File: {args.pattern_file}\n")

    # Generate validation report
    report = generate_validation_report(args.pattern_file)

    print(report)

    # Save report if requested
    if args.output_report:
        with open(args.output_report, 'w') as f:
            f.write(report)
        print(f"\n✓ Report saved to {args.output_report}")


def platform_command(args):
    """Show platform detection info"""
    print_platform_info()


if __name__ == '__main__':
    main()
