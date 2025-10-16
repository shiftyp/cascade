#!/usr/bin/env python3
"""
Process tournament-generated patterns for always-on center frequency design.

Takes the output from the tournament pattern generator and creates:
1. Center patterns - for continuous transmission
2. Lower patterns - for even symbols only
3. Upper patterns - for odd symbols only
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Optional


class AlwaysOnPatternProcessor:
    """Process tournament patterns for always-on center design."""

    def __init__(self):
        """Initialize pattern processor."""
        pass

    def load_tournament_patterns(self, pattern_file: Path) -> Dict:
        """
        Load patterns from tournament output.

        Args:
            pattern_file: Path to tournament output pickle file

        Returns:
            Dictionary containing pattern data
        """
        with open(pattern_file, 'rb') as f:
            data = pickle.load(f)

        print(f"Loaded patterns from: {pattern_file}")

        # Handle different possible formats from tournament
        if 'patterns' in data:
            patterns = data['patterns']
        elif 'nested_patterns' in data:
            # Extract from nested format
            nested = data['nested_patterns']
            # Get the longest available patterns
            max_len = max(nested.keys())
            print(f"Using patterns of length {max_len}")
            pattern_dict = nested[max_len]
            patterns = self._extract_patterns_from_nested(pattern_dict)
        else:
            # Assume data itself is the patterns
            patterns = data

        return patterns

    def _extract_patterns_from_nested(self, pattern_dict: Dict) -> List[np.ndarray]:
        """Extract patterns from nested tournament format."""
        patterns = []

        if 'cores' in pattern_dict:
            cores = pattern_dict['cores']
            rep_map = pattern_dict.get('repetition_map', None)

            # Extract each pattern (cores are already full patterns, no repetition map needed)
            for i in range(len(cores)):
                pattern = np.array(cores[i], dtype=np.int8)
                patterns.append(pattern)
        else:
            # Direct pattern list
            for i in range(4):  # Expecting 4 patterns
                if i < len(pattern_dict):
                    patterns.append(np.array(pattern_dict[i], dtype=np.int8))

        return patterns

    def create_always_on_patterns(self, base_pattern: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create center/lower/upper patterns from a base pattern.

        Args:
            base_pattern: Base ternary pattern from tournament

        Returns:
            Dictionary with 'center', 'lower', 'upper' patterns
        """
        pattern_length = len(base_pattern)

        # Ensure pattern is ternary (0, 1, 2)
        base_pattern = np.array(base_pattern, dtype=np.int8)
        base_pattern = np.clip(base_pattern, 0, 2)

        # Center pattern: continuous transmission with the base pattern
        center_pattern = base_pattern.copy()

        # Lower pattern: Active on even symbols only (0, 2, 4, ...)
        lower_pattern = np.full(pattern_length, -1, dtype=np.int8)
        lower_pattern[::2] = base_pattern[::2]

        # Upper pattern: Active on odd symbols only (1, 3, 5, ...)
        upper_pattern = np.full(pattern_length, -1, dtype=np.int8)
        upper_pattern[1::2] = base_pattern[1::2]

        return {
            'center': center_pattern,
            'lower': lower_pattern,
            'upper': upper_pattern
        }

    def validate_patterns(self, patterns: Dict[str, np.ndarray]) -> bool:
        """Validate always-on pattern requirements."""
        center = patterns['center']
        lower = patterns['lower']
        upper = patterns['upper']

        # Check that center has no gaps
        if np.any(center == -1):
            raise ValueError("Center pattern must not have gaps")

        # Check lower/upper alternation
        for i in range(len(center)):
            if i % 2 == 0:  # Even index
                if lower[i] == -1:
                    raise ValueError(f"Lower pattern missing at even index {i}")
                if upper[i] != -1:
                    raise ValueError(f"Upper pattern active at even index {i}")
                if lower[i] != center[i]:
                    raise ValueError(f"Lower/center mismatch at index {i}")
            else:  # Odd index
                if upper[i] == -1:
                    raise ValueError(f"Upper pattern missing at odd index {i}")
                if lower[i] != -1:
                    raise ValueError(f"Lower pattern active at odd index {i}")
                if upper[i] != center[i]:
                    raise ValueError(f"Upper/center mismatch at index {i}")

        return True

    def process_tournament_output(self, pattern_file: Path, output_dir: Path) -> Path:
        """
        Process tournament output and save always-on formatted patterns.

        Args:
            pattern_file: Path to tournament output file
            output_dir: Directory to save processed patterns

        Returns:
            Path to saved pattern file
        """
        # Load tournament patterns
        tournament_patterns = self.load_tournament_patterns(pattern_file)

        # Process each pattern
        always_on_patterns = {}

        if isinstance(tournament_patterns, list):
            # List of patterns
            for i, pattern in enumerate(tournament_patterns[:4]):  # Use first 4
                print(f"\nProcessing pattern {i}...")
                always_on = self.create_always_on_patterns(pattern)
                self.validate_patterns(always_on)
                always_on_patterns[i] = always_on

                # Show sample
                print(f"  Center: {always_on['center'][:20]}...")
                print(f"  Lower:  {always_on['lower'][:20]}...")
                print(f"  Upper:  {always_on['upper'][:20]}...")
        else:
            # Dictionary format
            for i in range(min(4, len(tournament_patterns))):
                if i in tournament_patterns:
                    pattern = tournament_patterns[i]
                    print(f"\nProcessing pattern {i}...")
                    always_on = self.create_always_on_patterns(pattern)
                    self.validate_patterns(always_on)
                    always_on_patterns[i] = always_on

                    # Show sample
                    print(f"  Center: {always_on['center'][:20]}...")
                    print(f"  Lower:  {always_on['lower'][:20]}...")
                    print(f"  Upper:  {always_on['upper'][:20]}...")

        # Package with metadata
        output_data = {
            'patterns': always_on_patterns,
            'num_patterns': len(always_on_patterns),
            'pattern_length': len(always_on_patterns[0]['center']),
            'design_type': 'always_on_center',
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'source_file': str(pattern_file),
            'description': 'Tournament-optimized patterns for always-on center frequency design'
        }

        # Save to output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"always_on_patterns_{timestamp}.pkl"

        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)

        print(f"\nSaved always-on patterns to: {output_file}")

        # Also save individual numpy arrays for inspection
        np_dir = output_dir / f"numpy_{timestamp}"
        np_dir.mkdir(exist_ok=True)

        for pattern_id, patterns in always_on_patterns.items():
            for ptype, pattern in patterns.items():
                np_file = np_dir / f"pattern_{pattern_id}_{ptype}.npy"
                np.save(np_file, pattern)

        print(f"NumPy arrays saved to: {np_dir}")

        return output_file


def find_latest_tournament_output(checkpoint_dir: Path) -> Optional[Path]:
    """Find the latest tournament output file."""
    output_dir = checkpoint_dir / "output"
    if not output_dir.exists():
        return None

    # Look for pattern files
    pattern_files = list(output_dir.glob("patterns_*.pkl"))
    if not pattern_files:
        return None

    # Return the most recent
    return max(pattern_files, key=lambda p: p.stat().st_mtime)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process tournament patterns for always-on center design"
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Path to tournament output file (auto-detect if not specified)'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints/always_on',
        help='Tournament checkpoint directory (for auto-detection)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./patterns/always_on',
        help='Output directory for processed patterns'
    )

    args = parser.parse_args()

    print("="*60)
    print("CASCADE Always-On Pattern Processor")
    print("="*60)

    # Find input file
    if args.input:
        input_file = Path(args.input)
        if not input_file.exists():
            print(f"❌ Input file not found: {input_file}")
            return 1
    else:
        # Auto-detect latest tournament output
        checkpoint_dir = Path(args.checkpoint_dir)
        input_file = find_latest_tournament_output(checkpoint_dir)
        if not input_file:
            print(f"❌ No tournament output found in: {checkpoint_dir}/output/")
            print("   Run the tournament first or specify --input")
            return 1
        print(f"Auto-detected: {input_file}")

    # Process patterns
    processor = AlwaysOnPatternProcessor()
    output_dir = Path(args.output_dir)

    try:
        output_file = processor.process_tournament_output(input_file, output_dir)
        print("\n✅ Processing complete!")
        print(f"   Output: {output_file}")
        return 0
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())