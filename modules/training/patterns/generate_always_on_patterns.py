#!/usr/bin/env python3
"""
Generate patterns for CASCADE always-on center frequency design.

Creates three synchronized patterns:
1. Center pattern - continuous transmission (all symbols)
2. Lower pattern - even symbols only
3. Upper pattern - odd symbols only

All three patterns carry the SAME data for maximum redundancy.
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse


class AlwaysOnPatternGenerator:
    """Generate patterns for always-on center frequency design."""

    def __init__(self,
                 num_patterns: int = 4,  # 0-3 for 4 patterns as per spec
                 pattern_length: int = 1024,  # Full master pattern length
                 seed: int = None):
        """
        Initialize pattern generator.

        Args:
            num_patterns: Number of pattern sets to generate (default 4)
            pattern_length: Length of master pattern in symbols (default 1024)
            seed: Random seed for reproducibility
        """
        self.num_patterns = num_patterns
        self.pattern_length = pattern_length

        if seed is not None:
            np.random.seed(seed)

    def generate_base_pattern(self, pattern_id: int) -> np.ndarray:
        """
        Generate a base ternary pattern using genetic algorithm principles.

        Args:
            pattern_id: Pattern ID (0-3)

        Returns:
            Base pattern of length pattern_length with values 0, 1, 2
        """
        # Use different seeds for each pattern to ensure uniqueness
        state = np.random.RandomState(42 + pattern_id * 1000)

        # Generate ternary pattern with good autocorrelation properties
        # Start with random ternary values
        pattern = state.randint(0, 3, self.pattern_length)

        # Apply some structure to improve autocorrelation
        # Use a simple repetition code with variation
        block_size = 16
        num_blocks = self.pattern_length // block_size

        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size

            # Create structured block with some randomness
            if i % 4 == 0:
                # Ascending pattern
                pattern[start:end] = np.tile([0, 1, 2, 1], block_size // 4)
            elif i % 4 == 1:
                # Descending pattern
                pattern[start:end] = np.tile([2, 1, 0, 1], block_size // 4)
            elif i % 4 == 2:
                # Alternating pattern
                pattern[start:end] = np.tile([0, 2], block_size // 2)
            else:
                # Keep random for diversity
                pass

        # Add pattern-specific modifications
        if pattern_id == 0:
            # Pattern 0: More 0s for lower frequency bias
            mask = state.random(self.pattern_length) < 0.4
            pattern[mask] = 0
        elif pattern_id == 1:
            # Pattern 1: Balanced
            pass
        elif pattern_id == 2:
            # Pattern 2: More 2s for upper frequency bias
            mask = state.random(self.pattern_length) < 0.4
            pattern[mask] = 2
        else:
            # Pattern 3: Alternating bias
            pattern[::2] = (pattern[::2] + 1) % 3

        return pattern

    def generate_always_on_patterns(self, pattern_id: int) -> Dict[str, np.ndarray]:
        """
        Generate synchronized patterns for always-on center design.

        Creates three patterns that work together:
        - Center: Always transmitting (all symbols)
        - Lower: Even symbols only
        - Upper: Odd symbols only

        Args:
            pattern_id: Pattern ID (0-3)

        Returns:
            Dictionary with 'center', 'lower', 'upper' patterns
        """
        # Generate base pattern
        base_pattern = self.generate_base_pattern(pattern_id)

        # Center pattern uses the full base pattern
        center_pattern = base_pattern.copy()

        # Lower pattern: Active on even symbols (0, 2, 4, ...)
        # Use -1 to indicate "no transmission" slots
        lower_pattern = np.full(self.pattern_length, -1, dtype=np.int8)
        lower_pattern[::2] = base_pattern[::2]  # Copy even indices

        # Upper pattern: Active on odd symbols (1, 3, 5, ...)
        upper_pattern = np.full(self.pattern_length, -1, dtype=np.int8)
        upper_pattern[1::2] = base_pattern[1::2]  # Copy odd indices

        return {
            'center': center_pattern,
            'lower': lower_pattern,
            'upper': upper_pattern,
            'base': base_pattern  # Include base for reference
        }

    def validate_patterns(self, patterns: Dict[str, np.ndarray]) -> bool:
        """
        Validate that patterns meet always-on requirements.

        Args:
            patterns: Dictionary of patterns to validate

        Returns:
            True if valid, raises ValueError if not
        """
        center = patterns['center']
        lower = patterns['lower']
        upper = patterns['upper']

        # Check lengths
        if len(center) != self.pattern_length:
            raise ValueError(f"Center pattern length {len(center)} != {self.pattern_length}")
        if len(lower) != self.pattern_length:
            raise ValueError(f"Lower pattern length {len(lower)} != {self.pattern_length}")
        if len(upper) != self.pattern_length:
            raise ValueError(f"Upper pattern length {len(upper)} != {self.pattern_length}")

        # Check center has no gaps
        if np.any(center == -1):
            raise ValueError("Center pattern must not have gaps (-1 values)")

        # Check lower is only active on even symbols
        for i in range(self.pattern_length):
            if i % 2 == 0:  # Even index
                if lower[i] == -1:
                    raise ValueError(f"Lower pattern missing at even index {i}")
            else:  # Odd index
                if lower[i] != -1:
                    raise ValueError(f"Lower pattern active at odd index {i}")

        # Check upper is only active on odd symbols
        for i in range(self.pattern_length):
            if i % 2 == 1:  # Odd index
                if upper[i] == -1:
                    raise ValueError(f"Upper pattern missing at odd index {i}")
            else:  # Even index
                if upper[i] != -1:
                    raise ValueError(f"Upper pattern active at even index {i}")

        # Check that active values match between patterns
        for i in range(self.pattern_length):
            if i % 2 == 0:  # Even
                if center[i] != lower[i]:
                    raise ValueError(f"Center/lower mismatch at index {i}")
            else:  # Odd
                if center[i] != upper[i]:
                    raise ValueError(f"Center/upper mismatch at index {i}")

        return True

    def generate_all_patterns(self) -> Dict:
        """
        Generate all pattern sets.

        Returns:
            Dictionary containing all pattern sets and metadata
        """
        all_patterns = {}

        for pattern_id in range(self.num_patterns):
            patterns = self.generate_always_on_patterns(pattern_id)
            self.validate_patterns(patterns)
            all_patterns[pattern_id] = patterns

            print(f"Generated pattern {pattern_id}:")
            print(f"  Center: {patterns['center'][:20]}... (always on)")
            print(f"  Lower:  {patterns['lower'][:20]}... (even symbols)")
            print(f"  Upper:  {patterns['upper'][:20]}... (odd symbols)")
            print()

        # Package with metadata
        result = {
            'patterns': all_patterns,
            'num_patterns': self.num_patterns,
            'pattern_length': self.pattern_length,
            'design_type': 'always_on_center',
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'description': 'Always-on center frequency patterns with alternating outer frequencies'
        }

        return result

    def save_patterns(self, output_dir: str = './patterns/always_on'):
        """
        Generate and save patterns to disk.

        Args:
            output_dir: Directory to save patterns
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate patterns
        pattern_data = self.generate_all_patterns()

        # Save as pickle
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"always_on_patterns_{timestamp}.pkl"

        with open(filename, 'wb') as f:
            pickle.dump(pattern_data, f)

        print(f"\nPatterns saved to: {filename}")

        # Also save as numpy arrays for easy inspection
        np_dir = output_path / f"numpy_{timestamp}"
        np_dir.mkdir(exist_ok=True)

        for pattern_id, patterns in pattern_data['patterns'].items():
            for ptype, pattern in patterns.items():
                np_file = np_dir / f"pattern_{pattern_id}_{ptype}.npy"
                np.save(np_file, pattern)

        print(f"NumPy arrays saved to: {np_dir}")

        return filename


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate patterns for CASCADE always-on center frequency design"
    )

    parser.add_argument(
        '--num-patterns',
        type=int,
        default=4,
        help='Number of pattern sets to generate (default: 4)'
    )

    parser.add_argument(
        '--pattern-length',
        type=int,
        default=1024,
        help='Length of master pattern in symbols (default: 1024)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./patterns/always_on',
        help='Output directory for patterns (default: ./patterns/always_on)'
    )

    args = parser.parse_args()

    print("="*60)
    print("CASCADE Always-On Center Pattern Generator")
    print("="*60)
    print(f"Patterns: {args.num_patterns}")
    print(f"Length: {args.pattern_length} symbols")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    print()

    # Create generator
    generator = AlwaysOnPatternGenerator(
        num_patterns=args.num_patterns,
        pattern_length=args.pattern_length,
        seed=args.seed
    )

    # Generate and save patterns
    generator.save_patterns(args.output_dir)

    print("\n✅ Pattern generation complete!")


if __name__ == "__main__":
    main()