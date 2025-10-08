"""Pattern loader for CASCADE V2 signal generator.

Loads pre-generated patterns from .pkl files produced by the genetic algorithm.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np


class PatternLoader:
    """Loads and caches CASCADE V2 patterns from pickle files.

    Patterns are generated offline by genetic algorithm and stored as .pkl files.
    This class provides efficient loading and caching of pattern data.

    Pattern file naming: pattern_{pattern_id}_{length}.pkl
    - pattern_id: 0-7 (8 patterns)
    - length: 64, 128, 256, 512, 1024, 2048 (6 nested lengths)

    Total: 48 pattern files (8 × 6)
    """

    def __init__(self, patterns_dir: Optional[Path] = None):
        """Initialize pattern loader.

        Args:
            patterns_dir: Directory containing pattern .pkl files.
                         Defaults to modules/training/patterns/tournament/
        """
        if patterns_dir is None:
            # Default to tournament patterns directory
            module_dir = Path(__file__).parent.parent.parent
            patterns_dir = module_dir / "patterns" / "tournament"

        self.patterns_dir = Path(patterns_dir)
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._loaded_count = 0

    def load_pattern(self, pattern_id: int, length: int) -> np.ndarray:
        """Load a specific pattern by ID and length.

        Args:
            pattern_id: Pattern ID (0-7)
            length: Pattern length in symbols (64, 128, 256, 512, 1024, 2048)

        Returns:
            np.ndarray: Pattern symbols, shape (length,), dtype uint8, values {0, 1, 2} for 3-FSK

        Raises:
            ValueError: If pattern_id or length is invalid
            FileNotFoundError: If pattern file doesn't exist
        """
        # Validate inputs
        if not (0 <= pattern_id <= 7):
            raise ValueError(f"pattern_id must be 0-7, got {pattern_id}")

        valid_lengths = [64, 128, 256, 512, 1024, 2048]
        if length not in valid_lengths:
            raise ValueError(f"length must be one of {valid_lengths}, got {length}")

        # Check cache first
        cache_key = (pattern_id, length)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load from file
        pattern_file = self.patterns_dir / f"pattern_{pattern_id}_{length}.pkl"

        if not pattern_file.exists():
            raise FileNotFoundError(
                f"Pattern file not found: {pattern_file}\n"
                f"Patterns must be generated first using the genetic algorithm.\n"
                f"Expected location: {self.patterns_dir}"
            )

        with open(pattern_file, 'rb') as f:
            pattern_data = pickle.load(f)

        # Extract pattern bits (format depends on genetic algorithm output)
        # Assuming it's either raw array or dict with 'pattern' key
        if isinstance(pattern_data, dict):
            pattern_bits = pattern_data['pattern']
        else:
            pattern_bits = pattern_data

        # Validate pattern format
        pattern_symbols = np.asarray(pattern_bits, dtype=np.uint8)

        if pattern_symbols.shape != (length,):
            raise ValueError(
                f"Pattern file {pattern_file} has shape {pattern_symbols.shape}, "
                f"expected ({length},)"
            )

        # Validate ternary symbols for 3-FSK
        if not np.all((pattern_symbols >= 0) & (pattern_symbols <= 2)):
            raise ValueError(
                f"Pattern file {pattern_file} contains values outside 0-2 range (3-FSK ternary)"
            )

        # Cache and return
        self._cache[cache_key] = pattern_symbols
        self._loaded_count += 1

        return pattern_symbols

    def load_all_patterns(self) -> int:
        """Pre-load all 48 patterns into cache.

        Returns:
            int: Number of patterns successfully loaded

        Note:
            This is recommended for production use to avoid loading delays.
            Takes ~1 second for all 48 patterns.
        """
        pattern_ids = range(8)
        lengths = [64, 128, 256, 512, 1024, 2048]

        loaded = 0
        for pattern_id in pattern_ids:
            for length in lengths:
                try:
                    self.load_pattern(pattern_id, length)
                    loaded += 1
                except FileNotFoundError:
                    # Pattern file doesn't exist yet, skip
                    continue

        return loaded

    def clear_cache(self):
        """Clear pattern cache to free memory."""
        self._cache.clear()
        self._loaded_count = 0

    def get_cache_info(self) -> Dict:
        """Get cache statistics.

        Returns:
            dict: Cache info with keys: 'cached_count', 'loaded_count', 'memory_mb'
        """
        memory_bytes = sum(pattern.nbytes for pattern in self._cache.values())
        memory_mb = memory_bytes / (1024 ** 2)

        return {
            'cached_count': len(self._cache),
            'loaded_count': self._loaded_count,
            'memory_mb': round(memory_mb, 2)
        }
