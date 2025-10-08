"""Pattern loader for CASCADE V2 signal generator.

Loads pre-generated patterns from .pkl files produced by the genetic algorithm.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np


class PatternLoader:
    """Loads and caches CASCADE V2 patterns from final_patterns_399968.pkl.

    Patterns are generated offline by genetic algorithm. The file contains 4 full-length
    patterns (1024 symbols each). Patterns support both partial extraction and looping.

    Pattern structure in final_patterns_399968.pkl:
    - pattern_id: 0-3 (4 patterns, 3-FSK with 3 tones each)
    - Master length: 1024 symbols
    - Partial patterns (< 1024): Extract first N symbols
    - Looped patterns (> 1024): Repeat pattern (up to 4096 symbols)
    - Format: nested_patterns[1024]['cores'][pattern_id][repetition_map]

    Total: 4 full-length patterns, supporting partial extraction and looping
    """

    MASTER_LENGTH = 1024  # Full pattern length
    MAX_LOOPED_LENGTH = 4096  # Maximum length with looping (4× repetition)

    def __init__(self, patterns_dir: Optional[Path] = None):
        """Initialize pattern loader.

        Args:
            patterns_dir: Directory containing final_patterns_399968.pkl.
                         Defaults to modules/training/patterns/
        """
        if patterns_dir is None:
            # Default to patterns directory (not tournament subdirectory)
            module_dir = Path(__file__).parent.parent.parent
            patterns_dir = module_dir / "patterns"

        self.patterns_dir = Path(patterns_dir)
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._master_patterns: Dict[int, np.ndarray] = {}  # Cache full-length patterns
        self._final_patterns_data = None  # Lazy load

    def _load_final_patterns(self):
        """Lazy load final_patterns_399968.pkl file."""
        if self._final_patterns_data is not None:
            return

        final_patterns_file = self.patterns_dir / "final_patterns_399968.pkl"

        if not final_patterns_file.exists():
            raise FileNotFoundError(
                f"Final patterns file not found: {final_patterns_file}\n"
                f"Expected location: {self.patterns_dir}\n"
                f"This file should contain the optimized patterns from the genetic algorithm."
            )

        with open(final_patterns_file, 'rb') as f:
            self._final_patterns_data = pickle.load(f)

        print(f"Loaded final patterns: {self._final_patterns_data.get('algorithm', 'unknown')}, "
              f"{self._final_patterns_data.get('best_score', 0):.2f} dB")

    def _load_master_pattern(self, pattern_id: int) -> np.ndarray:
        """Load full-length (1024) pattern from file.

        Args:
            pattern_id: Pattern ID (0-3)

        Returns:
            np.ndarray: Full pattern, shape (1024,), dtype uint8
        """
        if pattern_id in self._master_patterns:
            return self._master_patterns[pattern_id]

        # Load final patterns file if not already loaded
        self._load_final_patterns()

        # Extract master pattern (length 1024)
        try:
            nested_patterns = self._final_patterns_data['nested_patterns']
            patterns_dict = nested_patterns[self.MASTER_LENGTH]
            pattern_core = patterns_dict['cores'][pattern_id]
            repetition_map = patterns_dict['repetition_map']

            # Expand pattern using repetition map
            pattern_symbols = pattern_core[repetition_map]

        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Could not extract pattern_id={pattern_id} from final patterns file: {e}"
            )

        # Validate pattern format
        pattern_symbols = np.asarray(pattern_symbols, dtype=np.uint8)

        if pattern_symbols.shape[0] != self.MASTER_LENGTH:
            raise ValueError(
                f"Pattern has length {pattern_symbols.shape[0]}, expected {self.MASTER_LENGTH}"
            )

        # Validate ternary symbols for 3-FSK
        if not np.all((pattern_symbols >= 0) & (pattern_symbols <= 2)):
            raise ValueError(
                f"Pattern contains values outside 0-2 range (3-FSK ternary)"
            )

        # Cache master pattern
        self._master_patterns[pattern_id] = pattern_symbols
        return pattern_symbols

    def load_pattern(self, pattern_id: int, length: int) -> np.ndarray:
        """Load pattern with requested length (supports partial patterns and looping).

        For length <= 1024: Returns first N symbols of master pattern
        For length > 1024: Returns looped pattern (repeats master pattern)

        Args:
            pattern_id: Pattern ID (0-3, 4 patterns total)
            length: Pattern length in symbols (any positive integer up to 4096)

        Returns:
            np.ndarray: Pattern symbols, shape (length,), dtype uint8, values {0, 1, 2} for 3-FSK

        Raises:
            ValueError: If pattern_id or length is invalid
            FileNotFoundError: If final patterns file doesn't exist
        """
        # Validate inputs
        if not (0 <= pattern_id <= 3):
            raise ValueError(f"pattern_id must be 0-3 (4 patterns), got {pattern_id}")

        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")

        if length > self.MAX_LOOPED_LENGTH:
            raise ValueError(
                f"length cannot exceed {self.MAX_LOOPED_LENGTH} (max with looping), got {length}"
            )

        # Check cache first
        cache_key = (pattern_id, length)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load master pattern (1024 symbols)
        master_pattern = self._load_master_pattern(pattern_id)

        # Extract partial or loop pattern
        if length <= self.MASTER_LENGTH:
            # Partial pattern: first N symbols
            pattern_symbols = master_pattern[:length].copy()
        else:
            # Looped pattern: tile master pattern to reach desired length
            num_tiles = (length + self.MASTER_LENGTH - 1) // self.MASTER_LENGTH
            pattern_symbols = np.tile(master_pattern, num_tiles)[:length]

        # Cache and return
        self._cache[cache_key] = pattern_symbols

        return pattern_symbols

    def load_all_patterns(self) -> int:
        """Pre-load all 4 master patterns from final_patterns_399968.pkl.

        Returns:
            int: Number of patterns successfully loaded (4)

        Note:
            This loads the 4 full-length (1024) patterns into cache.
            Partial patterns are extracted on-demand from these masters.
        """
        pattern_ids = range(4)  # 0-3 (4 patterns)

        loaded = 0
        for pattern_id in pattern_ids:
            try:
                self._load_master_pattern(pattern_id)
                loaded += 1
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load pattern {pattern_id}: {e}")
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
