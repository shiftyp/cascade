"""Pattern loader for CASCADE V2 signal generator.

Loads pre-generated patterns from .pkl files produced by the genetic algorithm.
Supports both standard patterns and always-on center frequency design patterns.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np


class PatternLoader:
    """Loads and caches CASCADE V2 patterns.

    Supports two pattern formats:
    1. Standard patterns from final_patterns_319968.pkl
    2. Always-on patterns from always_on_patterns_*.pkl

    Always-on pattern structure:
    - pattern_id: 0-3 (4 patterns)
    - Each pattern has 3 variants:
      - 'center': Continuous transmission (all symbols)
      - 'lower': Even symbols only (for lower outer frequency)
      - 'upper': Odd symbols only (for upper outer frequency)

    Standard pattern structure:
    - pattern_id: 0-3 (4 patterns, 3-FSK with 3 tones each)
    - Master length: 1024 symbols
    - Format: nested_patterns[1024]['cores'][pattern_id][repetition_map]
    """

    MASTER_LENGTH = 512  # Full pattern length (from final_patterns_319968.pkl)
    MAX_LOOPED_LENGTH = 4096  # Maximum length with looping

    def __init__(self, patterns_dir: Optional[Path] = None, use_always_on: bool = False):
        """Initialize pattern loader.

        Args:
            patterns_dir: Directory containing pattern files.
                         Defaults to modules/training/patterns/
            use_always_on: If True, load always-on patterns instead of standard
        """
        if patterns_dir is None:
            # Default to patterns directory (not tournament subdirectory)
            module_dir = Path(__file__).parent.parent.parent
            patterns_dir = module_dir / "patterns"

        self.patterns_dir = Path(patterns_dir)
        self.use_always_on = use_always_on
        self._cache: Dict[Tuple[int, int, Optional[str]], np.ndarray] = {}
        self._master_patterns: Dict[int, np.ndarray] = {}  # Cache full-length patterns
        self._always_on_patterns: Dict[int, Dict[str, np.ndarray]] = {}  # Cache always-on patterns
        self._final_patterns_data = None  # Lazy load
        self._always_on_data = None  # Lazy load for always-on patterns

    def _load_always_on_patterns(self):
        """Lazy load always-on patterns from always_on/ subdirectory."""
        if self._always_on_data is not None:
            return

        # Look for patterns in always_on subdirectory
        always_on_dir = self.patterns_dir / "always_on"

        if not always_on_dir.exists():
            raise FileNotFoundError(
                f"Always-on patterns directory not found: {always_on_dir}\n"
                f"Run process_tournament_for_always_on.py to generate patterns."
            )

        # Find the most recent always_on_patterns_*.pkl file
        pattern_files = list(always_on_dir.glob("always_on_patterns_*.pkl"))

        if not pattern_files:
            raise FileNotFoundError(
                f"No always-on pattern files found in: {always_on_dir}\n"
                f"Run process_tournament_for_always_on.py to generate patterns."
            )

        # Use the most recent file
        latest_file = max(pattern_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, 'rb') as f:
            self._always_on_data = pickle.load(f)

        # Extract patterns into cache
        if 'patterns' in self._always_on_data:
            for pattern_id, pattern_set in self._always_on_data['patterns'].items():
                self._always_on_patterns[pattern_id] = pattern_set

    def _load_final_patterns(self):
        """Lazy load final_patterns_319968.pkl file."""
        if self._final_patterns_data is not None:
            return

        final_patterns_file = self.patterns_dir / "final_patterns_319968.pkl"

        # If in always-on subdir (patterns/patterns), check parent directory
        if not final_patterns_file.exists() and self.patterns_dir.name == "patterns":
            parent_dir = self.patterns_dir.parent
            final_patterns_file = parent_dir / "final_patterns_319968.pkl"

        if not final_patterns_file.exists():
            raise FileNotFoundError(
                f"Final patterns file not found: {final_patterns_file}\n"
                f"Expected location: {self.patterns_dir}\n"
                f"This file should contain the optimized patterns from the genetic algorithm."
            )

        with open(final_patterns_file, 'rb') as f:
            self._final_patterns_data = pickle.load(f)

        # Patterns loaded silently for cleaner output during batch processing

    def _load_master_pattern(self, pattern_id: int) -> np.ndarray:
        """Load full-length pattern from file.

        Args:
            pattern_id: Pattern ID (0-3)

        Returns:
            np.ndarray: Full pattern, shape (MASTER_LENGTH,), dtype uint8
        """
        if pattern_id in self._master_patterns:
            return self._master_patterns[pattern_id]

        # Load final patterns file if not already loaded
        self._load_final_patterns()

        # Extract master pattern
        try:
            nested_patterns = self._final_patterns_data['nested_patterns']

            # Use the longest available pattern length
            available_lengths = list(nested_patterns.keys())
            if self.MASTER_LENGTH in available_lengths:
                pattern_length = self.MASTER_LENGTH
            else:
                # Use the longest available
                pattern_length = max(available_lengths)

            patterns_dict = nested_patterns[pattern_length]

            # For final_patterns_319968.pkl, cores are already complete patterns
            # No need to apply repetition_map
            pattern_symbols = np.array(patterns_dict['cores'][pattern_id], dtype=np.uint8)

        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Could not extract pattern_id={pattern_id} from final patterns file: {e}"
            )

        # Validate pattern format
        pattern_symbols = np.asarray(pattern_symbols, dtype=np.uint8)

        # Pattern length should match what we loaded
        if pattern_symbols.shape[0] != pattern_length:
            raise ValueError(
                f"Pattern has length {pattern_symbols.shape[0]}, expected {pattern_length}"
            )

        # Validate ternary symbols for 3-FSK
        if not np.all((pattern_symbols >= 0) & (pattern_symbols <= 2)):
            raise ValueError(
                f"Pattern contains values outside 0-2 range (3-FSK ternary)"
            )

        # Cache master pattern
        self._master_patterns[pattern_id] = pattern_symbols
        return pattern_symbols

    def load_always_on_pattern(self, pattern_id: int, pattern_type: str, length: int) -> np.ndarray:
        """Load always-on pattern with specified type and length.

        Args:
            pattern_id: Pattern ID (0-3)
            pattern_type: Pattern type ('center', 'lower', 'upper')
            length: Pattern length in symbols

        Returns:
            np.ndarray: Pattern symbols with -1 for inactive slots

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not (0 <= pattern_id <= 3):
            raise ValueError(f"pattern_id must be 0-3, got {pattern_id}")

        if pattern_type not in ['center', 'lower', 'upper']:
            raise ValueError(f"pattern_type must be 'center', 'lower', or 'upper', got {pattern_type}")

        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")

        # Load always-on patterns if not already loaded
        if not self._always_on_patterns:
            self._load_always_on_patterns()

        # Get the pattern set for this ID
        if pattern_id not in self._always_on_patterns:
            raise ValueError(f"Pattern ID {pattern_id} not found in always-on patterns")

        pattern_set = self._always_on_patterns[pattern_id]

        # Get the specific pattern type
        if pattern_type not in pattern_set:
            raise ValueError(f"Pattern type '{pattern_type}' not found for pattern ID {pattern_id}")

        master_pattern = pattern_set[pattern_type]

        # Handle length adjustment
        if length <= self.MASTER_LENGTH:
            # Partial pattern: first N symbols
            pattern_symbols = master_pattern[:length].copy()
        else:
            # Looped pattern: tile master pattern to reach desired length
            num_tiles = (length + self.MASTER_LENGTH - 1) // self.MASTER_LENGTH
            pattern_symbols = np.tile(master_pattern, num_tiles)[:length]

        return pattern_symbols

    def load_pattern(self, pattern_id: int, length: int, pattern_type: Optional[str] = None) -> np.ndarray:
        """Load pattern with requested length (supports partial patterns and looping).

        For length <= 1024: Returns first N symbols of master pattern
        For length > 1024: Returns looped pattern (repeats master pattern)

        Args:
            pattern_id: Pattern ID (0-3, 4 patterns total)
            length: Pattern length in symbols (any positive integer up to 4096)
            pattern_type: For always-on mode: 'center', 'lower', or 'upper'

        Returns:
            np.ndarray: Pattern symbols, shape (length,), dtype uint8, values {0, 1, 2} for 3-FSK
                       For always-on patterns, -1 indicates inactive slots

        Raises:
            ValueError: If pattern_id or length is invalid
            FileNotFoundError: If patterns file doesn't exist
        """
        # If always-on mode and pattern_type specified, use always-on loader
        if self.use_always_on and pattern_type:
            return self.load_always_on_pattern(pattern_id, pattern_type, length)
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
        cache_key = (pattern_id, length, None)  # None for standard patterns
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
            except (FileNotFoundError, ValueError):
                # Silently skip patterns that can't be loaded
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
