"""Data models for pattern generation"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Pattern:
    """A single 4D orthogonal pattern in CASCADE

    Each pattern represents a unique signal shape in Time × Frequency × IQ space.
    Supports flip-orthogonality for adjacent-channel robustness.
    """
    pattern_id: int
    freq_sequence: np.ndarray  # 32 × uint8, tone indices 0-1 (2-FSK)
    iq_trajectory: np.ndarray  # 32 × complex64, IQ constellation points
    iq_complexity_lambda: float  # 0.0 to 0.9, IQ complexity parameter

    # Flip-orthogonality support (computed automatically)
    freq_sequence_inv: np.ndarray = field(init=False)  # Inverted frequency sequence
    flip_orthogonality_stats: dict = field(default_factory=dict)  # Flip performance metrics

    def __post_init__(self):
        """Auto-compute inverted frequency sequence for flip-orthogonality checks"""
        # For 2-FSK, invert means swapping 0↔1
        self.freq_sequence_inv = 1 - self.freq_sequence

        # Initialize flip stats (will be populated during validation)
        self.flip_orthogonality_stats = {
            'max_flip_correlation_db': None,
            'avg_flip_correlation_db': None,
            'adjacent_channel_safe': None
        }
