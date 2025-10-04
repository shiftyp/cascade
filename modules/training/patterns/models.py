"""Data models for pattern generation"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Pattern:
    """A single 4D orthogonal pattern in CASCADE

    Each pattern represents a unique signal shape in Time × Frequency × IQ space.
    """
    pattern_id: int
    freq_sequence: np.ndarray  # 32 × uint8, tone indices 0-3
    iq_trajectory: np.ndarray  # 32 × complex64, IQ constellation points
    iq_complexity_lambda: float  # 0.0 to 0.9, IQ complexity parameter
