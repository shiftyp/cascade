"""CASCADE Pattern Generation Module

Generates orthogonal 4D pattern sets (64 and 128 patterns) using:
- Zadoff-Chu sequences for base orthogonality
- Simulated annealing optimization to achieve -37.5 dB cross-correlation
- Adaptive IQ complexity (λ minimization)
"""

from .models import Pattern
from .generator import generate_pattern_set
from .correlation import compute_4d_correlation
from .validator import validate_orthogonality
from .binary_format import save_pattern_file, load_pattern_file

__all__ = [
    'Pattern',
    'generate_pattern_set',
    'compute_4d_correlation',
    'validate_orthogonality',
    'save_pattern_file',
    'load_pattern_file',
]
