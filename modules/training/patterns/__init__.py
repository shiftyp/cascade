"""CASCADE Pattern Generation Module

Generates orthogonal binary pattern sets (16 patterns, 512 symbols each) with:
- QR-like erasure coding (37.5% tolerance)
- Triple orthogonality: normal, flip, and erasure
- Repetition mapping for data redundancy
- Tournament-style optimization for -30 dB orthogonality

Use tournament subdirectory for pattern generation:
    python tournament/generate_patterns_tournament.py
"""

__all__ = []
