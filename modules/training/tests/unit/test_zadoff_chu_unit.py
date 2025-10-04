"""Unit Tests: Zadoff-Chu Sequence Generation"""

import pytest
import sys
import numpy as np

sys.path.insert(0, '/workspaces/cascade')

from modules.training.patterns.zadoff_chu import generate_zadoff_chu_pattern, generate_random_pattern


def test_zadoff_chu_sequence_length():
    """T024: Test sequence length is 32"""
    for u in range(5):
        seq = generate_zadoff_chu_pattern(u=u)
        assert seq.shape == (32,), f"Expected shape (32,), got {seq.shape}"
        assert seq.dtype == np.dtype('uint8')


def test_zadoff_chu_tone_indices_valid():
    """T024: Test tone indices are in [0, 3]"""
    for u in range(10):
        seq = generate_zadoff_chu_pattern(u=u)
        assert np.all(seq >= 0), "Found tone index < 0"
        assert np.all(seq <= 3), "Found tone index > 3"


def test_zadoff_chu_deterministic():
    """T024: Test sequences are deterministic"""
    seq1 = generate_zadoff_chu_pattern(u=5)
    seq2 = generate_zadoff_chu_pattern(u=5)
    assert (seq1 == seq2).all(), "Same u should produce same sequence"


def test_zadoff_chu_different_u_produces_different_sequences():
    """T024: Test u=0 to u=30 produce different sequences"""
    sequences = []
    for u in range(31):
        seq = generate_zadoff_chu_pattern(u=u)
        sequences.append(seq)

    # Check that all sequences are unique
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            # At least some elements should differ
            assert not (sequences[i] == sequences[j]).all(), \
                f"u={i} and u={j} produced identical sequences"


def test_random_pattern_deterministic_with_seed():
    """T024: Test random patterns are deterministic with seed"""
    seq1 = generate_random_pattern(seed=42)
    seq2 = generate_random_pattern(seed=42)
    assert (seq1 == seq2).all(), "Same seed should produce same pattern"


def test_random_pattern_different_seeds():
    """T024: Test different seeds produce different patterns"""
    seq1 = generate_random_pattern(seed=42)
    seq2 = generate_random_pattern(seed=43)
    # Should be different (very unlikely to match)
    assert not (seq1 == seq2).all(), "Different seeds should produce different patterns"


def test_random_pattern_valid_tones():
    """T024: Test random patterns have valid tone indices"""
    for seed in range(10):
        seq = generate_random_pattern(seed=seed)
        assert seq.shape == (32,)
        assert seq.dtype == np.dtype('uint8')
        assert np.all(seq >= 0)
        assert np.all(seq <= 3)
