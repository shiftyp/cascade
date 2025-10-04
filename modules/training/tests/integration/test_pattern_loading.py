"""Integration Test: Pattern Loading and Usage

Tests that generated patterns can be loaded and used correctly.
"""

import pytest
import sys
import tempfile
import os
import numpy as np

sys.path.insert(0, '/workspaces/cascade')

from modules.training.patterns import Pattern, save_pattern_file, load_pattern_file


def test_pattern_loading_from_file():
    """T023: Test loading patterns from binary file"""
    # Create test patterns
    patterns = []
    for i in range(16):
        freq = np.random.randint(0, 4, size=32, dtype='uint8')
        iq = (np.random.randn(32) + 1j * np.random.randn(32)).astype('complex64')
        # Normalize
        iq = iq / np.sqrt(np.mean(np.abs(iq)**2))

        patterns.append(Pattern(
            pattern_id=i,
            freq_sequence=freq,
            iq_trajectory=iq,
            iq_complexity_lambda=i * 0.05
        ))

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        temp_file = f.name

    try:
        save_pattern_file(patterns, temp_file)

        # Load back
        loaded = load_pattern_file(temp_file)

        # Verify patterns accessible by ID
        assert len(loaded) == 16
        for i in range(16):
            assert loaded[i].pattern_id == i

        # Verify data integrity
        for orig, load in zip(patterns, loaded):
            assert (orig.freq_sequence == load.freq_sequence).all()
            assert np.allclose(orig.iq_trajectory, load.iq_trajectory)
            assert abs(orig.iq_complexity_lambda - load.iq_complexity_lambda) < 1e-6

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_pattern_selection_by_lambda():
    """T023: Test selecting patterns by complexity pool"""
    # Create patterns with different lambda values
    patterns = []
    for i in range(20):
        freq = np.random.randint(0, 4, size=32, dtype='uint8')
        iq = np.ones(32, dtype='complex64')
        # Distribute lambdas: 0.0, 0.1, 0.2, ..., 0.9, then repeat
        lam = (i % 10) * 0.1

        patterns.append(Pattern(
            pattern_id=i,
            freq_sequence=freq,
            iq_trajectory=iq,
            iq_complexity_lambda=lam
        ))

    # Select emergency patterns (λ < 0.15)
    emergency = [p for p in patterns if p.iq_complexity_lambda < 0.15]
    assert len(emergency) == 4  # λ=0.0, 0.1 (twice each)

    # Select simple patterns (λ < 0.3)
    simple = [p for p in patterns if p.iq_complexity_lambda < 0.3]
    assert len(simple) == 6  # λ=0.0, 0.1, 0.2 (twice each)

    # Select complex patterns (λ >= 0.5)
    complex_patterns = [p for p in patterns if p.iq_complexity_lambda >= 0.5]
    assert len(complex_patterns) == 10  # λ=0.5-0.9 (twice each)


def test_pattern_usage_in_encoding():
    """T023: Verify patterns can be used for signal encoding (stub)"""
    # This would test using patterns for actual signal generation
    # Placeholder for future modem module integration
    pass
