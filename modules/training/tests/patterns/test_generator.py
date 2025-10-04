"""Contract tests for pattern generation orchestrator"""

import pytest
import sys
sys.path.insert(0, '/workspaces/cascade')

from modules.training.patterns import generate_pattern_set, Pattern


def test_generate_64_pattern_set():
    """T005: Contract test for 64-pattern generation"""
    patterns = generate_pattern_set(count=64, seed=42)

    # Verify returns list of Pattern objects
    assert isinstance(patterns, list)
    assert len(patterns) > 0
    assert all(isinstance(p, Pattern) for p in patterns)

    # Verify count matches request (64)
    assert len(patterns) == 64

    # Verify beacon count = 48
    beacon_patterns = [p for p in patterns if p.pattern_id < 48]
    assert len(beacon_patterns) == 48

    # Verify message count = 16 for 64-set
    message_patterns = [p for p in patterns if p.pattern_id >= 48]
    assert len(message_patterns) == 16

    # Verify all pattern IDs are unique and in range
    pattern_ids = [p.pattern_id for p in patterns]
    assert len(set(pattern_ids)) == 64
    assert all(0 <= pid < 64 for pid in pattern_ids)


def test_generate_128_pattern_set():
    """T005: Contract test for 128-pattern generation"""
    patterns = generate_pattern_set(count=128, seed=42)

    # Verify returns list of Pattern objects
    assert isinstance(patterns, list)
    assert len(patterns) == 128
    assert all(isinstance(p, Pattern) for p in patterns)

    # Verify beacon count = 48 for both sets
    beacon_patterns = [p for p in patterns if p.pattern_id < 48]
    assert len(beacon_patterns) == 48

    # Verify message count = 80 for 128-set
    message_patterns = [p for p in patterns if p.pattern_id >= 48]
    assert len(message_patterns) == 80

    # Verify all pattern IDs are unique and in range
    pattern_ids = [p.pattern_id for p in patterns]
    assert len(set(pattern_ids)) == 128
    assert all(0 <= pid < 128 for pid in pattern_ids)


def test_pattern_structure():
    """T005: Verify Pattern object has required fields"""
    patterns = generate_pattern_set(count=64, seed=42)

    for pattern in patterns:
        # Verify freq_sequence is 32 × uint8
        assert pattern.freq_sequence.shape == (32,)
        assert pattern.freq_sequence.dtype == 'uint8'
        assert all(0 <= tone <= 3 for tone in pattern.freq_sequence)

        # Verify iq_trajectory is 32 × complex64
        assert pattern.iq_trajectory.shape == (32,)
        assert pattern.iq_trajectory.dtype == 'complex64'

        # Verify iq_complexity_lambda in valid range
        assert 0.0 <= pattern.iq_complexity_lambda <= 0.9


def test_deterministic_with_seed():
    """T005: Verify same seed produces same patterns"""
    patterns1 = generate_pattern_set(count=64, seed=42)
    patterns2 = generate_pattern_set(count=64, seed=42)

    assert len(patterns1) == len(patterns2)
    for p1, p2 in zip(patterns1, patterns2):
        assert p1.pattern_id == p2.pattern_id
        assert (p1.freq_sequence == p2.freq_sequence).all()
        assert (p1.iq_trajectory == p2.iq_trajectory).all()
        assert p1.iq_complexity_lambda == p2.iq_complexity_lambda
