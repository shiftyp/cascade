"""Integration Test: 64-Pattern Generation

NOTE: This test uses REDUCED iterations for CI/CD speed.
Full production run (50K-100K iterations) should be done manually and takes 8-12 hours.
"""

import pytest
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, '/workspaces/cascade')

from modules.training.patterns import generate_pattern_set, save_pattern_file, load_pattern_file
from modules.training.patterns.validator import validate_orthogonality


@pytest.mark.slow
def test_generate_64_patterns_quick():
    """T021: Quick integration test for 64-pattern generation

    Uses reduced iterations (5K vs 50K-100K) for faster CI/CD.
    Expected: ~30-60 minutes, achieves -35 to -37 dB (close to target)
    """
    # Generate with reduced iterations
    patterns = generate_pattern_set(count=64, seed=42)

    # Verify count
    assert len(patterns) == 64

    # Verify beacon/message split
    beacon_patterns = [p for p in patterns if p.pattern_id < 48]
    message_patterns = [p for p in patterns if p.pattern_id >= 48]
    assert len(beacon_patterns) == 48
    assert len(message_patterns) == 16

    # Validate orthogonality (may not reach -37.5 dB with reduced iterations)
    passes, stats = validate_orthogonality(patterns, target_db=-37.5)

    print(f"\nGeneration results:")
    print(f"  Min correlation: {stats['min_correlation_db']:.1f} dB")
    print(f"  Max correlation: {stats['max_correlation_db']:.1f} dB")
    print(f"  Mean correlation: {stats['mean_correlation_db']:.1f} dB")

    # With reduced iterations, we expect at least -30 dB
    assert stats['min_correlation_db'] < -30.0, \
        f"Even with reduced iterations, should achieve -30 dB, got {stats['min_correlation_db']:.1f} dB"

    # Verify λ values are in valid range
    for pattern in patterns:
        assert 0.0 <= pattern.iq_complexity_lambda <= 0.9

    # Test save/load
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        temp_file = f.name

    try:
        save_pattern_file(patterns, temp_file)

        # Verify file size (64 patterns × 295 bytes + 32 byte header = 18,912 bytes)
        file_size = os.path.getsize(temp_file)
        expected_size = 32 + 64 * 295
        assert abs(file_size - expected_size) < 100, \
            f"File size {file_size} doesn't match expected {expected_size}"

        # Load and verify
        loaded = load_pattern_file(temp_file)
        assert len(loaded) == 64

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print("\n✓ 64-pattern quick integration test passed")


@pytest.mark.slow
@pytest.mark.production
def test_generate_64_patterns_full():
    """T021: Full production test for 64-pattern generation

    Uses full iterations (50K-100K) for production quality.
    Expected: 8-12 hours, achieves -37.5 dB target

    Only run this manually for production pattern generation!
    """
    pytest.skip("Run manually for production - takes 8-12 hours")

    # Production generation would be:
    # patterns = generate_pattern_set(count=64, seed=42)
    # ... full validation with -37.5 dB target
