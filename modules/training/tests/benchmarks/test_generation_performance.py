"""Performance Benchmarks: Pattern Generation

NOTE: These benchmarks test performance targets from NFR requirements.
They use reduced iterations for automated testing.
"""

import pytest
import sys
import time
import psutil
import os

sys.path.insert(0, '/workspaces/cascade')

from modules.training.patterns import generate_pattern_set
from modules.training.patterns.validator import validate_orthogonality


@pytest.mark.benchmark
def test_single_pattern_generation_time():
    """T026: Measure time for single pattern generation"""
    from modules.training.patterns.zadoff_chu import generate_zadoff_chu_pattern
    from modules.training.patterns.optimizer import optimize_pattern_direct_iq
    from modules.training.patterns.models import Pattern

    start = time.time()

    base = generate_zadoff_chu_pattern(u=0)
    freq, iq, lam = optimize_pattern_direct_iq(
        pattern_id=0,
        base_freq_sequence=base,
        existing_patterns=[],
        max_iterations=5000,
        seed=42
    )

    elapsed = time.time() - start

    print(f"\nSingle pattern (5K iterations): {elapsed:.2f} seconds")
    assert elapsed < 120, f"Single pattern should generate in <2 minutes, took {elapsed:.2f}s"


@pytest.mark.benchmark
def test_64_pattern_generation_time_quick():
    """T026: Measure time for 64-pattern generation (reduced iterations)

    Full production (50K-100K iterations) would take 8-12 hours.
    This test uses reduced iterations for CI/CD.
    """
    pytest.skip("Skip by default - takes 30-60 minutes even with reduced iterations")

    # Would measure:
    # start = time.time()
    # patterns = generate_pattern_set(count=64, seed=42)  # Uses default reduced iterations
    # elapsed = (time.time() - start) / 3600  # hours
    # assert elapsed < 1.0, "Quick test should complete in <1 hour"


@pytest.mark.benchmark
@pytest.mark.production
def test_128_pattern_generation_time_full():
    """T026: Measure time for full 128-pattern generation

    NFR-001: Should complete within 18-24 hours on 8-core CPU
    """
    pytest.skip("Production benchmark - run manually, takes 18-24 hours")


@pytest.mark.benchmark
def test_memory_usage():
    """T026: Measure memory usage during generation

    NFR-005: Must stay under 500 MB per trial
    """
    process = psutil.Process(os.getpid())
    baseline_mb = process.memory_info().rss / (1024**2)

    # Generate small set
    from modules.training.patterns.zadoff_chu import generate_zadoff_chu_pattern
    from modules.training.patterns.optimizer import optimize_pattern_direct_iq
    from modules.training.patterns.models import Pattern

    patterns = []
    for i in range(8):
        base = generate_zadoff_chu_pattern(u=i)
        freq, iq, lam = optimize_pattern_direct_iq(
            pattern_id=i,
            base_freq_sequence=base,
            existing_patterns=patterns,
            max_iterations=1000,
            seed=42 + i
        )
        patterns.append(Pattern(i, freq, iq, lam))

    peak_mb = process.memory_info().rss / (1024**2)
    delta_mb = peak_mb - baseline_mb

    print(f"\nMemory usage for 8 patterns: {delta_mb:.1f} MB")
    assert delta_mb < 500, f"Memory usage {delta_mb:.1f} MB exceeds 500 MB limit"


@pytest.mark.benchmark
def test_validation_time():
    """T026: Measure validation time

    NFR-004: Must complete within 5 minutes
    """
    # Create test patterns
    import numpy as np
    from modules.training.patterns.models import Pattern

    patterns = []
    for i in range(64):
        freq = np.random.randint(0, 4, size=32, dtype='uint8')
        iq = np.ones(32, dtype='complex64')
        patterns.append(Pattern(i, freq, iq, 0.0))

    start = time.time()
    passes, stats = validate_orthogonality(patterns, target_db=-37.5)
    elapsed = time.time() - start

    print(f"\nValidation time for 64 patterns: {elapsed:.2f} seconds")
    assert elapsed < 300, f"Validation took {elapsed:.2f}s, should be <5 minutes (300s)"


@pytest.mark.benchmark
def test_visualization_generation_time():
    """T026: Measure visualization generation time

    NFR-006: Must complete within 30 seconds per batch
    """
    import numpy as np
    from modules.training.patterns.models import Pattern
    from modules.training.patterns.visualization import generate_batch_report

    # Create test patterns
    patterns = []
    for i in range(16):
        freq = np.random.randint(0, 4, size=32, dtype='uint8')
        iq = (np.random.randn(32) + 1j * np.random.randn(32)).astype('complex64')
        iq = iq / np.sqrt(np.mean(np.abs(iq)**2))
        patterns.append(Pattern(i, freq, iq, i * 0.05))

    start = time.time()
    generate_batch_report(patterns, batch_num=999, output_dir='/tmp/viz_test')
    elapsed = time.time() - start

    print(f"\nVisualization generation time: {elapsed:.2f} seconds")
    assert elapsed < 30, f"Visualization took {elapsed:.2f}s, should be <30s"
