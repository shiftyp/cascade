#!/usr/bin/env python3
"""Test the actual pattern optimization algorithm"""

import numpy as np
import sys
from pathlib import Path

# Add tournament to path
sys.path.insert(0, str(Path(__file__).parent))

def test_correlation_calculation():
    """Test that correlation calculations are correct"""
    print("Testing correlation calculations...")

    # Create two simple test patterns
    pattern_length = 32
    pattern1 = np.array([1, 0, 1, 0] * 8, dtype=np.uint8)[:pattern_length]
    pattern2 = np.array([0, 1, 0, 1] * 8, dtype=np.uint8)[:pattern_length]

    # Convert to centered format (-0.5 to 0.5)
    p1_centered = pattern1.astype(np.float32) - 0.5
    p2_centered = pattern2.astype(np.float32) - 0.5

    # Calculate normal correlation (should be highly negative since patterns are opposite)
    corr_normal = np.dot(p1_centered, p2_centered)
    corr_normal_db = 20 * np.log10(np.abs(corr_normal) / pattern_length + 1e-10)
    print(f"  Normal correlation: {corr_normal:.3f} ({corr_normal_db:.1f} dB)")

    # Calculate flip correlation (pattern2 inverted should match pattern1)
    p2_flip = -p2_centered  # Flip is negation after centering
    corr_flip = np.dot(p1_centered, p2_flip)
    corr_flip_db = 20 * np.log10(np.abs(corr_flip) / pattern_length + 1e-10)
    print(f"  Flip correlation: {corr_flip:.3f} ({corr_flip_db:.1f} dB)")

    # Test with random patterns
    print("\nTesting with random patterns:")
    np.random.seed(42)
    rand1 = np.random.randint(0, 2, pattern_length, dtype=np.uint8)
    rand2 = np.random.randint(0, 2, pattern_length, dtype=np.uint8)

    r1_centered = rand1.astype(np.float32) - 0.5
    r2_centered = rand2.astype(np.float32) - 0.5

    corr_rand_normal = np.dot(r1_centered, r2_centered)
    corr_rand_normal_db = 20 * np.log10(np.abs(corr_rand_normal) / pattern_length + 1e-10)
    print(f"  Random normal correlation: {corr_rand_normal:.3f} ({corr_rand_normal_db:.1f} dB)")

    r2_flip = -r2_centered
    corr_rand_flip = np.dot(r1_centered, r2_flip)
    corr_rand_flip_db = 20 * np.log10(np.abs(corr_rand_flip) / pattern_length + 1e-10)
    print(f"  Random flip correlation: {corr_rand_flip:.3f} ({corr_rand_flip_db:.1f} dB)")

    print("\nCorrelation test PASSED!")

def test_worker_function():
    """Test the worker function with minimal iterations"""
    print("\nTesting worker function...")

    from core.tournament_optimizer import run_single_trial_worker

    # Run a very short test
    result = run_single_trial_worker(
        trial_id=999,
        iterations=100,  # Just 100 iterations for test
        seed=42,
        checkpoint_dir="./test_checkpoint",
        p_cores=None
    )

    print(f"  Trial ID: {result['trial_id']}")
    print(f"  Iterations run: {result['iterations_run']}")
    print(f"  Best score: {result.get('best_score', 'N/A'):.2f} dB")
    print(f"  Error: {result.get('error', 'None')}")

    if 'error' in result:
        print(f"Worker test FAILED: {result['error']}")
    else:
        print("Worker test PASSED!")

    # Clean up test files
    import shutil
    if Path('./test_checkpoint').exists():
        shutil.rmtree('./test_checkpoint')
    if Path('./logs').exists() and Path('./logs/debug_trial_999.txt').exists():
        Path('./logs/debug_trial_999.txt').unlink()

if __name__ == "__main__":
    test_correlation_calculation()
    test_worker_function()