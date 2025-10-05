#!/usr/bin/env python3
"""Verify that CASCADE pattern optimization has proper computational complexity"""

import time
import numpy as np
import sys

def test_correlation_speed():
    """Test the speed of CASCADE correlation calculations"""
    print("Testing CASCADE correlation computation speed...")

    # Generate test patterns (32 symbols as per CASCADE)
    pattern_length = 32
    num_patterns = 128

    patterns = []
    for i in range(num_patterns):
        pattern = np.random.randint(0, 2, pattern_length, dtype=np.uint8)
        patterns.append(pattern.astype(np.float32) - 0.5)

    print(f"Generated {num_patterns} patterns of length {pattern_length}")

    # Test single pattern evaluation speed
    print("\n1. Single pattern vs all others (254 correlations):")

    start = time.time()
    pattern_test = patterns[0]
    max_corr = -100

    for j in range(1, num_patterns):
        # Normal correlation with all shifts
        xcorr = np.correlate(pattern_test, patterns[j], mode='full')
        peak = np.max(np.abs(xcorr))

        # Flip correlation
        xcorr_flip = np.correlate(pattern_test, -patterns[j], mode='full')
        peak_flip = np.max(np.abs(xcorr_flip))

    single_time = time.time() - start
    print(f"   Time: {single_time:.3f} seconds")
    print(f"   Correlations/sec: {254/single_time:.1f}")

    # Test full pattern set evaluation (8,128 pairs)
    print("\n2. Full pattern set (8,128 pattern pairs):")

    start = time.time()
    max_corr = -100

    for i in range(num_patterns):
        for j in range(i + 1, num_patterns):
            # Normal
            xcorr = np.correlate(patterns[i], patterns[j], mode='full')
            peak = np.max(np.abs(xcorr))

            # Flip
            xcorr_flip = np.correlate(patterns[i], -patterns[j], mode='full')
            peak_flip = np.max(np.abs(xcorr_flip))

    full_time = time.time() - start
    print(f"   Time: {full_time:.3f} seconds")
    print(f"   Pattern pairs/sec: {8128/full_time:.1f}")

    # Estimate iteration speed
    print("\n3. Estimated optimization speed:")

    # Each iteration: evaluate mutation, sometimes recalc full set
    iter_time = single_time  # Most iterations just eval one mutation
    full_recalc_every = 10  # Recalc full set every 10 iterations

    avg_iter_time = iter_time + (full_time / full_recalc_every)
    iter_per_sec = 1.0 / avg_iter_time

    print(f"   Per iteration: ~{avg_iter_time:.3f} seconds")
    print(f"   Iterations/sec: ~{iter_per_sec:.1f}")
    print(f"   50k iterations: ~{50000/iter_per_sec/3600:.1f} hours")
    print(f"   3.2M iterations: ~{3200000/iter_per_sec/3600:.1f} hours (~{3200000/iter_per_sec/3600/24:.1f} days)")

if __name__ == "__main__":
    test_correlation_speed()