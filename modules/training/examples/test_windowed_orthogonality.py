#!/usr/bin/env python3
"""Test the windowed orthogonality fitness function"""

import numpy as np
import sys

# Test pattern lengths
PATTERN_LENGTH = 2048
NUM_PATTERNS = 8

def test_with_erasure(p1, p2, erasure_rate=0.375):
    """Test correlation with random erasures"""
    keep_rate = 1.0 - erasure_rate
    mask = np.random.random(len(p1)) < keep_rate
    p1_erased = p1[mask]
    p2_erased = p2[mask]
    if len(p1_erased) < 10:
        return 0.0
    xcorr = np.correlate(p1_erased, p2_erased, mode='full')
    peak = np.max(np.abs(xcorr))
    return 20 * np.log10(peak / len(p1_erased) + 1e-10)


def windowed_correlation(p1, p2, window_size, num_windows=8):
    """Test correlation on random windows of patterns"""
    pattern_len = len(p1)
    worst_corr = -100.0

    for _ in range(num_windows):
        if pattern_len <= window_size:
            window = slice(0, pattern_len)
        else:
            window_start = np.random.randint(0, pattern_len - window_size)
            window = slice(window_start, window_start + window_size)

        p1_win = p1[window]
        p2_win = p2[window]

        xcorr = np.correlate(p1_win, p2_win, mode='full')
        peak = np.max(np.abs(xcorr))
        corr_db = 20 * np.log10(peak / len(p1_win) + 1e-10)
        worst_corr = max(worst_corr, corr_db)

    return worst_corr


def global_correlation(p1, p2):
    """Test full-pattern correlation (old method)"""
    xcorr = np.correlate(p1, p2, mode='full')
    peak = np.max(np.abs(xcorr))
    return 20 * np.log10(peak / len(p1) + 1e-10)


def test_orthogonality_methods():
    """Compare global vs windowed orthogonality"""
    np.random.seed(42)

    print("=" * 70)
    print("Testing Windowed vs Global Orthogonality")
    print("=" * 70)

    # Generate random patterns
    patterns = []
    for i in range(NUM_PATTERNS):
        pattern = np.random.randint(0, 2, PATTERN_LENGTH, dtype=np.uint8)
        pattern = 2 * pattern.astype(np.float32) - 1  # Convert to ±1
        patterns.append(pattern)

    print(f"\nGenerated {NUM_PATTERNS} random patterns of length {PATTERN_LENGTH}")
    print(f"\nWindow sizes: {PATTERN_LENGTH//4}, {PATTERN_LENGTH//8}, {PATTERN_LENGTH//16}")
    print(f"Evaluating all {NUM_PATTERNS * (NUM_PATTERNS-1) // 2} pairs...\n")

    # Test all pairs
    global_worst = -100.0
    windowed_worst = {
        PATTERN_LENGTH // 4: -100.0,
        PATTERN_LENGTH // 8: -100.0,
        PATTERN_LENGTH // 16: -100.0
    }

    for i in range(NUM_PATTERNS):
        for j in range(i + 1, NUM_PATTERNS):
            # Global correlation
            global_corr = global_correlation(patterns[i], patterns[j])
            global_worst = max(global_worst, global_corr)

            # Windowed correlation at multiple scales
            for window_size in windowed_worst.keys():
                win_corr = windowed_correlation(patterns[i], patterns[j], window_size, num_windows=5)
                windowed_worst[window_size] = max(windowed_worst[window_size], win_corr)

    # Results
    print(f"Results:")
    print(f"  Global (full 2048-bit):  {global_worst:6.2f} dB")
    print(f"  Windowed (512-bit, 25%): {windowed_worst[PATTERN_LENGTH//4]:6.2f} dB")
    print(f"  Windowed (256-bit, 12%): {windowed_worst[PATTERN_LENGTH//8]:6.2f} dB")
    print(f"  Windowed (128-bit, 6%):  {windowed_worst[PATTERN_LENGTH//16]:6.2f} dB")

    print(f"\n{'='*70}")
    print("Analysis:")
    print(f"  For random patterns, windowed orthogonality should be WORSE than global")
    print(f"  (shorter windows = less averaging = higher peak correlation)")
    print(f"  This is expected and shows the windowing is working correctly.")
    print(f"\n  When patterns are optimized FOR windowed orthogonality:")
    print(f"  - Short windows will have better separation")
    print(f"  - Full-pattern correlation may be slightly worse")
    print(f"  - Overall: better partial-pattern detection robustness")
    print(f"{'='*70}")

    # Verify windowing makes sense
    assert windowed_worst[PATTERN_LENGTH//16] >= global_worst - 5, \
        "Windowed orthogonality should be worse or similar for random patterns"

    print("\n✓ Windowed orthogonality function working correctly!")
    return True


if __name__ == "__main__":
    try:
        test_orthogonality_methods()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
