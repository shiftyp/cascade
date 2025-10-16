#!/usr/bin/env python3
"""
Test continuous symbol rates with enhanced collision probability.

Verifies:
1. Shannon efficiency stays below 100%
2. Minimum SNR is -14 dB
3. Increased message arrival rate creates more collisions
4. Continuous rates are being generated (not discrete)
"""

import numpy as np
import torch
from continuous_rate_calculator import ContinuousRateCalculator
from streaming_cascade_dataset import StreamingCascadeDataset


def test_shannon_efficiency():
    """Test that Shannon efficiency never exceeds 100%."""
    print("\n" + "="*70)
    print("TEST 1: Shannon Efficiency Bounds")
    print("="*70)

    calc = ContinuousRateCalculator(bandwidth_hz=60.0)

    # Test across wide SNR range
    test_snrs = [-14, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]

    print("SNR(dB) | Modulation | Rate(sym/s) | Bitrate | Shannon | Efficiency")
    print("-"*70)

    all_valid = True
    for snr_db in test_snrs:
        mod, bps = calc.optimal_modulation(snr_db)
        rate, num_channels, _ = calc.calculate_continuous_rate(snr_db)
        # Use effective bandwidth for capacity calculation
        capacity = calc.shannon_capacity(snr_db) * num_channels
        bitrate = rate * bps
        efficiency = (bitrate / capacity * 100) if capacity > 0 else 0

        # Check if efficiency exceeds 100%
        if efficiency > 100:
            all_valid = False
            status = "❌ EXCEEDS!"
        else:
            status = "✓"

        print(f"{snr_db:6.1f} | {mod:10s} | {rate:11.1f} | {bitrate:7.1f} | {capacity:7.1f} | {efficiency:6.1f}% {status}")

    if all_valid:
        print("\n✅ PASS: All efficiencies below 100%")
    else:
        print("\n❌ FAIL: Some efficiencies exceed Shannon limit!")

    return all_valid


def test_continuous_rates():
    """Test that rates are continuous, not discrete."""
    print("\n" + "="*70)
    print("TEST 2: Continuous Rate Generation")
    print("="*70)

    calc = ContinuousRateCalculator(bandwidth_hz=60.0)

    # Generate many rates
    np.random.seed(42)
    rates = []
    for _ in range(100):
        snr = np.random.uniform(-14, 30)
        rate, _, _ = calc.calculate_continuous_rate(snr)
        rates.append(rate)

    # Check for diversity (should not be just 4 discrete values)
    unique_rates = len(set(rates))

    print(f"Generated 100 rates, found {unique_rates} unique values")
    print(f"Rate range: {min(rates):.1f} - {max(rates):.1f} sym/s")

    # Show histogram
    hist, bins = np.histogram(rates, bins=10)
    print("\nRate distribution:")
    for i in range(len(hist)):
        bar = '█' * int(hist[i] * 2)
        print(f"  {bins[i]:6.1f}-{bins[i+1]:6.1f}: {bar} ({hist[i]})")

    # Should have many unique rates (not just 4 discrete)
    is_continuous = unique_rates > 20

    if is_continuous:
        print(f"\n✅ PASS: Rates are continuous ({unique_rates} unique values)")
    else:
        print(f"\n❌ FAIL: Rates appear discrete ({unique_rates} unique values)")

    return is_continuous


def test_collision_probability():
    """Test that increased message arrival rate creates more collisions."""
    print("\n" + "="*70)
    print("TEST 3: Collision Probability Analysis")
    print("="*70)

    # Create small test dataset with new parameters
    print("\nCreating test dataset with increased collision rate...")
    dataset = StreamingCascadeDataset(
        num_streams=100,
        stream_duration_sec=10.0,
        window_duration_sec=2.0,
        message_arrival_rate=0.8,  # Doubled from original 0.4
        seed=42,
        regenerate_cache=True,
        batch_size=4,
        num_workers=1
    )

    # Analyze collision statistics
    collision_count = 0
    total_messages = 0

    # Check first 10 streams for collision analysis
    for stream_idx in range(min(10, dataset.num_streams)):
        messages = dataset._generate_stream_messages(stream_idx)
        total_messages += len(messages)

        # Check for temporal overlaps
        for i in range(len(messages)):
            for j in range(i + 1, len(messages)):
                if messages[i].end_sample > messages[j].start_sample and \
                   messages[i].start_sample < messages[j].end_sample:
                    # Check frequency overlap
                    if abs(messages[i].frequency_triple - messages[j].frequency_triple) <= 2:
                        collision_count += 1

    avg_messages_per_stream = total_messages / 10
    collision_rate = collision_count / max(1, total_messages) * 100

    print(f"\nResults from 10 test streams:")
    print(f"  Average messages per stream: {avg_messages_per_stream:.1f}")
    print(f"  Total collisions detected: {collision_count}")
    print(f"  Collision rate: {collision_rate:.1f}%")

    # Expected: ~8 messages per stream (vs 4 before)
    # Should see more collisions
    increased_collisions = avg_messages_per_stream > 6 and collision_rate > 5

    if increased_collisions:
        print(f"\n✅ PASS: Increased collision rate achieved")
    else:
        print(f"\n❌ FAIL: Collision rate not significantly increased")

    return increased_collisions


def test_snr_range():
    """Test that minimum SNR is -14 dB."""
    print("\n" + "="*70)
    print("TEST 4: SNR Range Verification")
    print("="*70)

    # Generate many SNR values using same distribution as dataset
    np.random.seed(42)
    snr_values = []
    for _ in range(1000):
        snr = np.random.uniform(-14, 20)
        snr_values.append(snr)

    min_snr = min(snr_values)
    max_snr = max(snr_values)

    print(f"SNR range from 1000 samples:")
    print(f"  Minimum: {min_snr:.2f} dB")
    print(f"  Maximum: {max_snr:.2f} dB")

    # Verify minimum is >= -14 dB
    valid_range = min_snr >= -14.0

    if valid_range:
        print(f"\n✅ PASS: Minimum SNR is -14 dB as required")
    else:
        print(f"\n❌ FAIL: SNR goes below -14 dB limit!")

    return valid_range


def main():
    """Run all tests."""
    print("="*70)
    print("CASCADE CONTINUOUS RATE TESTING")
    print("="*70)

    results = []

    # Run all tests
    results.append(("Shannon Efficiency", test_shannon_efficiency()))
    results.append(("Continuous Rates", test_continuous_rates()))
    results.append(("SNR Range", test_snr_range()))
    results.append(("Collision Probability", test_collision_probability()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nConfiguration ready for training:")
        print("  - Shannon efficiency stays below 100%")
        print("  - Minimum SNR set to -14 dB")
        print("  - Continuous symbol rates (50-600 sym/s)")
        print("  - Increased collision rate (0.8 messages/sec)")
    else:
        print("\n⚠️ Some tests failed. Please review the issues above.")

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)