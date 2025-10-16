#!/usr/bin/env python3
"""
Test performance of optimized always-on signal generation.
Compare speed before/after optimization.
"""

import time
import torch
import numpy as np
from pathlib import Path
from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters


def test_performance():
    """Test signal generation performance with always-on design."""

    print("Testing optimized always-on signal generation performance...")
    print("=" * 60)

    # Create generator with always-on patterns
    gen = GPUSignalGenerator(
        device='cuda',
        patterns_dir=Path(__file__).parent.parent / "patterns" / "patterns",
        use_gpu_polar=True,
        use_always_on=True
    )

    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256]

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        # Create batch parameters
        batch_params = BatchKernelParameters(
            pattern_ids=torch.randint(0, 4, (batch_size,), device='cuda'),
            frequency_triples=torch.randint(0, 61, (batch_size,), device='cuda'),
            modulations=['QPSK'] * batch_size,
            polar_rates=[(2, 3)] * batch_size,
            data_symbol_rates=torch.tensor([150] * batch_size, device='cuda')
        )

        # Create messages
        messages = [b"Test message for performance"] * batch_size

        # Warm up GPU
        for _ in range(3):
            _ = gen.generate_batch(batch_params, messages, fixed_length=None, num_centers=4)

        # Time the generation
        num_iterations = 10
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_iterations):
            signals, metadata = gen.generate_batch(
                batch_params,
                messages,
                fixed_length=None,
                num_centers=4  # Always-on with 4 centers
            )
            torch.cuda.synchronize()

        elapsed = time.time() - start_time

        # Calculate statistics
        total_signals = batch_size * num_iterations
        signals_per_sec = total_signals / elapsed
        ms_per_signal = (elapsed / total_signals) * 1000

        print(f"  Generated {total_signals} signals in {elapsed:.2f}s")
        print(f"  Speed: {signals_per_sec:.1f} signals/sec")
        print(f"  Per-signal time: {ms_per_signal:.2f}ms")

        # For streaming dataset, each "stream" contains multiple signals
        # Estimate stream generation rate (assuming ~50 signals per stream)
        streams_per_sec = signals_per_sec / 50
        print(f"  Estimated stream rate: {streams_per_sec:.1f} streams/sec")

    print("\n" + "=" * 60)
    print("Performance test complete!")
    print("\nOptimizations applied:")
    print("  ✓ Center tones use simple sinusoids (no GMSK)")
    print("  ✓ Pre-computed Gaussian filter for alternating tones")
    print("  ✓ Single ramp window for all center tones")
    print("  ✓ Batch frequency calculations")

    print("\nExpected improvement:")
    print("  Before: ~70 streams/sec (with unnecessary GMSK on centers)")
    print("  Target: ~260 streams/sec (matching original 3-FSK performance)")


if __name__ == "__main__":
    test_performance()