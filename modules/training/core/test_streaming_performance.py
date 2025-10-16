#!/usr/bin/env python3
"""
Test actual streaming dataset performance to identify bottlenecks.
"""

import time
import torch
import numpy as np
from streaming_cascade_dataset import StreamingCascadeDataset


def test_streaming_performance():
    """Test actual streaming dataset generation performance."""

    print("Testing Streaming Dataset Performance")
    print("=" * 60)

    # Create a small streaming dataset with similar parameters to production
    dataset = StreamingCascadeDataset(
        num_streams=100,  # Small test
        stream_duration_sec=10.0,
        window_duration_sec=2.0,
        message_arrival_rate=0.8,  # ~8 messages per stream
        seed=42,
        regenerate_cache=True,
        batch_size=128,  # Production batch size
        device='cuda',
        num_workers=1,  # Minimize CPU overhead for testing
        enable_tx_observations=False,
        compute_optimal_embeddings=True
    )

    print("\nDataset generated successfully!")
    print(f"Check the generation speed reported above.")

    # Also test direct signal generation to compare
    print("\n" + "=" * 60)
    print("Direct Signal Generation Test (for comparison)")
    print("=" * 60)

    from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters
    from pathlib import Path

    # Test with both standard and always-on patterns
    for use_always_on in [False, True]:
        print(f"\nTesting with always_on={use_always_on}")

        gen = GPUSignalGenerator(
            device='cuda',
            patterns_dir=Path(__file__).parent.parent / "patterns" / "patterns" if use_always_on else None,
            use_gpu_polar=True,
            use_always_on=use_always_on
        )

        # Simulate typical message batch from streaming dataset
        # Average ~4 messages per stream, 128 streams = ~512 messages
        batch_size = 512

        batch_params = BatchKernelParameters(
            pattern_ids=torch.randint(0, 4, (batch_size,), device='cuda'),
            frequency_triples=torch.randint(0, 61, (batch_size,), device='cuda'),
            modulations=['QPSK'] * batch_size,
            polar_rates=[(2, 3)] * batch_size,
            data_symbol_rates=torch.tensor([150] * batch_size, device='cuda')
        )

        messages = [b"Test message"] * batch_size

        # Warm up
        for _ in range(3):
            _ = gen.generate_batch(batch_params, messages, fixed_length=None,
                                  num_centers=4 if use_always_on else 0)

        # Time generation
        torch.cuda.synchronize()
        start = time.time()

        num_iterations = 10
        for _ in range(num_iterations):
            signals, metadata = gen.generate_batch(
                batch_params, messages, fixed_length=None,
                num_centers=4 if use_always_on else 0
            )
            torch.cuda.synchronize()

        elapsed = time.time() - start

        signals_per_sec = (batch_size * num_iterations) / elapsed
        print(f"  Speed: {signals_per_sec:.1f} signals/sec")
        print(f"  Per-signal: {(elapsed / (batch_size * num_iterations)) * 1000:.2f}ms")

        # Estimate stream rate (128 streams with ~4 messages each)
        streams_per_sec = signals_per_sec / 4  # ~4 messages per stream
        print(f"  Estimated: {streams_per_sec:.1f} streams/sec")


if __name__ == "__main__":
    test_streaming_performance()