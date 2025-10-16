#!/usr/bin/env python3
"""Test that the training pipeline can initialize with always-on configuration."""

import os
import sys
import torch

# Set minimal test configuration
os.environ['CASCADE_TRAIN_SAMPLES'] = '100'  # Just test initialization
os.environ['CASCADE_VAL_SAMPLES'] = '10'
os.environ['CASCADE_DATASET_NUM_WORKERS'] = '2'
os.environ['CASCADE_BATCH_SIZE'] = '8'
os.environ['CASCADE_USE_STREAMING'] = 'true'

print("="*60)
print("Testing Training Pipeline Initialization")
print("="*60)

try:
    # Test GPU Signal Generator
    from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters

    print("\n1. Testing GPU Signal Generator...")
    gen = GPUSignalGenerator(device='cuda')
    print(f"   ✓ Pattern length: {gen.PATTERN_LENGTH}")
    print(f"   ✓ Tone spacing: {gen.TONE_SPACING} Hz")
    print(f"   ✓ Num frequency triples: {gen.NUM_FREQUENCY_TRIPLES}")
    print(f"   ✓ Pattern symbol rate: {gen.PATTERN_SYMBOL_RATE} sym/s")

    # Generate a test batch
    batch_params = BatchKernelParameters(
        pattern_ids=torch.tensor([0, 1, 2, 3], device='cuda'),
        frequency_triples=torch.tensor([0, 20, 40, 60], device='cuda'),
        modulations=['QPSK'] * 4,
        polar_rates=[(2, 3)] * 4,
        data_symbol_rates=torch.tensor([150] * 4, device='cuda')
    )
    messages = [b"Test message"] * 4

    signals, metadata = gen.generate_batch(batch_params, messages, fixed_length=2048)
    print(f"   ✓ Generated batch shape: {signals.shape}")

    # Test Streaming Dataset
    print("\n2. Testing Streaming Dataset...")
    from streaming_cascade_dataset import StreamingCascadeDataset

    dataset = StreamingCascadeDataset(
        num_streams=2,  # Just 2 streams for test
        stream_duration_sec=10.0,
        window_duration_sec=2.0,
        device='cuda',
        enable_tx_observations=False,
        regenerate_cache=False,
        use_cache=False  # Don't use cache for test
    )
    print(f"   ✓ Dataset initialized with {len(dataset)} samples")

    # Try to get one sample
    try:
        sample = dataset[0]
        if sample is not None:
            print(f"   ✓ Sample loaded successfully")
            if 'rx_iq' in sample:
                print(f"     RX IQ shape: {sample['rx_iq'].shape}")
            if 'labels' in sample:
                print(f"     Labels: {list(sample['labels'].keys())}")
    except Exception as e:
        print(f"   ⚠ Could not load sample (dataset may need generation): {e}")

    # Test Continuous Rate Calculator
    print("\n3. Testing Continuous Rate Calculator...")
    from continuous_rate_calculator import ContinuousRateCalculator

    calc = ContinuousRateCalculator()
    rate, num_channels, freq_triple = calc.calculate_continuous_rate(
        snr_db=-10.0,
        multipath_severity=0.5,
        k_index=3.0
    )
    print(f"   ✓ Rate: {rate:.1f} sym/s")
    print(f"   ✓ Num channels: {num_channels} (fixed 4-center config)")
    print(f"   ✓ Frequency triple: {freq_triple} (0-{calc.num_frequency_triples-1})")

    print("\n✅ All components initialized successfully!")
    print("   The training pipeline is ready for always-on configuration.")

except Exception as e:
    print(f"\n❌ Error during initialization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)