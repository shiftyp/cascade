"""
Test script for GPU-accelerated enhanced dataset generation.

Tests all components:
1. GPU signal generator
2. GPU channel simulator (continuous fading, absorption, QRN)
3. Enhanced dataset with collisions and QRM
4. Performance benchmarking
"""

import torch
import numpy as np
import time
from pathlib import Path

print("=" * 80)
print("GPU-ACCELERATED CASCADE DATASET PIPELINE TEST")
print("=" * 80)

# Check GPU
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"\nGPU: {gpu_name}")
print(f"Memory: {gpu_memory:.1f} GB")

# Test 1: GPU Signal Generator
print("\n" + "=" * 80)
print("TEST 1: GPU Signal Generator")
print("=" * 80)

try:
    from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters

    gen = GPUSignalGenerator(device='cuda')

    # Generate small batch
    batch_size = 64
    batch_params = BatchKernelParameters(
        pattern_ids=torch.randint(0, 4, (batch_size,), device='cuda'),
        frequency_triples=torch.randint(0, 43, (batch_size,), device='cuda'),
        modulations=['QPSK'] * batch_size,
        polar_rates=[(2, 3)] * batch_size,
        data_symbol_rates=torch.tensor([150] * batch_size, device='cuda')
    )

    messages = [b"Test message for GPU generator!" for _ in range(batch_size)]

    start = time.time()
    signals, metadata = gen.generate_batch(batch_params, messages, fixed_length=2048)
    elapsed = time.time() - start

    print(f"✓ Generated {batch_size} signals in {elapsed*1000:.1f}ms")
    print(f"  Per-signal: {elapsed/batch_size*1000:.2f}ms")
    print(f"  Output shape: {signals.shape}")
    print(f"  Device: {signals.device}")
    print(f"  Mean power: {torch.mean(torch.abs(signals)**2):.3f}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: GPU Channel Simulator
print("\n" + "=" * 80)
print("TEST 2: GPU Channel Simulator")
print("=" * 80)

try:
    from gpu_channel_simulator import GPUChannelSimulator, MultipathProfile

    sim = GPUChannelSimulator(sample_rate=48000, device='cuda')

    # Create test signals
    batch_size = 32
    num_samples = 48000  # 1 second

    t = torch.arange(num_samples, dtype=torch.float32, device='cuda') / 48000
    test_signals = torch.exp(1j * 2 * torch.pi * 1500 * t.unsqueeze(0).expand(batch_size, -1))

    # Test 2a: Time-varying multipath
    print("\n2a. Time-varying multipath (10ms updates, 2048 freq bins)...")
    profile = MultipathProfile(
        delays_ms=torch.tensor([[0, 0.5, 1.5, 3.0, 5.0, 7.0]], device='cuda').expand(batch_size, -1),
        powers=torch.tensor([[0.4, 0.25, 0.15, 0.1, 0.07, 0.03]], device='cuda').expand(batch_size, -1),
        doppler_shifts_hz=torch.tensor([[0.0, 0.3, 0.6, 0.9, 1.2, 1.5]], device='cuda').expand(batch_size, -1),
        k_factors=torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device='cuda').expand(batch_size, -1)
    )

    start = time.time()
    faded = sim.apply_time_varying_multipath_batch(test_signals, profile, coherence_bandwidth_hz=50)
    elapsed = time.time() - start

    print(f"✓ Applied in {elapsed*1000:.1f}ms ({elapsed/batch_size*1000:.2f}ms per signal)")
    print(f"  Updates per signal: {int(num_samples / 48000 / (sim.update_interval_ms/1000))}")
    print(f"  Mean power: {torch.mean(torch.abs(faded)**2):.3f}")

    # Test 2b: Continuous D-layer absorption
    print("\n2b. Continuous D-layer absorption (f^-1.5 law)...")
    absorption_db = torch.tensor([5.0] * batch_size, device='cuda')
    sza = torch.tensor([30.0] * batch_size, device='cuda')

    start = time.time()
    absorbed = sim.apply_continuous_d_layer_absorption(faded, absorption_db, sza)
    elapsed = time.time() - start

    print(f"✓ Applied in {elapsed*1000:.1f}ms")

    # Verify absorption is frequency-dependent
    signal_fft = torch.fft.rfft(test_signals[0])
    absorbed_fft = torch.fft.rfft(absorbed[0])
    freq_bins = torch.fft.rfftfreq(num_samples, 1/48000, device='cuda')

    # Check absorption at 300 Hz vs 2860 Hz
    idx_300 = torch.argmin(torch.abs(freq_bins - 300))
    idx_2860 = torch.argmin(torch.abs(freq_bins - 2860))

    absorption_300 = -20 * torch.log10(torch.abs(absorbed_fft[idx_300]) / torch.abs(signal_fft[idx_300]))
    absorption_2860 = -20 * torch.log10(torch.abs(absorbed_fft[idx_2860]) / torch.abs(signal_fft[idx_2860]))

    print(f"  Absorption at 300 Hz: {absorption_300:.2f} dB")
    print(f"  Absorption at 2860 Hz: {absorption_2860:.2f} dB")
    print(f"  Ratio (should be ~5x): {absorption_300/absorption_2860:.2f}x")

    # Test 2c: Lightning QRN
    print("\n2c. Realistic lightning strikes (Poisson process)...")
    strike_rate = torch.tensor([10.0] * batch_size, device='cuda')

    start = time.time()
    qrn = sim.generate_lightning_strike_batch(batch_size, num_samples, strike_rate)
    elapsed = time.time() - start

    print(f"✓ Generated in {elapsed*1000:.1f}ms")
    print(f"  Peak amplitude: {torch.max(torch.abs(qrn)):.2f}")
    print(f"  RMS level: {torch.sqrt(torch.mean(torch.abs(qrn)**2)):.3f}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Enhanced Dataset (small batch)
print("\n" + "=" * 80)
print("TEST 3: Enhanced Physics Dataset")
print("=" * 80)

try:
    from enhanced_physics_dataset import EnhancedPhysicsDataset

    print("\nGenerating 256 samples (8 batches of 32)...")
    print("This tests full pipeline: generation + fading + absorption + QRN + collisions + QRM")

    start_total = time.time()

    dataset = EnhancedPhysicsDataset(
        num_samples=256,
        sample_rate=48000,
        for_test=False,
        seed=42,
        batch_size=32,
        collision_probability=0.3,
        qrm_probability=0.2,
        regenerate_cache=True,
        enable_visualization=False
    )

    elapsed_total = time.time() - start_total

    print(f"\n✓ Generated {len(dataset)} samples in {elapsed_total:.1f}s")
    print(f"  Per-sample: {elapsed_total/len(dataset)*1000:.1f}ms")
    print(f"  Throughput: {len(dataset)/elapsed_total:.1f} samples/sec")

    # Check a few samples
    print("\nSample inspection:")
    for idx in [0, 100, 200]:
        iq, labels = dataset[idx]
        print(f"\n  Sample {idx}:")
        print(f"    Pattern {labels['pattern_id']}, Freq {labels['frequency_triple']}")
        print(f"    SNR: {labels['snr_db']:.1f} dB, Mod: {labels['modulation']}")
        print(f"    Collision: {labels['has_collision']}, QRM: {labels['has_qrm']}")
        print(f"    Propagation: {labels['propagation_mode']}")

    # Statistics
    num_collisions = sum(1 for i in range(len(dataset)) if dataset[i][1]['has_collision'])
    num_qrm = sum(1 for i in range(len(dataset)) if dataset[i][1]['has_qrm'])

    print(f"\nDataset statistics:")
    print(f"  Collision rate: {num_collisions/len(dataset)*100:.1f}% (target: 30%)")
    print(f"  QRM rate: {num_qrm/len(dataset)*100:.1f}% (target: 20%)")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Performance Projection
print("\n" + "=" * 80)
print("TEST 4: Performance Projection for 500K Samples")
print("=" * 80)

if 'elapsed_total' in locals() and 'dataset' in locals():
    per_sample_time = elapsed_total / len(dataset)

    # Project for 500K samples
    total_time_500k = 500000 * per_sample_time

    print(f"\nMeasured performance:")
    print(f"  Per-sample time: {per_sample_time*1000:.2f}ms")
    print(f"  Throughput: {1/per_sample_time:.1f} samples/sec")

    print(f"\nProjection for 500,000 samples:")
    print(f"  Total time: {total_time_500k/60:.1f} minutes ({total_time_500k/3600:.1f} hours)")
    print(f"  vs CPU (6.1 days): {6.1*24*3600/total_time_500k:.0f}x speedup")

    # Memory usage
    gpu_memory_used = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\nGPU memory usage:")
    print(f"  Peak: {gpu_memory_used:.2f} GB / {gpu_memory:.1f} GB available")
    print(f"  Utilization: {gpu_memory_used/gpu_memory*100:.1f}%")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETE!")
print("=" * 80)
