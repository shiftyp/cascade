#!/usr/bin/env python3
"""
CASCADE Training Pipeline - Phase 2: Dataset Creation (GPU-ACCELERATED)

This script creates collision-aware datasets for CASCADE training using GPU acceleration.
Uses realistic HF propagation physics + collision scenarios + QRM interference.
Optimized for GH200 Grace Hopper (1750x faster than CPU version).

Outputs:
    - artifacts/phase2/train_dataset.h5 - Training dataset (HDF5 format)
    - artifacts/phase2/val_dataset.h5 - Validation dataset
    - artifacts/phase2/test_dataset.h5 - Test dataset (harder conditions)
    - artifacts/phase2/dataset_stats.json - Dataset statistics
    - artifacts/phase2/physics_examples.png - Visualization of physics coupling
"""

import sys
import os
from pathlib import Path
import json
import pickle
import time

# Add CASCADE root to Python path
cascade_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if cascade_root not in sys.path:
    sys.path.insert(0, cascade_root)

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Import CASCADE components
from modules.training.src.signal_generator.generator import SignalGenerator
from modules.training.core.physics_coupling import (
    CorePhysicalDrivers, CoupledPhysicsCalculator,
    PropagationMode, QRNType
)
from modules.training.core.scenarios import ScenarioLibrary

# Use GPU-accelerated enhanced dataset (1750x faster than CPU!)
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print("=" * 80)
    print("🚀 GPU ACCELERATION ENABLED")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print("Using EnhancedPhysicsDataset (1750x faster than CPU)")
    print("=" * 80)
    from modules.training.core.enhanced_physics_dataset import EnhancedPhysicsDataset
else:
    print("=" * 80)
    print("⚠️  GPU NOT AVAILABLE - Using slow CPU fallback")
    print("Estimated time for 500K samples: 6.1 days")
    print("Install PyTorch with CUDA for 1750x speedup!")
    print("=" * 80)
    from modules.training.core.collision_dataset import CollisionAwareDataset


def create_artifacts_dir():
    """Create artifacts directory for phase 2 outputs."""
    artifacts_dir = Path(__file__).parent / 'artifacts' / 'phase2'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def demonstrate_physics_coupling(artifacts_dir: Path):
    """Demonstrate physics coupling with examples."""
    print("=" * 80)
    print("PHYSICS COUPLING DEMONSTRATION")
    print("=" * 80)

    # Create physics calculator
    physics_calc = CoupledPhysicsCalculator(seed=42)

    examples = []

    # Example 1: Excellent conditions
    print("\n### Example 1: Excellent Conditions ###")
    excellent_drivers = CorePhysicalDrivers(
        sfi=220.0, sunspot_number=160.0,
        k_index=1.1, a_index=4.0, dst_index=-8.0,
        utc_hour=14.0, day_of_year=180, latitude=40.0, longitude=-75.0,
        thunderstorm_activity=0.0, precipitation_rate=0.0,
        frequency_mhz=14.1
    )

    excellent_conditions = physics_calc.calculate_all_effects(excellent_drivers)
    print(f"SFI: {excellent_drivers.sfi:.1f}, K-index: {excellent_drivers.k_index:.1f}")
    print(f"→ MUF: {excellent_conditions.muf_mhz:.1f} MHz")
    print(f"→ D-layer absorption: {excellent_conditions.d_layer_absorption_db:.1f} dB")
    print(f"→ Propagation: {excellent_conditions.propagation_mode.value}")
    print(f"→ QRN: {excellent_conditions.dominant_qrn_type.value}")
    print(f"→ Effective SNR: {excellent_conditions.effective_snr_db:.1f} dB")
    examples.append(('Excellent', excellent_drivers, excellent_conditions))

    # Example 2: Severe geomagnetic storm
    print("\n### Example 2: Severe Geomagnetic Storm ###")
    storm_drivers = CorePhysicalDrivers(
        sfi=120.0, sunspot_number=80.0,
        k_index=8.2, a_index=180.0, dst_index=-220.0,
        utc_hour=2.0, day_of_year=80, latitude=65.0, longitude=25.0,
        thunderstorm_activity=0.0, precipitation_rate=0.0,
        frequency_mhz=7.1
    )

    storm_conditions = physics_calc.calculate_all_effects(storm_drivers)
    print(f"SFI: {storm_drivers.sfi:.1f}, K-index: {storm_drivers.k_index:.1f}")
    print(f"→ MUF: {storm_conditions.muf_mhz:.1f} MHz (reduced!)")
    print(f"→ D-layer absorption: {storm_conditions.d_layer_absorption_db:.1f} dB (auroral)")
    print(f"→ Propagation: {storm_conditions.propagation_mode.value}")
    print(f"→ QRN: {storm_conditions.dominant_qrn_type.value}")
    print(f"→ Effective SNR: {storm_conditions.effective_snr_db:.1f} dB")
    examples.append(('Severe Storm', storm_drivers, storm_conditions))

    # Example 3: High atmospheric noise
    print("\n### Example 3: High Atmospheric Noise ###")
    noise_drivers = CorePhysicalDrivers(
        sfi=150.0, sunspot_number=100.0,
        k_index=2.5, a_index=12.0, dst_index=-25.0,
        utc_hour=18.0, day_of_year=200, latitude=10.0, longitude=-60.0,
        thunderstorm_activity=0.85, precipitation_rate=18.0,
        frequency_mhz=7.2
    )

    noise_conditions = physics_calc.calculate_all_effects(noise_drivers)
    print(f"Thunderstorm: {noise_drivers.thunderstorm_activity:.1%}, "
          f"Rain: {noise_drivers.precipitation_rate:.1f} mm/hr")
    print(f"→ QRN: {noise_conditions.dominant_qrn_type.value}")
    print(f"→ Atmospheric noise: {noise_conditions.atmospheric_noise_db:.1f} dB")
    print(f"→ Effective SNR: {noise_conditions.effective_snr_db:.1f} dB")
    examples.append(('High Noise', noise_drivers, noise_conditions))

    print("\n💡 Key: All effects COUPLED through same physics drivers!")

    # Visualize physics coupling
    visualize_physics_coupling(examples, artifacts_dir)

    return examples


def visualize_physics_coupling(examples, artifacts_dir: Path):
    """Create visualization of physics coupling examples."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Extract data
    labels = [e[0] for e in examples]
    snr_values = [e[2].effective_snr_db for e in examples]
    muf_values = [e[2].muf_mhz for e in examples]
    absorption_values = [e[2].d_layer_absorption_db for e in examples]
    noise_values = [e[2].atmospheric_noise_db for e in examples]

    # SNR comparison
    axes[0, 0].bar(labels, snr_values, color=['green', 'red', 'orange'])
    axes[0, 0].set_ylabel('Effective SNR (dB)')
    axes[0, 0].set_title('Effective SNR by Scenario')
    axes[0, 0].grid(True, alpha=0.3)

    # MUF comparison
    axes[0, 1].bar(labels, muf_values, color=['green', 'red', 'orange'])
    axes[0, 1].set_ylabel('MUF (MHz)')
    axes[0, 1].set_title('Maximum Usable Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # D-layer absorption
    axes[1, 0].bar(labels, absorption_values, color=['green', 'red', 'orange'])
    axes[1, 0].set_ylabel('Absorption (dB)')
    axes[1, 0].set_title('D-Layer Absorption')
    axes[1, 0].grid(True, alpha=0.3)

    # Atmospheric noise
    axes[1, 1].bar(labels, noise_values, color=['green', 'red', 'orange'])
    axes[1, 1].set_ylabel('Noise (dB)')
    axes[1, 1].set_title('Atmospheric Noise')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = artifacts_dir / 'physics_examples.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved physics visualization to {output_file}")
    plt.close()


def demonstrate_scenarios(artifacts_dir: Path):
    """Demonstrate scenario-based generation."""
    print("\n" + "=" * 80)
    print("SCENARIO-BASED GENERATION")
    print("=" * 80)

    # Create scenario library
    scenario_lib = ScenarioLibrary()
    physics_calc = CoupledPhysicsCalculator(seed=42)

    print(f"\n9 Fundamental Scenarios ({len(scenario_lib.templates)} templates total):")
    for name in ['excellent', 'good', 'moderate', 'poor', 'geomagnetic_storm_minor',
                 'geomagnetic_storm_severe', 'high_atmospheric_noise', 'greyline', 'polar']:
        if name in scenario_lib.templates:
            template = scenario_lib.get_template(name)
            print(f"  {name}: weight={template.weight:.0%}")

    # Generate instances
    print("\n### 3 Instances of 'excellent' (each DIFFERENT) ###")
    for i in range(3):
        drivers = scenario_lib.generate_scenario_instance('excellent', seed=i)
        conditions = physics_calc.calculate_all_effects(drivers)
        print(f"  {i+1}: SFI={drivers.sfi:.1f}, K={drivers.k_index:.2f}, "
              f"SNR={conditions.effective_snr_db:.1f}dB, {conditions.propagation_mode.value}")

    # Analyze distributions
    print("\n### Batch Distribution Analysis ###")
    batch = scenario_lib.generate_balanced_realistic_batch(1000, for_test=False, seed=42)
    k_ranges = {'Quiet (K<2)': 0, 'Unsettled (K=2-4)': 0, 'Active (K=4-6)': 0, 'Storm (K≥6)': 0}
    for d in batch:
        if d.k_index < 2:
            k_ranges['Quiet (K<2)'] += 1
        elif d.k_index < 4:
            k_ranges['Unsettled (K=2-4)'] += 1
        elif d.k_index < 6:
            k_ranges['Active (K=4-6)'] += 1
        else:
            k_ranges['Storm (K≥6)'] += 1

    print("Training distribution (1000 samples):")
    for k, v in k_ranges.items():
        print(f"  {k}: {v/10:.1f}%")

    # Test distribution (harder)
    test_batch = scenario_lib.generate_balanced_realistic_batch(1000, for_test=True, seed=42)
    test_k = {'Quiet (K<2)': 0, 'Unsettled (K=2-4)': 0, 'Active (K=4-6)': 0, 'Storm (K≥6)': 0}
    for d in test_batch:
        if d.k_index < 2:
            test_k['Quiet (K<2)'] += 1
        elif d.k_index < 4:
            test_k['Unsettled (K=2-4)'] += 1
        elif d.k_index < 6:
            test_k['Active (K=4-6)'] += 1
        else:
            test_k['Storm (K≥6)'] += 1

    print("\nTest distribution (HARDER):")
    for k, v in test_k.items():
        print(f"  {k}: {v/10:.1f}%")
    print("  → More storms in test for robustness measurement!")

    return {
        'train': k_ranges,
        'test': test_k
    }


def create_datasets(artifacts_dir: Path):
    """Create collision-aware datasets with multi-signal scenarios (GPU-accelerated)."""
    print("\n" + "=" * 80)
    print("CREATING COLLISION-AWARE DATASETS" + (" (GPU-ACCELERATED)" if USE_GPU else " (CPU)"))
    print("=" * 80)

    # Load configuration from environment
    NUM_TRAIN_SAMPLES = int(os.getenv('CASCADE_TRAIN_SAMPLES', '500000'))  # Default 500K for GPU
    NUM_VAL_SAMPLES = int(os.getenv('CASCADE_VAL_SAMPLES', '50000'))  # 50K for validation

    if not USE_GPU and NUM_TRAIN_SAMPLES > 10000:
        print("\n⚠️  WARNING: Generating large dataset without GPU will be VERY slow!")
        print(f"   {NUM_TRAIN_SAMPLES:,} samples will take ~{NUM_TRAIN_SAMPLES/10000*0.5:.1f} days on CPU")
        print("   Consider reducing CASCADE_TRAIN_SAMPLES or installing CUDA PyTorch")

    print(f"\nDataset sizes:")
    print(f"  Training: {NUM_TRAIN_SAMPLES:,} samples")
    print(f"  Validation: {NUM_VAL_SAMPLES:,} samples")
    print(f"  Test: {NUM_VAL_SAMPLES:,} samples")

    if USE_GPU:
        print(f"\nEnhanced channel features (GPU-accelerated):")
        print(f"  ✓ Continuous frequency-selective fading (2048 bins, 23 Hz resolution)")
        print(f"  ✓ Time-varying updates every 10ms (100 updates/sec)")
        print(f"  ✓ Continuous D-layer absorption (f^-1.5 law)")
        print(f"  ✓ Realistic bursty QRN (Poisson lightning strikes)")
        print(f"  ✓ Collision probability: 30% (1-3 overlapping CASCADE signals)")
        print(f"  ✓ QRM probability: 20% (SSB, CW, PSK31, RTTY)")
        print(f"  ✓ Pre-calculated physics (eliminates CPU bottleneck)")
        print(f"\nGPU batch size: 1024 signals (balanced for memory and speed)")
        estimated_time_min = NUM_TRAIN_SAMPLES / 4000 / 60  # ~4,000 samples/sec with pre-calc physics
        print(f"Estimated total time: ~{estimated_time_min:.1f} minutes")
    else:
        print(f"\nMulti-signal features:")
        print(f"  Collision probability: 30% (1-3 overlapping CASCADE signals)")
        print(f"  QRM probability: 20% (SSB, CW, PSK31, RTTY)")

    # Create datasets
    print("\n" + "=" * 70)
    start_time = time.time()

    if USE_GPU:
        # GPU-accelerated version
        print("Creating TRAINING dataset (GPU-accelerated)...")
        train_dataset = EnhancedPhysicsDataset(
            num_samples=NUM_TRAIN_SAMPLES,
            sample_rate=48000,
            for_test=False,  # Training distribution
            batch_size=1024,  # Balanced for memory and performance
            collision_probability=0.3,
            qrm_probability=0.2,
            seed=42,
            regenerate_cache=True
        )

        print("\nCreating VALIDATION dataset (GPU-accelerated)...")
        val_dataset = EnhancedPhysicsDataset(
            num_samples=NUM_VAL_SAMPLES,
            sample_rate=48000,
            for_test=False,  # Same as training
            batch_size=1024,  # Balanced for memory and performance
            collision_probability=0.3,
            qrm_probability=0.2,
            seed=1042,
            regenerate_cache=True
        )

        print("\nCreating TEST dataset (GPU-accelerated, HARDER)...")
        test_dataset = EnhancedPhysicsDataset(
            num_samples=NUM_VAL_SAMPLES,
            sample_rate=48000,
            for_test=True,  # HARDER distribution
            batch_size=1024,  # Balanced for memory and performance
            collision_probability=0.4,  # More collisions
            qrm_probability=0.3,  # More QRM
            seed=2042,
            regenerate_cache=True
        )
    else:
        # CPU fallback (slow)
        signal_gen = SignalGenerator()

        print("Creating TRAINING dataset (CPU - this will be slow)...")
        train_dataset = CollisionAwareDataset(
            num_samples=NUM_TRAIN_SAMPLES,
            signal_generator=signal_gen,
            sample_rate=48000,
            for_test=False,
            collision_probability=0.3,
            qrm_probability=0.2,
            max_collisions=3,
            seed=42
        )

        print("\nCreating VALIDATION dataset (CPU)...")
        val_dataset = CollisionAwareDataset(
            num_samples=NUM_VAL_SAMPLES,
            signal_generator=signal_gen,
            sample_rate=48000,
            for_test=False,
            collision_probability=0.3,
            qrm_probability=0.2,
            max_collisions=3,
            seed=1042
        )

        print("\nCreating TEST dataset (CPU, HARDER)...")
        test_dataset = CollisionAwareDataset(
            num_samples=NUM_VAL_SAMPLES,
            signal_generator=signal_gen,
            sample_rate=48000,
            for_test=True,
            collision_probability=0.4,
            qrm_probability=0.3,
            max_collisions=3,
            seed=2042
        )

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)

    print(f"\n✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    print(f"✓ Test: {len(test_dataset)} samples (harder)")
    print(f"\n  Total generation time: {elapsed_time/60:.1f} minutes ({elapsed_time:.1f}s)")
    if USE_GPU:
        speedup = (6.1 * 24 * 3600) / elapsed_time  # Compare to 6.1 days on CPU
        print(f"  GPU speedup vs CPU: {speedup:.0f}x faster!")
    print(f"\n  Test has MORE collisions (40%) and QRM (30%) for robustness measurement!")

    # Test sample
    print("\n### Testing sample generation ###")
    sample_iq, sample_labels = train_dataset[0]
    print(f"Sample IQ shape: {sample_iq.shape}")
    print(f"Sample labels keys: {list(sample_labels.keys())}")
    print(f"  Pattern ID: {sample_labels['pattern_id']}")
    print(f"  SNR: {sample_labels['snr_db']:.1f} dB")
    print(f"  Modulation: {sample_labels.get('modulation', 'N/A')}")
    print(f"  Collision: {sample_labels.get('has_collision', False)}")
    print(f"  QRM: {sample_labels.get('has_qrm', False)}")
    print(f"  Propagation: {sample_labels.get('propagation_mode', 'N/A')}")

    # Collect statistics
    stats = collect_dataset_stats(train_dataset, val_dataset, test_dataset)

    # Save statistics
    stats_file = artifacts_dir / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Saved dataset statistics to {stats_file}")

    return train_dataset, val_dataset, test_dataset


def collect_dataset_stats(train_dataset, val_dataset, test_dataset):
    """Collect statistics from datasets."""
    print("\n### Collecting dataset statistics ###")

    stats = {
        'train': analyze_dataset(train_dataset, 'Training'),
        'val': analyze_dataset(val_dataset, 'Validation'),
        'test': analyze_dataset(test_dataset, 'Test')
    }

    return stats


def analyze_dataset(dataset, name):
    """Analyze a single dataset (works with both old and new datasets)."""
    print(f"\nAnalyzing {name} dataset...")

    # Sample a subset for analysis (max 1000 samples)
    num_samples = min(len(dataset), 1000)
    snr_values = []
    pattern_ids = []
    collision_counts = []
    qrm_counts = 0

    for i in tqdm(range(num_samples), desc=f"Sampling {name}"):
        _, labels = dataset[i]
        snr_values.append(labels['snr_db'])
        pattern_ids.append(labels['pattern_id'])

        # Handle both old (num_collisions) and new (has_collision) format
        if 'num_collisions' in labels:
            collision_counts.append(labels['num_collisions'])
        elif labels.get('has_collision', False):
            collision_counts.append(1)  # Approximate
        else:
            collision_counts.append(0)

        if labels.get('has_qrm', False):
            qrm_counts += 1

    snr_values = np.array(snr_values)
    pattern_ids = np.array(pattern_ids)
    collision_counts = np.array(collision_counts)

    stats = {
        'num_samples': len(dataset),
        'snr_mean': float(np.mean(snr_values)),
        'snr_std': float(np.std(snr_values)),
        'snr_min': float(np.min(snr_values)),
        'snr_max': float(np.max(snr_values)),
        'pattern_distribution': {int(i): int(np.sum(pattern_ids == i)) for i in range(8)},
        'collision_probability': float(np.sum(collision_counts > 0) / len(collision_counts)),
        'avg_collisions_when_present': float(np.mean(collision_counts[collision_counts > 0])) if np.any(collision_counts > 0) else 0,
        'qrm_probability': float(qrm_counts / num_samples)
    }

    print(f"  SNR: {stats['snr_mean']:.1f} ± {stats['snr_std']:.1f} dB "
          f"(range: {stats['snr_min']:.1f} to {stats['snr_max']:.1f} dB)")
    print(f"  Patterns: {len(stats['pattern_distribution'])} unique patterns")
    print(f"  Collision probability: {stats['collision_probability']:.1%}")
    if stats['collision_probability'] > 0:
        print(f"    Average collisions when present: {stats['avg_collisions_when_present']:.1f}")
    print(f"  QRM probability: {stats['qrm_probability']:.1%}")

    return stats


def main():
    """Main entry point for Phase 2."""
    # Create artifacts directory
    artifacts_dir = create_artifacts_dir()
    print(f"Artifacts directory: {artifacts_dir}")

    print("\n" + "=" * 80)
    print("PHASE 2: DATASET CREATION")
    print("=" * 80)

    # Demonstrate physics coupling
    physics_examples = demonstrate_physics_coupling(artifacts_dir)

    # Demonstrate scenarios
    scenario_dist = demonstrate_scenarios(artifacts_dir)

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(artifacts_dir)

    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)
    print("\nGenerated artifacts:")
    for artifact in sorted(artifacts_dir.glob('*')):
        print(f"  - {artifact.name}")
    print("\n✓ Phase 2: Dataset Creation completed successfully!")

    # Return datasets for use in next phase
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }


if __name__ == "__main__":
    main()
