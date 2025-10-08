"""
Physics-Constrained Dataset Integration for cascade.ipynb

This file contains the code cells to add to cascade.ipynb for integrating
the physics-based scenario system.

ADD THIS AS A NEW CELL AFTER THE HybridCascadeDataset SECTION (around cell 10-11).
"""

# ============================================================================
# CELL 1: Import Physics-Based Scenario System
# ============================================================================

print("=" * 80)
print("PHYSICS-CONSTRAINED DATASET SYSTEM")
print("=" * 80)

# Import physics-based components
from physics_coupling import (
    CorePhysicalDrivers, CoupledPhysicsCalculator,
    PropagationMode, QRNType
)
from scenarios import ScenarioLibrary, ScenarioType
from physics_constrained_dataset import PhysicsConstrainedDataset

print("✓ Physics coupling module imported")
print("✓ Scenario library imported")
print("✓ Physics-constrained dataset imported")

# ============================================================================
# CELL 2: Demonstrate Physics Coupling
# ============================================================================

print("\n" + "=" * 80)
print("PHYSICS COUPLING DEMONSTRATION")
print("=" * 80)

# Create physics calculator
physics_calc = CoupledPhysicsCalculator(seed=42)

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

# Example 2: Severe geomagnetic storm
print("\n### Example 2: Severe Geomagnetic Storm ###")
storm_drivers = CorePhysicalDrivers(
    sfi=120.0, sunspot_number=80.0,
    k_index=8.2, a_index=180.0, dst_index=-220.0,  # Severe storm!
    utc_hour=2.0, day_of_year=80, latitude=65.0, longitude=25.0,
    thunderstorm_activity=0.0, precipitation_rate=0.0,
    frequency_mhz=7.1
)

storm_conditions = physics_calc.calculate_all_effects(storm_drivers)
print(f"SFI: {storm_drivers.sfi:.1f}, K-index: {storm_drivers.k_index:.1f}")
print(f"→ MUF: {storm_conditions.muf_mhz:.1f} MHz (reduced!)")
print(f"→ D-layer absorption: {storm_conditions.d_layer_absorption_db:.1f} dB (auroral)")
print(f"→ Propagation: {storm_conditions.propagation_mode.value}")
print(f"→ Delay spread: {storm_conditions.multipath_delay_spread_ms:.2f} ms")
print(f"→ Doppler spread: {storm_conditions.doppler_spread_hz:.2f} Hz")
print(f"→ QRN: {storm_conditions.dominant_qrn_type.value}")
print(f"→ Effective SNR: {storm_conditions.effective_snr_db:.1f} dB")

print("\n💡 Key Insight: All effects (MUF, absorption, propagation, QRN) are")
print("   COUPLED through the same physical drivers (SFI, K-index, etc.)")

# ============================================================================
# CELL 3: Demonstrate Continuous Variation
# ============================================================================

print("\n" + "=" * 80)
print("CONTINUOUS VARIATION (Anti-Overfitting)")
print("=" * 80)

from continuous_distributions import create_k_index_dist, create_solar_flux_dist

# Show continuous K-index sampling
print("\n### Severe Storm K-index: Continuous vs Discrete ###")
print("\nDiscrete bins (bad - model memorizes):")
print("  K ∈ {7, 8, 9}  ← Only 3 values!")

print("\nContinuous sampling (good - model generalizes):")
k_dist = create_k_index_dist('severe_storm')
samples = [k_dist.sample() for _ in range(10)]
print(f"  K = {[f'{s:.2f}' for s in samples]}")
print("  ← Every sample is unique!")

# Show SFI continuous variation
print("\n### Solar Flux: Continuous Sampling ###")
sfi_dist = create_solar_flux_dist('excellent')
samples = [sfi_dist.sample() for _ in range(10)]
print(f"SFI samples: {[f'{s:.1f}' for s in samples]}")
print(f"Mean: {sfi_dist.mean():.1f}, Std: {sfi_dist.std():.1f}")

print("\n✓ Continuous variation prevents overfitting!")

# ============================================================================
# CELL 4: Demonstrate Scenario Generation
# ============================================================================

print("\n" + "=" * 80)
print("SCENARIO-BASED GENERATION")
print("=" * 80)

# Create scenario library
scenario_lib = ScenarioLibrary()

print(f"\nAvailable scenarios: {len(scenario_lib.templates)}")
for name in scenario_lib.get_all_template_names():
    template = scenario_lib.get_template(name)
    print(f"  - {name}: {template.description} (weight={template.weight})")

# Generate instances from one scenario
print("\n### Generating 3 instances of 'excellent' scenario ###")
print("Notice: Each instance has DIFFERENT physics (continuous variation)\n")

for i in range(3):
    drivers = scenario_lib.generate_scenario_instance('excellent', seed=i)
    conditions = physics_calc.calculate_all_effects(drivers)

    print(f"Instance {i+1}:")
    print(f"  SFI={drivers.sfi:.1f}, K={drivers.k_index:.2f}, "
          f"Freq={drivers.frequency_mhz:.3f} MHz, Time={drivers.utc_hour:.1f}h")
    print(f"  → MUF={conditions.muf_mhz:.1f} MHz, SNR={conditions.effective_snr_db:.1f} dB, "
          f"Prop={conditions.propagation_mode.value}")

# Generate balanced-realistic batch
print("\n### Balanced-Realistic Batch Distribution ###")
batch = scenario_lib.generate_balanced_realistic_batch(1000, for_test=False, seed=42)

# Analyze distribution
k_ranges = {'K<2': 0, '2≤K<4': 0, '4≤K<6': 0, 'K≥6': 0}
for drivers in batch:
    if drivers.k_index < 2:
        k_ranges['K<2'] += 1
    elif drivers.k_index < 4:
        k_ranges['2≤K<4'] += 1
    elif drivers.k_index < 6:
        k_ranges['4≤K<6'] += 1
    else:
        k_ranges['K≥6'] += 1

print("K-index distribution (1000 samples, training):")
for range_name, count in k_ranges.items():
    print(f"  {range_name}: {count/10:.1f}%")

# Test distribution (harder)
test_batch = scenario_lib.generate_balanced_realistic_batch(1000, for_test=True, seed=42)
test_k_ranges = {'K<2': 0, '2≤K<4': 0, '4≤K<6': 0, 'K≥6': 0}
for drivers in test_batch:
    if drivers.k_index < 2:
        test_k_ranges['K<2'] += 1
    elif drivers.k_index < 4:
        test_k_ranges['2≤K<4'] += 1
    elif drivers.k_index < 6:
        test_k_ranges['4≤K<6'] += 1
    else:
        test_k_ranges['K≥6'] += 1

print("\nK-index distribution (1000 samples, test - HARDER):")
for range_name, count in test_k_ranges.items():
    print(f"  {range_name}: {count/10:.1f}%")

print("\n💡 Test set has MORE severe conditions (K≥6) than training!")

# ============================================================================
# CELL 5: Create Physics-Constrained Dataset for Training
# ============================================================================

print("\n" + "=" * 80)
print("PHYSICS-CONSTRAINED DATASET FOR TRAINING")
print("=" * 80)

# Create datasets
print("\n### Creating Training Dataset ###")
physics_train_dataset = PhysicsConstrainedDataset(
    num_samples=10000,  # Start with 10K for demo, use 200K+ for real training
    signal_generator=signal_gen,
    sample_rate=48000,
    for_test=False,  # Training distribution
    seed=42,
    enable_real_signal_augmentation=False  # Set True if you have real signals
)

print("\n### Creating Validation Dataset ###")
physics_val_dataset = PhysicsConstrainedDataset(
    num_samples=2000,
    signal_generator=signal_gen,
    sample_rate=48000,
    for_test=False,  # Same distribution as training
    seed=1042
)

print("\n### Creating Test Dataset (HARDER) ###")
physics_test_dataset = PhysicsConstrainedDataset(
    num_samples=2000,
    signal_generator=signal_gen,
    sample_rate=48000,
    for_test=True,  # Harder distribution (more storms, no excellent)
    seed=2042
)

# Test loading samples
print("\n### Testing Sample Generation ###")
sample_iq, sample_labels = physics_train_dataset[0]

print(f"\nSample shape: {sample_iq.shape}")
print(f"\nPhysical state:")
print(f"  SFI: {sample_labels['sfi']:.1f}")
print(f"  K-index: {sample_labels['k_index']:.2f}")
print(f"  Frequency: {sample_labels['frequency_mhz']:.3f} MHz")
print(f"  Time: {sample_labels['utc_hour']:.1f}h UTC")
print(f"  Latitude: {sample_labels['latitude']:.1f}°")

print(f"\nDerived conditions:")
print(f"  MUF: {sample_labels['muf_mhz']:.1f} MHz")
print(f"  D-layer absorption: {sample_labels['d_layer_absorption_db']:.1f} dB")
print(f"  Propagation: {sample_labels['propagation_mode']}")
print(f"  QRN type: {sample_labels['dominant_qrn_type']}")
print(f"  Delay spread: {sample_labels['delay_spread_ms']:.2f} ms")
print(f"  Doppler spread: {sample_labels['doppler_spread_hz']:.2f} Hz")

print(f"\nGround truth:")
print(f"  Pattern ID: {sample_labels['pattern_id']}")
print(f"  Frequency pair: {sample_labels['frequency_pair']}")
print(f"  SNR: {sample_labels['snr_db']:.1f} dB")

# Create DataLoaders
print("\n### Creating DataLoaders ###")
physics_train_loader = DataLoader(
    physics_train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Increase for faster loading
    pin_memory=True if torch.cuda.is_available() else False
)

physics_val_loader = DataLoader(
    physics_val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)

physics_test_loader = DataLoader(
    physics_test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f"✓ Train loader: {len(physics_train_loader)} batches")
print(f"✓ Val loader: {len(physics_val_loader)} batches")
print(f"✓ Test loader: {len(physics_test_loader)} batches")

# Test batch loading
print("\n### Testing Batch Loading ###")
for batch_iq, batch_labels in physics_train_loader:
    print(f"Batch IQ shape: {batch_iq.shape}")
    print(f"Batch SNR range: {batch_labels['snr_db'].min():.1f} to {batch_labels['snr_db'].max():.1f} dB")
    print(f"Batch K-index range: {batch_labels['k_index'].min():.2f} to {batch_labels['k_index'].max():.2f}")
    print(f"Batch propagation modes: {set([batch_labels['propagation_mode'][i] for i in range(len(batch_labels['propagation_mode']))])}")
    break

print("\n" + "=" * 80)
print("✓ PHYSICS-CONSTRAINED DATASET READY FOR TRAINING!")
print("=" * 80)

print("\n### Key Advantages Over Random Generation ###")
print("1. ✓ Physics coupling: All effects derived from same drivers")
print("2. ✓ Continuous variation: Prevents overfitting to discrete bins")
print("3. ✓ Realistic scenarios: 9 fundamental + many variations")
print("4. ✓ Balanced weighting: Rare conditions oversampled")
print("5. ✓ Harder test set: More severe conditions than training")

print("\n### Next Steps ###")
print("1. Replace CascadeDataset with PhysicsConstrainedDataset in training loops")
print("2. Train Stage 1 (IQ Encoder) with physics_train_loader")
print("3. Train Stage 2 (Experts) with physics-coupled channel metadata")
print("4. Train Stage 3 (Decoder) with realistic propagation scenarios")
print("5. Evaluate on physics_test_loader (harder distribution)")

print("\n### Optional: Real Signal Augmentation ###")
print("To apply physics to real HF recordings:")
print("1. Collect real signals from WebSDR/KiwiSDR")
print("2. Set enable_real_signal_augmentation=True")
print("3. Provide real_signal_path='/path/to/real_signals.npz'")
print("4. Model will apply physics-based propagation to real recordings")

# ============================================================================
# CELL 6: Comparison - Random vs Physics-Based
# ============================================================================

print("\n" + "=" * 80)
print("COMPARISON: Random vs Physics-Based Generation")
print("=" * 80)

print("\n### RANDOM GENERATION (OLD) ###")
print("Problems:")
print("  ❌ QRN and propagation independent (unrealistic)")
print("  ❌ Discrete parameter bins (overfitting)")
print("  ❌ No physical constraints (impossible combinations)")
print("  ❌ Uniform distribution (rare conditions undersampled)")
print("\nExample:")
print("  K-index=8 (severe storm) + QRN='quiet' + Prop='awgn'")
print("  → IMPOSSIBLE! High K causes auroral hiss and multipath!")

print("\n### PHYSICS-BASED GENERATION (NEW) ###")
print("Advantages:")
print("  ✓ QRN, propagation, absorption COUPLED via physics")
print("  ✓ Continuous variation (every sample unique)")
print("  ✓ Physics constraints (only realistic combinations)")
print("  ✓ Balanced-realistic weighting (rare conditions oversampled)")
print("\nExample:")
print("  K-index=8.2 → Automatically causes:")
print("    - Auroral hiss QRN (at high latitudes)")
print("    - Dense multipath propagation")
print("    - High D-layer absorption")
print("    - Low SNR")
print("  → ALL COUPLED through same physical state!")

print("\n" + "=" * 80)
print("✓ Use PhysicsConstrainedDataset for all CASCADE training!")
print("=" * 80)
