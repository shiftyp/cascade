# CASCADE Training Module

Clean, organized structure for CASCADE model training with physics-based channel simulation.

## Directory Structure

```
training/
├── core/                           # Main training code
│   ├── cascade.ipynb              # Complete training notebook
│   ├── physics_coupling.py        # Physics-coupled parameter calculations
│   ├── scenarios.py                # 9 scenario templates (excellent → storm)
│   ├── continuous_distributions.py # Anti-overfitting distributions
│   ├── physics_constrained_dataset.py  # PyTorch Dataset with physics
│   └── *.md                       # Documentation
│
├── examples/                      # Visualization examples
│   └── visualize_physics_scenarios.py  # 3x3 physics demonstration
│
├── src/                           # Source modules
│   ├── signal_generator/         # CASCADE signal generation
│   └── channel_simulator/        # Basic channel simulation
│
├── tests/                         # Unit tests
├── patterns/                      # Walsh-Hadamard patterns
└── datasets/                      # Generated datasets
```

## Quick Start

### 1. Open Training Notebook

```bash
cd core
jupyter notebook cascade.ipynb
```

Run cells 1-16 to initialize physics-based datasets.

### 2. Training

The notebook provides complete training for all stages:

- **Stage 1**: IQ Encoder (autoencoder bootstrap)
- **Stage 2**: Expert Networks (5 experts)
- **Stage 3**: Integration Decoder

All training uses `physics_train_loader`, `physics_val_loader`, `physics_test_loader`.

### 3. Generate Visualizations

```bash
cd examples
python visualize_physics_scenarios.py
```

Creates a 3x3 grid showing:
- Row 1: Different modulations (BPSK, QPSK, 8-PSK, 16-APSK) - Clean baseline
- Row 2: Excellent conditions (SNR = +20 dB) - Strong signals
- Row 3: Poor conditions (SNR = -10 dB) - Weak signals buried in noise

## Physics-Based System

### Key Features

✅ **Physics Coupling**: All effects (QRN, propagation, absorption) derived from same drivers
✅ **Continuous Variation**: Prevents overfitting to discrete bins
✅ **Realistic Scenarios**: 9 fundamental templates with variations
✅ **Balanced Weighting**: Rare conditions oversampled
✅ **Harder Test Set**: More severe conditions than training

### How It Works

```python
# 1. Core physical state
drivers = CorePhysicalDrivers(
    sfi=220.0,           # Solar flux
    k_index=1.2,         # Geomagnetic index
    thunderstorm_activity=0.0,
    frequency_mhz=14.1
)

# 2. Calculate ALL coupled effects
conditions = physics_calc.calculate_all_effects(drivers)
# → conditions.propagation_mode (from K-index + freq/MUF)
# → conditions.qrn_components (from K-index + thunderstorms)
# → conditions.d_layer_absorption_db (from time + K-index)
# ALL COUPLED!

# 3. Apply to signal
received = apply_physics_channel(clean, drivers, conditions)
```

## Training Data

### Datasets (in core/)

```python
physics_train_dataset = PhysicsConstrainedDataset(
    num_samples=200000,  # Recommended for full training
    for_test=False       # Training distribution
)

physics_test_dataset = PhysicsConstrainedDataset(
    num_samples=20000,
    for_test=True        # HARDER distribution (more storms)
)
```

### Scenario Distribution

**Training (Balanced-Realistic):**
- Excellent: 15%
- Good: 25%
- Moderate: 30%
- Poor: 15%
- Storms: 9% (oversampled vs 2-3% reality)
- Others: 6%

**Test (Harder):**
- Excellent: 0% (no easy cases!)
- Severe Storms: 10% (10× training)
- Major Storms: 15% (5× training)
- More challenging conditions overall

## What Was Removed

Cleaned up old code:
- ❌ Old HFChannelSimulator (random generation)
- ❌ CascadeDataset, HybridCascadeDataset (legacy)
- ❌ 40+ old visualization scripts and PNGs
- ❌ Comparison/deprecation code
- ❌ Real-world data references (not available)

## GPU Requirements

**Model Size**: 2.5M parameters

**Minimum**: RTX 3060 (300-700 MB VRAM)
**Recommended**: RTX 3070 or better

**Training Time**: 12-15 hours on RTX 3060 (200K samples, 3 stages)

## Documentation

See `core/` folder for detailed documentation:
- `PHYSICS_SCENARIO_README.md` - Complete system documentation
- `IMPLEMENTATION_COMPLETE.md` - Implementation details
- `INTEGRATION_SUMMARY.md` - Integration into notebook

## Examples

### Basic Training Loop

```python
# Stage 1: IQ Encoder
iq_trainer = IQEncoderTrainer(device='cuda')
iq_trainer.train(physics_train_loader, physics_val_loader, epochs=50)

# Stage 2: Experts
for expert_name in ['QRN', 'Signal', 'Timing', 'Channel', 'QRM']:
    expert_trainer = ExpertTrainer(expert_name, expert, iq_encoder)
    expert_trainer.train(physics_train_loader, physics_val_loader, epochs=30)

# Stage 3: Decoder
decoder_trainer = IntegrationDecoderTrainer(iq_encoder, experts)
decoder_trainer.train(physics_train_loader, physics_val_loader, epochs=50)

# Evaluate on HARDER test set
evaluator = CascadeEvaluator(model)
test_metrics = evaluator.evaluate(physics_test_loader)
```

### Visualization

```bash
cd examples
python visualize_physics_scenarios.py
# Creates: physics_scenarios_3x3.png
```

## Contributing

When adding new features:
1. Put core training code in `core/`
2. Put examples/demos in `examples/`
3. Keep visualization scripts separate from training
4. Update this README

## Support

- Notebook: `core/cascade.ipynb` has complete examples
- Docs: `core/*.md` for detailed documentation
- Issues: Check import paths if moving files

---

**Status**: ✅ Ready for Training
**Last Updated**: 2025-10-08
**Physics System**: Fully Integrated
