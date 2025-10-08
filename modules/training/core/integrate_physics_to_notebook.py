#!/usr/bin/env python3
"""
Integrate physics-based scenario system into cascade.ipynb
"""

import json
import sys

def create_code_cell(source_lines):
    """Create a Jupyter code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def create_markdown_cell(source_lines):
    """Create a Jupyter markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }

# Physics system cells
physics_cells = []

# Cell 1: Markdown header
physics_cells.append(create_markdown_cell([
    "# Physics-Based Scenario System\n",
    "\n",
    "**Replaces random QRN/propagation with physics-coupled, continuously-varying scenarios.**\n",
    "\n",
    "Key features:\n",
    "- ✓ **Physics coupling**: All effects (QRN, propagation, absorption) derived from same drivers\n",
    "- ✓ **Continuous variation**: Prevents overfitting to discrete bins\n",
    "- ✓ **Realistic scenarios**: 9 fundamental templates with many variations\n",
    "- ✓ **Balanced-realistic**: Rare conditions oversampled for robust learning\n",
    "- ✓ **Harder test set**: More severe conditions than training"
]))

# Cell 2: Import physics modules
physics_cells.append(create_code_cell([
    "print(\"=\" * 80)\n",
    "print(\"PHYSICS-CONSTRAINED DATASET SYSTEM\")\n",
    "print(\"=\" * 80)\n",
    "\n",
    "# Import physics-based components\n",
    "from physics_coupling import (\n",
    "    CorePhysicalDrivers, CoupledPhysicsCalculator,\n",
    "    PropagationMode, QRNType\n",
    ")\n",
    "from scenarios import ScenarioLibrary, ScenarioType\n",
    "from physics_constrained_dataset import PhysicsConstrainedDataset\n",
    "\n",
    "print(\"✓ Physics coupling module imported\")\n",
    "print(\"✓ Scenario library imported\")\n",
    "print(\"✓ Physics-constrained dataset imported\")\n"
]))

# Cell 3: Demonstrate physics coupling
physics_cells.append(create_code_cell([
    "print(\"\\n\" + \"=\" * 80)\n",
    "print(\"PHYSICS COUPLING DEMONSTRATION\")\n",
    "print(\"=\" * 80)\n",
    "\n",
    "# Create physics calculator\n",
    "physics_calc = CoupledPhysicsCalculator(seed=42)\n",
    "\n",
    "# Example 1: Excellent conditions\n",
    "print(\"\\n### Example 1: Excellent Conditions ###\")\n",
    "excellent_drivers = CorePhysicalDrivers(\n",
    "    sfi=220.0, sunspot_number=160.0,\n",
    "    k_index=1.1, a_index=4.0, dst_index=-8.0,\n",
    "    utc_hour=14.0, day_of_year=180, latitude=40.0, longitude=-75.0,\n",
    "    thunderstorm_activity=0.0, precipitation_rate=0.0,\n",
    "    frequency_mhz=14.1\n",
    ")\n",
    "\n",
    "excellent_conditions = physics_calc.calculate_all_effects(excellent_drivers)\n",
    "print(f\"SFI: {excellent_drivers.sfi:.1f}, K-index: {excellent_drivers.k_index:.1f}\")\n",
    "print(f\"→ MUF: {excellent_conditions.muf_mhz:.1f} MHz\")\n",
    "print(f\"→ D-layer absorption: {excellent_conditions.d_layer_absorption_db:.1f} dB\")\n",
    "print(f\"→ Propagation: {excellent_conditions.propagation_mode.value}\")\n",
    "print(f\"→ QRN: {excellent_conditions.dominant_qrn_type.value}\")\n",
    "print(f\"→ Effective SNR: {excellent_conditions.effective_snr_db:.1f} dB\")\n",
    "\n",
    "# Example 2: Severe geomagnetic storm\n",
    "print(\"\\n### Example 2: Severe Geomagnetic Storm ###\")\n",
    "storm_drivers = CorePhysicalDrivers(\n",
    "    sfi=120.0, sunspot_number=80.0,\n",
    "    k_index=8.2, a_index=180.0, dst_index=-220.0,\n",
    "    utc_hour=2.0, day_of_year=80, latitude=65.0, longitude=25.0,\n",
    "    thunderstorm_activity=0.0, precipitation_rate=0.0,\n",
    "    frequency_mhz=7.1\n",
    ")\n",
    "\n",
    "storm_conditions = physics_calc.calculate_all_effects(storm_drivers)\n",
    "print(f\"SFI: {storm_drivers.sfi:.1f}, K-index: {storm_drivers.k_index:.1f}\")\n",
    "print(f\"→ MUF: {storm_conditions.muf_mhz:.1f} MHz (reduced!)\")\n",
    "print(f\"→ D-layer absorption: {storm_conditions.d_layer_absorption_db:.1f} dB (auroral)\")\n",
    "print(f\"→ Propagation: {storm_conditions.propagation_mode.value}\")\n",
    "print(f\"→ QRN: {storm_conditions.dominant_qrn_type.value}\")\n",
    "print(f\"→ Effective SNR: {storm_conditions.effective_snr_db:.1f} dB\")\n",
    "\n",
    "print(\"\\n💡 Key: All effects COUPLED through same physics (K=8.2 causes all degradation)\")\n"
]))

# Cell 4: Demonstrate continuous variation
physics_cells.append(create_code_cell([
    "print(\"\\n\" + \"=\" * 80)\n",
    "print(\"CONTINUOUS VARIATION (Anti-Overfitting)\")\n",
    "print(\"=\" * 80)\n",
    "\n",
    "from continuous_distributions import create_k_index_dist, create_solar_flux_dist\n",
    "\n",
    "# Show continuous K-index sampling\n",
    "print(\"\\n### Severe Storm K-index: Continuous vs Discrete ###\")\n",
    "print(\"\\nDISCRETE bins (bad - overfitting):\")\n",
    "print(\"  K ∈ {7, 8, 9}  ← Model memorizes only 3 values!\")\n",
    "\n",
    "print(\"\\nCONTINUOUS sampling (good - generalization):\")\n",
    "k_dist = create_k_index_dist('severe_storm')\n",
    "samples = [k_dist.sample() for _ in range(10)]\n",
    "print(f\"  K = {[f'{s:.2f}' for s in samples]}\")\n",
    "print(\"  ← Every sample unique! Prevents overfitting.\")\n",
    "\n",
    "# Show SFI continuous variation\n",
    "print(\"\\n### Solar Flux (Excellent Conditions) ###\")\n",
    "sfi_dist = create_solar_flux_dist('excellent')\n",
    "samples = [sfi_dist.sample() for _ in range(10)]\n",
    "print(f\"SFI samples: {[f'{s:.1f}' for s in samples]}\")\n",
    "print(f\"Mean: {sfi_dist.mean():.1f}, Std: {sfi_dist.std():.1f}\")\n",
    "\n",
    "print(\"\\n✓ Continuous variation prevents overfitting!\")\n"
]))

# Cell 5: Demonstrate scenario generation
physics_cells.append(create_code_cell([
    "print(\"\\n\" + \"=\" * 80)\n",
    "print(\"SCENARIO-BASED GENERATION\")\n",
    "print(\"=\" * 80)\n",
    "\n",
    "# Create scenario library\n",
    "scenario_lib = ScenarioLibrary()\n",
    "\n",
    "print(f\"\\n9 Fundamental Scenarios ({len(scenario_lib.templates)} templates total):\")\n",
    "for name in ['excellent', 'good', 'moderate', 'poor', 'geomagnetic_storm_minor', \n",
    "             'geomagnetic_storm_severe', 'high_atmospheric_noise', 'greyline', 'polar']:\n",
    "    if name in scenario_lib.templates:\n",
    "        template = scenario_lib.get_template(name)\n",
    "        print(f\"  {name}: weight={template.weight:.0%}\")\n",
    "\n",
    "# Generate instances\n",
    "print(\"\\n### 3 Instances of 'excellent' (each DIFFERENT) ###\")\n",
    "for i in range(3):\n",
    "    drivers = scenario_lib.generate_scenario_instance('excellent', seed=i)\n",
    "    conditions = physics_calc.calculate_all_effects(drivers)\n",
    "    print(f\"  {i+1}: SFI={drivers.sfi:.1f}, K={drivers.k_index:.2f}, \"\n",
    "          f\"SNR={conditions.effective_snr_db:.1f}dB, {conditions.propagation_mode.value}\")\n",
    "\n",
    "# Show distribution\n",
    "print(\"\\n### Batch Distribution Analysis ###\")\n",
    "batch = scenario_lib.generate_balanced_realistic_batch(1000, for_test=False, seed=42)\n",
    "k_ranges = {'Quiet (K<2)': 0, 'Unsettled (K=2-4)': 0, 'Active (K=4-6)': 0, 'Storm (K≥6)': 0}\n",
    "for d in batch:\n",
    "    if d.k_index < 2: k_ranges['Quiet (K<2)'] += 1\n",
    "    elif d.k_index < 4: k_ranges['Unsettled (K=2-4)'] += 1\n",
    "    elif d.k_index < 6: k_ranges['Active (K=4-6)'] += 1\n",
    "    else: k_ranges['Storm (K≥6)'] += 1\n",
    "\n",
    "print(\"Training distribution (1000 samples):\")\n",
    "for k, v in k_ranges.items():\n",
    "    print(f\"  {k}: {v/10:.1f}%\")\n",
    "\n",
    "# Test distribution\n",
    "test_batch = scenario_lib.generate_balanced_realistic_batch(1000, for_test=True, seed=42)\n",
    "test_k = {'Quiet (K<2)': 0, 'Unsettled (K=2-4)': 0, 'Active (K=4-6)': 0, 'Storm (K≥6)': 0}\n",
    "for d in test_batch:\n",
    "    if d.k_index < 2: test_k['Quiet (K<2)'] += 1\n",
    "    elif d.k_index < 4: test_k['Unsettled (K=2-4)'] += 1\n",
    "    elif d.k_index < 6: test_k['Active (K=4-6)'] += 1\n",
    "    else: test_k['Storm (K≥6)'] += 1\n",
    "\n",
    "print(\"\\nTest distribution (HARDER):\")\n",
    "for k, v in test_k.items():\n",
    "    print(f\"  {k}: {v/10:.1f}%\")\n",
    "print(\"  → More storms in test for robustness measurement!\")\n"
]))

# Cell 6: Create physics datasets
physics_cells.append(create_code_cell([
    "print(\"\\n\" + \"=\" * 80)\n",
    "print(\"PHYSICS-CONSTRAINED DATASETS FOR TRAINING\")\n",
    "print(\"=\" * 80)\n",
    "\n",
    "# Create datasets with physics-coupled scenarios\n",
    "print(\"\\nCreating datasets...\")\n",
    "\n",
    "physics_train_dataset = PhysicsConstrainedDataset(\n",
    "    num_samples=10000,  # Increase to 200K for full training\n",
    "    signal_generator=signal_gen,\n",
    "    sample_rate=48000,\n",
    "    for_test=False,  # Training distribution\n",
    "    seed=42\n",
    ")\n",
    "\n",
    "physics_val_dataset = PhysicsConstrainedDataset(\n",
    "    num_samples=2000,\n",
    "    signal_generator=signal_gen,\n",
    "    sample_rate=48000,\n",
    "    for_test=False,  # Same as training\n",
    "    seed=1042\n",
    ")\n",
    "\n",
    "physics_test_dataset = PhysicsConstrainedDataset(\n",
    "    num_samples=2000,\n",
    "    signal_generator=signal_gen,\n",
    "    sample_rate=48000,\n",
    "    for_test=True,  # HARDER distribution\n",
    "    seed=2042\n",
    ")\n",
    "\n",
    "print(f\"✓ Train: {len(physics_train_dataset)} samples\")\n",
    "print(f\"✓ Val: {len(physics_val_dataset)} samples\")\n",
    "print(f\"✓ Test: {len(physics_test_dataset)} samples (harder)\")\n",
    "\n",
    "# Test sample\n",
    "print(\"\\n### Sample from Physics Dataset ###\")\n",
    "sample_iq, sample_labels = physics_train_dataset[0]\n",
    "print(f\"IQ shape: {sample_iq.shape}\")\n",
    "print(f\"Physical state: SFI={sample_labels['sfi']:.1f}, K={sample_labels['k_index']:.2f}, \"\n",
    "      f\"Freq={sample_labels['frequency_mhz']:.3f}MHz\")\n",
    "print(f\"Derived: MUF={sample_labels['muf_mhz']:.1f}MHz, \"\n",
    "      f\"Absorption={sample_labels['d_layer_absorption_db']:.1f}dB\")\n",
    "print(f\"Channel: {sample_labels['propagation_mode']}, QRN={sample_labels['dominant_qrn_type']}, \"\n",
    "      f\"SNR={sample_labels['snr_db']:.1f}dB\")\n",
    "\n",
    "# Create DataLoaders\n",
    "print(\"\\n### Creating DataLoaders ###\")\n",
    "physics_train_loader = DataLoader(\n",
    "    physics_train_dataset, batch_size=32, shuffle=True,\n",
    "    num_workers=4, pin_memory=torch.cuda.is_available()\n",
    ")\n",
    "physics_val_loader = DataLoader(\n",
    "    physics_val_dataset, batch_size=32, shuffle=False,\n",
    "    num_workers=4, pin_memory=torch.cuda.is_available()\n",
    ")\n",
    "physics_test_loader = DataLoader(\n",
    "    physics_test_dataset, batch_size=32, shuffle=False,\n",
    "    num_workers=4, pin_memory=torch.cuda.is_available()\n",
    ")\n",
    "\n",
    "print(f\"✓ Train loader: {len(physics_train_loader)} batches\")\n",
    "print(f\"✓ Val loader: {len(physics_val_loader)} batches\")\n",
    "print(f\"✓ Test loader: {len(physics_test_loader)} batches\")\n",
    "\n",
    "print(\"\\n\" + \"=\" * 80)\n",
    "print(\"✓ PHYSICS-CONSTRAINED DATASETS READY!\")\n",
    "print(\"=\" * 80)\n",
    "print(\"\\nUse physics_train_loader / physics_val_loader in training loops below.\")\n",
    "print(\"Test on physics_test_loader (harder distribution) for true robustness.\")\n"
]))

# Cell 7: Comparison
physics_cells.append(create_markdown_cell([
    "## Comparison: Random vs Physics-Based Generation\n",
    "\n",
    "### ❌ Random Generation (CascadeDataset - OLD)\n",
    "\n",
    "**Problems:**\n",
    "- QRN and propagation **independent** (unrealistic)\n",
    "- **Discrete** parameter bins (overfitting)\n",
    "- No physics constraints (impossible combinations like K=8 + quiet QRN)\n",
    "- Uniform distribution (rare conditions undersampled)\n",
    "\n",
    "**Example:**\n",
    "```python\n",
    "# Randomly selected (WRONG!):\n",
    "K=8, QRN='quiet', Prop='awgn'\n",
    "# → IMPOSSIBLE! K=8 must cause auroral noise & multipath!\n",
    "```\n",
    "\n",
    "### ✓ Physics-Based (PhysicsConstrainedDataset - NEW)\n",
    "\n",
    "**Advantages:**\n",
    "- All effects **COUPLED** through same physical drivers\n",
    "- **Continuous** variation (every sample unique)\n",
    "- Physics constraints enforced (only realistic combinations)\n",
    "- Balanced-realistic weighting (rare conditions oversampled)\n",
    "- Harder test distribution for true robustness measurement\n",
    "\n",
    "**Example:**\n",
    "```python\n",
    "# Physics-coupled (CORRECT!):\n",
    "drivers = CorePhysicalDrivers(k_index=8.2, ...)\n",
    "# → Automatically causes:\n",
    "#   - Auroral hiss QRN (high lat)\n",
    "#   - Dense multipath\n",
    "#   - High absorption\n",
    "#   - Low SNR\n",
    "# → ALL COUPLED!\n",
    "```\n",
    "\n",
    "---\n",
    "\n",
    "**Use `physics_train_loader` and `physics_val_loader` in all training below!**"
]))

# Load notebook
with open('cascade.ipynb', 'r') as f:
    nb = json.load(f)

# Find insertion point (after HybridCascadeDataset)
hybrid_idx = None
for i, cell in enumerate(nb['cells']):
    if 'source' in cell:
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'class HybridCascadeDataset' in source:
            hybrid_idx = i
            break

if hybrid_idx is None:
    print("ERROR: Could not find HybridCascadeDataset cell")
    sys.exit(1)

# Insert physics cells after HybridCascadeDataset
insert_pos = hybrid_idx + 1
for i, cell in enumerate(physics_cells):
    nb['cells'].insert(insert_pos + i, cell)

# Save updated notebook
with open('cascade.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"✓ Successfully integrated physics system into cascade.ipynb")
print(f"✓ Added {len(physics_cells)} cells after position {hybrid_idx}")
print(f"✓ Total cells now: {len(nb['cells'])}")
print(f"\nNext: Open cascade.ipynb and use physics_train_loader in training loops!")
