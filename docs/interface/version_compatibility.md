# CASCADE Version Compatibility System

CASCADE ensures long-term interoperability through a version-based compatibility system where newer models are supersets of older ones, allowing mixed-version networks to communicate seamlessly.

## Design Philosophy

**Key principle**: Frozen protocol elements + evolvable model behavior

**Frozen forever** (ensures interoperability):
- 64 orthogonal patterns (generated once, never changed)
- Base symbol duration (50ms for messages, 160ms/500ms for beacons)
- Emergency frequencies [468, 1093 Hz]
- Normal beacon frequencies [78, 234, 1718, 1953 Hz]
- Message tone frequencies [0, 312, 625, 937, 1250, 1562, 1875, 2187 Hz]

**Can evolve** (negotiated via kernel version):
- Model weights and architectures
- Kernel formats and compression
- Constellation adaptation strategies
- FEC schemes
- Advanced features (mesh routing, etc.)

## Version Number in Kernel

All kernels include a 4-bit version field (16 possible versions):

```python
# Standard 64-bit kernel
kernel_with_version = {
    'version': 4 bits,             # 0-15 (current version)
    'modulation_pref': 3 bits,
    'hardware_tier': 2 bits,
    'capacity_users': 5 bits,
    'snr_floor': 5 bits,
    'interference_map': 8 bits,
    'frequency_pref': 8 bits,
    'timing_offset': 4 bits,
    'noise_floor': 5 bits,
    'power_request': 4 bits,
    'features': 4 bits,
    'reserved': 12 bits
}
# Total: 64 bits

# Extended 256-bit kernel
extended_kernel_with_version = {
    'version': 4 bits,             # Same version field
    'extended_fields': 252 bits     # Additional capabilities
}
```

## Version Evolution Strategy

### v1.0 (Initial Release - 2026)

**Capabilities:**
```python
v1_0 = {
    'patterns': 'walsh_hadamard_64',        # Fixed
    'modulation': '8-QAM → BPSK collapse',
    'kernel_sizes': [64],                   # Only standard kernel
    'fec': 'LDPC rate 0.3-0.8',
    'constellation': 'parametric_collapse',
    'relay': 'basic (3 hops max)',
    'sync': 'blind_pattern_correlation',
    'max_users': 50
}
```

### v2.0 (Enhanced - 2027+)

**New capabilities** (backward compatible):
```python
v2_0 = {
    'patterns': 'walsh_hadamard_64',        # SAME (frozen)
    'modulation': '16-QAM → BPSK collapse', # Enhanced! Can still do 8-QAM
    'kernel_sizes': [16, 64, 256],          # Extended kernels added
    'fec': 'Polar + LDPC',                  # Additional FEC, still supports LDPC
    'constellation': 'neural_learned',      # Improved, but can fall back
    'relay': 'mesh-aware (5 hops)',         # Enhanced, compatible with basic
    'sync': 'improved_blind_sync',          # Better, backward compatible
    'max_users': 80,                        # Improved Signal Expert

    # v2.0 model contains v1.0 mode
    'v1_compatibility_mode': True           # Can operate as v1.0 when needed
}
```

## Backward Compatibility Training

v2.0 models are trained to operate in both v2.0 and v1.0 modes:

```python
def train_v2_with_v1_compatibility():
    """Train v2.0 to be superset of v1.0"""

    for batch in training_data:
        # 50% train with v2.0 features
        if random.random() < 0.5:
            mode = 'v2_full'
            constraints = {
                'allow_16qam': True,
                'use_extended_kernels': True,
                'use_polar_codes': True,
                'advanced_relay': True
            }

        # 50% train with v1.0 constraints (compatibility mode)
        else:
            mode = 'v1_compatibility'
            constraints = {
                'allow_16qam': False,       # Limit to 8-QAM
                'use_extended_kernels': False,  # Only 64-bit kernels
                'use_polar_codes': False,   # Only LDPC
                'advanced_relay': False     # Basic relay only
            }

        # Train model to work under both constraint sets
        output = model(signal, mode=mode, constraints=constraints)
        loss = compute_loss(output, ground_truth)
        optimizer.step(loss)
```

**Result**: v2.0 model can seamlessly operate as v1.0 when communicating with v1.0 peers.

## Version Negotiation Protocol

**When two stations exchange kernels:**

```python
def negotiate_version(my_kernel, their_kernel):
    """Determine protocol level for communication"""

    my_version = my_kernel.version      # e.g., 2
    their_version = their_kernel.version  # e.g., 1

    # Use lowest common version
    protocol_version = min(my_version, their_version)

    if protocol_version < my_version:
        # I'm newer - activate compatibility mode
        model.set_mode(f'v{protocol_version}_compatibility')
        log(f"Using v{protocol_version} mode for compatibility with peer")

    return {
        'protocol_version': protocol_version,
        'my_mode': 'compatibility' if protocol_version < my_version else 'full',
        'features_available': get_features_for_version(protocol_version)
    }
```

**Example exchange:**

```markdown
**Alice (v1.0) beacons** → No version in beacon (beacons minimal)
**Bob (v2.0) ACKs** with kernel containing version=2
**Alice sends kernel** with version=1

**Bob sees**: Alice is v1.0
**Bob activates**: v1.0 compatibility mode
  - Limits to 8-QAM (no 16-QAM)
  - Uses 64-bit kernels only (not 256-bit)
  - Uses LDPC FEC only (not Polar codes)
  - All communication at v1.0 level

**Alice sees**: Bob is v2.0 (but doesn't matter, Alice can only do v1.0)
**Alice operates**: Normal v1.0 behavior

**Result**: Perfect communication at v1.0 level
```

## Version Compatibility Matrix

| Your Version | Their Version | Result | Mode |
|--------------|---------------|--------|------|
| 1.0 | 1.0 | v1.0 protocol | Full features (for v1.0) |
| 1.0 | 2.0 | v1.0 protocol | v1.0 full / v2.0 compatibility |
| 2.0 | 1.0 | v1.0 protocol | v2.0 compatibility / v1.0 full |
| 2.0 | 2.0 | v2.0 protocol | Full features (for v2.0) |
| 2.0 | 3.0 | v2.0 protocol | v2.0 full / v3.0 compatibility |
| 1.0 | 3.0 | v1.0 protocol | v1.0 full / v3.0 compatibility |

**Rule**: Always use minimum version, newer model activates compatibility mode.

## Future Version Planning

**v3.0 (Hypothetical - 2028+):**
- Must train on v1.0 AND v2.0 modes
- Can communicate with both
- Adds new features only when both stations are v3.0

**Long-term (v10+):**
- May drop v1.0 compatibility (after 5+ years, most network upgraded)
- Document breaking changes clearly
- Provide upgrade path

## See Also

- **[Signal Specification](../protocol/signal_specification.md)** - Frozen protocol elements
- **[Model Updates](../deployment/updates.md)** - How updates are deployed
- **[Training Strategy](../training/README.md)** - How compatibility modes are trained
