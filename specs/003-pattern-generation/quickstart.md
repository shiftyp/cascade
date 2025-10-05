# Quickstart: Pattern Generation (2-FSK Architecture)

## Overview

CASCADE uses **2-FSK** (2 adjacent tones per pattern) for optimal low-SNR performance:
- **75 patterns (59%)** at λ=0 (BPSK) - maximum robustness for emergency communications
- **Average λ: 0.08-0.10** (vs 0.17 with 4-FSK) - 47% improvement
- **Modular scaling**: Transmit 1×, 2×, 4×, or 8× 2-FSK patterns based on power budget

**Equipment Throughput** (corrected with realistic symbol rates):

| Equipment | Cost | Symbol Rate | Patterns TX | Throughput |
|-----------|------|-------------|-------------|------------|
| QRP (QMX 5W) | $180 | 200 sym/s | 1×2-FSK | 44 bps |
| Modern (IC-7300) | $1400 | 100 sym/s | 2-4×2-FSK | 44-88 bps |
| QMX+amp (50W) | $330 | 200 sym/s | 4×2-FSK | 175 bps |
| Premium SDR | $2000+ | 300 sym/s | 4×2-FSK | 261 bps |

**Best value**: QMX ($180) provides 44-175 bps at 1/8 the cost of IC-7300!

## Platform Detection

```bash
cd /workspaces/cascade

# Check your CPU capabilities
python -m modules.training.patterns platform

# Shows: cores, RAM, optimal worker count, suggested iterations
```

## Generate 64-Pattern Set

### Local (Auto-Tuned)
```bash
# Generate with auto-tuning (recommended)
python -m modules.training.patterns generate \
    --count 64 \
    --seed 42 \
    --output modules/training/data/cascade_patterns_64.bin

# Expected: 18-24 hours, ~19 KB file
# Auto-detects CPU: uses optimal workers and iterations
```

### Distributed (Fly.io - Better Quality)
```bash
# 32 workers for better quality (+1.5 dB), ~$6
python -m modules.training.patterns generate \
    --count 64 \
    --distributed \
    --workers 32 \
    --seed 42

# Expected: 18-24 hours, -39.5 dB quality
```

### Validate
```bash
python -m modules.training.patterns validate \
    modules/training/data/cascade_patterns_64.bin

# Expected output:
# ✓ 64 patterns loaded
# ✓ All 2,016 pairs < -37.5 dB
# ✓ Min: ~-40 dB, Max: ~-37.8 dB
# ✓ PASS
```

## Generate 128-Pattern Set (Production)

### Local High-End CPU (RECOMMENDED - Best Quality)
```bash
# Optimal: 8 trials × 400K iterations (depth strategy)
# Hardware: Core Ultra 7 265K, Ryzen 9, similar 8+ core CPUs
python -m modules.training.patterns generate \
    --count 128 \
    --trials 8 \
    --iterations 400000 \
    --seed 42

# Expected (2-FSK architecture):
# - Time: 72-96 hours (3-4 days, leave running over long weekend)
# - Cost: $0 (free!)
# - Quality: -42.6 to -43 dB separation
# - Average λ: 0.08-0.10 (exceptional - 47% lower than 4-FSK!)
# - BPSK patterns: 75 (59%) achieve λ=0 - maximum low-SNR robustness
```

### Local Mid-Range CPU (Balanced)
```bash
# Balanced: 8 trials × 200K iterations (2-FSK)
# Hardware: 6-8 core desktop
python -m modules.training.patterns generate \
    --count 128 \
    --trials 8 \
    --iterations 200000 \
    --seed 42

# Expected: 36-48 hours, $0 cost, -41.5 dB quality, λ=0.12-0.14
```

### Cloud (Fast Alternative - No Local Hardware)
```bash
# Breadth strategy: 32 workers × 100K iterations (2-FSK)
# Best for: Users without capable local CPU or need faster results
python -m modules.training.patterns generate \
    --count 128 \
    --distributed \
    --workers 32 \
    --iterations 100000 \
    --seed 42

# Expected: 30-40 hours, $9.60 cost, -40.7 dB quality, λ=0.14-0.16
# Note: Faster but lower quality and costs money
```

### Validate
```bash
python -m modules.training.patterns validate cascade_patterns_128.bin

# Shows:
# - Orthogonality results (all pairs < -37.5 dB)
# - λ distribution (discovered minimum complexity)
# - Phase robustness (degradation under HF distortion)
```

## Use in Phase 0 Vetting

```python
# Load 64 patterns for faster vetting
patterns = load_pattern_file("cascade_patterns_64.bin")

# Run vetting with 64 patterns
# (validates architecture with smaller set)
```
