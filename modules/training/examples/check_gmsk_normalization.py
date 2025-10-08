#!/usr/bin/env python3
"""Check if GMSK filtered signal reaches ±1 after normalization."""

import numpy as np
import sys
sys.path.insert(0, '/workspaces/cascade/modules/training')
from src.signal_generator.gmsk import generate_gaussian_filter

# Simple test pattern with long runs
pattern_bits = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# GMSK parameters
SAMPLE_RATE = 48000
PATTERN_SYMBOL_RATE = 75
samples_per_symbol = SAMPLE_RATE // PATTERN_SYMBOL_RATE

# Convert to NRZ
nrz = 2 * pattern_bits.astype(np.float32) - 1
print(f"NRZ values: {nrz}")

# Upsample
nrz_upsampled = np.zeros(len(pattern_bits) * samples_per_symbol, dtype=np.float32)
nrz_upsampled[::samples_per_symbol] = nrz

# Apply Gaussian filter
gaussian_filter = generate_gaussian_filter(BT=0.3, span_symbols=4,
                                          samples_per_symbol=samples_per_symbol)
filtered = np.convolve(nrz_upsampled, gaussian_filter, mode='same')

print(f"\nBefore normalization:")
print(f"  Min: {np.min(filtered):.4f}")
print(f"  Max: {np.max(filtered):.4f}")
print(f"  Range: {np.max(np.abs(filtered)):.4f}")

# Normalize
max_abs = np.max(np.abs(filtered))
if max_abs > 0:
    filtered_norm = filtered / max_abs
else:
    filtered_norm = filtered

print(f"\nAfter normalization:")
print(f"  Min: {np.min(filtered_norm):.4f}")
print(f"  Max: {np.max(filtered_norm):.4f}")
print(f"  Range: {np.max(np.abs(filtered_norm)):.4f}")

# Check values at symbol centers for steady-state symbols
print(f"\nFiltered values at symbol centers:")
for i in range(len(pattern_bits)):
    center_idx = int((i + 0.5) * samples_per_symbol)
    if center_idx < len(filtered_norm):
        bit = pattern_bits[i]
        expected = -1 if bit == 0 else +1
        actual = filtered_norm[center_idx]
        print(f"  Symbol {i:2d}: bit={bit}, filtered={actual:+.4f} (expect {expected:+.4f})")

print(f"\n{'='*70}")
print("Analysis:")
print("  - Long steady runs (symbols 0-7, 8-15) should approach ±1.0")
print("  - Transition symbols (around index 7-8) will be between -1 and +1")
print("  - If normalization is working, max should be exactly 1.0")
