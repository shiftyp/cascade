#!/usr/bin/env python3
"""Verify that the GMSK generator is using the 2-FSK pattern."""

import numpy as np
import sys
import pickle
sys.path.insert(0, '/workspaces/cascade/modules/training')
from src.signal_generator.gmsk import generate_gmsk_fsk

# Load a real pattern
with open('/workspaces/cascade/modules/training/patterns/tournament/pattern_0_2048.pkl', 'rb') as f:
    pattern_bits = pickle.load(f)

# Use first 20 bits for analysis
pattern = pattern_bits[:20]
print(f"Pattern first 20 bits: {pattern}")

# Generate GMSK signal
SAMPLE_RATE = 48000
PATTERN_SYMBOL_RATE = 75
tone_a = 1000.0
tone_b = 1020.0

iq_signal = generate_gmsk_fsk(
    pattern, tone_a, tone_b,
    sample_rate=SAMPLE_RATE,
    symbol_rate=PATTERN_SYMBOL_RATE
)

# Measure instantaneous frequency at symbol centers
samples_per_symbol = SAMPLE_RATE // PATTERN_SYMBOL_RATE  # 640

print(f"\nVerifying frequency matches pattern bits:")
print(f"Expected: bit=0 → {tone_a:.0f} Hz, bit=1 → {tone_b:.0f} Hz\n")

phase = np.unwrap(np.angle(iq_signal))
inst_freq = np.diff(phase) / (2 * np.pi) * SAMPLE_RATE

matches = 0
mismatches = 0

for i, bit in enumerate(pattern):
    # Sample frequency at symbol center
    sample_idx = int((i + 0.5) * samples_per_symbol)

    if sample_idx < len(inst_freq):
        freq = inst_freq[sample_idx]
        expected = tone_a if bit == 0 else tone_b

        # Allow ±5 Hz tolerance due to GMSK filtering
        error = abs(freq - expected)
        match = error < 8

        status = "✓" if match else "✗"
        print(f"{status} Symbol {i:2d}: bit={bit}, freq={freq:7.2f} Hz (expect {expected:.0f} Hz, error={error:.1f} Hz)")

        if match:
            matches += 1
        else:
            mismatches += 1

print(f"\n{'='*70}")
print(f"Results: {matches}/{len(pattern)} symbols match expected frequency (±8 Hz)")
print(f"Match rate: {100*matches/len(pattern):.1f}%")

if matches >= len(pattern) * 0.8:  # 80% threshold
    print("✓ PASS: Pattern is being used correctly in 2-FSK modulation")
else:
    print("✗ FAIL: Pattern not being used correctly!")
    print("Issue: GMSK generator may not be applying pattern bits to frequency")
