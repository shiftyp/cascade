#!/usr/bin/env python3
"""Verify GMSK frequency AVERAGES correctly over symbol period (not instantaneous)."""

import numpy as np
import sys
import pickle
sys.path.insert(0, '/workspaces/cascade/modules/training')
from src.signal_generator.gmsk import generate_gmsk_fsk

# Load pattern
with open('/workspaces/cascade/modules/training/patterns/tournament/pattern_0_2048.pkl', 'rb') as f:
    pattern_bits = pickle.load(f)

pattern = pattern_bits[:20]
print(f"Pattern first 20 bits: {pattern}")

# Generate GMSK 3-FSK
SAMPLE_RATE = 48000
PATTERN_SYMBOL_RATE = 25  # Testing slower rate
tone_a = 1000.0
tone_b = 1020.0
tone_c = 1040.0  # 3-FSK: third tone

iq_signal = generate_gmsk_3fsk(
    pattern, tone_a, tone_b, tone_c,
    sample_rate=SAMPLE_RATE,
    symbol_rate=PATTERN_SYMBOL_RATE
)

# Measure AVERAGE frequency over entire symbol period
samples_per_symbol = SAMPLE_RATE // PATTERN_SYMBOL_RATE

phase = np.unwrap(np.angle(iq_signal))
inst_freq = np.diff(phase) / (2 * np.pi) * SAMPLE_RATE

print(f"\nVerifying AVERAGE frequency over symbol period (3-FSK):")
print(f"Expected: symbol=0 → {tone_a:.0f} Hz, symbol=1 → {tone_b:.0f} Hz, symbol=2 → {tone_c:.0f} Hz\n")

matches = 0

for i, symbol in enumerate(pattern):
    # Average frequency over ENTIRE symbol period
    start_sample = i * samples_per_symbol
    end_sample = (i + 1) * samples_per_symbol

    if end_sample < len(inst_freq):
        freq_avg = np.mean(inst_freq[start_sample:end_sample])

        # 3-FSK: map symbol to expected frequency
        if symbol == 0:
            expected = tone_a
        elif symbol == 1:
            expected = tone_b
        else:  # symbol == 2
            expected = tone_c

        error = abs(freq_avg - expected)
        match = error < 5  # ±5 Hz tolerance

        status = "✓" if match else "✗"
        print(f"{status} Symbol {i:2d}: symbol={symbol}, avg_freq={freq_avg:7.2f} Hz (expect {expected:.0f} Hz, error={error:.1f} Hz)")

        if match:
            matches += 1

print(f"\n{'='*70}")
print(f"Results: {matches}/{len(pattern)} symbols have correct AVERAGE frequency")
print(f"Match rate: {100*matches/len(pattern):.1f}%")

if matches >= len(pattern) * 0.8:
    print("✓ PASS: GMSK 3-FSK is correctly using the ternary pattern")
    print("Note: Instantaneous frequency varies due to GMSK smoothing (BT=0.3)")
    print("      This is EXPECTED behavior to prevent spectral splatter")
else:
    print("✗ FAIL: Pattern not being used correctly")
