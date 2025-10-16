#!/usr/bin/env python3
"""Test the new channel allocation strategy."""

from continuous_rate_calculator import ContinuousRateCalculator
import numpy as np

np.random.seed(42)  # For reproducibility
calc = ContinuousRateCalculator(bandwidth_hz=60.0)

print('Channel Allocation Strategy: Prefer Single Channel with Higher Rate')
print('=' * 80)
print('SNR(dB) | Channels | BW(Hz) | Modulation | Rate(sym/s) | Bitrate | Efficiency')
print('-' * 80)

# Test across SNR range
for snr_db in [-10, -8, -6, -4, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
    mod, bps = calc.optimal_modulation(snr_db)
    rate, channels, _ = calc.calculate_continuous_rate(snr_db)
    bandwidth = channels * 60
    bitrate = rate * bps

    # Calculate Shannon capacity
    snr_linear = 10 ** (snr_db / 10)
    capacity = bandwidth * np.log2(1 + snr_linear)
    efficiency = (bitrate / capacity * 100) if capacity > 0 else 0

    # Mark interesting transitions
    marker = ''
    if efficiency > 95:
        marker = ' WARNING'

    print(f'{snr_db:6.1f} | {channels:8d} | {bandwidth:6d} | {mod:10s} | {rate:11.1f} | {bitrate:7.1f} | {efficiency:6.1f}%{marker}')

print()
print('Key Observations:')
print('-' * 50)

# Check where transitions happen
transition_points = []
prev_ch = None
for snr in range(-10, 21):
    _, ch, _ = calc.calculate_continuous_rate(float(snr))
    if prev_ch is not None and ch != prev_ch:
        transition_points.append((snr, prev_ch, ch))
    prev_ch = ch

for snr, old_ch, new_ch in transition_points:
    print(f'  Transition at {snr} dB: {old_ch} → {new_ch} channel(s)')

print()
print('Bitrate Progression (detailed):')
print('-' * 50)

# Check bitrate progression
prev_br = None
prev_ch = None
for snr in range(-6, 11):
    mod, bps = calc.optimal_modulation(float(snr))
    rate, ch, _ = calc.calculate_continuous_rate(float(snr))
    bitrate = rate * bps

    if prev_br is not None:
        change = (bitrate / prev_br - 1) * 100
        symbol = '↗' if change > 0 else '↘'

        # Mark channel transitions
        if ch != prev_ch:
            print(f'  {snr-1:2d}→{snr:2d} dB: {prev_ch}→{ch}ch {mod:6s} {prev_br:5.1f}→{bitrate:5.1f} bps ({symbol} {abs(change):4.1f}%) [TRANSITION]')
        else:
            print(f'  {snr-1:2d}→{snr:2d} dB: {ch}ch {mod:6s} {prev_br:5.1f}→{bitrate:5.1f} bps ({symbol} {abs(change):4.1f}%)')

    prev_br = bitrate
    prev_ch = ch

print()
print('Summary:')
print('  - Single channel used when it can maintain 50 sym/s')
print('  - Multiple channels only at low SNR for synchronization')
print('  - Maximizes bitrate on single channel when possible')