#!/usr/bin/env python3
"""
3D visualization of IQ constellation trajectory over time

Shows how QPSK signals evolve through I/Q/time space, with phase coloring.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import pickle

# Add paths
examples_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.dirname(examples_dir)
sys.path.insert(0, os.path.join(training_dir, 'core'))
sys.path.insert(0, os.path.join(training_dir, 'src'))

from signal_generator import gmsk, modulation

# Load final patterns
final_patterns_path = os.path.join(training_dir, 'patterns', 'final_patterns_399968.pkl')
with open(final_patterns_path, 'rb') as f:
    final_data = pickle.load(f)

print(f"Loaded final patterns: {final_data['algorithm']}, {final_data['best_score']:.2f} dB")

# Use 1024-symbol patterns
pattern_length = 1024
patterns_dict = final_data['nested_patterns'][pattern_length]
pattern_0_core = patterns_dict['cores'][0]
repetition_map = patterns_dict['repetition_map']
pattern_0 = pattern_0_core[repetition_map]

# Generate QPSK data layer signal
SAMPLE_RATE = 48000
PATTERN_SYMBOL_RATE = 75
DATA_SYMBOL_RATE = 150  # QPSK data rate

# Generate pattern layer (3-FSK GMSK)
duration_symbols = 75  # 1 second at 75 sym/s
pattern_subset = pattern_0[:duration_symbols]
tone_a, tone_b, tone_c = 1500, 1520, 1540

gmsk_signal = gmsk.generate_gmsk_3fsk(
    pattern_subset, tone_a, tone_b, tone_c,
    SAMPLE_RATE, PATTERN_SYMBOL_RATE
)

# Generate QPSK data layer
# Create some test data
test_message = b"CASCADE QPSK 3D VISUALIZATION"
message_bits = np.unpackbits(np.frombuffer(test_message, dtype=np.uint8))

# Pad to even number for QPSK
if len(message_bits) % 2 != 0:
    message_bits = np.append(message_bits, 0)

# Map to QPSK constellation
qpsk_symbols = modulation.map_to_constellation(message_bits, 'QPSK')

# Upsample QPSK symbols to match sample rate
samples_per_symbol = SAMPLE_RATE // DATA_SYMBOL_RATE
num_data_symbols = int((len(gmsk_signal) / SAMPLE_RATE) * DATA_SYMBOL_RATE)

# Repeat symbols to fill duration
qpsk_symbols_extended = np.tile(qpsk_symbols, (num_data_symbols // len(qpsk_symbols)) + 1)[:num_data_symbols]

# Create upsampled data signal
data_upsampled = np.zeros(len(gmsk_signal), dtype=np.complex64)
for i, sym in enumerate(qpsk_symbols_extended):
    idx = i * samples_per_symbol
    if idx < len(data_upsampled):
        data_upsampled[idx] = sym

# Apply raised cosine pulse shaping
def raised_cosine_filter(beta=0.35, span_symbols=8, samples_per_symbol=320):
    span_samples = span_symbols * samples_per_symbol
    t = np.arange(-span_samples // 2, span_samples // 2) / samples_per_symbol

    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2)
        singular_points = np.abs(np.abs(t) - 1/(2*beta)) < 1e-10
        h[singular_points] = np.pi / 4 * np.sinc(1 / (2 * beta))
        h[np.abs(t) < 1e-10] = 1.0

    h /= np.sqrt(np.sum(h**2))
    return h

rc_filter = raised_cosine_filter(beta=0.35, span_symbols=8, samples_per_symbol=samples_per_symbol)
data_shaped = np.convolve(data_upsampled, rc_filter, mode='same')

# Combine pattern and data layers
combined_signal = gmsk_signal * data_shaped

print(f"\nGenerated {len(combined_signal)/SAMPLE_RATE:.2f}s QPSK signal")
print(f"Pattern symbols: {len(pattern_subset)}")
print(f"Data symbols: {len(qpsk_symbols_extended)}")

# Create 3D visualization
fig = plt.figure(figsize=(20, 12))

# Subsample for visualization (every Nth sample)
subsample_factor = 200  # Show every 200th sample
sig_plot = combined_signal[::subsample_factor]
time_plot = np.arange(len(sig_plot)) * subsample_factor / SAMPLE_RATE

# Extract I, Q, and phase
I = sig_plot.real
Q = sig_plot.imag
phase = np.angle(sig_plot)

# Normalize for better visualization
max_amp = np.max(np.abs(sig_plot))
if max_amp > 0:
    I = I / max_amp
    Q = Q / max_amp

print(f"\nPlotting {len(sig_plot)} samples over {time_plot[-1]:.2f}s")

# Plot 1: 3D trajectory colored by time
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
scatter1 = ax1.scatter(I, Q, time_plot, c=time_plot, cmap='viridis', s=2, alpha=0.6)
ax1.plot(I, Q, time_plot, color='blue', alpha=0.1, linewidth=0.5)

ax1.set_xlabel('In-phase (I)')
ax1.set_ylabel('Quadrature (Q)')
ax1.set_zlabel('Time (s)')
ax1.set_title('IQ Trajectory Over Time\n(colored by time)', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=ax1, label='Time (s)', pad=0.1)

# Plot 2: 3D trajectory colored by phase
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
scatter2 = ax2.scatter(I, Q, time_plot, c=phase, cmap='hsv', s=2, alpha=0.6, vmin=-np.pi, vmax=np.pi)
ax2.plot(I, Q, time_plot, color='gray', alpha=0.1, linewidth=0.5)

ax2.set_xlabel('In-phase (I)')
ax2.set_ylabel('Quadrature (Q)')
ax2.set_zlabel('Time (s)')
ax2.set_title('IQ Trajectory Over Time\n(colored by phase)', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=ax2, label='Phase (rad)', pad=0.1)

# Plot 3: Top-down view (IQ plane)
ax3 = fig.add_subplot(2, 3, 3)
scatter3 = ax3.scatter(I, Q, c=time_plot, cmap='viridis', s=5, alpha=0.4)
ax3.set_xlabel('In-phase (I)')
ax3.set_ylabel('Quadrature (Q)')
ax3.set_title('IQ Plane (top view)\n(colored by time)', fontsize=12, fontweight='bold')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

# Mark QPSK constellation points
qpsk_points = np.array([
    [1/np.sqrt(2), 1/np.sqrt(2)],
    [-1/np.sqrt(2), 1/np.sqrt(2)],
    [-1/np.sqrt(2), -1/np.sqrt(2)],
    [1/np.sqrt(2), -1/np.sqrt(2)]
])
ax3.scatter(qpsk_points[:, 0], qpsk_points[:, 1],
           marker='x', s=200, linewidths=3, color='red', zorder=10, label='QPSK points')
ax3.legend()
plt.colorbar(scatter3, ax=ax3, label='Time (s)')

# Plot 4: Time slices (show IQ at different time points)
ax4 = fig.add_subplot(2, 3, 4, projection='3d')

# Select 5 time windows
num_slices = 5
time_windows = np.linspace(0, time_plot[-1], num_slices+1)
colors_slices = plt.cm.plasma(np.linspace(0, 1, num_slices))

for i in range(num_slices):
    t_start = time_windows[i]
    t_end = time_windows[i+1]
    mask = (time_plot >= t_start) & (time_plot < t_end)

    ax4.scatter(I[mask], Q[mask], time_plot[mask],
               color=colors_slices[i], s=3, alpha=0.7, label=f't={t_start:.2f}s')

ax4.set_xlabel('In-phase (I)')
ax4.set_ylabel('Quadrature (Q)')
ax4.set_zlabel('Time (s)')
ax4.set_title('Time-Sliced Trajectory', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8, loc='upper right')

# Plot 5: Phase evolution over time
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(time_plot, phase, linewidth=1, alpha=0.7)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Phase (rad)')
ax5.set_title('Phase Evolution', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_ylim([-np.pi, np.pi])
ax5.axhline(0, color='k', linestyle='--', alpha=0.3)

# Plot 6: Amplitude evolution over time
ax6 = fig.add_subplot(2, 3, 6)
amplitude = np.abs(sig_plot)
ax6.plot(time_plot, amplitude, linewidth=1, alpha=0.7, color='orange')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Amplitude')
ax6.set_title('Amplitude Evolution', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Overall title
fig.suptitle(
    f'QPSK Signal: 3D IQ Trajectory\n'
    f'Pattern layer: 3-FSK GMSK @ {PATTERN_SYMBOL_RATE} sym/s | Data layer: QPSK @ {DATA_SYMBOL_RATE} sym/s',
    fontsize=14, fontweight='bold', y=0.98
)

plt.tight_layout()

output_path = os.path.join(examples_dir, 'iq_3d_trajectory_qpsk.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved {output_path}")

# Create a second figure with different viewing angles
fig2 = plt.figure(figsize=(20, 10))

viewing_angles = [
    (30, 45),   # Default view
    (0, 90),    # Side view (I-time)
    (90, 0),    # Side view (Q-time)
    (0, 0),     # Front view (I-Q)
]

angle_names = ['Perspective', 'I-Time Side', 'Q-Time Side', 'Top (I-Q Plane)']

for idx, ((elev, azim), name) in enumerate(zip(viewing_angles, angle_names)):
    ax = fig2.add_subplot(2, 2, idx+1, projection='3d')

    scatter = ax.scatter(I, Q, time_plot, c=phase, cmap='hsv', s=2, alpha=0.6, vmin=-np.pi, vmax=np.pi)
    ax.plot(I, Q, time_plot, color='gray', alpha=0.1, linewidth=0.5)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('In-phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.set_zlabel('Time (s)')
    ax.set_title(f'{name} View\n(elev={elev}°, azim={azim}°)', fontsize=11, fontweight='bold')

    if idx == 3:
        plt.colorbar(scatter, ax=ax, label='Phase (rad)', pad=0.1)

fig2.suptitle(
    f'QPSK 3D Trajectory: Multiple Viewing Angles',
    fontsize=14, fontweight='bold', y=0.98
)

plt.tight_layout()

output_path2 = os.path.join(examples_dir, 'iq_3d_trajectory_angles.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Saved {output_path2}")

print("\nVisualization complete!")
print("\nKey features:")
print("- 3D trajectory shows signal path through I/Q/time space")
print("- GMSK creates continuous phase transitions (smooth curves)")
print("- QPSK data layer creates distinct regions around 4 constellation points")
print("- Phase coloring shows gradual transitions (no discontinuities)")
print("- Combined pattern + data layers create complex 3D trajectory")
