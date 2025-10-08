#!/usr/bin/env python3
"""
4D Visualization: Time-Frequency-Amplitude-Phase (Manifold)

Shows CASCADE signals in 4D space as continuous manifolds:
- X-axis: Time
- Y-axis: Frequency
- Z-axis: Amplitude
- Color: Instantaneous Phase

Compares:
1. Clean CASCADE signal (pattern layer + data layer)
2. Signal with noise + propagation effects (multipath, fading)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.interpolate import RectBivariateSpline
import sys
import os
import pickle

# Add paths
examples_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.dirname(examples_dir)
sys.path.insert(0, os.path.join(training_dir, 'core'))
sys.path.insert(0, os.path.join(training_dir, 'src'))

from signal_generator import gmsk, modulation
from physics_coupling import CorePhysicalDrivers, CoupledPhysicsCalculator
from scenarios import ScenarioLibrary

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

# Generate BPSK signal
SAMPLE_RATE = 48000
PATTERN_SYMBOL_RATE = 75
DATA_SYMBOL_RATE = 100  # BPSK data rate

duration_symbols = 75  # 1 second
pattern_subset = pattern_0[:duration_symbols]
tone_a, tone_b, tone_c = 1500, 1520, 1540

# Pattern layer (3-FSK GMSK)
gmsk_signal = gmsk.generate_gmsk_3fsk(
    pattern_subset, tone_a, tone_b, tone_c,
    SAMPLE_RATE, PATTERN_SYMBOL_RATE
)

# BPSK data layer
test_message = b"CASCADE BPSK 4D VIS"
message_bits = np.unpackbits(np.frombuffer(test_message, dtype=np.uint8))
bpsk_symbols = modulation.map_to_constellation(message_bits, 'BPSK')

# Upsample
samples_per_symbol = SAMPLE_RATE // DATA_SYMBOL_RATE
num_data_symbols = int((len(gmsk_signal) / SAMPLE_RATE) * DATA_SYMBOL_RATE)
bpsk_symbols_extended = np.tile(bpsk_symbols, (num_data_symbols // len(bpsk_symbols)) + 1)[:num_data_symbols]

data_upsampled = np.zeros(len(gmsk_signal), dtype=np.complex64)
for i, sym in enumerate(bpsk_symbols_extended):
    idx = i * samples_per_symbol
    if idx < len(data_upsampled):
        data_upsampled[idx] = sym

# Pulse shaping
def raised_cosine_filter(beta=0.35, span_symbols=8, samples_per_symbol=480):
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

# Combined signal (CLEAN)
sig_clean = gmsk_signal * data_shaped * 0.5

print(f"\nGenerated {len(sig_clean)/SAMPLE_RATE:.2f}s BPSK signal")

# Apply propagation effects to create degraded signal
def apply_multipath_fading(sig, delay_spread_ms=2.0, num_paths=3):
    """Apply multipath with Rayleigh fading"""
    output = sig.copy()
    t = np.arange(len(sig)) / SAMPLE_RATE

    for path in range(1, num_paths):
        delay_samples = int(np.random.uniform(0.5, delay_spread_ms) * SAMPLE_RATE / 1000)
        attenuation = np.random.uniform(0.3, 0.7)
        phase_shift = np.random.uniform(0, 2 * np.pi)
        doppler_hz = np.random.uniform(0.1, 0.5)
        phase_variation = 2 * np.pi * doppler_hz * t

        delayed = np.zeros_like(output)
        if delay_samples < len(sig):
            delayed[delay_samples:] = sig[:-delay_samples] * attenuation * \
                                      np.exp(1j * (phase_shift + phase_variation[:-delay_samples]))
            output += delayed

    return output

def add_awgn(sig, snr_db=5.0):
    """Add AWGN to achieve target SNR"""
    signal_power = np.mean(np.abs(sig) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))) * np.sqrt(noise_power / 2)
    return sig + noise

# Create degraded signal
print("\nApplying propagation effects...")
sig_multipath = apply_multipath_fading(sig_clean, delay_spread_ms=2.0, num_paths=3)
sig_degraded = add_awgn(sig_multipath, snr_db=5.0)

print("  - Multipath fading (3 paths, 2ms delay spread)")
print("  - AWGN (SNR = 5 dB)")

# Compute STFT for both signals (to get time-frequency-amplitude-phase)
# Use very large nperseg for smooth waterfall-like surface (SETI@home style)
nperseg = 4096
noverlap = int(nperseg * 0.98)

f_clean, t_clean, Zxx_clean = signal.stft(
    sig_clean, fs=SAMPLE_RATE, window='hann',
    nperseg=nperseg, noverlap=noverlap,
    return_onesided=False
)

f_degraded, t_degraded, Zxx_degraded = signal.stft(
    sig_degraded, fs=SAMPLE_RATE, window='hann',
    nperseg=nperseg, noverlap=noverlap,
    return_onesided=False
)

# Shift to center frequency
f_clean = np.fft.fftshift(f_clean)
f_degraded = np.fft.fftshift(f_degraded)
Zxx_clean = np.fft.fftshift(Zxx_clean, axes=0)
Zxx_degraded = np.fft.fftshift(Zxx_degraded, axes=0)

# Focus on signal bandwidth (±150 Hz around carrier for smooth surface)
freq_center = (tone_a + tone_c) / 2
freq_mask = (f_clean >= freq_center - 150) & (f_clean <= freq_center + 150)

f_plot = f_clean[freq_mask]
Zxx_clean_plot = Zxx_clean[freq_mask, :]
Zxx_degraded_plot = Zxx_degraded[freq_mask, :]

# Extract amplitude and phase
amp_clean = np.abs(Zxx_clean_plot)
phase_clean = np.angle(Zxx_clean_plot)

amp_degraded = np.abs(Zxx_degraded_plot)
phase_degraded = np.angle(Zxx_degraded_plot)

# Create meshgrid for 3D plotting
T_clean, F_clean = np.meshgrid(t_clean, f_plot)
T_degraded, F_degraded = np.meshgrid(t_degraded, f_plot)

print(f"\nSTFT computed: {len(f_plot)} freq bins × {len(t_clean)} time bins")

# Interpolate for smooth surface
# Upsample by 10x in both dimensions for ultra-smooth rendering
upsample_factor = 10
t_interp = np.linspace(t_clean.min(), t_clean.max(), len(t_clean) * upsample_factor)
f_interp = np.linspace(f_plot.min(), f_plot.max(), len(f_plot) * upsample_factor)

# Interpolate amplitude and phase for clean signal
amp_clean_interp_func = RectBivariateSpline(f_plot, t_clean, amp_clean)
phase_clean_interp_func = RectBivariateSpline(f_plot, t_clean, phase_clean)

amp_clean_interp = amp_clean_interp_func(f_interp, t_interp)
phase_clean_interp = phase_clean_interp_func(f_interp, t_interp)

# Interpolate amplitude and phase for degraded signal
amp_degraded_interp_func = RectBivariateSpline(f_plot, t_degraded, amp_degraded)
phase_degraded_interp_func = RectBivariateSpline(f_plot, t_degraded, phase_degraded)

amp_degraded_interp = amp_degraded_interp_func(f_interp, t_interp)
phase_degraded_interp = phase_degraded_interp_func(f_interp, t_interp)

# Create interpolated meshgrids
T_clean_sub, F_clean_sub = np.meshgrid(t_interp, f_interp)
T_degraded_sub, F_degraded_sub = np.meshgrid(t_interp, f_interp)

amp_clean_sub = amp_clean_interp
phase_clean_sub = phase_clean_interp
amp_degraded_sub = amp_degraded_interp
phase_degraded_sub = phase_degraded_interp

print(f"Interpolated to: {len(f_interp)} freq bins × {len(t_interp)} time bins (10x upsampling)")

# Create 4D visualization with surface manifold
fig = plt.figure(figsize=(22, 10))

# Clean signal (left column)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# Use plot_surface for smooth continuous surface
surf1 = ax1.plot_surface(
    T_clean_sub,
    F_clean_sub,
    amp_clean_sub,
    facecolors=plt.cm.hsv((phase_clean_sub + np.pi) / (2 * np.pi)),
    shade=True,
    alpha=0.95,
    linewidth=0,
    antialiased=True
)

ax1.set_xlabel('Time (s)', fontsize=11)
ax1.set_ylabel('Frequency (Hz)', fontsize=11)
ax1.set_zlabel('Amplitude', fontsize=11)
ax1.set_title('Clean CASCADE Signal\n4D: Time × Frequency × Amplitude × Phase',
             fontsize=12, fontweight='bold')

# Set viewing angle
ax1.view_init(elev=20, azim=45)

# Create colorbar manually for phase
norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
sm1 = plt.cm.ScalarMappable(cmap='hsv', norm=norm)
sm1.set_array([])
cbar1 = plt.colorbar(sm1, ax=ax1, pad=0.1, shrink=0.8)
cbar1.set_label('Phase (rad)', fontsize=10)

# Add info text
info_text1 = (
    f'Pattern: 3-FSK GMSK @ {PATTERN_SYMBOL_RATE} sym/s\n'
    f'Data: BPSK @ {DATA_SYMBOL_RATE} sym/s\n'
    f'No noise or propagation effects'
)
ax1.text2D(0.05, 0.95, info_text1, transform=ax1.transAxes,
          fontsize=9, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Degraded signal (right column)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Use plot_surface for smooth continuous surface
surf2 = ax2.plot_surface(
    T_degraded_sub,
    F_degraded_sub,
    amp_degraded_sub,
    facecolors=plt.cm.hsv((phase_degraded_sub + np.pi) / (2 * np.pi)),
    shade=True,
    alpha=0.95,
    linewidth=0,
    antialiased=True
)

ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('Frequency (Hz)', fontsize=11)
ax2.set_zlabel('Amplitude', fontsize=11)
ax2.set_title('CASCADE with Noise + Propagation\n4D: Time × Frequency × Amplitude × Phase',
             fontsize=12, fontweight='bold')

ax2.view_init(elev=20, azim=45)

# Create colorbar manually for phase
sm2 = plt.cm.ScalarMappable(cmap='hsv', norm=norm)
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=ax2, pad=0.1, shrink=0.8)
cbar2.set_label('Phase (rad)', fontsize=10)

# Add info text
info_text2 = (
    f'SNR: 5 dB (AWGN)\n'
    f'Multipath: 3 paths, 2ms spread\n'
    f'Rayleigh fading with Doppler'
)
ax2.text2D(0.05, 0.95, info_text2, transform=ax2.transAxes,
          fontsize=9, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

fig.suptitle(
    '4D Signal Visualization: Clean vs. Degraded CASCADE\n'
    'Spatial: Time × Frequency × Amplitude | Color: Instantaneous Phase',
    fontsize=14, fontweight='bold', y=0.98
)

plt.tight_layout()

output_path = os.path.join(examples_dir, '4d_clean_vs_noisy.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved {output_path}")

# Create a second figure with multiple viewing angles
fig2 = plt.figure(figsize=(22, 16))

viewing_angles = [
    (20, 45, 'Perspective'),
    (0, 0, 'Front (Time-Freq)'),
    (0, 90, 'Side (Freq-Amp)'),
    (90, 0, 'Top (Time-Amp)'),
]

for idx, (elev, azim, name) in enumerate(viewing_angles):
    # Clean signal
    ax_clean = fig2.add_subplot(4, 2, 2*idx+1, projection='3d')

    surf_c = ax_clean.plot_surface(
        T_clean_sub,
        F_clean_sub,
        amp_clean_sub,
        facecolors=plt.cm.hsv((phase_clean_sub + np.pi) / (2 * np.pi)),
        shade=True,
        alpha=0.95,
        linewidth=0,
        antialiased=True
    )

    ax_clean.view_init(elev=elev, azim=azim)
    ax_clean.set_xlabel('Time (s)', fontsize=9)
    ax_clean.set_ylabel('Frequency (Hz)', fontsize=9)
    ax_clean.set_zlabel('Amplitude', fontsize=9)
    ax_clean.set_title(f'Clean - {name} View', fontsize=10, fontweight='bold')

    if idx == 0:
        sm = plt.cm.ScalarMappable(cmap='hsv', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax_clean, pad=0.1, shrink=0.7)

    # Degraded signal
    ax_deg = fig2.add_subplot(4, 2, 2*idx+2, projection='3d')

    surf_d = ax_deg.plot_surface(
        T_degraded_sub,
        F_degraded_sub,
        amp_degraded_sub,
        facecolors=plt.cm.hsv((phase_degraded_sub + np.pi) / (2 * np.pi)),
        shade=True,
        alpha=0.95,
        linewidth=0,
        antialiased=True
    )

    ax_deg.view_init(elev=elev, azim=azim)
    ax_deg.set_xlabel('Time (s)', fontsize=9)
    ax_deg.set_ylabel('Frequency (Hz)', fontsize=9)
    ax_deg.set_zlabel('Amplitude', fontsize=9)
    ax_deg.set_title(f'Degraded - {name} View', fontsize=10, fontweight='bold')

    if idx == 0:
        sm = plt.cm.ScalarMappable(cmap='hsv', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax_deg, pad=0.1, shrink=0.7)

fig2.suptitle(
    '4D Comparison: Multiple Viewing Angles\n'
    'Left: Clean Signal | Right: Degraded Signal (5 dB SNR + Multipath)',
    fontsize=14, fontweight='bold', y=0.995
)

plt.tight_layout()

output_path2 = os.path.join(examples_dir, '4d_clean_vs_noisy_angles.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Saved {output_path2}")

print("\nVisualization complete!")
print("\nKey observations:")
print("Clean CASCADE signal:")
print("  - Smooth, continuous manifold structure")
print("  - Coherent phase patterns (organized HSV colors)")
print("  - Sharp frequency localization around 3-FSK carriers")
print("  - Predictable time evolution across pattern + data layers")
print("\nDegraded CASCADE signal:")
print("  - Ragged manifold surface (multipath interference)")
print("  - Phase distortion (chaotic color patterns)")
print("  - Frequency spreading (reduced sharpness)")
print("  - Amplitude fluctuations from Rayleigh fading")
print("\nThe 4D manifold visualization shows how noise and propagation")
print("destroy the clean geometric structure of CASCADE signals.")
