"""Diagnose GMSK pattern encoding to verify irregularity is preserved."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import sys
import pickle

sys.path.insert(0, '/workspaces/cascade/modules/training')
from src.signal_generator.gmsk import generate_gmsk_fsk

# Load a real pattern
with open('/workspaces/cascade/modules/training/patterns/tournament/pattern_0_2048.pkl', 'rb') as f:
    pattern_bits = pickle.load(f)

# Use first 100 bits for analysis
pattern = pattern_bits[:100]
print(f"Pattern first 30 bits: {pattern[:30]}")
print(f"Hamming weight: {np.sum(pattern)}/100")

# Count runs
runs = []
current_val = pattern[0]
current_run = 1
for i in range(1, len(pattern)):
    if pattern[i] == current_val:
        current_run += 1
    else:
        runs.append(current_run)
        current_val = pattern[i]
        current_run = 1
runs.append(current_run)
print(f"Run lengths: {runs[:20]}...")
print(f"Mean run length: {np.mean(runs):.2f}, std: {np.std(runs):.2f}")

# Generate GMSK signal
SAMPLE_RATE = 48000
PATTERN_SYMBOL_RATE = 75  # Pattern layer always 75 sym/s
tone_a = 1000.0
tone_b = 1020.0

iq_signal = generate_gmsk_fsk(
    pattern, tone_a, tone_b,
    sample_rate=SAMPLE_RATE,
    symbol_rate=PATTERN_SYMBOL_RATE
)

# Measure instantaneous frequency
# Method 1: From phase derivative
phase = np.unwrap(np.angle(iq_signal))
inst_freq = np.diff(phase) / (2 * np.pi) * SAMPLE_RATE

# Downsample to symbol rate for visualization
samples_per_symbol = SAMPLE_RATE // PATTERN_SYMBOL_RATE  # 640
symbol_times = np.arange(len(pattern)) * samples_per_symbol
symbol_freqs = inst_freq[symbol_times[:-1]]  # -1 because diff reduces length

print(f"\nInstantaneous frequency analysis:")
print(f"  Mean: {np.mean(inst_freq):.2f} Hz (expect {(tone_a + tone_b)/2:.1f} Hz)")
print(f"  Min: {np.min(inst_freq):.2f} Hz (expect ~{tone_a:.1f} Hz)")
print(f"  Max: {np.max(inst_freq):.2f} Hz (expect ~{tone_b:.1f} Hz)")
print(f"  Std: {np.std(inst_freq):.2f} Hz")

# Create diagnostic plot
fig, axes = plt.subplots(4, 1, figsize=(16, 12))

# Subplot 1: Pattern bits
ax = axes[0]
ax.step(np.arange(len(pattern)), pattern, where='post', linewidth=2)
ax.set_ylabel('Bit Value', fontsize=12)
ax.set_title('Pattern Bits (Should be Irregular)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.1)

# Subplot 2: IQ constellation over time (showing phase transitions)
ax = axes[1]
# Plot I and Q separately
t = np.arange(len(iq_signal)) / SAMPLE_RATE
ax.plot(t, iq_signal.real, label='I (real)', alpha=0.7, linewidth=1)
ax.plot(t, iq_signal.imag, label='Q (imag)', alpha=0.7, linewidth=1)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_title('GMSK IQ Signal (I/Q Components)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100 * samples_per_symbol / SAMPLE_RATE)  # First 100 symbols

# Subplot 3: Instantaneous frequency
ax = axes[2]
t_freq = np.arange(len(inst_freq)) / SAMPLE_RATE
ax.plot(t_freq, inst_freq, linewidth=1, alpha=0.8)
ax.axhline(tone_a, color='blue', linestyle='--', label=f'Tone A ({tone_a:.0f} Hz)', linewidth=2)
ax.axhline(tone_b, color='red', linestyle='--', label=f'Tone B ({tone_b:.0f} Hz)', linewidth=2)
ax.axhline((tone_a + tone_b)/2, color='green', linestyle=':', label='Center', linewidth=2)
ax.set_ylabel('Frequency (Hz)', fontsize=12)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_title('Instantaneous Frequency (Should Follow Pattern Irregularity)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100 * samples_per_symbol / SAMPLE_RATE)
ax.set_ylim(tone_a - 20, tone_b + 20)

# Subplot 4: Spectrogram (what we see in the main visualization)
ax = axes[3]
f, t, Zxx = scipy_signal.stft(
    iq_signal, fs=SAMPLE_RATE, window='hann',
    nperseg=2048, noverlap=int(2048 * 0.875),
    return_onesided=False
)
f = np.fft.fftshift(f)
Zxx = np.fft.fftshift(Zxx, axes=0)

# Focus on signal bandwidth
freq_mask = (f >= tone_a - 50) & (f <= tone_b + 50)
amplitude_db = 20 * np.log10(np.abs(Zxx[freq_mask]) + 1e-20)

im = ax.imshow(
    amplitude_db.T, aspect='auto', origin='lower',
    extent=[f[freq_mask][0], f[freq_mask][-1], t[0], t[-1]],
    cmap='viridis', interpolation='nearest'
)
ax.axhline(tone_a, color='yellow', linestyle='--', linewidth=2, label=f'Tone A')
ax.axhline(tone_b, color='yellow', linestyle='--', linewidth=2, label=f'Tone B')
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)
ax.set_title('Spectrogram (Power Distribution)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='dB')

plt.tight_layout()
plt.savefig('gmsk_pattern_diagnosis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: gmsk_pattern_diagnosis.png")

# Additional analysis: Check if frequency actually varies with pattern
print(f"\nPattern-to-frequency correlation:")
for i in range(min(20, len(pattern)-1)):
    bit = pattern[i]
    # Sample frequency at symbol center
    sample_idx = int((i + 0.5) * samples_per_symbol)
    if sample_idx < len(inst_freq):
        freq_at_symbol = inst_freq[sample_idx]
        expected = tone_a if bit == 0 else tone_b
        print(f"  Symbol {i}: bit={bit}, freq={freq_at_symbol:.1f} Hz (expect {expected:.1f} Hz)")
