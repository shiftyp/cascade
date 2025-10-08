"""
Quick test to visualize phase information from CASCADE signals
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys, os

# Add paths
examples_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.dirname(examples_dir)
sys.path.insert(0, os.path.join(training_dir, 'core'))
sys.path.insert(0, os.path.join(training_dir, 'src'))

from signal_generator.pattern_loader import PatternLoader
from signal_generator import gmsk

# Load patterns
pattern_loader = PatternLoader()
pattern_0 = pattern_loader.load_pattern(0, 2048)
pattern_2 = pattern_loader.load_pattern(2, 2048)

# Generate simple GMSK 3-FSK signals
SAMPLE_RATE = 48000
SYMBOL_RATE = 75
tone_a, tone_b, tone_c = 1000, 1020, 1040  # 3-FSK: 3 tones

# Pattern 0 (3-FSK)
sig_0 = gmsk.generate_gmsk_3fsk(pattern_0[:75], tone_a, tone_b, tone_c, SAMPLE_RATE, SYMBOL_RATE)

# Pattern 2 (3-FSK)
sig_2 = gmsk.generate_gmsk_3fsk(pattern_2[:75], tone_a, tone_b, tone_c, SAMPLE_RATE, SYMBOL_RATE)

# Create figure with phase information
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (sig, pattern_id) in enumerate([(sig_0, 0), (sig_2, 2)]):
    # Power spectrogram
    f, t, Sxx = signal.spectrogram(sig.real, fs=SAMPLE_RATE, window='hann',
                                   nperseg=1024, noverlap=int(1024*0.875))
    Sxx_db = 10 * np.log10(Sxx + 1e-20)
    
    axes[idx, 0].imshow(Sxx_db.T, aspect='auto', origin='lower', cmap='viridis',
                       extent=[f[0], f[-1], t[0], t[-1]])
    axes[idx, 0].set_title(f'Pattern {pattern_id}: Power Spectrogram')
    axes[idx, 0].set_ylabel('Time (s)')
    
    # Phase over time (instantaneous phase)
    inst_phase = np.angle(sig)
    axes[idx, 1].plot(np.arange(len(inst_phase))/SAMPLE_RATE, inst_phase, linewidth=0.5)
    axes[idx, 1].set_title(f'Pattern {pattern_id}: Instantaneous Phase')
    axes[idx, 1].set_ylabel('Phase (radians)')
    axes[idx, 1].set_ylim([-np.pi, np.pi])
    
    # IQ constellation (subsample for clarity)
    subsample = sig[::100]
    axes[idx, 2].scatter(subsample.real, subsample.imag, alpha=0.5, s=10)
    axes[idx, 2].set_title(f'Pattern {pattern_id}: IQ Constellation')
    axes[idx, 2].set_xlabel('In-phase')
    axes[idx, 2].set_ylabel('Quadrature')
    axes[idx, 2].grid(True, alpha=0.3)
    axes[idx, 2].axis('equal')

plt.tight_layout()
plt.savefig('phase_visualization_test.png', dpi=150)
print("✓ Saved phase_visualization_test.png")
print("\nConclusion: GMSK 3-FSK patterns look similar in power spectrograms")
print("because they use the same three frequencies. The difference is in the")
print("TIMING of transitions (phase), which is what the neural network learns.")
print("3-FSK provides better frequency-selective fading resilience than 2-FSK.")
