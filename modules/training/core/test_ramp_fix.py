#!/usr/bin/env python3
"""Quick test to verify Tukey windowing fix for signal ramp-down."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from streaming_cascade_dataset import StreamingCascadeDataset

# Create small test dataset
dataset = StreamingCascadeDataset(
    num_streams=1,
    stream_duration_sec=10.0,
    sample_rate=48000,
    cache_dir=None,  # No caching
    scenario_name='contest',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Generate one sample
sample = dataset[0]

# Extract I/Q stream
iq_stream = sample['iq_stream'].cpu().numpy()

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Time domain
t = np.arange(len(iq_stream)) / dataset.sample_rate
axes[0].plot(t, np.abs(iq_stream), linewidth=0.5)
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Stream I/Q - Time Domain (check for smooth ramp-downs)')
axes[0].grid(True, alpha=0.3)

# Spectrogram
from scipy import signal
f, t_spec, Sxx = signal.spectrogram(
    iq_stream,
    fs=dataset.sample_rate,
    nperseg=2048,
    noverlap=1536
)
axes[1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('Spectrogram')

# Power envelope
power = np.abs(iq_stream) ** 2
window_len = 480  # 10ms window
power_smooth = np.convolve(power, np.ones(window_len)/window_len, mode='same')
axes[2].plot(t, 10 * np.log10(power_smooth + 1e-10), linewidth=1)
axes[2].set_ylabel('Power (dB)')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('Power Envelope (should show smooth ramp-downs)')
axes[2].grid(True, alpha=0.3)

# Add message boundaries from labels
if 'messages' in sample:
    for msg in sample['messages']:
        for ax in axes:
            start_t = msg['start_sample'] / dataset.sample_rate
            end_t = msg['end_sample'] / dataset.sample_rate
            ax.axvline(start_t, color='green', alpha=0.3, linestyle='--')
            ax.axvline(end_t, color='red', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('test_ramp_fix.png', dpi=150)
print(f"Saved visualization to test_ramp_fix.png")
print(f"Generated {len(sample.get('messages', []))} messages in stream")
