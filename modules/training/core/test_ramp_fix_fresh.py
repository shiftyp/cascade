#!/usr/bin/env python3
"""Generate fresh stream with ramp fixes to verify."""

import torch
import numpy as np
import os
import sys

# Set visualization on for this test
os.environ['CASCADE_VIS_INTERVAL'] = '1'

from streaming_cascade_dataset import StreamingCascadeDataset

print("Generating 1 fresh stream with ramp fixes...")
print("(Cache disabled, will generate from scratch)")

# Create dataset with cache disabled
dataset = StreamingCascadeDataset(
    num_streams=1,
    stream_duration_sec=10.0,
    use_cache=False,  # Force fresh generation
    device='cuda' if torch.cuda.is_available() else 'cpu',
    seed=99999  # Different seed to avoid any caching
)

print(f"\nDataset created: {len(dataset)} samples (windows)")
print("Generating sample 0...")

# Generate one sample - should create visualization
sample = dataset[0]

print(f"✓ Generated sample with {len(sample.get('messages', []))} messages")
print("\nCheck visualizations/ directory for output")
print("Should show smooth 150ms ramp-downs (not abrupt cutoffs)")
