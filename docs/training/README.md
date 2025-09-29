# Training Strategy

CASCADE's training approach emphasizes learning from real-world conditions while maintaining privacy and enabling continuous improvement.

## Two-Pass Kernel Training

### Pass 1: Random Kernel Robustness
Train the model to decode signals without knowing the kernel configuration, building robustness to unknown conditions.

```python
for epoch in range(100):
    kernel = generate_random_kernel()
    signal = encode_with_kernel(data, kernel)
    decoded = model(signal, kernel=None)  # Must discover kernel
    loss = compute_loss(decoded, data)
    optimizer.step(loss)
```

### Pass 2: Generated Kernel Optimization
Use Pass 1 model to generate realistic kernels, then train with those kernels for optimized performance.

```python
generator = Pass1Model.kernel_generator
for epoch in range(100):
    kernel = generator(channel_state)
    signal = encode_with_kernel(data, kernel)
    decoded = model(signal, kernel)
    loss = compute_loss(decoded, data)
    optimizer.step(loss)
```

## Three-Stage Expert Training

### Stage 1: Independent Expert Training
Train all expert networks in parallel with their specific data:
- **Noise Expert**: QRN/QRM recordings from WebSDRs
- **Signal Expert**: Multi-user synthetic scenarios
- **Propagation Expert**: FT8/WSPR propagation reports
- **Pattern Expert**: Shannon optimization tasks
- **Spectrum Expert**: Interference avoidance scenarios

### Stage 2: Conductor Training
Freeze expert networks and train conductor to coordinate:
```python
for expert in experts:
    expert.freeze()

for batch in training_data:
    expert_outputs = [e(batch) for e in experts]
    final_output = conductor(expert_outputs)
    loss = compute_loss(final_output, target)
    optimizer.step(loss)
```

### Stage 3: Joint Fine-Tuning
Unfreeze all networks and fine-tune together:
```python
for expert in experts:
    expert.unfreeze()

for batch in training_data:
    output = full_model(batch)
    loss = compute_loss(output, target)
    optimizer.step(loss)
```

Adjust compute allocation based on results - if conductor struggles, allocate more Stage 2 training.

## Data Sources

### Real-World Data
- **QRN/QRM**: 10,000+ hours WebSDR recordings
- **Propagation**: 100M+ FT8/WSPR contacts
- **Multipath**: Amateur repeater recordings
- **Interference**: CASCADE self-generated patterns

### Synthetic Data
- Multi-user collision scenarios
- Edge cases (extreme SNR, heavy QRM)
- Adversarial examples for robustness
- Rare propagation conditions

## Continuous Learning

### ACK-Based Adaptation
Learn from every ACK received:
```python
def process_ack(ack_message):
    # Update link quality matrix
    link_quality[ack['from']] = ack['snr']

    # Track pattern success
    for pattern in ack['patterns_decoded']:
        pattern_success[pattern] += 1

    # Store kernel hints
    if 'kernel_generated' in ack:
        kernel_cache[ack['from']] = ack['kernel_generated']

    # Online learning step
    if accumulated_acks > threshold:
        model.adaptation_step(accumulated_acks)
        accumulated_acks.clear()
```

### Privacy-Preserving Telemetry
- **Differential Privacy**: ε=1.0 on all samples
- **No PII**: No callsigns or message content
- **Generalization**: Grid squares to 1000km
- **K-Anonymity**: Minimum K=10 samples

### Model Updates
- Collect telemetry for 30 days
- Train improved model offline
- Deploy if 5%+ improvement
- Maintain backward compatibility

## Training Infrastructure

### Hardware Requirements
- **Initial Training**: 4× RTX 4090 for 1 week
- **Fine-Tuning**: 1× GPU for 24 hours
- **Continuous Learning**: CPU sufficient

### Software Stack
- PyTorch 2.0+ for training
- Mixed precision (FP16) for efficiency
- Distributed training across GPUs
- Checkpointing every epoch

### Validation Strategy
- 80/10/10 train/val/test split
- Cross-validation on propagation conditions
- Holdout test set from different geography
- A/B testing in deployment

## Performance Metrics

### Primary Metrics
- **Shannon Efficiency**: Target 83-93%
- **Multi-User Capacity**: 1-50 users
- **SNR Range**: -25 to +15 dB
- **Inference Speed**: <10ms on Pi 4

### Secondary Metrics
- Pattern collision rate
- Kernel hint effectiveness
- ACK success rate
- Link quality prediction accuracy

## Model Deployment

### Optimization for Edge Devices
- Quantization to INT8 (75% size reduction)
- TorchScript compilation
- ONNX export for embedded
- Total model size: ~365MB → ~90MB quantized

### Versioning Strategy
- Semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking protocol changes
- MINOR: New capabilities
- PATCH: Performance improvements

### Rollback Capability
- Keep previous model version
- Monitor performance metrics
- Automatic rollback if degradation
- User option to force version