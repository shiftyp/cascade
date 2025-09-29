# Training Strategy

CASCADE's training approach emphasizes learning from real-world conditions while maintaining privacy and enabling continuous improvement. The complete training pipeline spans 21 months, from initial data collection through final model deployment.

## Training Pipeline Overview

The CASCADE training process follows a carefully orchestrated sequence:

1. **Months 1-18**: Collect 35-75TB of raw IQ recordings from global KiwiSDR network
2. **Month 19**: Create diversity-biased training dataset (5TB) and train embedding models
3. **Months 19-20**: Generate channel embeddings from curated dataset
4. **Months 20-21**: Train CASCADE model using correlated embedding pairs

This approach maximizes the diversity of training conditions while maintaining computational efficiency. For detailed information on each stage, see:
- [Data Pipeline](data_pipeline.md) - Complete data flow from collection to training
- [Embedding Models](embedding_models.md) - VAE architectures for channel representation
- [Propagation Augmentation](propagation_augmentation.md) - Applying real propagation to synthetic signals
- [Embedding Analytics](embedding_analytics.md) - Storage strategies and clustering insights

## Advanced Training Strategies from Embedding Analytics

The channel embeddings reveal rich structure that directly informs training optimization:

### Curriculum Learning from Clustering

Embeddings naturally cluster into propagation modes, enabling progressive training from typical to challenging conditions:

```python
def curriculum_from_embeddings(embeddings):
    # Find cluster centers (typical propagation)
    kmeans = KMeans(n_clusters=50)
    kmeans.fit(embeddings)

    # Distance to center = difficulty
    difficulties = []
    for emb in embeddings:
        dist_to_center = min([
            np.linalg.norm(emb - center)
            for center in kmeans.cluster_centers_
        ])
        difficulties.append(dist_to_center)

    # Progressive curriculum
    curriculum = {
        'stage_1': np.where(difficulties < percentile(difficulties, 33)),  # Easy
        'stage_2': np.where(percentile(difficulties, 33) <= difficulties < percentile(difficulties, 67)),  # Medium
        'stage_3': np.where(difficulties >= percentile(difficulties, 67)),  # Hard
        'stage_4': find_transitions(embeddings)  # Propagation transitions
    }
    return curriculum
```

### Anomaly-Weighted Training

Rare propagation events discovered through anomaly detection receive increased training weight:

```python
# Isolation Forest identifies rare propagation
iso_forest = IsolationForest(contamination=0.01)
anomaly_scores = iso_forest.fit_predict(embeddings)

# 10x weight for anomalous propagation
training_weights = np.ones(len(embeddings))
training_weights[anomaly_scores == -1] = 10.0
```

These anomalies often represent scientifically interesting propagation modes like trans-equatorial propagation, anomalous daytime DX, or potentially unknown phenomena.

### Diversity-Based Batch Selection

Select training batches that maximize coverage of the embedding space:

```python
def select_diverse_batch(embeddings, batch_size):
    selected = []
    remaining = list(range(len(embeddings)))

    # Start with furthest from mean
    mean_emb = embeddings.mean(axis=0)
    first = np.argmax([np.linalg.norm(e - mean_emb) for e in embeddings])
    selected.append(first)

    # Iteratively add most distant from selected
    while len(selected) < batch_size:
        distances = [
            min([np.linalg.norm(embeddings[r] - embeddings[s])
                 for s in selected])
            for r in remaining
        ]
        next_idx = remaining[np.argmax(distances)]
        selected.append(next_idx)
        remaining.remove(next_idx)

    return selected
```

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

### Real-World Data Collection

The training data comes entirely from real-world recordings, with no synthetic propagation mixing:

- **Scale**: 150,000-300,000 hours of IQ recordings over 18 months
- **Sources**: 30+ KiwiSDR receivers worldwide, rotated for geographic diversity
- **Bands**: Six HF amateur bands (80m, 40m, 20m, 15m, 10m, 6m)
- **Content**: FT8/WSPR signals for propagation, quiet periods for noise characterization

### Diversity-Biased Training Set

From the full collection, a 5TB curated subset is created that oversamples rare events:

- **Ultra-rare events** (100% included): K≥8 storms, X-class flares, exotic propagation
- **Very rare events** (80% sampled): K=6-7 storms, M-class flares, aurora
- **Rare events** (30% sampled): Sporadic-E, extreme SNRs
- **Common conditions** (1% sampled): Regular F2 propagation

This ensures CASCADE learns to handle edge cases without being dominated by common conditions.

### Natural Correlation Preservation

Noise and propagation characteristics from the same time and location remain paired:

```python
# Correlated extraction from same recording
recording = load("20m_storm_2025-06-15.iq")
noise_embedding = extract_noise(recording)      # Storm noise
prop_embedding = extract_propagation(recording) # Storm propagation
training_pair = (noise_embedding, prop_embedding)  # Keep together!
```

This avoids impossible combinations like Arctic noise with tropical propagation.

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

The complete pipeline requires different computational resources at each stage:

- **Data Collection** (Months 1-18): Minimal compute, mainly storage I/O
- **Embedding Model Training** (Week 2-3 of Month 19): 1× RTX 4090 for 3-5 days
- **Embedding Generation** (Weeks 3-4 of Month 19): 4× GPUs + 32 CPU cores for 2-3 weeks
- **CASCADE Training** (Months 20-21): 4× RTX 4090 for 1-2 weeks
- **Continuous Learning**: CPU sufficient for online updates

### Storage Architecture

Different storage tiers optimize cost and performance:

- **Archive Storage** (35-75TB): Cold storage/NAS for complete IQ collection
- **Training Storage** (5TB): Fast NVMe array for curated dataset
- **Embedding Database** (25GB): RAM/SSD for rapid random access
- **Model Storage** (1GB): Version-controlled model checkpoints

### Software Stack
- PyTorch 2.0+ for training
- Mixed precision (FP16) for efficiency
- Distributed training across GPUs
- PostgreSQL with pgvector for embedding storage
- Memory-mapped arrays for fast data loading

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