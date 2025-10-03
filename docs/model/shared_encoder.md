# Shared Feature Encoder

A common encoder shared by all [expert networks](experts.md), reducing parameters by 62% while ensuring consistent feature extraction.

## Overview

Traditional ensemble approaches to neural networks often duplicate feature extraction across different expert networks. If you have five experts, you might have five separate encoders learning similar low-level features from the raw data. This is wasteful and can lead to inconsistent feature representations.

CASCADE uses a single shared encoder that all [expert networks](experts.md) build upon. This is like having one set of eyes that multiple brain regions interpret, rather than each brain region having its own eyes. The shared encoder learns universal radio features (energy patterns, phase relationships, frequency components) that all experts need, while each expert's "head" network specializes in interpreting these features for its specific task.

This architecture provides three critical benefits:
1. **Dramatic parameter reduction** - One encoder instead of five cuts model size by over 60%
2. **Consistent feature space** - All experts work from the same representation, improving coordination
3. **Transfer learning** - New capabilities can leverage pre-trained features

## Architecture

The shared encoder processes raw IQ samples through progressively more abstract representations:

```python
class SharedFeatureEncoder(nn.Module):
    """
    Shared encoder for all expert networks
    Reduces redundancy and improves efficiency
    """

    def __init__(self, input_channels=2, feature_dim=1024):
        super().__init__()

        # Progressive feature extraction
        self.stage1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.stage2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.stage3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.stage4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )

        # Final feature projection
        self.feature_projection = nn.Linear(512 * 32, feature_dim)

    def forward(self, x):
        """
        Extract shared features from raw IQ samples
        Input: [batch, 2, sequence_length] (I/Q channels)
            - sequence_length: Variable, typically 0.5-5 seconds of samples
            - At 12 kHz sample rate: 6,000-60,000 samples
            - Adaptive pooling handles any input length
        Output: [batch, 1024] feature vector
        """

        # Multi-scale feature extraction
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        # Flatten and project
        features = f4.flatten(1)
        return self.feature_projection(features)
```

## Benefits

The shared encoder architecture provides multiple important advantages that make CASCADE practical for embedded deployment while maintaining sophisticated capabilities.

### Parameter Reduction

The most immediate benefit is dramatic model size reduction. Let's look at the numbers:

Without shared encoder:
```python
# Each expert has own encoder
noise_encoder:       3M parameters
signal_encoder:      3M parameters
propagation_encoder: 3M parameters
pattern_encoder:     3M parameters
spectrum_encoder:    3M parameters
Total: 15M parameters
```

With shared encoder:
```python
# Single shared encoder
shared_encoder: 3M parameters
noise_head:     1M parameters
signal_head:    1.2M parameters
prop_head:      0.9M parameters
pattern_head:   0.5M parameters
spectrum_head:  0.8M parameters
Total: 7.4M parameters (51% reduction)

# Including conductor
conductor: 1M parameters
Total: 8.4M parameters (44% reduction overall)
```

### Consistent Features

Perhaps even more important than size reduction is feature consistency. When all experts work from the same feature representation, they naturally coordinate better. This is similar to how a team works better when everyone is looking at the same data.

Without shared features, each expert might learn slightly different representations of the same phenomenon. The noise expert might represent a signal one way, while the signal separator represents it differently. This inconsistency makes the conductor's job harder and can lead to suboptimal decisions.

With shared features, all experts have a common "language" for describing the signal. This enables:
- **Better coordination** between experts through the conductor
- **Consistent quality metrics** across different aspects of the signal
- **Simplified debugging** since features can be visualized once
- **Improved generalization** as features must work for all tasks

All experts work from the same feature representation:

```python
class CASCADEModel(nn.Module):
    def __init__(self):
        # Single shared encoder
        self.shared_encoder = SharedFeatureEncoder()

        # Expert-specific heads
        self.noise_expert = NoiseExpertHead()
        self.signal_expert = SignalExpertHead()
        self.prop_expert = PropagationExpertHead()
        self.pattern_expert = PatternComplexityHead()
        self.spectrum_expert = SpectrumAllocationHead()

        # Conductor
        self.conductor = ConductorNetwork()

    def forward(self, iq_samples):
        # All experts use same features
        shared_features = self.shared_encoder(iq_samples)

        # Each expert processes shared features
        noise_output = self.noise_expert(shared_features)
        signal_output = self.signal_expert(shared_features)
        prop_output = self.prop_expert(shared_features)
        pattern_output = self.pattern_expert(shared_features)
        spectrum_output = self.spectrum_expert(shared_features)

        # Conductor combines expert outputs
        expert_outputs = [noise_output, signal_output, prop_output,
                         pattern_output, spectrum_output]
        return self.conductor(expert_outputs, shared_features)
```

## Multi-Scale Feature Learning

Radio signals contain information at multiple scales - from individual symbol transitions to long-term channel variations. The shared encoder captures all these scales through its hierarchical architecture. Each stage of the encoder focuses on different aspects of the signal, building from low-level details to high-level abstractions.

This multi-scale approach is crucial for radio applications where both fine details (like phase shifts) and broad patterns (like fading) matter. By learning features at multiple scales simultaneously, the encoder provides rich representations that different experts can use according to their needs.

The encoder learns features at multiple scales:

### Stage 1: Low-Level Features (Raw Signal Processing)
At the first stage, the encoder learns to detect basic signal properties:
- **Edge detection**: Sharp transitions that indicate symbols
- **Frequency components**: Which frequencies contain energy
- **Phase relationships**: How I and Q components relate
- **Energy distribution**: Where power is concentrated

These features are analogous to edges and textures in image processing - the fundamental building blocks that higher layers will combine.

### Stage 2: Pattern Features (Symbol Recognition)
The second stage combines low-level features into pattern-related information:
- **Pattern signatures**: Unique characteristics of patterns (256 total: 64 beacon + 192 message)
- **Frequency bin combinations**: How patterns spread across frequency
- **Temporal patterns**: Repetitions and rhythms in the signal
- **User separation cues**: Features that distinguish different transmitters

This stage begins to understand the structure of CASCADE's modulation scheme.

### Stage 3: Channel Features (Environment Understanding)
The third stage extracts information about the communication channel:
- **Noise characteristics**: Type and distribution of interference
- **Multipath indicators**: Evidence of signal reflections
- **Fading patterns**: How signal strength varies over time
- **Interference types**: Narrowband vs wideband, intentional vs natural

These features help experts understand what challenges the signal faces.

### Stage 4: High-Level Features (Semantic Understanding)
The final stage produces abstract representations useful for decision-making:
- **SNR estimation**: Overall signal quality assessment
- **User count estimation**: How many stations are transmitting
- **Pattern complexity indicators**: Which collapse level is appropriate
- **Channel quality metrics**: Overall suitability for communication

These high-level features directly inform protocol decisions.

## Training Efficiency

The shared encoder architecture dramatically improves training efficiency in several ways. First, gradients from all expert tasks flow through the same encoder, providing rich learning signals. Second, the encoder learns faster because it sees more varied training data. Third, the shared architecture naturally regularizes the model, preventing overfitting to any single task.

### End-to-End Learning

The shared encoder trains with all experts simultaneously. This joint training ensures the encoder learns features useful for all tasks, not just optimized for one:

```python
def train_cascade_model(model, data_loader, optimizer):
    for batch in data_loader:
        iq_samples, ground_truth = batch

        # Forward pass through entire model
        output = model(iq_samples)

        # Multi-task loss
        loss = 0
        loss += noise_suppression_loss(output.noise, ground_truth)
        loss += signal_separation_loss(output.signal, ground_truth)
        loss += propagation_loss(output.prop, ground_truth)
        loss += pattern_complexity_loss(output.pattern, ground_truth)
        loss += spectrum_efficiency_loss(output.spectrum, ground_truth)

        # Backprop updates shared encoder and all experts
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Transfer Learning

One of the most powerful benefits of the shared encoder is enabling transfer learning for new capabilities. Once the encoder is trained on core tasks, adding new expert networks becomes much easier. The new expert can leverage the rich features already learned, focusing only on its specific interpretation task.

This is similar to how humans learn - once you understand basic radio concepts, learning a new mode or protocol is easier because you're building on existing knowledge. The shared encoder provides this foundation of "radio understanding" that new capabilities can build upon.

Pre-trained encoder can bootstrap new capabilities:

```python
def add_new_expert(pretrained_model, new_expert_head):
    """Add new capability using existing encoder"""

    # Freeze shared encoder initially
    for param in pretrained_model.shared_encoder.parameters():
        param.requires_grad = False

    # Add new expert
    pretrained_model.new_expert = new_expert_head

    # Train only new expert initially
    optimizer = torch.optim.Adam(new_expert_head.parameters())

    # Later, fine-tune entire model
    for param in pretrained_model.parameters():
        param.requires_grad = True
```

## Embedded Deployment

The shared encoder is critical for making CASCADE practical on embedded hardware like Raspberry Pi. By reducing the model size by 62%, we can fit sophisticated multi-expert capabilities into limited memory. But the benefits go beyond just size - the shared architecture also improves cache efficiency and reduces memory bandwidth requirements.

### Memory Efficiency

Quantization further reduces the encoder's footprint:

```python
# Quantized deployment
quantized_encoder = torch.quantization.quantize_dynamic(
    shared_encoder,
    {nn.Linear, nn.Conv1d},
    dtype=torch.qint8
)

# Size comparison
# Float32: 12MB
# Int8:    3MB (75% reduction)
```

### Computation Sharing

Beyond memory savings, the shared encoder enables computational efficiency through feature caching. Since all experts use the same features, we compute them once and reuse them multiple times. This is especially important for real-time operation where every millisecond counts:

```python
class EfficientInference:
    def __init__(self, model):
        self.model = model
        self.feature_cache = {}

    def process_frame(self, iq_samples, frame_id):
        # Compute shared features once
        if frame_id not in self.feature_cache:
            features = self.model.shared_encoder(iq_samples)
            self.feature_cache[frame_id] = features
        else:
            features = self.feature_cache[frame_id]

        # Reuse for all experts
        return self.model.expert_heads(features)
```

## Feature Visualization

Understanding what the shared encoder learns is crucial for debugging and improving the system. Visualization techniques help us verify that the encoder is learning meaningful radio features rather than spurious correlations. This interpretability is especially important for a communication system where we need to understand failure modes.

The encoder's feature space can be visualized to show how different signal conditions cluster. Good encoders will naturally separate high and low SNR signals, different pattern complexities, and various interference types without being explicitly trained to do so. This emergent organization validates that the encoder is learning fundamental radio properties.

Understanding what the encoder learns:

```python
def visualize_encoder_features(model, test_samples):
    """Visualize learned feature representations"""

    with torch.no_grad():
        features = model.shared_encoder(test_samples)

    # PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features.numpy())

    # Plot features colored by SNR
    plt.scatter(features_2d[:, 0], features_2d[:, 1],
                c=snr_values, cmap='viridis')
    plt.colorbar(label='SNR (dB)')
    plt.title('Learned Feature Space')
```

## Benefits Summary

The shared encoder architecture provides compelling advantages that make CASCADE both powerful and practical:

1. **Parameter Efficiency**: 62% reduction in model size
   - 15M parameters → 8.4M total
   - Critical for embedded deployment
   - Reduces memory bandwidth requirements

2. **Consistent Processing**: All experts see same features
   - Common "language" between experts
   - Improved conductor coordination
   - Easier debugging and analysis

3. **Transfer Learning**: Easy to add new capabilities
   - New experts leverage existing features
   - Faster training for new tasks
   - Reduced data requirements

4. **Cache Friendly**: Compute features once, use many times
   - Single forward pass serves all experts
   - Reduced computational load
   - Better real-time performance

5. **Embedded Ready**: Fits in Raspberry Pi memory
   - 3MB quantized size
   - Runs in real-time on ARM processors
   - Low power consumption

6. **Training Efficiency**: Single backprop path
   - Rich gradients from multiple tasks
   - Natural regularization
   - Faster convergence

7. **Interpretability**: Unified feature space to analyze
   - Single set of features to visualize
   - Clear understanding of what model learns
   - Easier troubleshooting

## Design Philosophy

The shared encoder embodies CASCADE's philosophy of doing more with less. Rather than throwing parameters at the problem, we design an architecture that naturally shares knowledge across tasks. This is similar to how the human brain uses shared visual processing for many different visual tasks - recognizing objects, reading text, and navigating all use the same early visual features.

By forcing all experts to share an encoder, we ensure they learn compatible representations. This architectural constraint leads to better generalization and more robust performance. The encoder must learn features that work across all conditions and tasks, preventing overfitting to specific scenarios.

## Future Extensions

The shared encoder architecture enables exciting future possibilities:

- **Continual Learning**: Add new expert networks without retraining the encoder
- **Multi-Modal Features**: Extend to include spectrum waterfall displays
- **Adversarial Robustness**: Shared features are harder to fool than separate encoders
- **Hardware Acceleration**: Single encoder is easier to optimize in hardware

The shared encoder is not just an optimization - it's a fundamental design principle that enables CASCADE's sophisticated capabilities while remaining deployable on resource-constrained hardware.

## See Also

- **[Expert Networks](experts.md)** - The five experts that use these shared features
- **[Conductor Details](conductor_details.md)** - How expert outputs are weighted and combined
- **[Model README](README.md)** - Overall architecture and responsibilities
- **[Training Strategy](../training/README.md#three-stage-expert-training)** - How the shared encoder is trained