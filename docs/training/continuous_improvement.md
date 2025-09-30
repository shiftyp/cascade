# Continuous Model Improvement

CASCADE improves continuously through privacy-preserving telemetry and federated learning. The system collects anonymized performance data and computes model updates locally, enabling ongoing refinement while protecting user privacy.

## Overview

Post-deployment learning operates through two complementary mechanisms:

1. **Privacy-Preserving Telemetry**: Collects anonymized performance metrics to identify improvement opportunities
2. **Federated Learning**: Computes model updates locally without sharing raw data

Both systems prioritize privacy, efficiency, and robustness against adversarial contributions.

## Privacy-Preserving Telemetry

CASCADE collects anonymized performance data to understand real-world usage patterns and identify areas for improvement. No callsigns, message content, or exact locations are ever collected.

### Telemetry Data Structure

The telemetry system collects three categories of information, all quantized and anonymized:

```python
class TelemetrySample:
    """Anonymized performance data"""

    def __init__(self):
        # NO personally identifiable information
        # NO message content
        # NO exact locations
        # NO callsigns

        self.metadata = {
            'timestamp': round_to_hour(),  # Hour precision only
            'grid_square': grid[:4],       # 4-char grid (70×35 km area)
            'band': frequency_band,         # e.g., "20m"
            'mode': 'CASCADE-1.0'
        }

        self.channel_features = {
            'snr_class': quantize_snr(),   # Low/Med/High bins
            'qrm_type': classify_qrm(),    # Generic categories
            'qrn_level': round(qrn_db, 5), # 5 dB steps
            'multipath': boolean,           # Present/absent
            'time_of_day': hour_utc // 6   # 4 time bins
        }

        self.performance = {
            'decode_success': boolean,
            'patterns_used': pattern_count,  # Not which patterns
            'confidence': round(conf, 0.1),
            'computation_ms': round(time),
            'retry_count': min(retries, 3)
        }
```

### Privacy Guarantees

The telemetry system implements multiple layers of privacy protection:

#### Differential Privacy (ε=1.0)

Laplace noise is added to all continuous values to prevent exact reconstruction:

```python
def add_differential_privacy(value, sensitivity=1.0, epsilon=1.0):
    """Add Laplace noise for differential privacy"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

# Applied to all continuous values
sample.performance['confidence'] += add_differential_privacy(0, 0.1, 1.0)
sample.channel_features['qrn_level'] += add_differential_privacy(0, 5, 1.0)
```

#### K-Anonymity (K≥10)

Rare combinations are not transmitted to prevent re-identification:

```python
def ensure_k_anonymity(samples, k=10):
    """Don't send rare combinations"""

    # Group by identifying features
    groups = defaultdict(list)
    for sample in samples:
        key = (
            sample.metadata['grid_square'],
            sample.metadata['band'],
            sample.channel_features['snr_class']
        )
        groups[key].append(sample)

    # Only send groups with K or more samples
    anonymized = []
    for group in groups.values():
        if len(group) >= k:
            anonymized.extend(group)

    return anonymized
```

#### Location Generalization

Grid squares are truncated to reduce location precision from ~10×5 km to ~70×35 km:

```python
def generalize_grid_square(full_grid):
    """Reduce location precision"""

    # Full grid: "FN42mc" (10×5 km)
    # Generalized: "FN42" (70×35 km)

    if len(full_grid) >= 4:
        return full_grid[:4]
    return full_grid

# Further generalization for privacy
def ultra_generalize(grid):
    """1000km precision for sensitive cases"""
    # Only keep field (first 2 chars)
    return grid[:2] + "55"  # Center of field
```

### Compression Strategy

Aggressive compression minimizes bandwidth usage while maintaining data utility:

```python
class TelemetryCompressor:
    """Minimize bandwidth usage"""

    def __init__(self):
        self.compressor = zstd.ZstdCompressor(level=22)  # Max compression
        self.dictionary = self.train_dictionary()

    def train_dictionary(self):
        """Train Zstandard dictionary on typical samples"""
        typical_samples = generate_typical_telemetry(n=10000)
        return zstd.train_dictionary(
            typical_samples,
            dict_size=8192  # 8KB dictionary
        )

    def compress_batch(self, samples):
        """Compress batch of telemetry"""

        # Convert to efficient binary format
        binary_batch = msgpack.packb({
            'version': 1,
            'samples': [s.to_dict() for s in samples],
            'checksum': compute_checksum(samples)
        })

        # Compress with dictionary
        compressed = self.compressor.compress(
            binary_batch,
            dict_data=self.dictionary
        )

        # Typical: 100 samples × 500 bytes → 500 bytes compressed (100:1)
        return compressed
```

### Batching Strategy

Telemetry is collected locally and sent in batches to reduce transmission overhead:

```python
def batch_telemetry():
    """Collect and send in batches"""

    buffer = []
    last_send = time.time()

    while True:
        sample = collect_telemetry()
        buffer.append(sample)

        # Send when buffer full or timeout
        if len(buffer) >= 100 or time.time() - last_send > 3600:
            anonymized = ensure_k_anonymity(buffer)
            compressed = compress_batch(anonymized)

            # Only ~500 bytes to send
            send_telemetry(compressed)

            buffer = []
            last_send = time.time()
```

### Telemetry Processing Pipeline

The complete pipeline ensures privacy at every stage:

```python
def process_telemetry_locally(raw_data):
    """Process before sending"""

    # 1. Remove PII
    cleaned = remove_identifiers(raw_data)

    # 2. Quantize values
    quantized = quantize_all_values(cleaned)

    # 3. Add DP noise
    private = add_differential_privacy(quantized)

    # 4. Check K-anonymity
    if is_unique_combination(private):
        return None  # Don't send

    return private
```

Server-side aggregation further protects privacy by never storing individual samples:

```python
def aggregate_telemetry(samples):
    """Server aggregates samples"""

    aggregated = {
        'total_samples': len(samples),
        'success_rate': np.mean([s.decode_success for s in samples]),
        'snr_distribution': compute_histogram([s.snr_class for s in samples]),
        'geographic_distribution': count_grid_squares([s.grid for s in samples]),
        'time_distribution': count_time_bins([s.time_of_day for s in samples])
    }

    # No individual samples stored
    return aggregated
```

### User Control and Transparency

Users maintain full control over telemetry with explicit opt-in:

```python
class TelemetryControl:
    def __init__(self):
        self.enabled = self.load_user_preference()
        self.level = self.load_privacy_level()

    def collect_if_allowed(self):
        if not self.enabled:
            return None

        if self.level == 'minimal':
            return collect_minimal_telemetry()
        elif self.level == 'standard':
            return collect_standard_telemetry()
        elif self.level == 'detailed':
            return collect_detailed_telemetry()

    def load_user_preference(self):
        # Default: opt-in required
        return config.get('telemetry_enabled', False)
```

Users can inspect exactly what will be collected:

```python
def show_telemetry_sample():
    """Show user what we collect"""

    sample = collect_telemetry()
    anonymized = process_telemetry_locally(sample)

    print("This is what CASCADE collects:")
    print(json.dumps(anonymized.to_dict(), indent=2))
    print("\nNotice: No callsigns, no messages, no exact location")
```

## Federated Learning

CASCADE uses federated learning to improve models without sharing raw data. Gradients are computed locally on each user's device and aggregated centrally using Byzantine-robust methods that protect against malicious contributions.

### Local Gradient Computation

Each CASCADE deployment maintains a local experience buffer and computes gradients from its own transmission history:

```python
class LocalLearner:
    """Compute gradients locally without sharing data"""

    def __init__(self, base_model):
        self.model = base_model.clone()
        self.experience_buffer = deque(maxlen=1000)
        self.min_samples = 100

    def collect_experience(self, transmission):
        """Store local transmission results"""

        # Keep only channel characteristics and outcome
        experience = {
            'channel_sample': extract_channel_features(transmission),
            'encoding_used': transmission.encoding_params,
            'success': transmission.decode_success,
            'snr_estimate': transmission.measured_snr
        }

        self.experience_buffer.append(experience)

    def compute_local_update(self):
        """Compute gradient from local experience"""

        if len(self.experience_buffer) < self.min_samples:
            return None  # Not enough data

        # Local training step
        local_gradient = []

        for batch in batch_iterator(self.experience_buffer):
            # Forward pass on local data
            predictions = self.model(batch.channel_samples)

            # Compute loss
            loss = cascade_loss(predictions, batch.ground_truth)

            # Backward pass
            grad = torch.autograd.grad(loss, self.model.parameters())
            local_gradient.append(grad)

        # Average gradients
        avg_gradient = average_gradients(local_gradient)

        # Add differential privacy noise
        private_gradient = add_dp_noise(avg_gradient, epsilon=1.0)

        return private_gradient
```

The key advantage is that raw transmission data never leaves the local device. Only aggregated gradients are shared, and these are further protected by differential privacy noise.

### Byzantine-Robust Aggregation

The central server must aggregate gradients from potentially thousands of users, some of whom may be malicious or misconfigured. CASCADE implements multiple layers of defense:

```python
class ByzantineRobustAggregator:
    """Aggregate gradients robustly"""

    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.history = deque(maxlen=100)

    def aggregate(self, client_gradients):
        """Byzantine-robust aggregation"""

        # 1. Statistical filtering
        filtered = self.statistical_filter(client_gradients)

        # 2. Krum selection
        selected = self.krum_selection(filtered)

        # 3. Trimmed mean
        aggregated = self.trimmed_mean(selected)

        # 4. Norm clipping
        clipped = self.clip_norms(aggregated)

        return clipped

    def statistical_filter(self, gradients):
        """Remove statistical outliers"""
        norms = [torch.norm(g) for g in gradients]
        median = np.median(norms)
        mad = np.median(np.abs(norms - median))

        # Keep gradients within 3 MAD
        filtered = []
        for g, norm in zip(gradients, norms):
            if abs(norm - median) < 3 * mad:
                filtered.append(g)

        return filtered

    def krum_selection(self, gradients, f=2):
        """Select f-resilient subset using Krum algorithm"""
        n = len(gradients)
        scores = []

        for i, g_i in enumerate(gradients):
            # Compute distances to all other gradients
            distances = []
            for j, g_j in enumerate(gradients):
                if i != j:
                    dist = torch.norm(g_i - g_j)
                    distances.append(dist)

            # Score is sum of n-f-1 closest
            distances.sort()
            score = sum(distances[:n-f-1])
            scores.append(score)

        # Select gradient with minimum score
        best_idx = np.argmin(scores)
        return gradients[best_idx]

    def trimmed_mean(self, gradients, trim_ratio=0.1):
        """Compute trimmed mean"""
        # Stack gradients
        stacked = torch.stack(gradients)

        # Sort along client dimension
        sorted_grads, _ = torch.sort(stacked, dim=0)

        # Trim top and bottom
        trim_count = int(len(gradients) * trim_ratio)
        trimmed = sorted_grads[trim_count:-trim_count]

        # Average remaining
        return torch.mean(trimmed, dim=0)

    def clip_norms(self, gradient, max_norm=1.0):
        """Clip gradient norms"""
        norm = torch.norm(gradient)
        if norm > max_norm:
            gradient = gradient * (max_norm / norm)
        return gradient
```

This multi-layered approach provides strong protection against Byzantine attacks while maintaining model improvement quality.

### Model Update Strategy

New models are deployed cautiously with gradual rollout and performance monitoring:

```python
def should_update_model(new_model, current_model, test_set):
    """Decide if new model is better"""

    current_performance = evaluate(current_model, test_set)
    new_performance = evaluate(new_model, test_set)

    # Require 5% improvement to justify update
    improvement = (new_performance - current_performance) / current_performance

    if improvement > 0.05:
        return True, improvement
    else:
        return False, improvement
```

```python
class GradualRollout:
    """Slowly deploy new models"""

    def __init__(self):
        self.rollout_percentage = 0
        self.performance_history = []

    def update_rollout(self, performance_metrics):
        """Adjust rollout based on performance"""

        self.performance_history.append(performance_metrics)

        if len(self.performance_history) < 10:
            # Not enough data
            return

        # Check if performance improving
        trend = np.polyfit(range(10), self.performance_history[-10:], 1)[0]

        if trend > 0:
            # Performance improving, increase rollout
            self.rollout_percentage = min(100, self.rollout_percentage + 10)
        elif trend < -0.01:
            # Performance degrading, rollback
            self.rollout_percentage = max(0, self.rollout_percentage - 20)
        else:
            # Stable, continue
            self.rollout_percentage = min(100, self.rollout_percentage + 5)

    def should_use_new_model(self):
        """Decide which model to use"""
        return random.random() * 100 < self.rollout_percentage
```

### Secure Aggregation

For maximum privacy, CASCADE can optionally use homomorphic encryption to aggregate gradients without the server ever seeing individual contributions:

```python
class HomomorphicAggregator:
    """Aggregate encrypted gradients"""

    def __init__(self):
        self.context = seal.EncryptionParameters(seal.scheme_type.ckks)
        self.context.set_poly_modulus_degree(8192)
        self.context.set_coeff_modulus(seal.CoeffModulus.Create(8192, [60, 40, 40, 60]))

    def aggregate_encrypted(self, encrypted_gradients):
        """Sum encrypted gradients"""

        # Initialize with first gradient
        sum_gradient = encrypted_gradients[0]

        # Add remaining gradients (homomorphically)
        for gradient in encrypted_gradients[1:]:
            sum_gradient = self.add_encrypted(sum_gradient, gradient)

        # Server cannot decrypt individual gradients
        # Only the sum can be decrypted with threshold crypto
        return sum_gradient
```

Alternative secure multi-party computation approach using secret sharing:

```python
def secure_aggregation_protocol():
    """MPC-based aggregation"""

    # Phase 1: Secret sharing
    shares = []
    for client in clients:
        secret = client.compute_gradient()
        client_shares = shamir_secret_share(secret, threshold=len(clients)//2)
        shares.append(client_shares)

    # Phase 2: Aggregation
    aggregated_shares = []
    for i in range(len(shares[0])):
        share_sum = sum(s[i] for s in shares)
        aggregated_shares.append(share_sum)

    # Phase 3: Reconstruction
    aggregated_gradient = shamir_reconstruct(aggregated_shares)

    return aggregated_gradient
```

### Contribution Verification

To prevent free-riding and Sybil attacks, CASCADE verifies that clients are genuinely computing gradients:

```python
def verify_contribution(client_id, gradient):
    """Verify client did work"""

    # Client must provide proof of work
    challenge = generate_challenge(client_id)
    expected_proof = compute_proof(challenge, gradient)

    if not verify_proof(client_proof, expected_proof):
        return False  # Reject contribution

    # Check gradient is reasonable
    if not is_valid_gradient(gradient):
        return False

    return True
```

A reputation system tracks contribution quality over time:

```python
class ClientReputation:
    """Track client contribution quality"""

    def __init__(self):
        self.reputation = defaultdict(lambda: 1.0)
        self.contribution_count = defaultdict(int)

    def update_reputation(self, client_id, contribution_quality):
        """Update client reputation"""

        old_rep = self.reputation[client_id]
        self.contribution_count[client_id] += 1

        # Quality metrics
        if contribution_quality > 0.9:
            delta = 0.1
        elif contribution_quality > 0.7:
            delta = 0.0
        else:
            delta = -0.1

        # Update with momentum
        new_rep = 0.9 * old_rep + 0.1 * (old_rep + delta)
        self.reputation[client_id] = np.clip(new_rep, 0.1, 2.0)

    def get_weight(self, client_id):
        """Get aggregation weight based on reputation"""
        return self.reputation[client_id]
```

### Communication Efficiency

Federated learning can generate significant network traffic. CASCADE implements compression and asynchronous updates to minimize bandwidth:

```python
def compress_gradient(gradient, compression_ratio=0.01):
    """Compress gradients before sending"""

    # Top-k sparsification
    flat_gradient = flatten(gradient)
    k = int(len(flat_gradient) * compression_ratio)

    # Keep only top-k values
    top_k_indices = torch.topk(torch.abs(flat_gradient), k).indices
    sparse_gradient = torch.zeros_like(flat_gradient)
    sparse_gradient[top_k_indices] = flat_gradient[top_k_indices]

    # Further compress with quantization
    quantized = quantize_gradient(sparse_gradient, bits=8)

    return {
        'indices': top_k_indices,
        'values': quantized,
        'shape': gradient.shape
    }
```

Asynchronous updates allow clients to contribute when convenient:

```python
class AsynchronousFederated:
    """Handle async client updates"""

    def __init__(self):
        self.pending_updates = []
        self.last_aggregation = time.time()

    def receive_update(self, client_id, gradient):
        """Receive async update"""

        self.pending_updates.append({
            'client': client_id,
            'gradient': gradient,
            'timestamp': time.time()
        })

        # Aggregate if enough updates or timeout
        if len(self.pending_updates) >= 10 or \
           time.time() - self.last_aggregation > 3600:
            return self.aggregate_pending()

        return None

    def aggregate_pending(self):
        """Aggregate pending updates"""

        # Weight by staleness
        weights = []
        gradients = []

        for update in self.pending_updates:
            staleness = time.time() - update['timestamp']
            weight = np.exp(-staleness / 3600)  # Exponential decay
            weights.append(weight)
            gradients.append(update['gradient'])

        # Weighted average
        aggregated = weighted_average(gradients, weights)

        self.pending_updates = []
        self.last_aggregation = time.time()

        return aggregated
```

## Security Considerations

Both telemetry and federated learning require robust security measures:

### Transport Security

All communications use TLS with certificate verification:

```python
def send_telemetry(data):
    """Secure transmission"""

    # Encrypt with public key
    encrypted = encrypt_with_cascade_public_key(data)

    # Send over TLS
    response = requests.post(
        'https://telemetry.cascade-radio.org',
        data=encrypted,
        verify=True  # Verify certificate
    )
```

### No Correlation Attacks

Random delays and batch mixing prevent linking samples to specific users:

```python
def prevent_correlation():
    """Prevent linking samples"""

    # Random delays
    delay = random.uniform(0, 3600)  # 0-1 hour
    time.sleep(delay)

    # Random ordering
    samples = random.shuffle(samples)

    # Batch mixing
    samples = mix_with_other_users(samples)
```

## Benefits and Metrics

### Model Improvement Benefits

Continuous learning enables CASCADE to:
- Identify and fix common failure modes
- Discover unexpected propagation conditions
- Optimize for real-world usage patterns
- Adapt to changing solar and geomagnetic conditions

### Network Benefits

Aggregated telemetry helps improve overall system performance:
- Better understanding of spectrum usage patterns
- Improved multi-user coordination
- Enhanced interference avoidance
- More efficient resource allocation

### User Benefits

Individual users benefit from collective improvement:
- Better performance over time without manual updates
- Adaptation to local propagation conditions
- Reduced failure rates in challenging conditions
- Access to improvements from global user community

## Summary

CASCADE's continuous improvement system achieves multiple goals simultaneously:

**Privacy Protection**:
- ε=1.0 differential privacy on all telemetry
- K≥10 anonymity for rare combinations
- Local gradient computation (no raw data sharing)
- Optional homomorphic encryption

**Efficiency**:
- 100:1 telemetry compression
- Gradient sparsification and quantization
- Asynchronous updates with staleness weighting
- ~500 bytes per telemetry batch

**Robustness**:
- Byzantine-fault tolerant aggregation
- Multi-layered outlier detection
- Reputation-weighted contributions
- Gradual rollout with automatic rollback

**Effectiveness**:
- 5%+ performance improvement threshold
- Continuous model refinement
- Collective learning from global deployments
- No manual intervention required

This architecture enables CASCADE to improve indefinitely while respecting user privacy and maintaining system security.

## See Also

- **[Privacy Protection](../privacy.md)** - Core privacy principles and anonymization methods
- **[Data Pipeline](data_pipeline.md#phase-2-telemetry-based-gap-closure-post-deployment)** - How telemetry addresses geographic gaps
- **[Long-Term Roadmap](long_term_roadmap.md#transition-to-telemetry-primary-2027)** - Transition from SDR to telemetry-primary data collection
- **[Training README](README.md)** - Overall training and continuous learning strategy