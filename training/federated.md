# Federated Learning

CASCADE uses federated learning to improve models without sharing raw data. Gradients are computed locally and aggregated centrally with Byzantine-robust methods.

## Local Gradient Computation

### Experience Buffer
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

## Byzantine-Robust Aggregation

### Defending Against Bad Updates
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
        """Select f-resilient subset"""
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

## Model Update Strategy

### Update Criteria
```python
def should_update_model(new_model, current_model, test_set):
    """Decide if new model is better"""

    current_performance = evaluate(current_model, test_set)
    new_performance = evaluate(new_model, test_set)

    # Require 5% improvement
    improvement = (new_performance - current_performance) / current_performance

    if improvement > 0.05:
        return True, improvement
    else:
        return False, improvement
```

### Gradual Rollout
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

## Secure Aggregation

### Homomorphic Encryption
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

### Secure Multi-Party Computation
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

## Contribution Verification

### Proof of Work
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

### Reputation System
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

## Communication Efficiency

### Gradient Compression
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

### Asynchronous Updates
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

## Summary

CASCADE's federated learning:
- **Privacy**: Local computation, encrypted aggregation
- **Robustness**: Byzantine-fault tolerant
- **Efficiency**: Compressed, asynchronous updates
- **Verification**: Proof of work, reputation tracking
- **Improvement**: 5%+ performance gains required