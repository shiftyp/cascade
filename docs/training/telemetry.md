# Privacy-Preserving Telemetry

CASCADE improves continuously through telemetry while protecting user privacy. No callsigns, message content, or exact locations are ever collected.

## Telemetry Data Structure

### What We Collect
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

## Differential Privacy

### Noise Addition (ε=1.0)
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

### K-Anonymity (K≥10)
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

## Compression Strategy

### Aggressive Compression (100:1)
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

        # Typical: 100 samples × 500 bytes → 500 bytes compressed
        return compressed
```

### Batching Strategy
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

## Grid Square Generalization

### Location Privacy
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
    """1000km precision"""
    # Only keep field (first 2 chars)
    return grid[:2] + "55"  # Center of field
```

## Telemetry Categories

### Channel Quality Metrics
```python
channel_telemetry = {
    'noise_floor': quantize(-120, 10),    # 10 dB bins
    'occupancy': round(percent, 10),      # 10% bins
    'multipath_spread': boolean,          # Present/absent
    'fading_rate': ['slow', 'medium', 'fast'][category]
}
```

### Performance Metrics
```python
performance_telemetry = {
    'decode_rate': round(rate, 0.1),      # 0.0-1.0
    'throughput_class': ['low', 'medium', 'high'],
    'latency_ms': round(latency, 50),     # 50ms bins
    'cpu_usage': round(percent, 25)       # 25% bins
}
```

### Error Patterns
```python
error_telemetry = {
    'error_type': ['timeout', 'crc', 'pattern', 'other'],
    'error_rate': round(rate, 0.05),      # 5% bins
    'recovery_success': boolean,
    'retry_helpful': boolean
}
```

## Telemetry Processing Pipeline

### Local Processing
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

### Server-Side Aggregation
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

## Opt-Out and Control

### User Control
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

### Transparency
```python
def show_telemetry_sample():
    """Show user what we collect"""

    sample = collect_telemetry()
    anonymized = process_telemetry_locally(sample)

    print("This is what CASCADE collects:")
    print(json.dumps(anonymized.to_dict(), indent=2))
    print("\nNotice: No callsigns, no messages, no exact location")
```

## Benefits of Telemetry

### Model Improvement
- Identify common failure modes
- Discover unexpected propagation
- Optimize for real conditions

### Network Benefits
- Understand usage patterns
- Improve spectrum efficiency
- Enhance multi-user performance

### User Benefits
- Better performance over time
- Adaptation to local conditions
- Reduced failure rates

## Security Considerations

### Transport Security
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

## Summary

CASCADE's telemetry system:
- **Protects Privacy**: ε=1.0 differential privacy, K≥10 anonymity
- **Minimizes Bandwidth**: 100:1 compression, ~500 bytes per batch
- **Improves System**: Enables continuous learning
- **Respects Users**: Opt-in, transparent, no PII