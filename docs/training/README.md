# Training Strategy

CASCADE's training approach emphasizes learning from real-world conditions while maintaining privacy and enabling continuous improvement. The complete training pipeline spans 21 months, from initial data collection through final model deployment.

## Table of Contents

1. [Training Pipeline Overview](#training-pipeline-overview)
2. [Advanced Training Strategies](#advanced-training-strategies-from-embedding-analytics)
3. [Two-Pass Kernel Training](#two-pass-kernel-training)
4. [Three-Stage Expert Training](#three-stage-expert-training)
5. [Data Sources](#data-sources)
6. [Temporal Diversity Assurance](#temporal-diversity-assurance)
7. [Validation Through Real-World Testing](#validation-through-real-world-testing)
8. [Continuous Learning](#continuous-learning)
9. [Training Infrastructure](#training-infrastructure)

## Executive Summary

CASCADE's training strategy solves a fundamental challenge: how to train an adaptive HF radio system without existing CASCADE deployments. The solution uses **three-stage knowledge transfer**:

1. **Data Collection (Months 1-18)**: Collect 200k-250k hours of real HF propagation from [800-1100 SDRs](data_pipeline.md#sdr-usage-management) worldwide, capturing FT8/WSPR signals and atmospheric noise (see [Data Pipeline](data_pipeline.md))
2. **Knowledge Compression (Month 19)**: Train [embedding VAEs](embedding_models.md) that compress 40-50TB of IQ data into 15-25GB of embeddings, preserving propagation diversity
3. **CASCADE Training (Months 20-21)**: Train CASCADE on synthetic signals augmented with real embeddings, learning to handle authentic propagation conditions

**Key Innovation**: Embeddings are used **only during training** to create realistic conditions. At inference, CASCADE processes raw IQ samples end-to-end, requiring no separate embedding computation.

**Training Philosophy**:
- **Real data only**: No synthetic propagation models, only authentic recordings
- **[Diversity-biased sampling](data_pipeline.md#diversity-biased-sampling)**: Rare events (K≥5 storms, M/X flares) get 100x training weight
- **[Natural correlation preservation](data_pipeline.md#correlation-preservation)**: Noise and propagation from same time/place stay paired
- **[Privacy-first](../privacy.md)**: Callsigns hashed, grid squares preserved, no message content stored

**Training Scale**:
- Data: 200k-250k hours raw → 3-5TB curated → 15-25GB [embeddings](embedding_models.md)
- Duration: 18 months collection + 3 months processing/training
- Compute: 4× RTX 4090 GPUs for 1-2 weeks (CASCADE training)
- Storage: [40-50TB cold storage](data_pipeline.md#storage-requirements), 3-5TB hot NVMe, 15-25GB embeddings

## Training Pipeline Overview

The CASCADE training process follows a carefully orchestrated sequence:

1. **Months 1-18**: Collect 40-50TB of raw IQ recordings from global KiwiSDR/WebSDR network
2. **Month 19**: Create diversity-biased training dataset (3-5TB) and train embedding models
3. **Months 19-20**: Generate channel embeddings from curated dataset
4. **Months 20-21**: Train CASCADE model using correlated embedding pairs

This approach maximizes the diversity of training conditions while maintaining computational efficiency. For detailed information on each stage, see:
- [Data Pipeline](data_pipeline.md) - Complete data flow from collection to training including geographic diversity strategies
- [Embedding Models](embedding_models.md) - VAE architectures, propagation augmentation, and embedding analytics
- [Continuous Improvement](continuous_improvement.md) - Privacy-preserving telemetry and federated learning
- [Long-Term Roadmap](long_term_roadmap.md) - Multi-decadal planning, climate adaptation, and scientific legacy (2025-2040)

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
Train all [expert networks](../model/experts.md) in parallel with their specific data:
- **[Noise Expert](../model/experts.md#noise-expert-network)**: QRN/QRM recordings from WebSDRs
- **[Signal Expert](../model/experts.md#signal-expert-network)**: Multi-user synthetic scenarios
- **[Propagation Expert](../model/experts.md#propagation-expert-network)**: FT8/WSPR propagation reports
- **[Pattern Expert](../model/experts.md#pattern-complexity-expert-network)**: Shannon optimization tasks
- **[Spectrum Expert](../model/experts.md#spectrum-allocation-expert-network)**: Interference avoidance scenarios

### Stage 2: Conductor Training
Freeze expert networks and train [conductor](../model/conductor_details.md) to coordinate:
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

- **Scale**: 200,000-250,000 hours of IQ recordings over 18 months
- **Sources**: 800-1100 receivers globally (600-800 KiwiSDRs + 200-300 WebSDRs), with 50-100 concurrent connections rotating through usage limits
- **Bands**: Six HF amateur bands (80m, 40m, 20m, 15m, 10m, 6m)
- **Content**: FT8/WSPR signals for propagation, quiet periods for noise characterization

### Diversity-Biased Training Set

From the full collection, a 3-5TB curated subset is created that oversamples rare events:

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

## Temporal Diversity Assurance

CASCADE's training data spans multiple natural cycles affecting HF propagation. The 18-month collection period requires systematic strategies to ensure comprehensive temporal coverage despite the limited timeframe relative to longer cycles.

### Systematic Cycle Coverage

**Complete Cycles Captured (18 months):**
- **Diurnal**: 548 complete 24-hour cycles across all time zones
- **Lunar**: 20 complete lunar cycles (new moon to new moon)
- **Solar Rotation**: 24 complete 27-day solar rotations
- **Seasonal**: 1.5 complete annual cycles with systematic balancing

**Partial Cycles with Compensation:**
- **Solar Cycle (11 years)**: Only solar minimum captured, requires enhanced rare event collection
- **QBO (28 months)**: Single phase transition captured, enhanced equatorial monitoring

### Solar Minimum Compensation Strategy

Since data collection occurs during Solar Cycle 25 minimum (2025-2026), the system implements several compensation strategies:

```python
def solar_minimum_compensation():
    """
    Adjust collection parameters for low solar activity period
    """
    compensation_factors = {
        # Aggressive rare event boost strategy
        'storm_threshold': 3,        # K≥3 vs normal K≥5
        'flare_threshold': 'C',      # Include ALL C-class flares
        'capture_rate': 1.0,         # 100% capture for any activity

        # Aggressive boost multipliers
        'k3_storm_boost': 50,        # 50x vs normal 8x (6x increase)
        'k4_storm_boost': 100,       # 100x vs normal 25x (4x increase)
        'k5_storm_boost': 500,       # 500x vs normal 50x (10x increase)
        'c_flare_boost': 25,         # NEW: Include C-class (was 0)
        'm_flare_boost': 200,        # 200x vs normal 50x (4x increase)
        'x_flare_boost': 2000,       # 2000x vs normal 500x (4x increase)

        # Opportunity-cost driven philosophy
        'philosophy': 'Maximize limited window - hoard all rare events',
        'future_strategy': 'Phase 2 collection during solar max (2028-2030)'
    }

    return compensation_factors
```

### Training Bias Management for Aggressive Boost Strategy

The aggressive solar minimum boost strategy intentionally creates a biased dataset that overrepresents rare events. This requires careful training compensation:

```python
def solar_minimum_training_bias_correction():
    """
    Compensate for aggressive rare event boosting during training
    """
    # True natural activity distribution
    natural_frequencies = {
        'quiet_sun': 0.70,    # 70% of time during solar cycle
        'moderate': 0.25,     # 25% of time (K=1-3, C-class)
        'active': 0.05        # 5% of time (K≥4, M+ class)
    }

    # Our intentionally biased collection (aggressive boost)
    collected_distribution = {
        'quiet_sun': 0.30,    # Underrepresented due to boost strategy
        'moderate': 0.40,     # Better represented
        'active': 0.30        # Overrepresented due to 5x-10x boost
    }

    # Training weights to correct for bias
    bias_correction_weights = {}
    for activity_level in natural_frequencies:
        natural_freq = natural_frequencies[activity_level]
        collected_freq = collected_distribution[activity_level]
        bias_correction_weights[activity_level] = natural_freq / collected_freq

    # Result: quiet_sun=2.33x, moderate=0.625x, active=0.167x
    return bias_correction_weights

def apply_bias_correction_in_training(batch_data):
    """
    Apply bias correction weights during CASCADE training
    """
    correction_weights = solar_minimum_training_bias_correction()

    for sample in batch_data:
        # Classify sample activity level
        activity_level = classify_activity_level(sample)

        # Apply correction weight
        sample.training_weight *= correction_weights[activity_level]

        # Note: Model still learns from rare events, but understands
        # their true frequency for realistic performance expectations

    return batch_data
```

### Two-Phase Training Strategy

**Phase 1 Training (2025-2026): Solar Minimum Model**
- Train on aggressively boosted solar minimum dataset
- Apply bias correction weights during training
- Model learns rare event handling extremely well
- Accept some bias toward unusual conditions

**Phase 2 Training (2028-2030): Balanced Model**
- Retrain on combined solar minimum + solar maximum data
- Create truly representative long-term model
- Benefit from Phase 1's excellent rare event learning

```python
def phase_1_vs_phase_2_training():
    """
    Different training strategies for different collection phases
    """
    phase_1_strategy = {
        'data_source': 'Solar minimum boost collection (2025-2026)',
        'bias_correction': 'Apply 2.33x quiet sun upweighting',
        'strength': 'Excellent rare event handling',
        'limitation': 'Biased toward unusual conditions',
        'use_case': 'Initial CASCADE deployment'
    }

    phase_2_strategy = {
        'data_source': 'Combined min+max collection (2025-2030)',
        'bias_correction': 'Natural balance, minimal correction',
        'strength': 'Truly representative of all solar conditions',
        'limitation': 'Requires 3+ additional years',
        'use_case': 'Long-term CASCADE optimization'
    }

    return phase_1_strategy, phase_2_strategy
```

This approach acknowledges that we're creating an intentionally biased but scientifically valuable dataset, with explicit plans for bias correction both in training and future data collection.

### Seasonal Balance Enforcement

The system maintains strict seasonal quotas to prevent bias toward any particular season:

```python
class TemporalDiversityManager:
    """
    Ensures balanced representation across all temporal scales
    """
    def __init__(self):
        # Target 25% ±5% per season
        self.seasonal_targets = {
            'WINTER': 0.30,   # Enhanced for solar minimum rarity
            'SPRING': 0.25,   # Standard equinoctial
            'SUMMER': 0.20,   # Reduced (common conditions)
            'AUTUMN': 0.25    # Standard equinoctial
        }

        # Ensure diurnal balance within each season
        self.diurnal_targets = {
            hour: 1/24 for hour in range(24)  # Equal UTC hour coverage
        }

    def check_temporal_balance(self, collected_data):
        """
        Verify temporal diversity meets requirements
        """
        # Seasonal balance check
        seasonal_distribution = calculate_seasonal_distribution(collected_data)
        for season, actual in seasonal_distribution.items():
            target = self.seasonal_targets[season]
            if abs(actual - target) > 0.05:
                raise TemporalImbalanceError(f"{season}: {actual:.3f} vs target {target:.3f}")

        # Diurnal balance within seasons
        for season in ['WINTER', 'SPRING', 'SUMMER', 'AUTUMN']:
            season_data = filter_by_season(collected_data, season)
            hour_distribution = calculate_hour_distribution(season_data)

            # Each hour should have 4.17% ±1% of season's data
            for hour, fraction in hour_distribution.items():
                if abs(fraction - 1/24) > 0.01:
                    raise DiurnalImbalanceError(f"{season} {hour}UTC: {fraction:.3f}")
```

### Cycle-Aware Diversity Scoring

The diversity-biased sampling now incorporates natural cycle awareness:

```python
def calculate_temporal_diversity_score(recording):
    """
    Score recording value considering all natural cycle contexts
    """
    score = 1.0

    # Base rarity (K-index, flares, etc.)
    score *= calculate_base_rarity(recording)

    # Solar cycle phase adjustment
    if recording.solar_cycle_phase == 'MINIMUM':
        # Any activity more valuable during minimum
        if recording.k_index >= 3:
            score *= 1.5
        if recording.xray_class in ['C', 'M', 'X']:
            score *= 1.3

    # Seasonal context
    seasonal_weights = {
        'WINTER': 1.3,  # Rarer conditions at solar minimum
        'SPRING': 1.1,  # Equinoctial enhancement
        'SUMMER': 0.9,  # Common sporadic-E
        'AUTUMN': 1.1   # Equinoctial enhancement
    }
    score *= seasonal_weights[recording.season]

    # Equinoctial period boost
    if recording.equinoctial_enhancement:
        score *= 1.3

    # QBO transition enhancement
    if (recording.qbo_phase == 'TRANSITION' and
        recording.is_equatorial_path):
        score *= 1.4

    # Lunar extremes (new/full moon)
    if recording.lunar_phase in [0.0, 0.5]:
        score *= 1.1

    # Rare cycle combinations
    if (recording.solar_cycle_phase == 'MINIMUM' and
        recording.season == 'WINTER' and
        recording.qbo_phase == 'TRANSITION'):
        score *= 2.0  # Triple rarity combination

    return min(score, 10000)
```

### Cycle Metadata Integration

Every training sample includes comprehensive temporal context:

```python
def generate_cycle_aware_training_batch(embeddings):
    """
    Create training batches with full cycle context
    """
    for embedding in embeddings:
        training_sample = {
            'channel_embedding': embedding.channel_vector,
            'noise_embedding': embedding.noise_vector,
            'path_embedding': embedding.path_vector,

            # Full temporal context
            'cycle_context': {
                'solar_cycle': {
                    'phase': embedding.solar_cycle_phase,
                    'number': 25,
                    'activity_level': 'MINIMUM'
                },
                'seasonal': {
                    'season': embedding.season,
                    'equinoctial': embedding.equinoctial_enhancement,
                    'balance_weight': embedding.seasonal_balance_factor
                },
                'lunar': {
                    'phase': embedding.lunar_phase,
                    'age_days': embedding.lunar_age_days
                },
                'qbo': {
                    'phase': embedding.qbo_phase,
                    'index': embedding.qbo_index
                }
            }
        }

        yield training_sample
```

### Training Implications

The cycle-aware approach impacts training in several ways:

1. **Bias Correction**: Model learns solar minimum conditions aren't "normal"
2. **Seasonal Adaptation**: Equal experience across all seasons prevents seasonal bias
3. **Cycle Extrapolation**: Model can potentially extrapolate to unseen cycle phases
4. **Scientific Validity**: Training data temporal context enables performance correlation analysis

### Validation Metrics

```python
def validate_temporal_diversity(training_dataset):
    """
    Comprehensive temporal diversity validation
    """
    metrics = {
        'seasonal_balance': check_seasonal_quotas(training_dataset),
        'diurnal_coverage': check_24h_coverage(training_dataset),
        'lunar_coverage': check_lunar_phases(training_dataset),
        'solar_minimum_bias': check_activity_distribution(training_dataset),
        'qbo_representation': check_qbo_phases(training_dataset),
        'cycle_combinations': check_rare_combinations(training_dataset)
    }

    # All metrics must pass for training data acceptance
    for metric, passed in metrics.items():
        if not passed:
            raise TemporalDiversityError(f"Failed {metric} validation")

    return metrics
```

This systematic temporal diversity assurance ensures CASCADE trains on truly representative HF propagation conditions across all relevant natural cycles, providing robust performance despite the 18-month solar minimum data collection period.

## Validation Through Real-World Testing

CASCADE validation employs a comprehensive two-stage approach combining controlled laboratory testing with innovative real-world geographic diversity testing using the global amateur radio infrastructure.

### Laboratory Testing Infrastructure

**Software-Defined Radio Simulation:**
```python
class ControlledPropagationTesting:
    """
    Laboratory validation using SDR hardware and GNU Radio channel models
    """
    def __init__(self):
        self.tx_sdr = USRP_B210()  # Or HackRF, BladeRF
        self.rx_sdr = USRP_B210()
        self.channel_sim = GNURadio_Channel_Model()

    def test_cascade_adaptation(self, test_conditions):
        results = {}

        for condition in test_conditions:
            # Configure controlled channel
            self.channel_sim.configure(
                multipath_delays=condition['delays'],
                fading_rate=condition['fading_hz'],
                doppler_spread=condition['doppler_hz'],
                snr_db=condition['target_snr']
            )

            # Generate CASCADE transmission
            test_signal = cascade_encode(test_data, adaptive_params=None)

            # Apply controlled propagation
            received = self.channel_sim.apply(test_signal)

            # Test CASCADE adaptation
            decoded, adaptation_time = cascade_decode_with_timing(received)

            results[condition['name']] = {
                'ber': calculate_ber(test_data, decoded),
                'adaptation_ms': adaptation_time,
                'snr_threshold': find_minimum_snr(condition),
                'success_rate': calculate_success_rate(decoded)
            }

        return results
```

**Multi-User Interference Testing:**
```python
def validate_pattern_system(num_users=10):
    """
    Test CASCADE's interference avoidance with multiple simultaneous users
    """
    # Generate unique patterns for each user
    patterns = [cascade_generate_pattern(user_id=i) for i in range(num_users)]

    # Create test signals for each user
    user_signals = []
    for i, pattern in enumerate(patterns):
        signal = cascade_encode(test_data[i], pattern=pattern)
        user_signals.append(signal)

    # Combine all signals (simulate shared frequency)
    combined_signal = sum(user_signals) + add_realistic_noise(snr=-15)

    # Test each user's decoder independently
    success_rates = []
    for i, pattern in enumerate(patterns):
        decoded = cascade_decode(combined_signal, pattern=pattern)
        success_rate = calculate_success_rate(test_data[i], decoded)
        success_rates.append(success_rate)

    return {
        'users_successful': len([r for r in success_rates if r > 0.9]),
        'average_success_rate': np.mean(success_rates),
        'pattern_collision_rate': calculate_pattern_collisions(patterns),
        'interference_rejection_db': measure_interference_rejection(combined_signal)
    }
```

### Real-World Multi-Path Testing

**Remote Transceiver Network Coordination:**
```python
class GeographicDiversityTesting:
    """
    Coordinate remote transmissions with global SDR reception network
    """
    def __init__(self):
        self.remote_tx_pool = RemoteTransceiverPool()
        self.sdr_rx_network = HybridSDRNetwork()
        self.real_time_optimizer = CascadeOptimizer()

    def execute_geographic_diversity_test(self):
        # Schedule coordinated transmissions from multiple continents
        test_session = {
            'tx_stations': [
                {'service': 'RemoteHams', 'location': 'VK2', 'grid': 'QF56'},
                {'service': 'University_WebSDR', 'location': 'JA1', 'grid': 'PM95'},
                {'service': 'Amateur_Volunteer', 'location': 'G0', 'grid': 'IO91'}
            ],
            'frequency': 14080000,  # 20m test frequency
            'duration': 600,        # 10-minute coordinated test
            'test_data': generate_test_payload()
        }

        # Coordinate simultaneous reception across global SDR network
        rx_stations = self.sdr_rx_network.select_geographic_coverage(
            tx_locations=[s['grid'] for s in test_session['tx_stations']],
            path_types=['short', 'long', 'polar', 'equatorial'],
            min_receivers_per_tx=5  # At least 5 receivers per transmitter
        )

        # Execute real-time multi-path optimization
        results = {}
        for tx_station in test_session['tx_stations']:
            path_results = self.coordinate_real_time_test(
                tx_station, rx_stations, test_session['test_data']
            )
            results[tx_station['location']] = path_results

        return self.analyze_geographic_performance(results)

    def coordinate_real_time_test(self, tx_station, rx_stations, test_data):
        """
        Real-time optimization during live transmission
        """
        transmission_results = []

        for minute in range(10):  # 10-minute test
            # Current CASCADE parameters
            current_params = self.real_time_optimizer.get_current_params()

            # Transmit CASCADE signal with current parameters
            tx_result = self.remote_tx_pool.transmit(
                station=tx_station,
                data=test_data,
                cascade_params=current_params
            )

            # Collect simultaneous reception across multiple SDRs
            rx_results = []
            for rx_sdr in rx_stations:
                reception = self.sdr_rx_network.receive_and_decode(
                    sdr=rx_sdr,
                    target_tx=tx_station,
                    expected_data=test_data
                )
                rx_results.append(reception)

            # Analyze path performance
            path_analysis = self.analyze_propagation_paths(
                tx_station, rx_results
            )

            # Real-time parameter optimization
            optimized_params = self.real_time_optimizer.optimize_for_paths(
                current_performance=path_analysis,
                target_objectives=['reliability', 'throughput', 'fairness']
            )

            # Update parameters for next transmission
            self.real_time_optimizer.update_params(optimized_params)

            transmission_results.append({
                'minute': minute,
                'tx_params': current_params,
                'path_performance': path_analysis,
                'optimization_delta': calculate_parameter_delta(
                    current_params, optimized_params
                )
            })

        return transmission_results
```

**Multi-Path Optimization Validation:**
- **Adaptive Convergence**: Measure how quickly CASCADE optimizes for multiple paths
- **Path Fairness**: Ensure optimization doesn't sacrifice weak paths for strong ones
- **Geographic Scaling**: Validate performance improvement with increased receiver diversity
- **Real-Time Response**: Test adaptation speed to changing propagation conditions
- **Emergency Readiness**: Simulate emergency scenarios with multiple coordinated stations

This enhanced testing methodology demonstrates CASCADE's capabilities through rigorous, innovative validation that leverages the global amateur radio infrastructure for unprecedented real-world testing while maintaining amateur radio community focus and regulatory compliance.

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

**Geographic Gap Closure Strategy:**
Telemetry becomes CASCADE's primary tool for addressing geographic bias post-deployment:

```python
def geographic_telemetry_strategy():
    """
    Use telemetry to progressively achieve global coverage
    """
    return {
        'privacy_guarantees': {
            'differential_privacy': 1.0,      # ε=1.0 noise addition
            'k_anonymity': 10,                # Minimum 10 reports per region
            'grid_precision': 'maidenhead_4', # ~70x35 mile areas
            'temporal_precision': 'hourly',   # Round to nearest hour
            'no_pii': True                    # No callsigns or content
        },
        'collection_priority': {
            'africa': 5.0,                    # Maximum priority
            'pacific_islands': 5.0,           # Maximum priority
            'polar_regions': 4.5,             # Very high priority
            'south_america': 4.0,             # High priority
            'middle_east': 3.5,               # High priority
            'north_america': 0.5,             # Low (well-covered)
            'europe': 0.4                     # Lowest (overcovered)
        },
        'progressive_improvement': {
            'month_0': '40-50% in gaps',      # Initial deployment
            'month_6': '75-85% in gaps',      # First telemetry update
            'month_12': '80-90% in gaps',     # Seasonal coverage
            'month_18': '85-95% globally'     # Full integration
        }
    }
```

**Telemetry Schema:**
- **Location**: Grid square only (FK29 = Fiji region)
- **Propagation**: Detected mode, multipath, Doppler
- **Noise**: QRN level, type, temporal stability
- **Performance**: Adaptation time, efficiency achieved
- **No Content**: Never transmit decoded messages

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

- **Archive Storage** (40-50TB): Tigris S3 cold storage for complete IQ collection
- **Training Storage** (3-5TB): Fast NVMe array for curated dataset
- **Embedding Database** (15-25GB): RAM/SSD for rapid random access
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