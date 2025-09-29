# Propagation Augmentation for CASCADE Training

This document describes how CASCADE leverages real-world propagation characteristics extracted from FT8/WSPR recordings to create realistic training conditions. Rather than relying on simplified channel models, the system applies authentic propagation effects captured during 18 months of global monitoring.

## Core Concept

The propagation augmentation system uses a Conditional Variational Autoencoder (CVAE) approach to transfer real channel characteristics from FT8/WSPR signals to synthetic CASCADE transmissions. This ensures the model trains on signals that have experienced genuine ionospheric propagation, multipath, fading, and noise conditions.

The key insight is that FT8 and WSPR signals, with their known transmitted content, allow precise extraction of channel transfer functions. These extracted characteristics can then be applied to CASCADE signals during training, providing authentic propagation without requiring CASCADE deployments during the data collection phase.

## Channel Transfer Learning Algorithm

The algorithm operates in two main phases: channel extraction from real signals and application to training data.

### Phase 1: Channel Characteristic Extraction

When an FT8 signal is successfully decoded, the system can compare the received signal with an ideal reconstruction to extract the channel effects:

```python
def extract_channel_characteristics(received_signal, decoded_message):
    """
    Extract propagation effects by comparing received and ideal signals
    """
    # Generate ideal FT8 signal from decoded message
    ideal_signal = generate_ft8(decoded_message)

    # Align signals in time and frequency
    aligned_received = time_frequency_align(received_signal, ideal_signal)

    # Extract channel in frequency domain
    channel_response = fft(aligned_received) / fft(ideal_signal)

    # Compute propagation metrics
    characteristics = {
        'multipath_profile': extract_multipath(channel_response),
        'fading_pattern': extract_fading(aligned_received, ideal_signal),
        'doppler_spectrum': extract_doppler(channel_response),
        'phase_evolution': extract_phase(aligned_received, ideal_signal)
    }

    return characteristics
```

The extraction process captures several key propagation phenomena:

**Multipath Profile**: By analyzing the impulse response, the system identifies multiple propagation paths with their respective delays and amplitudes. A signal might travel via ground wave, single F2 hop, and double F2 hop simultaneously, each arriving with different delays.

**Fading Characteristics**: Comparing amplitude variations over the transmission reveals fading depth, rate, and correlation time. Slow, deep fades indicate stable ionospheric propagation, while rapid fluctuations suggest disturbed conditions.

**Doppler Effects**: Frequency analysis shows both shifts (from ionospheric motion) and spreading (from multipath with different Doppler on each path). This captures the dynamic nature of the propagation medium.

**Phase Evolution**: Tracking phase changes across symbols reveals path length variations as the ionosphere moves and changes density.

### Phase 2: Application to CASCADE Signals

During training, these extracted characteristics augment synthetic CASCADE signals:

```python
def apply_propagation_to_cascade(cascade_signal, propagation_embedding):
    """
    Apply real propagation characteristics to synthetic CASCADE signal
    """
    # Decode embedding to channel parameters
    channel = cvae_decoder(propagation_embedding)

    # Apply multipath
    multipath_signal = convolve(cascade_signal, channel.impulse_response)

    # Apply fading
    faded_signal = multipath_signal * channel.fading_envelope

    # Apply Doppler
    doppler_signal = apply_doppler_shift(faded_signal, channel.doppler)

    # Apply phase rotation
    rotated_signal = apply_phase_evolution(doppler_signal, channel.phase)

    return rotated_signal
```

## CVAE Architecture

The Conditional Variational Autoencoder learns to encode and decode propagation characteristics in a continuous latent space, enabling smooth interpolation between different channel conditions.

### Encoder Network

The encoder compresses raw channel measurements into a 128-dimensional embedding:

```python
class PropagationEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Process time-domain characteristics
        self.temporal_encoder = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2
        )

        # Process frequency-domain characteristics
        self.spectral_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=32),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128)
        )

        # Combine features for VAE latent space
        self.fc_mu = nn.Linear(256, 128)
        self.fc_logvar = nn.Linear(256, 128)

    def encode(self, temporal_features, spectral_features):
        temporal_encoded, _ = self.temporal_encoder(temporal_features)
        spectral_encoded = self.spectral_encoder(spectral_features)

        combined = torch.cat([temporal_encoded[:, -1],
                             spectral_encoded.squeeze()], dim=1)

        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)

        return mu, logvar
```

### Decoder Network

The decoder reconstructs channel parameters from embeddings:

```python
class PropagationDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Expand embedding to parameter space
        self.expansion = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )

        # Generate specific channel components
        self.multipath_head = nn.Linear(1024, 50)    # Delay taps
        self.fading_head = nn.Linear(1024, 200)      # Fading envelope
        self.doppler_head = nn.Linear(1024, 64)      # Doppler spectrum
        self.phase_head = nn.Linear(1024, 200)       # Phase trajectory

    def decode(self, z):
        expanded = self.expansion(z)

        channel_params = {
            'multipath_taps': self.multipath_head(expanded),
            'fading_envelope': torch.sigmoid(self.fading_head(expanded)),
            'doppler_spectrum': self.doppler_head(expanded),
            'phase_evolution': self.phase_head(expanded) * 2 * np.pi
        }

        return channel_params
```

## Training Process

The CVAE training occurs after the 18-month data collection, using the curated 5TB dataset.

### Training Data Preparation

The system creates training pairs of channel observations and their embeddings:

```python
def prepare_cvae_training_data(recordings):
    training_pairs = []

    for recording in recordings:
        # Extract FT8 signals
        ft8_signals = detect_ft8(recording)

        for signal in ft8_signals:
            # Decode to get reference
            if decoded := ft8_decode(signal):
                # Extract channel characteristics
                channel = extract_channel_characteristics(
                    signal.iq,
                    decoded.message
                )

                # Add contextual information for conditioning
                context = {
                    'snr': signal.snr,
                    'k_index': recording.k_index,
                    'solar_flux': recording.solar_flux,
                    'band': recording.band,
                    'distance': calculate_distance(decoded.grids),
                    'time_of_day': recording.timestamp.hour
                }

                training_pairs.append({
                    'channel': channel,
                    'context': context,
                    'quality': signal.decode_confidence
                })

    return training_pairs
```

### Loss Functions

The CVAE optimizes multiple objectives to ensure realistic channel generation:

```python
def cvae_loss(original_channel, reconstructed_channel, mu, logvar):
    # Reconstruction loss - can we recreate the channel?
    recon_loss = F.mse_loss(reconstructed_channel, original_channel)

    # KL divergence - maintain smooth latent space
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Physics consistency losses
    physics_loss = 0.0

    # Multipath delays must be positive and ordered
    delays = reconstructed_channel['multipath_delays']
    physics_loss += F.relu(-delays).mean()

    # Fading must be bounded
    fading = reconstructed_channel['fading_envelope']
    physics_loss += F.relu(fading - 1.0).mean()

    # Total path power should be conserved
    path_powers = reconstructed_channel['multipath_amplitudes'].pow(2)
    physics_loss += (path_powers.sum() - 1.0).pow(2)

    return recon_loss + 0.01 * kl_loss + 0.1 * physics_loss
```

### Conditioning on Context

The CVAE conditions on contextual information to generate appropriate propagation:

```python
def conditional_generation(context):
    """
    Generate propagation embedding conditioned on context
    """
    # Encode context to condition vector
    condition = encode_context(context)

    # Sample from conditional distribution
    z = sample_conditional_latent(condition)

    # Decode to channel parameters
    channel = decoder(z, condition)

    return channel
```

This allows the system to generate propagation appropriate for specific conditions - for example, generating storm-like propagation when training CASCADE for high K-index scenarios.

## Geographic Context in Propagation Embeddings

FT8/WSPR signals provide unique geographic information through grid squares that pure QRN recordings lack. This path geometry critically affects propagation characteristics and must be captured alongside the channel response.

### Dual Embedding Strategy

The system uses two complementary embeddings:
- **Channel Embedding (128D)**: Pure propagation characteristics from IQ data
- **Path Context Embedding (32D)**: Geographic and geometric features

```python
class PathContextEncoder:
    """
    Encodes geographic path information into a 32D vector
    """
    def encode_path_context(self, tx_grid, rx_grid, timestamp):
        # Basic path geometry
        distance = great_circle_distance(tx_grid, rx_grid)
        bearing = calculate_bearing(tx_grid, rx_grid)

        # Solar illumination geometry
        midpoint = calculate_midpoint(tx_grid, rx_grid)
        solar_zenith_angle = calculate_sza(midpoint, timestamp)
        terminator_distance = distance_to_terminator(midpoint, timestamp)
        tx_day_night = is_daylight(tx_grid, timestamp)
        rx_day_night = is_daylight(rx_grid, timestamp)

        # Geomagnetic context
        mag_lat_tx = geographic_to_magnetic_latitude(tx_grid)
        mag_lat_rx = geographic_to_magnetic_latitude(rx_grid)
        auroral_zone_proximity = min_distance_to_auroral_oval(
            [tx_grid, midpoint, rx_grid], timestamp
        )

        # Ionospheric path features
        likely_hop_count = estimate_hop_count(distance, frequency)
        max_hop_height = estimate_f_layer_height(midpoint, timestamp)
        crosses_equator = path_crosses_magnetic_equator(tx_grid, rx_grid)
        crosses_auroral = path_crosses_auroral_zone(tx_grid, rx_grid)

        # Critical frequencies along path
        foF2_tx = estimate_foF2(tx_grid, timestamp)
        foF2_mid = estimate_foF2(midpoint, timestamp)
        foF2_rx = estimate_foF2(rx_grid, timestamp)

        # Encode as 32D vector
        return np.array([
            distance / 20000,                    # [0-1] normalized
            bearing / 360,                        # [0-1] normalized
            np.sin(bearing * np.pi / 180),       # Circular encoding
            np.cos(bearing * np.pi / 180),       # Circular encoding
            solar_zenith_angle / 180,            # [0-1] normalized
            terminator_distance / 1000,          # km from terminator
            float(tx_day_night),                  # Binary day/night
            float(rx_day_night),                  # Binary day/night
            mag_lat_tx / 90,                      # [-1,1] normalized
            mag_lat_rx / 90,                      # [-1,1] normalized
            auroral_zone_proximity / 1000,       # km to oval
            likely_hop_count / 4,                 # [0-1] typical 1-4 hops
            max_hop_height / 400,                 # km F-layer height
            float(crosses_equator),               # Binary
            float(crosses_auroral),               # Binary
            foF2_tx / 15,                         # MHz critical freq
            foF2_mid / 15,                        # MHz critical freq
            foF2_rx / 15,                         # MHz critical freq
            # ... additional features to reach 32D
        ])
```

### Why Geographic Context Matters

Different propagation paths exhibit distinct characteristics:

**Polar Paths**: Subject to aurora, high absorption, rapid changes
**Equatorial Paths**: TEP enhancement, spread-F, evening anomalies
**East-West Paths**: Follow day-night terminator, gray-line enhancement
**North-South Paths**: Cross different ionospheric zones, seasonal asymmetry

The path context embedding allows the CVAE to learn these relationships:

**Important**: Geographic context is used ONLY during training to create realistic propagation conditions. At inference time, CASCADE operates without any geographic information - it must adaptively decode signals based solely on the received IQ data. This ensures true blind adaptivity in operational deployment.

```python
def cvae_with_path_context(channel_iq, path_context):
    # Channel characteristics from IQ
    channel_embedding = channel_encoder(channel_iq)

    # Condition decoder on both embeddings
    combined = torch.cat([channel_embedding, path_context])

    # Generate channel parameters aware of path geometry
    channel_params = decoder(combined)

    return channel_params
```

## Integration with CASCADE Training

During CASCADE model training, the propagation augmentation system provides realistic channel conditions informed by both IQ characteristics and geographic context:

```python
class CascadeTrainer:
    def __init__(self):
        self.prop_cvae = load_trained_cvae()
        self.path_encoder = PathContextEncoder()
        self.embedding_db = load_embedding_database()

    def train_step(self, batch):
        # Generate fresh CASCADE transmission
        cascade_tx = cascade_encoder(batch.data)

        # Fetch appropriate propagation embedding with path context
        context = {
            'band': batch.band,
            'time': batch.timestamp.hour,
            'k_index': current_k_index(),
            'target_snr': batch.target_snr,
            'distance_range': batch.target_distance  # New!
        }

        # Find similar real-world propagation
        prop_record = self.embedding_db.query_similar(context)

        # Extract both embeddings
        channel_embedding = prop_record.channel_embedding

        # Generate synthetic path context for CASCADE scenario
        if batch.scenario_path:
            # Use specified training path
            path_context = self.path_encoder.encode_path_context(
                batch.scenario_path.tx_grid,
                batch.scenario_path.rx_grid,
                batch.timestamp
            )
        else:
            # Use path context from the FT8 recording
            path_context = prop_record.path_embedding

        # Apply propagation with geographic awareness
        cascade_rx = apply_propagation_with_context(
            cascade_tx,
            channel_embedding,
            path_context
        )

        # Add correlated noise (from same recording)
        noise_embedding = prop_record.paired_noise
        cascade_rx = apply_noise_embedding(cascade_rx, noise_embedding)

        # Train CASCADE decoder
        decoded = cascade_decoder(cascade_rx)
        loss = compute_loss(decoded, batch.data)

        return loss

def apply_propagation_with_context(signal, channel_emb, path_emb):
    """
    Apply propagation aware of geographic context
    """
    # Combine embeddings for conditioning
    combined_context = torch.cat([channel_emb, path_emb])

    # Decoder generates path-appropriate channel
    channel_params = cvae_decoder(combined_context)

    # Path distance affects delay spread
    if path_emb[0] > 0.5:  # Long path
        channel_params['multipath_spread'] *= 1.5

    # Auroral proximity affects fading
    if path_emb[10] < 0.1:  # Near auroral zone
        channel_params['fading_rate'] *= 2.0

    return apply_channel(signal, channel_params)
```

## Benefits of Real Propagation Augmentation

This approach provides several critical advantages over synthetic channel models:

### Authentic Physics

Traditional channel models like Watterson or ITU-R make simplifying assumptions that don't capture the full complexity of HF propagation. Real FT8/WSPR recordings include:

- Non-stationary channel behavior during mode transitions
- Coupled multipath and Doppler effects
- Realistic noise correlations with propagation
- Geographic and temporal dependencies

### Rare Event Coverage

The 18-month collection ensures CASCADE trains on genuine examples of:

- Geomagnetic storm effects (K≥7)
- Solar flare impacts (M and X class)
- Unusual propagation modes (TEP, aurora, ducting)
- Extreme multipath conditions

These rare events are crucial for robust operation but difficult to model synthetically.

### Natural Correlations

By preserving the pairing between noise and propagation from the same recordings, the system maintains physical relationships:

```python
# Storm conditions affect both noise and propagation
storm_recording = load("K7_storm_recording.iq")
storm_noise = extract_noise(storm_recording)
storm_prop = extract_propagation(storm_recording)
# These naturally correlate - both show storm effects
```

### Efficient Training

Channel embeddings enable rapid experimentation:

- 200x compression from IQ to embeddings
- Millisecond retrieval of propagation conditions
- Smooth interpolation between similar conditions
- Reusable across multiple CASCADE variants

## Validation Methods

Several approaches validate the propagation augmentation quality:

### Channel Reconstruction Accuracy

The system verifies it can recreate original propagation:

```python
def validate_reconstruction():
    test_signal = load_test_ft8()
    channel = extract_channel(test_signal)
    embedding = cvae.encode(channel)
    reconstructed = cvae.decode(embedding)

    # Apply reconstructed channel to ideal signal
    augmented = apply_channel(ideal_ft8, reconstructed)

    # Should closely match original
    correlation = correlate(augmented, test_signal)
    assert correlation > 0.9
```

### Physical Plausibility

Generated channels must obey propagation physics:

```python
def validate_physics(generated_channel):
    # Multipath delays increase with hop distance
    assert all(np.diff(generated_channel.delays) > 0)

    # Doppler limited by ionospheric velocity
    assert generated_channel.doppler_spread < 10  # Hz

    # Conservation of energy
    assert 0.8 < generated_channel.total_power < 1.2
```

### Training Improvement

Most importantly, CASCADE performance improves:

```python
def validate_training_benefit():
    # Train with and without propagation augmentation
    cascade_augmented = train_with_propagation()
    cascade_synthetic = train_with_watterson()

    # Real propagation should improve robustness
    assert cascade_augmented.ber < cascade_synthetic.ber
    assert cascade_augmented.multipath_resilience > cascade_synthetic.multipath_resilience
```

## Future Enhancements

Several improvements could further enhance propagation augmentation:

### Continuous Learning

As new propagation data arrives, the CVAE could be fine-tuned:

```python
def online_cvae_update(new_recordings):
    # Extract new propagation examples
    new_channels = extract_channels(new_recordings)

    # Fine-tune CVAE on new data
    cvae.adaptation_step(new_channels)

    # Particularly valuable for rare events
    if is_rare_propagation(new_channels):
        cvae.focused_learning(new_channels)
```

### Multi-Band Coordination

Currently independent per band, the system could learn cross-band relationships:

```python
def multi_band_propagation(band_20m_prop):
    # Infer likely propagation on other bands
    band_40m_prop = cvae.cross_band_inference(band_20m_prop, target='40m')
    return band_40m_prop
```

### Adversarial Robustness

Training could include adversarial propagation conditions:

```python
def generate_adversarial_propagation(cascade_signal):
    # Find propagation that maximally challenges CASCADE
    worst_case = cvae.adversarial_search(cascade_signal)
    return worst_case
```

## Conclusion

The propagation augmentation system transforms CASCADE training from simplified channel models to authentic HF propagation. By extracting real channel characteristics from FT8/WSPR signals and applying them to synthetic CASCADE transmissions, the model learns to handle the full complexity of ionospheric propagation.

This approach ensures CASCADE will perform robustly in real-world deployments, having trained on genuine examples of everything from calm nighttime conditions to severe geomagnetic storms. The 18-month data collection provides the diversity, the CVAE provides the mechanism, and the result is a model truly prepared for the challenges of HF communication.