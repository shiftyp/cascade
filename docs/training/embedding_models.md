# Channel Embedding Models

CASCADE uses specialized neural networks to create compact representations of radio channel conditions. These embedding models transform raw IQ recordings into dense vectors that capture the essential characteristics of noise environments and propagation effects, enabling efficient and realistic training.

## Triple Encoder Architecture

The system employs three specialized encoders for comprehensive channel characterization:

1. **Quiet Channel VAE**: Generates 64-dimensional embeddings representing noise characteristics
2. **FT8 Propagation VAE**: Generates 128-dimensional embeddings encoding propagation effects from IQ data
3. **Path Context Encoder**: Generates 32-dimensional embeddings from geographic path information

The VAEs learn compressed representations from raw signals, while the Path Context Encoder extracts deterministic features from FT8 grid squares. This separation allows each model to specialize in extracting relevant features from fundamentally different data types while maintaining the ability to combine them naturally during training.

## Understanding Channel Embeddings

Channel embeddings are learned vector representations that capture radio propagation characteristics in a compact numerical format. Think of them as a compressed "fingerprint" of channel conditions - just as word embeddings in natural language processing capture semantic meaning in vectors, channel embeddings capture propagation physics in numbers.

Each embedding encodes complex, high-dimensional propagation measurements into a dense vector:

```python
# Raw propagation data - thousands of parameters
raw_channel = {
    'fading_pattern': [0.8, 0.6, 0.9, ...],    # Time series
    'multipath_delays': [0, 2.3, 5.1, ...],    # Variable length
    'doppler_shifts': [-5, 0, 3, ...],         # Frequency domain
    'phase_rotations': [0.1, 0.3, ...],        # Per symbol
}

# Compressed to embedding - essential characteristics
channel_embedding = [0.23, -0.45, 0.67, ...]  # 64 or 128 values
```

## Quiet Channel VAE Architecture

The Quiet Channel VAE specializes in characterizing noise environments without reference signals. It focuses on statistical properties, spectral characteristics, and temporal patterns in the absence of intentional transmissions.

### Architecture Details

The model uses a multi-scale approach to capture noise features at different temporal and spectral resolutions:

```python
class QuietChannelVAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Multi-scale spectral analysis
        self.spectral_encoders = nn.ModuleList([
            nn.Conv1d(2, 32, kernel_size=64, stride=4),   # Wide features
            nn.Conv1d(2, 32, kernel_size=16, stride=4),   # Medium features
            nn.Conv1d(2, 32, kernel_size=4, stride=4),    # Narrow features
        ])

        # Statistical feature processor
        self.stats_processor = nn.Sequential(
            nn.Linear(96, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Temporal pattern detection for QRN bursts
        self.temporal_lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2
        )

        # VAE latent space - 64D for noise
        self.fc_mu = nn.Linear(192, 64)
        self.fc_logvar = nn.Linear(192, 64)
```

The multi-scale convolutions capture noise characteristics at different bandwidths, essential for distinguishing between broadband atmospheric noise and narrowband interference. The LSTM component identifies temporal patterns like lightning-induced static crashes that have characteristic time profiles.

### What Noise Embeddings Capture

The 64-dimensional noise embeddings encode several key characteristics:

**Noise Floor Level**: The baseline sensitivity limit in dBm, which varies with atmospheric conditions, time of day, and geographic location.

**Spectral Shape**: Whether the noise is white (flat across frequency) or colored (frequency-dependent), indicating different noise sources.

**Impulsive Components**: The rate and intensity of QRN bursts from lightning and other atmospheric phenomena.

**Narrowband Interference**: Presence and characteristics of QRM from other services or equipment.

**Statistical Distribution**: Whether noise follows Gaussian, impulsive, or other statistical patterns, critical for optimal detection strategies.

## Multi-Channel QRN Embeddings

Rather than generating a single embedding for the entire 12 kHz recording, the system divides the spectrum into overlapping 2.5 kHz channels. This approach captures the frequency-dependent nature of atmospheric noise and interference that would otherwise be averaged out in a single wideband embedding.

### Why Frequency Division Matters

QRN is not uniform across frequency. Real-world noise exhibits significant variation due to several factors:

**Frequency-Selective Atmospheric Effects**: Lower frequencies experience stronger atmospheric noise due to propagation characteristics. A recording might show -110 dBm noise floor at 2 kHz but -118 dBm at 10 kHz within the same 12 kHz window.

**Narrowband Interference Patterns**: QRM from electronic devices, power lines, or other transmitters often affects specific frequencies. A plasma TV might generate strong interference at 3.58 kHz while leaving adjacent frequencies relatively clean. A single 12 kHz embedding would average this spike into the overall noise floor, losing critical information.

**Selective Fading of Noise**: The noise floor itself experiences frequency-selective fading as it propagates through the ionosphere. This creates a non-uniform noise spectrum that varies with time and propagation conditions.

### Overlapping Channel Extraction Strategy

The system extracts nine overlapping 2.5 kHz channels from each 12 kHz recording, with 50% overlap to ensure smooth coverage:

```python
def extract_multichannel_qrn_embeddings(iq_12khz, sample_rate=12000):
    """
    Extract overlapping narrow-band noise embeddings
    """
    # Define overlapping channels with 50% overlap
    channels = [
        {'center': 1250,  'span': 2500},   # 0-2.5 kHz
        {'center': 2500,  'span': 2500},   # 1.25-3.75 kHz
        {'center': 3750,  'span': 2500},   # 2.5-5 kHz
        {'center': 5000,  'span': 2500},   # 3.75-6.25 kHz
        {'center': 6250,  'span': 2500},   # 5-7.5 kHz
        {'center': 7500,  'span': 2500},   # 6.25-8.75 kHz
        {'center': 8750,  'span': 2500},   # 7.5-10 kHz
        {'center': 10000, 'span': 2500},   # 8.75-11.25 kHz
        {'center': 11250, 'span': 2500},   # 10-12 kHz
    ]

    embeddings = []
    for ch in channels:
        # Bandpass filter to isolate channel
        channel_iq = bandpass_filter(
            iq_12khz,
            ch['center'] - ch['span']/2,
            ch['center'] + ch['span']/2
        )

        # Generate embedding for this narrow channel
        embedding = noise_vae.encode(channel_iq)

        embeddings.append({
            'embedding': embedding,
            'center_freq': ch['center'],
            'noise_floor': measure_noise_floor(channel_iq),
            'occupancy': calculate_occupancy(channel_iq)
        })

    return embeddings  # 9 embeddings per recording
```

The 50% overlap ensures no "blind spots" between channels while providing smooth interpolation capabilities when CASCADE operates at frequencies between channel centers.

### Multi-Channel VAE Architecture

The multi-channel approach requires a modified VAE architecture that can process multiple frequency bands while capturing their correlations:

```python
class MultiChannelNoiseVAE(nn.Module):
    """
    Optimized for simultaneous multi-channel processing
    """
    def __init__(self, n_channels=9):
        super().__init__()

        # Shared encoder captures common features
        self.shared_encoder = nn.Conv1d(2, 128, kernel_size=256)

        # Channel-specific heads for unique characteristics
        self.channel_heads = nn.ModuleList([
            nn.Linear(128, 64) for _ in range(n_channels)
        ])

        # Cross-channel attention captures correlations
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4
        )

        # VAE latent space - 64D per channel
        self.fc_mu = nn.Linear(64 * n_channels, 64 * n_channels)
        self.fc_logvar = nn.Linear(64 * n_channels, 64 * n_channels)
```

The shared encoder extracts features common to all channels (like general QRN bursts), while channel-specific heads capture unique characteristics (like narrowband QRM at specific frequencies). The cross-channel attention mechanism learns correlations between channels, understanding that strong QRN at low frequencies often correlates with elevated noise at higher frequencies.

### Frequency-Matched Training Application

During CASCADE training, the system applies noise embeddings that match CASCADE's operating frequency:

```python
def apply_frequency_matched_noise(cascade_signal, cascade_freq, qrn_embeddings):
    """
    Apply noise at CASCADE's exact frequency
    """
    # Find closest matching channel
    best_match = min(qrn_embeddings,
                     key=lambda e: abs(e['center_freq'] - cascade_freq))

    # Or interpolate between adjacent channels
    below = [e for e in qrn_embeddings if e['center_freq'] < cascade_freq][-1]
    above = [e for e in qrn_embeddings if e['center_freq'] > cascade_freq][0]

    # Linear interpolation weighted by frequency distance
    alpha = (cascade_freq - below['center_freq']) / \
            (above['center_freq'] - below['center_freq'])

    interpolated = (1-alpha) * below['embedding'] + alpha * above['embedding']

    return apply_noise_embedding(cascade_signal, interpolated)
```

This ensures CASCADE trains with noise characteristics specific to its operating frequency rather than averaged across the entire 12 kHz band.

### Benefits of Multi-Channel Approach

The multi-channel embedding strategy provides several key advantages:

**Frequency-Specific Training**: CASCADE experiences noise conditions specific to its operating frequency, improving real-world performance.

**Narrowband Interference Modeling**: Strong QRM spikes are preserved rather than averaged out, training CASCADE to handle real interference.

**9x Diversity Increase**: Each recording yields nine distinct noise environments instead of one, dramatically expanding training diversity.

**Spectral Awareness**: CASCADE can learn about adjacent channel conditions, improving its ability to adapt to the spectral environment.

**Future Flexibility**: If CASCADE's operating frequencies change, the embeddings already cover the new frequencies without reprocessing.

## Fixed Frequency Optimization

Even when CASCADE operates on fixed frequencies per band (such as 3573 kHz on 80m), the system should still generate embeddings across all channels rather than just the primary operating frequency. This comprehensive approach provides critical benefits for training robustness and real-world performance.

### Why All Channels Matter for Fixed-Frequency Operation

**Adjacent Channel Interference**: CASCADE's receiver bandwidth extends beyond its transmitted signal. A CASCADE signal at 3573 kHz with 50 Hz bandwidth still operates within a typical 500 Hz SSB filter. Noise and interference from adjacent frequencies directly affect reception quality. Strong QRM at 3650 kHz will raise the noise floor at 3573 kHz through filter skirts and receiver nonlinearities.

**Frequency Uncertainty in Real Deployments**: Multiple factors cause frequency variations in practice:
- Oscillator drift: ±10 Hz over temperature ranges
- Ionospheric Doppler: ±5 Hz from layer movement
- Tuning errors: ±50 Hz from manual operation
- Reference calibration: ±20 Hz from GPS or timebase errors

A CASCADE station nominally transmitting at 3573 kHz might actually operate anywhere from 3520 to 3620 kHz. Training with noise embeddings across this range ensures robustness to real-world frequency uncertainty.

**Wideband QRN Effects**: Atmospheric noise from lightning creates broadband impulses that affect all frequencies simultaneously. A strong QRN burst has primary energy below 5 kHz but still significantly impacts receivers at any HF frequency. Training needs awareness of the full QRN spectrum to properly model its effects on CASCADE's specific operating frequency.

**Spectral Awareness Benefits**: CASCADE's neural network can learn to use spectral context for improved performance. Knowing that strong QRM exists 200 Hz above its operating frequency allows the model to adjust detection thresholds or apply targeted filtering. This spectral awareness requires training with multi-channel embeddings that preserve the relationship between CASCADE's frequency and adjacent spectrum conditions.

### Optimized Sampling Strategy for Fixed Frequencies

When CASCADE uses fixed frequencies, the system optimizes embedding generation with a hierarchical approach:

```python
class FixedFrequencyOptimizedSampling:
    def __init__(self, cascade_freqs):
        # CASCADE operating frequencies per band
        self.cascade_freqs = {
            '80m': 3573,
            '40m': 7074,
            '20m': 14074,
            '15m': 21074,
            '10m': 28074
        }

    def generate_hierarchical_embeddings(self, recording):
        embeddings = {}

        # Always generate primary channels (100% sampling)
        for band, freq in self.cascade_freqs.items():
            if recording.band == band:
                primary = extract_channel(recording, freq, width=500)
                embeddings[f'{band}_primary'] = embed(primary)

        # Frequently sample adjacent channels (50% sampling)
        if random.random() < 0.5:
            for offset in [-1250, +1250]:
                adj = extract_channel(recording, freq+offset, width=2500)
                embeddings[f'{band}_adj_{offset}'] = embed(adj)

        # Occasionally sample distant channels (10% sampling)
        if random.random() < 0.1:
            for offset in [-5000, -2500, +2500, +5000]:
                distant = extract_channel(recording, freq+offset, width=2500)
                embeddings[f'{band}_dist_{offset}'] = embed(distant)

        # Rarely sample full spectrum (5% sampling)
        if random.random() < 0.05:
            embeddings[f'{band}_full'] = embed(recording.full_12khz)

        return embeddings
```

This hierarchical approach ensures perfect coverage at CASCADE frequencies while maintaining awareness of the broader spectral environment, achieving a balance between storage efficiency and training completeness.

### Storage Optimization for Fixed-Frequency Systems

The storage schema prioritizes CASCADE's operating frequencies while maintaining spectral context:

```sql
CREATE TABLE cascade_optimized_embeddings (
    embedding_id UUID PRIMARY KEY,
    session_id UUID REFERENCES recording_sessions(session_id),

    -- Primary embeddings at CASCADE frequencies (always present)
    primary_embedding FLOAT[64] NOT NULL,
    primary_freq_hz INTEGER NOT NULL,
    primary_noise_floor_dbm FLOAT,

    -- Adjacent channel embeddings (frequently present)
    adjacent_embeddings JSONB,  -- Array of ±1.25kHz embeddings

    -- Distant channel embeddings (occasionally present)
    distant_embeddings JSONB,  -- Array of ±2.5-5kHz embeddings

    -- Full spectrum embedding (rarely present)
    full_spectrum_embedding FLOAT[64],

    -- Metadata for intelligent retrieval
    has_adjacent_qrm BOOLEAN,
    spectral_occupancy_percent FLOAT,
    primary_channel_quality FLOAT,

    -- Indexes for CASCADE training
    INDEX idx_cascade_primary (primary_freq_hz, primary_channel_quality DESC),
    INDEX idx_spectral_context (has_adjacent_qrm, spectral_occupancy_percent)
);
```

This schema reduces storage from 115 GB (full multi-channel) to approximately 30-40 GB while maintaining the most valuable information for CASCADE training.

### Training Benefits with Fixed-Frequency Optimization

The optimized multi-channel approach provides several advantages for fixed-frequency CASCADE systems:

**Primary Coverage**: 100% of recordings include embeddings at CASCADE's exact operating frequencies, ensuring comprehensive training coverage where it matters most.

**Interference Robustness**: Adjacent channel embeddings train CASCADE to handle real-world QRM that affects reception despite frequency separation.

**Drift Tolerance**: The ±1.25 kHz adjacent embeddings provide natural training for frequency uncertainty without explicitly modeling drift.

**Spectral Intelligence**: Occasional full-spectrum embeddings teach CASCADE about the broader RF environment, improving its adaptive capabilities.

**Storage Efficiency**: The hierarchical sampling reduces storage by 65-75% compared to full multi-channel extraction while preserving essential diversity.

## Path Context Encoder Architecture

The Path Context Encoder is a deterministic feature extractor that encodes geographic and ionospheric path information from FT8 grid squares into a 32-dimensional vector. Unlike the VAEs which learn representations from data, this encoder uses physics-based feature engineering.

### Key Geographic Features

The encoder extracts features that directly influence propagation:

```python
class PathContextEncoder:
    """
    Encodes path geometry and ionospheric context into 32D vector
    """
    def __init__(self):
        # Precompute magnetic pole locations for efficiency
        self.mag_north = (86.5, -164.04)  # 2025 position
        self.mag_south = (-64.07, 136.59)

        # Load auroral oval model
        self.auroral_model = load_auroral_predictor()

        # Load foF2 prediction model
        self.foF2_model = load_foF2_predictor()

    def encode_path_context(self, tx_grid, rx_grid, timestamp, k_index):
        features = []

        # 1. Basic geometry (4D)
        distance = great_circle_distance(tx_grid, rx_grid)
        bearing = calculate_bearing(tx_grid, rx_grid)
        features.extend([
            distance / 20000,              # [0-1] Earth half circumference
            bearing / 360,                 # [0-1] azimuth
            np.sin(bearing * 2*np.pi/360), # Circular encoding
            np.cos(bearing * 2*np.pi/360)
        ])

        # 2. Solar illumination (6D)
        tx_sza = solar_zenith_angle(tx_grid, timestamp)
        rx_sza = solar_zenith_angle(rx_grid, timestamp)
        midpoint = calculate_midpoint(tx_grid, rx_grid)
        mid_sza = solar_zenith_angle(midpoint, timestamp)
        features.extend([
            tx_sza / 180,                  # [0-1] solar angle
            rx_sza / 180,
            mid_sza / 180,
            float(tx_sza < 90),            # Day/night binary
            float(rx_sza < 90),
            float(mid_sza < 90)
        ])

        # 3. Geomagnetic context (6D)
        mag_lat_tx = to_magnetic_latitude(tx_grid)
        mag_lat_rx = to_magnetic_latitude(rx_grid)
        auroral_tx = distance_to_auroral_oval(tx_grid, k_index)
        auroral_rx = distance_to_auroral_oval(rx_grid, k_index)
        features.extend([
            mag_lat_tx / 90,               # [-1,1] normalized
            mag_lat_rx / 90,
            auroral_tx / 2000,             # km to oval
            auroral_rx / 2000,
            float(abs(mag_lat_tx) > 60),  # In auroral zone
            float(abs(mag_lat_rx) > 60)
        ])

        # 4. Ionospheric parameters (6D)
        foF2_tx = self.foF2_model.predict(tx_grid, timestamp)
        foF2_rx = self.foF2_model.predict(rx_grid, timestamp)
        foF2_mid = self.foF2_model.predict(midpoint, timestamp)
        features.extend([
            foF2_tx / 15,                  # MHz critical freq
            foF2_rx / 15,
            foF2_mid / 15,
            estimate_muf(distance) / 30,   # Maximum usable freq
            estimate_hop_count(distance) / 4,
            estimate_skip_distance(frequency) / 3000
        ])

        # 5. Path topology (6D)
        features.extend([
            float(crosses_magnetic_equator(tx_grid, rx_grid)),
            float(crosses_terminator(tx_grid, rx_grid, timestamp)),
            float(is_transequatorial(tx_grid, rx_grid)),
            float(is_grayline_path(tx_grid, rx_grid, timestamp)),
            latitude_difference(tx_grid, rx_grid) / 180,
            longitude_difference(tx_grid, rx_grid) / 360
        ])

        # 6. Time context (4D)
        features.extend([
            timestamp.hour / 24,           # UTC hour [0-1]
            timestamp.month / 12,          # Season proxy
            np.sin(timestamp.hour * 2*np.pi/24),  # Circular time
            np.cos(timestamp.hour * 2*np.pi/24)
        ])

        return np.array(features)  # Total: 32D
```

### Training vs Inference Usage

**Critical Design Point**: The Path Context Encoder is used ONLY during training to ensure CASCADE learns from geographically-diverse propagation. At inference time, CASCADE operates blindly without any geographic information:

```python
# TRAINING: Use path context to create realistic conditions
def training_mode(iq_data, tx_grid, rx_grid):
    path_context = path_encoder.encode(tx_grid, rx_grid, timestamp)
    channel = cvae.decode(channel_emb, path_context)
    propagated = apply_channel(cascade_signal, channel)

    # CASCADE must decode WITHOUT path information
    decoded = cascade_model(propagated)  # No geographic input!

# INFERENCE: No geographic information available or needed
def inference_mode(received_signal):
    decoded = cascade_model(received_signal)  # Fully adaptive
    return decoded
```

This design ensures CASCADE develops true adaptive capability rather than relying on geographic priors.

### Integration with Propagation VAE

During training, the path context conditions the propagation decoder, allowing it to generate geographically-appropriate channel characteristics:

```python
class GeographicallyAwarePropagationVAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Standard propagation encoder for IQ data
        self.prop_encoder = FT8PropagationVAE()

        # Path context encoder for geographic features
        self.path_encoder = PathContextEncoder()

        # Conditional decoder uses both embeddings
        self.decoder = ConditionalDecoder(
            channel_dim=128,
            context_dim=32,
            output_dim=512
        )

    def forward(self, iq_data, tx_grid, rx_grid, timestamp):
        # Extract propagation from IQ
        prop_embedding = self.prop_encoder.encode(iq_data)

        # Extract path context
        path_context = self.path_encoder.encode_path_context(
            tx_grid, rx_grid, timestamp
        )

        # Condition decoder on both
        combined = torch.cat([prop_embedding, path_context])
        channel_params = self.decoder(combined)

        return channel_params
```

## FT8 Propagation VAE Architecture

The FT8 Propagation VAE extracts channel characteristics by comparing received FT8 signals with their known transmitted form. This model leverages the structured nature of FT8 transmissions to precisely characterize propagation effects.

### Architecture Details

The model incorporates symbol-aware processing and attention mechanisms:

```python
class FT8PropagationVAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Symbol-aware processing (79 symbols in FT8)
        self.symbol_encoder = nn.Conv1d(
            in_channels=2,
            out_channels=128,
            kernel_size=160,  # Samples per symbol
            stride=160        # Non-overlapping
        )

        # Symbol correlation patterns
        self.symbol_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8
        )

        # Frequency-domain channel estimation
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=256),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(79)  # Match symbol count
        )

        # VAE latent space - 128D for propagation
        self.fc_mu = nn.Linear(256, 128)
        self.fc_logvar = nn.Linear(256, 128)
```

The symbol-level processing aligns with FT8's structure, allowing the model to track how propagation affects each symbol differently. The attention mechanism captures correlations between symbols, identifying consistent propagation patterns versus rapid channel changes.

### What Propagation Embeddings Capture

The 128-dimensional propagation embeddings encode rich channel information:

**Multipath Profile**: The delays and amplitudes of different propagation paths, from direct ground wave to multiple ionospheric hops.

**Fading Characteristics**: How signal amplitude varies over time, including fade depth, rate, and correlation time.

**Doppler Effects**: Frequency shifts and spreading caused by ionospheric movement and path variations.

**Phase Evolution**: How signal phase rotates across symbols, indicating path length changes.

**Propagation Mode**: Whether the signal traveled via F2 layer, sporadic-E, aurora, or other mechanisms.

## Training Process

Both VAE models follow similar training procedures but with different objectives and data preparation.

### Data Preparation

For the Quiet Channel VAE, the system identifies periods in recordings without detected signals:

```python
def prepare_noise_training_data(recording):
    quiet_segments = []
    for window in recording.sliding_windows(duration=2.0):
        if not contains_signals(window):
            quiet_segments.append({
                'iq': window,
                'noise_floor': measure_noise_floor(window),
                'k_index': recording.k_index,
                'time': recording.timestamp
            })
    return quiet_segments
```

For the FT8 Propagation VAE, the system pairs received signals with their decoded content:

```python
def prepare_propagation_training_data(recording):
    training_pairs = []
    for signal in detect_ft8(recording):
        if decoded := ft8_decode(signal):
            ideal = generate_ideal_ft8(decoded.message)
            training_pairs.append({
                'received': signal.iq,
                'ideal': ideal,
                'snr': signal.snr,
                'distance': calculate_distance(decoded.grids)
            })
    return training_pairs
```

### Training Objectives

Both models optimize a combination of reconstruction accuracy and latent space regularity:

```python
def vae_loss(original, reconstructed, mu, logvar):
    # Reconstruction loss - how well can we recreate the channel?
    reconstruction_loss = F.mse_loss(reconstructed, original)

    # KL divergence - keep embeddings well-distributed
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Balance reconstruction quality with embedding regularity
    return reconstruction_loss + 0.01 * kl_loss
```

The reconstruction loss ensures embeddings contain sufficient information to recreate channel conditions, while the KL divergence term prevents the latent space from developing discontinuities or sparse regions.

### Frequency-Matched Training Process

When using multi-channel QRN embeddings, the training process incorporates frequency-specific noise application:

```python
def train_with_frequency_matching(cascade_model, embedding_db):
    """
    Train CASCADE with frequency-matched noise embeddings
    """
    for epoch in range(n_epochs):
        for batch in training_loader:
            # Generate CASCADE signal at specific frequency
            cascade_freq = batch.operating_frequency  # e.g., 3573 kHz
            cascade_signal = cascade_encoder(batch.data)

            # Fetch multi-channel noise embeddings
            noise_embeddings = embedding_db.get_multichannel(
                timestamp=batch.timestamp,
                k_index=batch.k_index
            )

            # Apply frequency-matched noise
            if cascade_freq in CASCADE_FREQUENCIES:
                # Use primary embedding at exact CASCADE frequency
                noise = noise_embeddings.get_primary(cascade_freq)
            else:
                # Interpolate between channels for other frequencies
                noise = interpolate_channels(noise_embeddings, cascade_freq)

            # Apply correlated propagation
            prop_embedding = noise_embeddings.paired_propagation
            cascade_signal = apply_propagation(cascade_signal, prop_embedding)
            cascade_signal = apply_noise(cascade_signal, noise)

            # Train with realistic conditions
            decoded = cascade_model(cascade_signal)
            loss = compute_loss(decoded, batch.data)
            optimizer.step(loss)
```

### Spectral Context Awareness

The multi-channel approach enables training with spectral context, where CASCADE learns about adjacent channel conditions:

```python
def train_with_spectral_context(cascade_signal, multi_channel_embeddings):
    """
    Provide CASCADE with awareness of surrounding spectrum
    """
    cascade_freq = 3573  # CASCADE on 80m

    # Get noise at CASCADE frequency and adjacent channels
    context = {
        'primary': multi_channel_embeddings.get_channel(3750),  # Closest center
        'below': multi_channel_embeddings.get_channel(2500),    # 1.25 kHz below
        'above': multi_channel_embeddings.get_channel(5000),    # 1.25 kHz above
    }

    # CASCADE's attention mechanism can use spectral context
    # to adapt detection thresholds or filtering
    cascade_with_context = cascade_model(
        signal=cascade_signal,
        noise_context=context
    )

    # The model learns that strong QRM above affects reception
    # even though CASCADE transmits at a different frequency
    return cascade_with_context
```

This spectral awareness helps CASCADE adapt to real-world conditions where adjacent channel interference affects reception quality.

### Physics-Informed Constraints

Training incorporates domain knowledge to ensure physically plausible embeddings:

```python
def physics_constraints(channel_parameters):
    losses = []

    # Multipath delays must be positive and ordered
    delays = channel_parameters['multipath_delays']
    losses.append(F.relu(-delays).mean())  # Penalize negative
    losses.append(F.relu(delays[:-1] - delays[1:]).mean())  # Ensure ordering

    # Doppler spread limited by physics
    doppler = channel_parameters['doppler_spread']
    max_doppler = 10.0  # Hz for HF
    losses.append(F.relu(doppler.abs() - max_doppler).mean())

    # Power conservation
    path_powers = channel_parameters['path_amplitudes'].pow(2)
    losses.append((path_powers.sum() - 1.0).pow(2))  # Should sum to 1

    return sum(losses)
```

These constraints guide the model toward learning representations consistent with radio propagation physics rather than arbitrary mathematical transformations.

## Training Timeline

The embedding models are trained after the 18-month data collection completes:

**Week 1**: Create curated diverse dataset (5TB from 75TB total)
- Apply rarity scoring to identify valuable training examples
- Ensure coverage of all propagation modes and conditions
- Balance rare events with baseline coverage

**Weeks 2-3**: Train both VAE models on curated dataset
- 3-5 days training per model on single GPU
- Validate reconstruction quality and embedding distributions
- Fine-tune hyperparameters based on validation metrics

**Weeks 3-4**: Generate embeddings for training dataset
- Process 5TB of IQ data through trained models
- Generate ~25GB of embeddings
- Build indexes for efficient retrieval

## Embedding Space Properties

The learned embedding spaces exhibit useful mathematical properties that enable effective training:

### Smoothness and Continuity

Similar channel conditions produce similar embeddings. A quiet winter night and a quiet winter morning have nearby embeddings, while a quiet winter night and a stormy summer afternoon are far apart. This smoothness allows the model to interpolate between conditions.

### Meaningful Interpolation

Linear combinations of embeddings produce plausible intermediate conditions:

```python
# Interpolate between day and night propagation
day_embedding = encode(noon_recording)
night_embedding = encode(midnight_recording)

# Create dawn/dusk conditions
dawn = 0.7 * night_embedding + 0.3 * day_embedding
dusk = 0.3 * night_embedding + 0.7 * day_embedding
```

### Cluster Formation

Natural clusters emerge corresponding to physical phenomena. Sporadic-E propagation forms one cluster, F2 propagation another, with subclusters for different solar conditions. This structure confirms the model has learned meaningful representations.

## Validation Methods

Multiple validation approaches ensure embedding quality:

### Reconstruction Testing

Can the decoder recreate original channel conditions from embeddings?

```python
def test_reconstruction(vae, test_data):
    for sample in test_data:
        embedding = vae.encode(sample)
        reconstructed = vae.decode(embedding)
        correlation = correlate(sample, reconstructed)
        assert correlation > 0.85  # High similarity required
```

### Clustering Analysis

Do similar conditions cluster appropriately?

```python
def test_clustering(embeddings, labels):
    # Calculate cluster metrics
    silhouette = silhouette_score(embeddings, labels)
    davies_bouldin = davies_bouldin_score(embeddings, labels)

    # Good clustering has high silhouette, low Davies-Bouldin
    assert silhouette > 0.6
    assert davies_bouldin < 1.5
```

### Physics Consistency

Do decoded parameters obey physical constraints?

```python
def test_physics(vae, test_samples):
    for sample in test_samples:
        embedding = vae.encode(sample)
        parameters = vae.decode(embedding)

        # Verify physical plausibility
        assert all(parameters['delays'] >= 0)
        assert parameters['doppler'].abs().max() < 10  # Hz
        assert 0.5 < parameters['powers'].sum() < 1.5
```

### Downstream Performance

Most importantly, does CASCADE train better with these embeddings?

```python
def test_cascade_improvement():
    cascade_with_embeddings = train_cascade(use_embeddings=True)
    cascade_without = train_cascade(use_embeddings=False)

    # Embeddings should improve performance
    assert cascade_with_embeddings.ber < cascade_without.ber
    assert cascade_with_embeddings.convergence_time < cascade_without.convergence_time
```

## Why VAEs Over Alternatives

The Variational Autoencoder architecture was chosen after considering alternatives:

**Standard Autoencoders** lack the continuous latent space property, making interpolation between conditions unreliable.

**Contrastive Learning** (like SimCLR) could work but requires careful negative sampling strategies for radio data where "different" isn't always clear.

**Transformer Models** are powerful but overkill for the relatively short FT8 transmissions and would require much more training data.

**Physics-Informed Neural Networks** could incorporate Maxwell's equations directly but are challenging to train and require deep domain expertise to implement correctly.

VAEs provide the right balance of mathematical elegance, training stability, and useful properties for the channel embedding task. The continuous latent space, natural interpolation capabilities, and well-understood training dynamics make them ideal for learning radio channel representations.

## Storage and Retrieval

The storage architecture adapts to whether the system uses single-channel or multi-channel embeddings, with optimizations for CASCADE's fixed operating frequencies.

### Multi-Channel Storage Schema

For systems using the multi-channel approach, embeddings are stored hierarchically:

```sql
-- Multi-channel noise embeddings with frequency-specific indexing
CREATE TABLE multichannel_noise_embeddings (
    embedding_id UUID PRIMARY KEY,
    session_id UUID REFERENCES recording_sessions(session_id),

    -- Array of channel embeddings
    channel_embeddings JSONB NOT NULL,
    /* Structure:
    {
        "channels": [
            {
                "center_hz": 1250,
                "embedding": [0.23, -0.45, ...],  -- 64D vector
                "noise_floor_dbm": -110,
                "occupancy": 0.15
            },
            ...
        ]
    }
    */

    -- CASCADE-specific embeddings for fast access
    cascade_primary_embedding FLOAT[64],  -- At CASCADE frequency
    cascade_primary_freq_hz INTEGER,

    -- Metadata for intelligent retrieval
    avg_noise_floor_dbm FLOAT,
    spectral_variation FLOAT,  -- Std dev across channels
    has_narrowband_qrm BOOLEAN,

    -- Context preservation
    timestamp TIMESTAMP NOT NULL,
    k_index INTEGER,
    solar_flux INTEGER,

    -- Optimized indexes for CASCADE training
    INDEX idx_cascade_freq (cascade_primary_freq_hz, timestamp),
    INDEX idx_spectral_features (spectral_variation, has_narrowband_qrm),
    INDEX idx_conditions (k_index, solar_flux)
);

-- Propagation embeddings remain single-channel but correlated
CREATE TABLE propagation_embeddings (
    embedding_id UUID PRIMARY KEY,
    session_id UUID REFERENCES recording_sessions(session_id),

    embedding FLOAT[128] NOT NULL,
    frequency_hz INTEGER NOT NULL,

    -- Propagation characteristics
    snr_db FLOAT,
    propagation_mode VARCHAR(20),
    distance_km FLOAT,
    multipath_spread_ms FLOAT,

    -- Link to paired noise for correlation preservation
    paired_noise_embedding_id UUID REFERENCES multichannel_noise_embeddings(embedding_id),

    timestamp TIMESTAMP NOT NULL,

    -- Indexes for efficient retrieval
    INDEX idx_frequency (frequency_hz),
    INDEX idx_propagation_mode (propagation_mode),
    INDEX idx_paired_retrieval (paired_noise_embedding_id)
);
```

### Optimized Retrieval for Training

The system uses multiple retrieval strategies depending on training requirements:

```python
class EmbeddingRetriever:
    def __init__(self):
        # Load frequently-used CASCADE frequencies into memory
        self.cascade_freq_cache = self.build_cascade_cache()

        # KD-tree for similarity searches
        self.kdtree = self.build_kdtree()

    def get_frequency_matched_embedding(self, target_freq, conditions):
        """
        Retrieve embedding matching CASCADE's operating frequency
        """
        # Check cache first for CASCADE frequencies
        if target_freq in self.cascade_freq_cache:
            return self.cascade_freq_cache[target_freq].get_similar(conditions)

        # Otherwise query database with frequency interpolation
        query = """
            SELECT
                channel_embeddings,
                cascade_primary_embedding
            FROM multichannel_noise_embeddings
            WHERE
                ABS(cascade_primary_freq_hz - %s) < 1000
                AND k_index = %s
                AND timestamp::date = %s
            ORDER BY
                ABS(cascade_primary_freq_hz - %s)
            LIMIT 1
        """

        result = db.execute(query, (target_freq, conditions.k_index,
                                   conditions.date, target_freq))

        # Extract best matching channel or interpolate
        return self.interpolate_to_frequency(result, target_freq)

    def get_correlated_pair(self, session_id):
        """
        Retrieve naturally correlated noise and propagation embeddings
        """
        query = """
            SELECT
                n.channel_embeddings,
                n.cascade_primary_embedding,
                p.embedding as prop_embedding
            FROM multichannel_noise_embeddings n
            JOIN propagation_embeddings p
                ON p.paired_noise_embedding_id = n.embedding_id
            WHERE n.session_id = %s
        """

        return db.execute(query, (session_id,))
```

### Memory-Mapped Arrays for Fast Access

For production training, frequently-accessed embeddings are stored in memory-mapped arrays:

```python
class MemoryMappedEmbeddings:
    def __init__(self, cascade_frequencies):
        # Pre-load CASCADE frequency embeddings
        self.primary_embeddings = {}

        for band, freq in cascade_frequencies.items():
            # Memory-map embeddings at CASCADE frequencies
            embeddings_file = f'embeddings_{band}_{freq}.npy'
            self.primary_embeddings[freq] = np.memmap(
                embeddings_file,
                dtype='float32',
                mode='r',
                shape=(n_recordings, 64)
            )

        # Build KD-tree for each frequency
        self.kdtrees = {}
        for freq, embeddings in self.primary_embeddings.items():
            self.kdtrees[freq] = KDTree(embeddings)

    def get_nearest(self, target_embedding, freq, k=10):
        """
        Fast nearest-neighbor search at specific frequency
        """
        distances, indices = self.kdtrees[freq].query(
            target_embedding.reshape(1, -1),
            k=k
        )
        return self.primary_embeddings[freq][indices[0]]
```

### Storage Optimization Strategies

The system implements several strategies to balance storage efficiency with training effectiveness:

**Hierarchical Storage**: Primary CASCADE frequencies are stored at full resolution (100% sampling), adjacent channels at medium resolution (50% sampling), and distant channels at low resolution (10% sampling).

**Compression**: Embeddings use float16 where precision allows, reducing storage by 50% with minimal impact on training quality.

**Temporal Decimation**: For long-term storage, the system keeps hourly samples for common conditions but retains all samples for rare events (K≥7, X-class flares).

**Smart Caching**: The most frequently accessed embeddings (CASCADE primary frequencies during normal conditions) remain in RAM, while rare event embeddings are loaded on demand.

### Database Performance Optimization

PostgreSQL configuration for optimal embedding retrieval:

```sql
-- Enable vector extension for similarity searches
CREATE EXTENSION IF NOT EXISTS vector;

-- Optimize for embedding queries
ALTER TABLE multichannel_noise_embeddings
    SET (fillfactor = 90);  -- Leave space for updates

-- Partial indexes for common queries
CREATE INDEX idx_cascade_80m
    ON multichannel_noise_embeddings(timestamp, k_index)
    WHERE cascade_primary_freq_hz = 3573;

CREATE INDEX idx_cascade_40m
    ON multichannel_noise_embeddings(timestamp, k_index)
    WHERE cascade_primary_freq_hz = 7074;

-- Materialized view for training batch generation
CREATE MATERIALIZED VIEW training_batch_embeddings AS
SELECT
    n.cascade_primary_embedding as noise_emb,
    p.embedding as prop_emb,
    n.k_index,
    n.timestamp
FROM multichannel_noise_embeddings n
JOIN propagation_embeddings p
    ON p.paired_noise_embedding_id = n.embedding_id
WHERE n.cascade_primary_freq_hz IN (3573, 7074, 14074, 21074, 28074)
WITH DATA;

-- Refresh periodically
REFRESH MATERIALIZED VIEW CONCURRENTLY training_batch_embeddings;
```

This storage architecture provides millisecond retrieval times for CASCADE training while maintaining the flexibility to access the full spectrum of recorded conditions when needed.

## Hybrid Storage Architecture

Based on cost and performance analysis, CASCADE benefits most from a hybrid storage approach that combines database flexibility with filesystem efficiency. This architecture leverages PostgreSQL for metadata and queries while using HDF5 for bulk embedding storage, reducing costs by 60-70% compared to pure database storage while maintaining analytical capabilities.

### Cost-Performance Tradeoffs

For 50 million embeddings (approximately 48GB raw):

**Pure Database Approach**
- Storage cost: ~$6.10/month (AWS RDS)
- Query performance: Excellent (1-5ms with indexes)
- Similarity search: Native with pgvector
- Concurrent access: Built-in MVCC
- Maintenance: Automated backups, scaling

**Pure Filesystem Approach**
- Storage cost: ~$0.78/month (AWS EBS)
- Sequential read: Excellent (<1ms memory-mapped)
- Random access: Poor without indexes
- Query capability: None without loading all data
- Concurrent access: Manual locking required

**Hybrid Approach (Recommended)**
- Storage cost: ~$2-3/month
- Combines PostgreSQL metadata queries with HDF5 bulk storage
- Best of both worlds: SQL flexibility + file I/O performance

### Embedding Analytics Capabilities

The embeddings form a rich analytical space that reveals patterns and guides training. Key discoveries include:

**Propagation Mode Clustering**: Embeddings naturally cluster into distinct propagation modes (F2, Sporadic-E, Aurora), with some clusters potentially representing unknown phenomena.

**Anomaly Detection**: Outliers in embedding space identify rare propagation events that deserve 10x training weight, including trans-equatorial propagation, anomalous daytime DX, and unexplained propagation modes.

**Temporal Evolution**: Tracking embedding changes over time reveals propagation transitions (sunrise/sunset, mode changes) that challenge decoders and require focused training.

**Geographic Patterns**: Path-dependent clustering shows that polar paths, oceanic paths, and mountain diffraction paths each form distinct embedding signatures.

For detailed analysis of embedding analytics and advanced storage strategies, see [Embedding Analytics](embedding_analytics.md).

## Future Enhancements

Several improvements could further enhance the embedding models:

**Multi-Band Awareness**: Currently separate models per band could be unified with band-aware architectures.

**Temporal Dynamics**: Incorporating time series of embeddings to capture channel evolution.

**Adversarial Robustness**: Training with adversarial examples to improve embedding stability.

**Few-Shot Learning**: Adapting quickly to new propagation modes not seen during training.

The embedding models form a critical bridge between raw radio recordings and efficient CASCADE training, transforming overwhelming amounts of IQ data into compact, meaningful representations that preserve the full richness of HF radio propagation.