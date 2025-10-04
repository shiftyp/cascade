# Channel Embedding Models

CASCADE uses specialized neural networks to create compact representations of radio channel conditions. These embedding models transform raw IQ recordings into dense vectors that capture the essential characteristics of noise environments and propagation effects, enabling efficient and realistic training.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Training vs Inference Data Flow](#training-vs-inference-data-flow)
   - [Single-User Training](#single-user-training)
   - **[Multi-User Training](#multi-user-training-critical)** ← CRITICAL: Per-user embeddings
3. [Quiet Channel VAE](#quiet-channel-vae-architecture)
4. [FT8 Propagation VAE](#ft8-propagation-vae)
5. [Path Context Encoder](#path-context-encoder)
6. [Station Fingerprinting](#station-fingerprinting-with-persistent-identifiers)
7. [Equipment Signature Extraction](#advanced-equipment-signature-extraction)
8. [Propagation Augmentation](#propagation-augmentation)
9. [Embedding Analytics and Storage](#embedding-analytics-and-storage)

## Architecture Overview

### Triple Encoder Architecture

The system employs three specialized encoders for comprehensive channel characterization:

1. **Quiet Channel VAE**: Generates 64-dimensional embeddings representing noise characteristics **from IQ samples**
2. **FT8 Propagation VAE**: Generates 128-dimensional embeddings encoding propagation effects **from IQ samples**
3. **Path Context Encoder**: Generates 32-dimensional embeddings from geographic path information (grid squares, distance, bearing)

All three encoders process **raw IQ data** or **metadata**, not pre-computed embeddings. The VAEs learn compressed representations from raw signals, while the Path Context Encoder extracts deterministic features from FT8 decoded metadata.

### Station Fingerprinting Enhancement

Station fingerprinting adds a fourth component that enhances propagation embeddings:

4. **Station Fingerprint Encoder**: Generates 16-dimensional embeddings from equipment characteristics (phase noise, frequency drift, PA linearity)

This creates station-aware propagation embeddings (128D + 16D = 144D total) that capture both channel effects and equipment signatures.

## Training vs Inference Data Flow

**Critical Distinction**: The embedding models are **only used during training data preparation**. CASCADE's operational model receives raw IQ samples directly, not embeddings.

### Training Data Preparation (Offline, Months 19-20)

```
KiwiSDR Recordings (IQ samples)
        ↓
    Embedding VAEs
        ↓
   Embeddings stored
        ↓
CASCADE Training (learns from embeddings)
        ↓
  Trained CASCADE Model
```

During training preparation:
1. Collect 200,000-300,000 hours of IQ recordings from KiwiSDRs
2. Train embedding VAEs on curated 3-5TB subset
3. Generate embeddings for all curated recordings
4. Store embeddings (~15-25GB compressed)
5. Use embeddings to train CASCADE model

### Inference (Real-time Operation)

```
Live Radio Signal (IQ samples)
        ↓
CASCADE Shared Encoder (processes raw IQ)
        ↓
   Expert Networks
        ↓
    Decoded Data
        ↓
  [Telemetry: Copy internal state] → Upload for fine-tuning
```

During inference:
1. CASCADE receives raw IQ samples from radio
2. Shared encoder processes IQ directly (NOT embeddings)
3. Expert networks operate on encoded features
4. No embedding VAEs involved at inference time
5. **Telemetry captures CASCADE's internal activations** (zero overhead - just copying existing state)

### Telemetry Usage (Post-Deployment)

After CASCADE is deployed, telemetry provides continuous training data:

```
Live CASCADE Operation
        ↓
Internal State (shared encoder + all experts) → Quantize to INT8
        ↓
Upload batched telemetry (3MB/hour)
        ↓
Dequantize to FP32 + Fine-tune CASCADE model
        ↓
Deploy improved model
```

**Key insight**: CASCADE's internal activations (3581-D) serve double duty:
1. During operation: Feed to conductor for decoding decisions
2. For telemetry: Capture complete model state for fine-tuning

This eliminates the need for separate embedding models during deployment - CASCADE's own representations are the telemetry.

### Why This Architecture?

**Training Efficiency**: Embeddings compress 40-50TB of IQ data into 15-25GB, making training computationally feasible while preserving propagation diversity.

**Inference Simplicity**: CASCADE's shared encoder learns to extract similar features to the embedding VAEs, but does so end-to-end from raw IQ. No separate embedding computation needed.

**Domain Transfer**: The embedding VAEs teach CASCADE what propagation features matter, but CASCADE learns to extract them directly during inference.

**Telemetry Efficiency**: Capturing CASCADE's internal state provides perfect training signal with zero computational overhead - the features are already computed for normal operation.

```python
# TRAINING: Using pre-computed embeddings

## Single-User Training
def train_cascade_single_user():
    """Train with one CASCADE signal per sample"""
    # Load pre-computed embeddings
    noise_emb = sample_embedding('noise_embeddings.h5')      # 64D
    prop_emb = sample_embedding('propagation_embeddings.h5') # 128D
    station_emb = sample_embedding('station_embeddings.h5')  # 16D

    # Generate synthetic CASCADE signal
    cascade_signal = generate_cascade_transmission(data)

    # Apply embeddings to simulate real conditions
    augmented_signal = apply_channel_embeddings(
        cascade_signal,
        noise_emb,
        prop_emb,
        station_emb
    )

    # Train CASCADE to decode
    decoded = cascade_model(augmented_signal)  # Raw IQ input!
    loss = criterion(decoded, data)
    loss.backward()

## Multi-User Training (CRITICAL)
def train_cascade_multi_user():
    """
    Train with multiple CASCADE signals - each gets independent propagation!
    """
    num_users = random.randint(2, 20)
    mixed_signal = np.zeros(signal_length)

    # IMPORTANT: Each user gets INDEPENDENT embeddings
    for user_id in range(num_users):
        # Sample DIFFERENT embeddings for each user
        user_prop_emb = sample_embedding('propagation_embeddings.h5')  # Independent!
        user_station_emb = sample_embedding('station_embeddings.h5')   # Independent!

        # Generate CASCADE signal for this user
        user_data = generate_user_data(user_id)
        user_signal = generate_cascade_transmission(user_data)

        # Apply THIS USER'S propagation and station signature
        user_augmented = apply_channel_embeddings(
            user_signal,
            user_prop_emb,
            user_station_emb
        )

        # Add to mix at appropriate power level
        user_power = sample_power_level()
        mixed_signal += user_augmented * (10 ** (user_power / 20))

    # Shared noise affects ALL users (applied once to mixed signal)
    shared_noise_emb = sample_embedding('noise_embeddings.h5')
    final_signal = add_noise_from_embedding(mixed_signal, shared_noise_emb)

    # Train to separate and decode all users
    decoded_users = cascade_model(final_signal)
    loss = multi_user_separation_loss(decoded_users, ground_truth_data)
    loss.backward()

# INFERENCE: Direct IQ processing
def inference_with_cascade(iq_samples):
    # CASCADE processes raw IQ directly
    # No embedding computation needed
    decoded = cascade_model(iq_samples)  # Same input type as training!
    return decoded
```

**Key Insight**:
- **Single-user training**: 1 propagation + 1 station embedding per sample
- **Multi-user training**: N independent (propagation + station) embeddings, 1 shared noise
- **Embeddings used during training only** to create realistic conditions
- **Inference always processes raw IQ** end-to-end, no embeddings needed

**Why This Matters**:
Each user in a multi-user scenario experiences different propagation (different paths, equipment, timing). Applying a single propagation embedding to all users would create unrealistic training conditions where everyone experiences identical channels.

**Physical Reality**:
- **Different propagation per user**: User A might have strong F2 propagation while User B experiences fading
- **Different equipment per user**: User A has GPS-locked OCXO, User B has drifting TCXO
- **Shared atmospheric noise**: All users at receiver experience the same local QRN/QRM

This matches reality: if you're receiving 10 stations simultaneously, they all arrive through different propagation paths with different equipment signatures, but they all experience your local noise environment.

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

## Processing Timeline: Collection vs Training

### During Data Collection (Months 1-18)
The data collection phase stores raw 10-minute IQ recordings at 12 kHz bandwidth without any channelization or temporal chunking. This preserves maximum flexibility for future processing decisions and reduces collection system complexity.

### During Embedding Model Training (Month 19)
After collection completes, the system applies multi-scale processing:
- **Frequency Channelization**: Extract 250-500 Hz channels from the 12 kHz recordings
- **Temporal Chunking**: Create sliding windows of 0.5, 1.0, 2.0, and 5.0 seconds
- **Multi-Scale Tiles**: Generate embeddings for each frequency-time combination

This deferred processing approach allows optimization based on CASCADE's actual requirements and learned insights from the collection phase.

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

Even when CASCADE operates on fixed frequencies per band (such as 3573 kHz on 80m), the system generates embeddings across all channels during the embedding model training phase rather than during initial collection. This comprehensive approach provides critical benefits for training robustness and real-world performance.

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
    Encodes path geometry, ionospheric context, and natural cycles into 42D vector
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

        # 6. Natural cycle context (10D)
        features.extend([
            timestamp.hour / 24,           # UTC hour [0-1]
            timestamp.month / 12,          # Season proxy
            np.sin(timestamp.hour * 2*np.pi/24),    # Circular time
            np.cos(timestamp.hour * 2*np.pi/24),
            metadata.get('lunar_phase', 0.0),       # Lunar phase [0-1]
            metadata.get('qbo_index', 0.0) / 40,    # QBO normalized
            float(metadata.get('solar_cycle_phase') == 'MINIMUM'),  # Binary
            float(metadata.get('equinoctial_enhancement', False)),  # Binary
            metadata.get('seasonal_balance_factor', 1.0),  # [0.8-1.3]
            float(metadata.get('season') == 'WINTER')      # Winter flag
        ])

        return np.array(features)  # Total: 42D (expanded for cycle context)
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
            context_dim=42,  # Updated for cycle-aware features
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

### Cycle-Aware Training Integration

The enhanced path context encoder enables training that systematically accounts for natural cycles:

```python
def cycle_aware_training_step(batch_data):
    """
    Training step that integrates natural cycle context
    """
    for sample in batch_data:
        # Extract IQ characteristics
        channel_embedding = prop_vae.encode(sample.iq_data)

        # Enhanced path context with cycle information
        cycle_metadata = {
            'solar_cycle_phase': sample.solar_cycle_phase,
            'season': sample.season,
            'lunar_phase': sample.lunar_phase,
            'qbo_index': sample.qbo_index,
            'equinoctial_enhancement': sample.equinoctial_enhancement,
            'seasonal_balance_factor': sample.seasonal_balance_factor
        }

        path_context = path_encoder.encode_path_context(
            sample.tx_grid,
            sample.rx_grid,
            sample.timestamp,
            metadata=cycle_metadata
        )

        # Combined conditioning for realistic propagation
        combined_context = torch.cat([channel_embedding, path_context])

        # Generate cycle-appropriate channel characteristics
        channel_params = decoder(combined_context)

        # Apply to CASCADE signal for training
        cascade_rx = apply_propagation_with_cycles(
            sample.cascade_signal,
            channel_params,
            cycle_context=cycle_metadata
        )

        # Train CASCADE to decode without cycle knowledge
        decoded = cascade_model(cascade_rx)  # No cycle info at inference!
        loss = compute_loss(decoded, sample.original_data)

        return loss

def apply_propagation_with_cycles(signal, channel_params, cycle_context):
    """
    Apply propagation effects conditioned on natural cycles
    """
    # Base propagation effects
    propagated = apply_base_propagation(signal, channel_params)

    # Aggressive boost-aware adjustments for solar minimum
    if cycle_context['solar_cycle_phase'] == 'MINIMUM':
        # Aggressive enhancement of rare activity (boost strategy)
        if channel_params['storm_effects'] > 0:
            # Apply aggressive multiplier based on activity level
            k_index = cycle_context.get('k_index', 0)
            if k_index >= 5:
                channel_params['storm_effects'] *= 10.0  # 10x boost for major storms
            elif k_index >= 4:
                channel_params['storm_effects'] *= 5.0   # 5x boost for moderate storms
            elif k_index >= 3:
                channel_params['storm_effects'] *= 3.0   # 3x boost for minor storms

        # Aggressive flare effect enhancement
        if cycle_context.get('xray_class') in ['C', 'M', 'X']:
            flare_multipliers = {'C': 2.0, 'M': 5.0, 'X': 10.0}
            multiplier = flare_multipliers[cycle_context['xray_class']]
            channel_params['ionospheric_disturbance'] *= multiplier

    if cycle_context['season'] == 'WINTER':
        # Enhanced atmospheric noise during winter
        propagated = add_winter_noise_enhancement(propagated)

    if cycle_context['equinoctial_enhancement']:
        # Boost propagation efficiency during equinoxes
        propagated = apply_equinoctial_boost(propagated)

    if cycle_context['lunar_phase'] in [0.0, 0.5]:  # New/full moon
        # Subtle EME and tidal effects
        propagated = apply_lunar_effects(propagated)

    return propagated
```

### Temporal Context Validation

The system validates that training data includes adequate temporal diversity:

```python
class CycleContextValidator:
    """
    Ensures training data represents all natural cycle phases
    """
    def __init__(self):
        self.cycle_requirements = {
            'seasonal': {
                'WINTER': (0.25, 0.35),   # 25-35% (enhanced for solar min)
                'SPRING': (0.20, 0.30),   # 20-30%
                'SUMMER': (0.15, 0.25),   # 15-25% (reduced common conditions)
                'AUTUMN': (0.20, 0.30)    # 20-30%
            },
            'lunar_phases': {
                'new_moon': (0.08, 0.12),      # ±10% around 10%
                'full_moon': (0.08, 0.12),     # ±10% around 10%
                'other_phases': (0.76, 0.84)   # Remaining 80%
            },
            'solar_activity': {
                'quiet': (0.25, 0.35),         # Reduced due to boost strategy
                'moderate': (0.35, 0.45),      # Increased representation
                'active': (0.25, 0.35)         # Heavily boosted (vs natural 5%)
            }
        }

    def validate_cycle_coverage(self, training_embeddings):
        """
        Verify training data meets cycle coverage requirements
        """
        # Check seasonal distribution
        seasonal_dist = self.calculate_seasonal_distribution(training_embeddings)
        for season, (min_pct, max_pct) in self.cycle_requirements['seasonal'].items():
            actual_pct = seasonal_dist[season]
            if not (min_pct <= actual_pct <= max_pct):
                raise CycleValidationError(
                    f"Season {season}: {actual_pct:.3f} outside range [{min_pct:.3f}, {max_pct:.3f}]"
                )

        # Check lunar phase distribution
        lunar_dist = self.calculate_lunar_distribution(training_embeddings)
        for phase, (min_pct, max_pct) in self.cycle_requirements['lunar_phases'].items():
            actual_pct = lunar_dist[phase]
            if not (min_pct <= actual_pct <= max_pct):
                raise CycleValidationError(
                    f"Lunar phase {phase}: {actual_pct:.3f} outside range [{min_pct:.3f}, {max_pct:.3f}]"
                )

        # Check solar activity distribution (adjusted for solar minimum)
        activity_dist = self.calculate_activity_distribution(training_embeddings)
        for level, (min_pct, max_pct) in self.cycle_requirements['solar_activity'].items():
            actual_pct = activity_dist[level]
            if not (min_pct <= actual_pct <= max_pct):
                raise CycleValidationError(
                    f"Activity level {level}: {actual_pct:.3f} outside range [{min_pct:.3f}, {max_pct:.3f}]"
                )

        return True  # All validations passed
```

### Solar Minimum Boost Strategy Integration

The embedding models are designed to handle the aggressive rare event boost strategy implemented during solar minimum collection:

```python
class BoostAwareEmbeddingTrainer:
    """
    Training that accounts for aggressive solar minimum boost strategy
    """
    def __init__(self):
        self.boost_compensation = {
            'quiet_conditions': 2.33,    # Upweight underrepresented
            'moderate_activity': 0.625,  # Downweight moderately overrepresented
            'high_activity': 0.167       # Downweight heavily overrepresented
        }

    def train_embedding_with_boost_awareness(self, data_batch):
        """
        Train embedding models accounting for collection bias
        """
        for sample in data_batch:
            # Classify activity level
            activity_level = self.classify_solar_activity(sample)

            # Apply boost compensation weight
            compensation_weight = self.boost_compensation[activity_level]

            # Train with bias-corrected weighting
            loss = compute_embedding_loss(sample)
            weighted_loss = loss * compensation_weight

            # Update model with corrected gradients
            optimizer.step(weighted_loss)

    def classify_solar_activity(self, sample):
        """
        Classify samples for boost compensation
        """
        if sample.k_index >= 4 or sample.xray_class in ['M', 'X']:
            return 'high_activity'
        elif sample.k_index >= 2 or sample.xray_class == 'C':
            return 'moderate_activity'
        else:
            return 'quiet_conditions'
```

### Validation Adjustments for Boost Strategy

The validation requirements are updated to reflect the intentional bias from aggressive boosting:

```python
def boost_aware_validation_requirements():
    """
    Validation thresholds adjusted for solar minimum boost strategy
    """
    # Note: These reflect COLLECTED distribution, not natural distribution
    validation_targets = {
        'solar_activity_distribution': {
            'quiet': (0.25, 0.35),      # Reduced from natural ~70%
            'moderate': (0.35, 0.45),   # Increased from natural ~25%
            'active': (0.25, 0.35)      # Heavily boosted from natural ~5%
        },
        'rare_event_coverage': {
            'K5_storms': 0.95,          # 95%+ capture rate
            'K6_storms': 1.0,           # 100% capture rate
            'M_class_flares': 1.0,      # 100% capture rate
            'X_class_flares': 1.0       # 100% capture rate
        },
        'boost_effectiveness': {
            'min_rare_event_multiplier': 5.0,   # At least 5x boost
            'max_rare_event_multiplier': 10.0,  # Up to 10x boost
            'c_class_inclusion': True           # Must include C-class
        }
    }

    return validation_targets
```

This boost-aware approach ensures the embedding models properly handle the intentionally biased training data while maintaining the ability to generate realistic propagation for all activity levels during CASCADE training.

### Real-World Validation Integration

The embedding models undergo comprehensive validation using both laboratory testing and innovative real-world geographic diversity testing:

```python
class EmbeddingValidationFramework:
    """
    Comprehensive validation of embedding quality using real-world testing
    """
    def __init__(self):
        self.lab_test_suite = ControlledPropagationTesting()
        self.real_world_tester = GeographicDiversityTesting()
        self.embedding_models = {
            'noise_vae': QuietChannelVAE(),
            'prop_vae': FT8PropagationVAE(),
            'path_encoder': PathContextEncoder()
        }

    def validate_embedding_quality(self):
        """
        Validate embeddings produce realistic propagation when applied to CASCADE
        """
        validation_results = {}

        # Test 1: Laboratory validation with known conditions
        lab_conditions = [
            {'name': 'quiet', 'snr': 15, 'multipath': None},
            {'name': 'storm', 'snr': -15, 'multipath': [0, 2.1, 5.3]},
            {'name': 'aurora', 'snr': -10, 'fading': 0.5},
            {'name': 'flutter', 'snr': 5, 'doppler': 2.0}
        ]

        for condition in lab_conditions:
            # Generate embedding from real data
            real_sample = self.get_real_sample(condition['name'])
            embedding = self.embedding_models['prop_vae'].encode(real_sample)

            # Apply to test CASCADE signal
            test_signal = generate_cascade_test_signal()
            propagated = apply_embedding_to_signal(test_signal, embedding)

            # Validate in controlled environment
            lab_result = self.lab_test_suite.test_condition(
                signal=propagated,
                expected_condition=condition
            )

            validation_results[f"lab_{condition['name']}"] = lab_result

        # Test 2: Real-world path validation
        real_world_paths = [
            {'tx': 'VK2', 'rx': 'W1', 'type': 'long_path'},
            {'tx': 'JA1', 'rx': 'G0', 'type': 'short_path'},
            {'tx': 'VK2', 'rx': 'OH', 'type': 'polar_path'}
        ]

        for path in real_world_paths:
            # Get embedding from similar real path data
            path_embedding = self.find_similar_path_embedding(path)

            # Test with real remote transmission
            real_world_result = self.real_world_tester.test_embedding_realism(
                embedding=path_embedding,
                tx_location=path['tx'],
                rx_locations=self.select_rx_coverage(path['rx'])
            )

            validation_results[f"real_{path['type']}"] = real_world_result

        return validation_results

    def validate_multi_path_optimization(self):
        """
        Test embedding-driven optimization across multiple simultaneous paths
        """
        # Coordinate simultaneous transmissions from multiple locations
        multi_tx_test = {
            'tx_stations': ['VK2', 'JA1', 'G0'],  # Three continents
            'rx_network': self.select_global_sdr_coverage(),
            'test_duration': 600  # 10-minute coordinated test
        }

        # Test embeddings for multi-path optimization
        results = {}
        for minute in range(10):
            # Get current propagation embeddings for all paths
            current_embeddings = {}
            for tx in multi_tx_test['tx_stations']:
                for rx in multi_tx_test['rx_network']:
                    path_key = f"{tx}→{rx.grid}"
                    # Use real-time SDR data to generate embedding
                    live_data = self.collect_live_sdr_data(rx, tx_frequency=14080000)
                    current_embeddings[path_key] = self.embedding_models['prop_vae'].encode(live_data)

            # Optimize CASCADE parameters for all paths simultaneously
            optimized_params = self.multi_path_optimizer.optimize(
                embeddings=current_embeddings,
                objectives=['min_path_reliability', 'max_total_throughput'],
                fairness_weight=0.3
            )

            # Test optimized parameters via remote transmission
            test_results = self.coordinate_multi_tx_test(
                tx_stations=multi_tx_test['tx_stations'],
                rx_network=multi_tx_test['rx_network'],
                cascade_params=optimized_params
            )

            results[f"minute_{minute}"] = {
                'embeddings_used': len(current_embeddings),
                'optimization_time_ms': measure_optimization_time(),
                'path_performance': test_results,
                'fairness_score': calculate_path_fairness(test_results)
            }

        return results
```

This validation framework ensures that embeddings not only compress propagation data effectively but also produce realistic and useful propagation effects when applied to CASCADE signals in both controlled and real-world testing scenarios.

## FT8 Propagation VAE Architecture

The FT8 Propagation VAE extracts channel characteristics by comparing received FT8 signals with their known transmitted form. This model leverages the structured nature of FT8 transmissions to precisely characterize propagation effects.

### Handling Crowded FT8 Bands

FT8 bands present a unique challenge - dozens of stations transmit simultaneously every 15 seconds within a narrow 2.7 kHz segment. Each signal occupies approximately 50 Hz of bandwidth, with signals packed closely together. To extract individual propagation characteristics from this crowded spectrum, the system must isolate each signal before generating embeddings.

The solution leverages FT8's structured nature and the decoder's ability to identify individual signals:

```python
def isolate_ft8_signals(recording, ft8_band_center=14074000):
    """
    Extract individual FT8 propagation from crowded band

    The FT8 decoder identifies all signals in the band, providing their
    exact frequencies, decoded messages, and SNR. We use this information
    to isolate each signal for individual propagation analysis.
    """
    # Step 1: Decode all FT8 signals in the recording
    # The decoder handles the crowded band, identifying each signal
    decoded_signals = ft8_decoder.decode_all(recording)
    # Returns: [{frequency: 14074237, message: "CQ VK2ABC QF56",
    #            snr: -15, tx_grid: "QF56", ...}, ...]

    isolated_propagations = []

    for signal in decoded_signals:
        # Step 2: Apply narrow bandpass filter around THIS signal only
        # FT8 signals are ~50 Hz wide, we use 60 Hz to ensure complete capture
        signal_freq = signal['frequency']
        isolated_signal = bandpass_filter(
            recording,
            center_freq=signal_freq,
            bandwidth=60  # Slightly wider than signal for guard band
        )

        # Step 3: Generate ideal signal for comparison
        # The decoder provides the message content, allowing us to
        # reconstruct what was transmitted
        ideal_signal = generate_ideal_ft8(signal['message'])

        # Step 4: Extract propagation by comparing received vs ideal
        # This isolated comparison reveals the specific propagation
        # effects on this individual path
        prop_embedding = propagation_vae.encode(
            received=isolated_signal,
            ideal=ideal_signal,
            snr=signal['snr']
        )

        isolated_propagations.append({
            'frequency': signal_freq,
            'embedding': prop_embedding,
            'path': f"{signal['tx_grid']}->{rx_grid}",
            'distance_km': calculate_distance(signal['tx_grid'], rx_grid),
            'snr_db': signal['snr']
        })

    return isolated_propagations
```

This frequency isolation approach transforms a crowded band with 20-30 simultaneous signals into 20-30 individual propagation embeddings, each representing a unique transmission path. A single 15-second FT8 period can yield embeddings for paths ranging from local ground wave (100 km) to antipodal multi-hop (20,000 km), each experiencing different propagation conditions despite transmitting at the same moment.

### Privacy-Preserving Approach for FT8 Data

The system must balance privacy protection with the need for accurate propagation analysis. Grid squares are essential for calculating propagation distances and understanding path geometry, but callsigns must be anonymized to protect operator privacy:

```python
def anonymize_ft8_with_preserved_geography(decoded_signal):
    """
    Anonymize personal identifiers while preserving propagation geography

    Critical insight: Grid squares (like QF56) represent geographic regions
    (~100x70 km) containing thousands of operators, not individual locations.
    Callsigns uniquely identify individuals and must be anonymized.
    """
    # Anonymize the callsign using one-way hash
    callsign_hash = hashlib.sha256(
        decoded_signal['callsign'].encode()
    ).hexdigest()[:8]  # Keep 8 chars for uniqueness tracking

    # PRESERVE grid squares - essential for propagation analysis
    tx_grid = decoded_signal['tx_grid']  # e.g., "QF56" (Sydney region)
    rx_grid = decoded_signal['rx_grid']   # e.g., "FN42" (New England)

    # Calculate propagation metrics from preserved grids
    distance_km = calculate_great_circle_distance(tx_grid, rx_grid)
    bearing_deg = calculate_bearing(tx_grid, rx_grid)

    # Determine propagation characteristics
    if distance_km < 500:
        likely_mode = "ground_wave"
    elif distance_km < 2000:
        likely_mode = "single_hop"
    else:
        likely_mode = "multi_hop"

    return {
        'tx_hash': callsign_hash,      # Anonymized identifier
        'tx_grid': tx_grid,             # Geographic region preserved
        'rx_grid': rx_grid,             # Receiver location preserved
        'distance_km': distance_km,     # Essential for training
        'bearing_deg': bearing_deg,     # Path direction
        'likely_mode': likely_mode,     # Propagation classification
        'snr_db': decoded_signal['snr'],
        'frequency_hz': decoded_signal['frequency']
    }
```

**Why Grid Squares Can Be Preserved:**
1. **Non-identifying**: Each grid square covers ~7,000 km² with thousands of potential operators
2. **Essential for physics**: Propagation behavior fundamentally depends on distance and path geometry
3. **Already public**: Grid squares are openly shared in FT8/WSPR protocols
4. **Scientific value**: Enables research on ionospheric propagation patterns

**What Gets Anonymized:**
- Callsigns (direct personal identifiers)
- Message content (may contain personal information)
- Exact timestamps (replaced with relative timing within recordings)

This approach ensures CASCADE can learn accurate propagation models while respecting amateur radio operator privacy.

### Station Fingerprinting for Enhanced Propagation Modeling

The anonymized callsign hashes enable sophisticated station fingerprinting that captures persistent equipment characteristics and operating patterns. This provides valuable training signal diversity without compromising privacy, as amateur radio transmissions are inherently public.

#### Equipment Signature Extraction

Each station's equipment leaves subtle but persistent signatures in their transmissions:

```python
def extract_station_fingerprint(signal_history, tx_hash):
    """
    Build equipment and operational fingerprint from signal history

    Amateur radio signals are publicly transmitted and receivable by anyone.
    We extract technical characteristics to improve propagation modeling.
    """
    # Equipment characteristics (implicit from signal analysis)
    equipment_signature = {
        'phase_noise_db': calculate_phase_noise(signal_history),
        'freq_stability_ppb': measure_frequency_drift(signal_history),
        'pa_linearity': estimate_amplifier_compression(signal_history),
        'modulation_quality': assess_signal_fidelity(signal_history),

        # These characteristics naturally emerge from:
        # - Different transceiver architectures (SDR vs analog)
        # - Reference oscillator quality (TCXO vs OCXO)
        # - Power amplifier design (Class A/AB/E)
        # - DSP implementation differences
    }

    # Operating patterns (behavioral fingerprint)
    operating_patterns = {
        'typical_power_class': classify_power_level(signal_history),  # QRP/QRO
        'active_hours_utc': extract_activity_schedule(signal_history),
        'band_preferences': analyze_frequency_usage(signal_history),
        'message_patterns': categorize_usage(signal_history)  # CQ/QSO/beacon
    }

    # Propagation consistency (station-specific paths)
    path_characteristics = {
        'common_paths': identify_regular_contacts(signal_history),
        'antenna_pattern': infer_directionality(signal_history),
        'ground_conductivity': estimate_local_ground(signal_history),
        'noise_environment': characterize_local_qrm(signal_history)
    }

    return {
        'tx_hash': tx_hash,
        'equipment_embedding': encode_equipment(equipment_signature),
        'pattern_embedding': encode_patterns(operating_patterns),
        'path_embedding': encode_paths(path_characteristics),
        'observation_count': len(signal_history),
        'confidence': calculate_statistical_confidence(signal_history)
    }
```

#### Advanced Equipment Signature Extraction

The system implements sophisticated signal processing techniques to separate equipment characteristics from propagation effects. This separation is crucial because equipment signatures remain consistent across multiple observations while propagation varies with atmospheric conditions. By isolating these signatures, CASCADE can better model the true diversity of real-world transmissions.

##### Phase Noise Analysis from Known Symbols

Every transmitter's local oscillator exhibits unique phase noise characteristics determined by its crystal quality (TCXO vs OCXO), temperature stability, and circuit design. Known symbol positions (from FT8 or CASCADE's own transmitted symbols) allow precise measurement:

```python
def analyze_phase_noise_from_symbols(iq_signal, decoded_symbols, symbol_type='ft8'):
    """
    Extract phase noise characteristics from symbol centers
    Works for both FT8 (training) and CASCADE (telemetry)
    Task T076a implementation
    """
    # Extract symbol positions based on type
    if symbol_type == 'ft8':
        # FT8 uses 8-FSK with known symbol positions
        symbol_centers = extract_symbol_centers(iq_signal, decoded_symbols)
        ideal_points = ft8_ideal_constellation()
    elif symbol_type == 'cascade':
        # CASCADE uses its own learned patterns
        symbol_centers = extract_cascade_symbols(iq_signal, decoded_symbols)
        ideal_points = cascade_pattern_points(decoded_symbols)

    # Measure phase deviation from ideal constellation points
    phase_deviations = []
    for actual, ideal in zip(symbol_centers, ideal_points):
        phase_error = np.angle(actual) - np.angle(ideal)
        phase_deviations.append(phase_error)

    # Analyze phase noise spectrum
    phase_noise_profile = {
        'rms_phase_error': np.std(phase_deviations),
        'phase_noise_10hz': calculate_phase_noise_at_offset(phase_deviations, 10),
        'phase_noise_100hz': calculate_phase_noise_at_offset(phase_deviations, 100),
        'phase_noise_1khz': calculate_phase_noise_at_offset(phase_deviations, 1000),
        'oscillator_quality': classify_oscillator(phase_deviations)  # TCXO/OCXO/GPS
    }

    return phase_noise_profile
```

**For CASCADE Telemetry:**

During live operation, CASCADE characterizes its own transmitted signal:

```python
def generate_tx_station_fingerprint(cascade_tx):
    """
    Extract station fingerprint from CASCADE's own transmission
    Used in TX telemetry (16-D component)
    """
    # CASCADE knows what it transmitted
    transmitted_symbols = cascade_tx.symbols
    transmitted_iq = cascade_tx.iq_samples

    # Measure own equipment characteristics
    fingerprint = analyze_phase_noise_from_symbols(
        transmitted_iq,
        transmitted_symbols,
        symbol_type='cascade'
    )

    # Encode to 16-D embedding
    station_embedding = station_encoder.encode(fingerprint)

    return station_embedding  # Used in TX telemetry
```

This approach works identically for FT8 (training data) and CASCADE (telemetry), providing consistent equipment characterization across both contexts.

##### Multi-Diversity Equipment-Propagation Separation

When the same transmission is received simultaneously by multiple geographically distributed receivers, features that remain consistent across all receivers must originate from the transmitter's equipment, while features that vary are due to different propagation paths. This fundamental principle enables clean separation:

```python
def separate_equipment_from_propagation(multi_rx_observations):
    """
    Use multiple simultaneous receivers to isolate TX equipment characteristics
    Task T076d, T076f implementation
    """
    # Common TX characteristics across all receivers = equipment
    # Varying characteristics = propagation

    common_features = []
    varying_features = []

    for feature in ['phase_noise', 'freq_drift', 'spectral_regrowth']:
        values = [obs[feature] for obs in multi_rx_observations]

        if np.std(values) / np.mean(values) < 0.1:  # Low variance
            # Consistent across receivers = equipment signature
            common_features.append({
                'feature': feature,
                'value': np.median(values),
                'confidence': 1.0 - (np.std(values) / np.mean(values))
            })
        else:
            # Variable across receivers = propagation effect
            varying_features.append({
                'feature': feature,
                'range': [np.min(values), np.max(values)],
                'propagation_dependency': np.std(values)
            })

    return {
        'equipment_signatures': common_features,
        'propagation_effects': varying_features,
        'separation_confidence': calculate_separation_quality(multi_rx_observations)
    }
```

##### Temporal Scale Separation

Equipment and propagation effects operate on fundamentally different time scales. Phase noise and oscillator jitter change in microseconds to milliseconds, while ionospheric fading and Doppler shifts evolve over seconds to minutes. This temporal separation provides another dimension for isolating equipment signatures:

```python
def separate_temporal_scales(signal_history):
    """
    Separate fast equipment variations from slow propagation changes
    Task T076g implementation
    """
    # Equipment: microsecond to millisecond variations (phase noise, jitter)
    # Propagation: second to minute variations (fading, Doppler)

    time_scales = {
        'microsecond': {'window': 1e-6, 'features': []},
        'millisecond': {'window': 1e-3, 'features': []},
        'second': {'window': 1.0, 'features': []},
        'minute': {'window': 60.0, 'features': []}
    }

    for scale_name, scale_info in time_scales.items():
        window = scale_info['window']

        # Analyze variations at this time scale
        variations = analyze_at_timescale(signal_history, window)

        if scale_name in ['microsecond', 'millisecond']:
            # Fast variations = equipment
            scale_info['features'] = extract_equipment_features(variations)
            scale_info['source'] = 'equipment'
        else:
            # Slow variations = propagation
            scale_info['features'] = extract_propagation_features(variations)
            scale_info['source'] = 'propagation'

    return time_scales
```

##### Reciprocal Path Analysis

HF propagation follows the principle of reciprocity - the path from station A to B experiences nearly identical propagation as the path from B to A. However, the equipment at each end is different. By analyzing bidirectional communications between station pairs, we can isolate each station's unique equipment signature:

```python
def analyze_reciprocal_paths(station_a_to_b, station_b_to_a):
    """
    Use bidirectional paths to isolate equipment at each end
    Task T076e implementation
    """
    # Propagation is reciprocal, equipment is not
    # A→B path propagation ≈ B→A path propagation
    # A's TX signature ≠ B's TX signature

    # Extract common propagation from both directions
    propagation = {
        'multipath_profile': average_multipath(station_a_to_b, station_b_to_a),
        'path_loss': average_path_loss(station_a_to_b, station_b_to_a),
        'doppler_spread': average_doppler(station_a_to_b, station_b_to_a)
    }

    # Isolate equipment signatures
    equipment_a = {
        'tx_signature': extract_tx_only_features(station_a_to_b),
        'rx_signature': extract_rx_only_features(station_b_to_a)
    }

    equipment_b = {
        'tx_signature': extract_tx_only_features(station_b_to_a),
        'rx_signature': extract_rx_only_features(station_a_to_b)
    }

    return {
        'shared_propagation': propagation,
        'station_a_equipment': equipment_a,
        'station_b_equipment': equipment_b,
        'reciprocity_score': calculate_path_reciprocity(station_a_to_b, station_b_to_a)
    }
```

##### PA Linearity Estimation

Power amplifier (PA) non-linearity creates distinctive spectral regrowth patterns adjacent to the intended signal. Different PA classes (A, AB, D, E) exhibit characteristic distortion signatures that persist across all transmissions from that station. By analyzing the spectral shoulders and intermodulation products, we can identify the PA type and its operating point:

```python
def estimate_pa_linearity_from_spectral_regrowth(iq_signal):
    """
    Analyze power amplifier linearity from spectral regrowth
    Task T076c implementation
    """
    # Non-linear PAs cause spectral regrowth adjacent to signal
    spectrum = np.fft.fft(iq_signal)

    # Measure in-band vs out-of-band power
    signal_bw = 50  # Hz for FT8
    in_band_power = calculate_power(spectrum, -signal_bw/2, signal_bw/2)

    # Adjacent channel power (spectral regrowth)
    left_adjacent = calculate_power(spectrum, -signal_bw*1.5, -signal_bw/2)
    right_adjacent = calculate_power(spectrum, signal_bw/2, signal_bw*1.5)

    # Third-order intermodulation products
    im3_left = calculate_power(spectrum, -signal_bw*2.5, -signal_bw*1.5)
    im3_right = calculate_power(spectrum, signal_bw*1.5, signal_bw*2.5)

    linearity_metrics = {
        'acpr_db': 10 * np.log10(in_band_power / np.mean([left_adjacent, right_adjacent])),
        'im3_suppression_db': 10 * np.log10(in_band_power / np.mean([im3_left, im3_right])),
        'pa_class_estimate': classify_pa_type(spectrum),  # Class A/AB/D/E
        'compression_point_dbm': estimate_p1db(spectrum),
        'efficiency_estimate': estimate_pa_efficiency(spectrum)
    }

    return linearity_metrics
```

##### Frequency Drift Tracking

Oscillator frequency stability varies dramatically between equipment types. A GPS-disciplined oscillator drifts less than 1 PPB, while a basic TCXO might drift 10 PPM with temperature changes. By tracking frequency offsets across multiple observations over days or weeks, we build a stability profile that uniquely characterizes each station's reference oscillator:

```python
def track_frequency_drift_over_time(signal_observations):
    """
    Build frequency stability profile across multiple observations
    Task T076b implementation
    """
    drift_profile = []

    for obs in signal_observations:
        # Measure carrier frequency offset
        carrier_offset = measure_carrier_frequency(obs.iq_signal) - obs.nominal_freq

        drift_profile.append({
            'timestamp': obs.timestamp,
            'temperature': obs.ambient_temp if available else estimate_from_time(),
            'offset_hz': carrier_offset,
            'offset_ppm': carrier_offset / obs.nominal_freq * 1e6
        })

    # Analyze drift characteristics
    stability_metrics = {
        'total_drift_ppm': max([d['offset_ppm'] for d in drift_profile]) -
                          min([d['offset_ppm'] for d in drift_profile]),
        'drift_rate_ppm_per_hour': calculate_drift_rate(drift_profile),
        'temperature_coefficient': correlate_with_temperature(drift_profile),
        'oscillator_type': classify_oscillator_type(drift_profile),  # TCXO/OCXO/GPS
        'aging_rate_ppb_per_day': estimate_aging(drift_profile)
    }

    return stability_metrics
```

##### Statistical Confidence Scoring

Not all equipment signatures are equally reliable. A signature extracted from hundreds of observations across multiple bands and receivers is far more trustworthy than one from a handful of weak signals. This confidence scoring system weights multiple factors to assess the statistical reliability of each extracted signature:

```python
def calculate_signature_confidence(equipment_observations):
    """
    Score confidence in extracted equipment signatures
    Task T076j implementation
    """
    confidence_factors = {
        'observation_count': min(len(equipment_observations) / 100, 1.0),
        'temporal_span': min(calculate_time_span(equipment_observations) / (30*24*3600), 1.0),
        'snr_quality': np.mean([obs.snr for obs in equipment_observations]) / 30,
        'multi_band_confirmation': count_unique_bands(equipment_observations) / 6,
        'multi_rx_correlation': calculate_multi_rx_agreement(equipment_observations),
        'feature_stability': calculate_feature_variance(equipment_observations),
        'reciprocal_validation': check_reciprocal_consistency(equipment_observations)
    }

    # Weighted confidence score
    weights = {
        'observation_count': 0.20,
        'temporal_span': 0.15,
        'snr_quality': 0.15,
        'multi_band_confirmation': 0.15,
        'multi_rx_correlation': 0.20,
        'feature_stability': 0.10,
        'reciprocal_validation': 0.05
    }

    overall_confidence = sum(confidence_factors[k] * weights[k] for k in weights)

    return {
        'overall_confidence': overall_confidence,
        'confidence_factors': confidence_factors,
        'reliability_class': classify_reliability(overall_confidence)
    }
```

#### Integration of Equipment Signatures

These advanced signal processing techniques work together to create a comprehensive equipment fingerprint for each station. The multi-dimensional approach ensures robust separation even in challenging conditions:

1. **Phase noise** reveals oscillator quality and temperature stability
2. **Frequency drift** tracks long-term stability and aging characteristics
3. **PA linearity** identifies amplifier type and operating conditions
4. **Multi-diversity** separates transmitter from propagation effects
5. **Reciprocal paths** isolate equipment at both ends of a QSO
6. **Temporal analysis** leverages different time scales of variation
7. **Confidence scoring** ensures only statistically valid signatures are used

This rich equipment characterization significantly improves CASCADE's ability to model realistic signal diversity, as the same message transmitted by different stations will exhibit unique equipment-induced variations even under identical propagation conditions.

#### Persistent Path Learning

Regular communication paths (like daily nets) provide valuable consistent propagation data:

```python
class PersistentPathModel:
    """
    Learn propagation characteristics for frequently-used station pairs
    """

    def __init__(self):
        self.path_models = {}

    def update_path(self, tx_hash, rx_grid, propagation_observation):
        """
        Build path-specific propagation models for regular contacts
        """
        path_key = f"{tx_hash}_{rx_grid}"

        if path_key not in self.path_models:
            self.path_models[path_key] = {
                'observations': [],
                'typical_snr': None,
                'diurnal_pattern': None,
                'seasonal_variation': None,
                'equipment_factor': None
            }

        # Add observation
        self.path_models[path_key]['observations'].append({
            'timestamp': propagation_observation.timestamp,
            'snr': propagation_observation.snr,
            'propagation_mode': propagation_observation.mode,
            'k_index': propagation_observation.k_index
        })

        # Update model after sufficient observations
        if len(self.path_models[path_key]['observations']) > 50:
            self.train_path_specific_model(path_key)

    def train_path_specific_model(self, path_key):
        """
        Learn the specific propagation characteristics of this station pair

        This captures:
        - Antenna patterns at both ends
        - Local noise floors
        - Equipment capabilities
        - Typical operating conditions
        - Path-specific propagation modes
        """
        observations = self.path_models[path_key]['observations']

        # Extract patterns
        self.path_models[path_key]['typical_snr'] = np.percentile(
            [o['snr'] for o in observations], [10, 50, 90]
        )
        self.path_models[path_key]['diurnal_pattern'] = extract_time_pattern(
            observations
        )
        self.path_models[path_key]['equipment_factor'] = estimate_station_capability(
            observations
        )
```

#### Station-Aware Embedding Architecture

Integrating station fingerprints into propagation embeddings provides richer training data:

```python
class StationAwarePropagationVAE(nn.Module):
    """
    Enhanced propagation VAE that includes station fingerprints
    """

    def __init__(self, prop_dim=128, station_dim=16):
        super().__init__()

        # Standard propagation encoder
        self.propagation_encoder = PropagationVAE(output_dim=prop_dim)

        # Station fingerprint encoder
        self.station_encoder = nn.Sequential(
            nn.Linear(48, 64),  # Equipment + pattern features
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, station_dim)
        )

        # Combined embedding
        self.combined_dim = prop_dim + station_dim

    def forward(self, signal, station_fingerprint):
        # Extract propagation characteristics
        prop_embedding = self.propagation_encoder(signal)

        # Extract station signature
        station_embedding = self.station_encoder(station_fingerprint)

        # Concatenate for station-aware propagation embedding
        combined = torch.cat([prop_embedding, station_embedding], dim=-1)

        return combined
```

#### Benefits for CASCADE Training

Station fingerprinting provides several advantages:

1. **Equipment Diversity**: Learn how different transceivers and antennas affect propagation
2. **Persistent Patterns**: Leverage regular nets and QSOs for consistent training data
3. **Anomaly Detection**: Identify unusual propagation by comparing to station norms
4. **Population Statistics**: Understand the real distribution of equipment and operating patterns

#### Data Collection Ethics

```python
"""
CASCADE Station Fingerprinting Ethics Statement:

This system analyzes publicly transmitted amateur radio signals in accordance
with international regulations. Amateur radio explicitly operates without
privacy expectations - transmissions must be unencrypted and identifiable.

What we do:
- Analyze technical characteristics of public transmissions
- Use one-way hashed identifiers (more private than callsigns)
- Extract equipment signatures to improve propagation modeling
- Learn from persistent paths to enhance prediction accuracy

Comparison to existing systems:
- PSK Reporter: Shows actual callsigns publicly (we hash them)
- RBN: Tracks CW signals with full callsign disclosure
- WSPRnet: Complete database with exact locations
- WebSDR: Many already record and archive everything

Our approach is MORE private than these widely-accepted systems while
advancing propagation science for the amateur radio community.
"""
```

This station fingerprinting approach transforms the hashed callsigns from simple anonymization tokens into valuable features that capture the real diversity of amateur radio equipment and operating patterns, significantly improving CASCADE's ability to model realistic propagation conditions.

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

## Temporal Correlation Preservation

One of the most critical aspects of the embedding system is maintaining exact temporal correlation between noise and propagation characteristics. The physics of HF radio demands that noise conditions and propagation effects from the same moment in time remain paired throughout the training pipeline.

### Why Temporal Correlation Matters

Radio propagation is not random - noise and signal propagation are intrinsically linked through physical processes:
- **Geomagnetic storms** simultaneously increase QRN and create auroral propagation
- **D-layer absorption** during solar flares affects both noise floor and signal strength
- **Thunderstorms** create QRN while also affecting local propagation
- **Diurnal changes** at sunrise/sunset impact both noise and propagation together

Training CASCADE with artificially mixed conditions (e.g., quiet night noise with disturbed daytime propagation) would create unrealistic scenarios that never occur in nature.

### Correlation Implementation

The system maintains temporal correlation through unique correlation IDs that encode the exact time relationship:

```python
def extract_correlated_embeddings(recording_10min):
    """
    Extract temporally-aligned embeddings from same time windows
    """
    correlated_pairs = []

    # Detect FT8 signals with precise timing
    ft8_signals = detect_ft8_with_timing(recording_10min)

    for ft8 in ft8_signals:
        # For each temporal window size
        for duration in [0.5, 1.0, 2.0, 5.0]:
            stride = duration / 2

            for window_start in np.arange(0, ft8.duration - duration, stride):
                # Absolute time in the recording
                absolute_time = ft8.start_time + window_start

                # Extract FT8 propagation from this exact window
                ft8_window = ft8.signal[window_start:window_start + duration]
                prop_embedding = propagation_vae.encode(ft8_window)

                # Extract QRN from the SAME time window
                qrn_window = recording_10min[absolute_time:absolute_time + duration]

                # Multi-channel QRN extraction
                noise_embeddings = []
                for freq_channel in frequency_channels:
                    qrn_channel = bandpass_filter(qrn_window, freq_channel)
                    noise_emb = noise_vae.encode(qrn_channel)
                    noise_embeddings.append({
                        'frequency': freq_channel.center,
                        'embedding': noise_emb
                    })

                # Store with correlation ID
                correlation_id = f"{recording_id}_{absolute_time}_{duration}"
                correlated_pairs.append({
                    'correlation_id': correlation_id,
                    'absolute_time': absolute_time,
                    'duration': duration,
                    'prop_embedding': prop_embedding,
                    'noise_embeddings': noise_embeddings,
                    'space_weather': recording_10min.space_weather
                })

    return correlated_pairs
```

### Storage Schema for Correlated Embeddings

The database schema ensures temporal relationships are preserved while maintaining privacy:

```sql
CREATE TABLE correlated_embeddings (
    correlation_id VARCHAR PRIMARY KEY,  -- Encodes session_time_duration
    session_id UUID NOT NULL,
    absolute_start_time FLOAT NOT NULL,  -- Seconds from recording start
    window_duration FLOAT NOT NULL,       -- 0.5, 1.0, 2.0, or 5.0 seconds

    -- Propagation embedding from FT8
    prop_embedding FLOAT[128] NOT NULL,
    ft8_frequency_hz INTEGER NOT NULL,
    snr_db FLOAT,

    -- Geographic data (preserved for propagation analysis)
    tx_grid VARCHAR(6),  -- Maidenhead grid (e.g., "QF56")
    rx_grid VARCHAR(6),  -- Receiver grid square
    distance_km FLOAT,   -- Calculated great circle distance
    bearing_deg FLOAT,   -- Path bearing

    -- Anonymized identifiers (privacy preserved)
    tx_hash VARCHAR(8),  -- SHA256 hash of callsign (first 8 chars)

    -- Noise embeddings (multiple frequency channels)
    noise_embeddings JSONB NOT NULL,  -- Array of {freq_hz, embedding[64]}

    -- Preserved correlations
    k_index INTEGER NOT NULL,
    solar_flux INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,  -- Actual UTC time

    -- Indexes for efficient retrieval
    INDEX idx_duration (window_duration),
    INDEX idx_temporal (session_id, absolute_start_time),
    INDEX idx_space_weather (k_index, solar_flux),
    INDEX idx_distance (distance_km),
    INDEX idx_path (tx_grid, rx_grid)
);
```

This schema ensures that during CASCADE training, noise and propagation embeddings are always fetched as naturally correlated pairs, preserving the physical relationships present in the real world.

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

For the FT8 Propagation VAE, the system pairs received signals with their decoded content, applying temporal slicing to match CASCADE's operating timescales:

```python
def prepare_propagation_training_data(recording):
    training_pairs = []
    for signal in detect_ft8(recording):
        if decoded := ft8_decode(signal):
            ideal = generate_ideal_ft8(decoded.message)

            # Temporal slicing for FT8 (12.64 second signals)
            for window_duration in [0.5, 1.0, 2.0, 5.0]:
                stride = window_duration / 2  # 50% overlap

                for start in np.arange(0, 12.64 - window_duration, stride):
                    training_pairs.append({
                        'received': signal.iq[start:start+window_duration],
                        'ideal': ideal[start:start+window_duration],
                        'snr': signal.snr,
                        'distance': calculate_distance(decoded.grids),
                        'window_start': signal.start_time + start,
                        'window_duration': window_duration
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

### Frequency-Matched and Time-Aligned Training Process

When using multi-channel QRN embeddings, the training process incorporates both frequency-specific noise application and temporal correlation preservation:

```python
def train_with_correlated_embeddings(cascade_model, embedding_db):
    """
    Train CASCADE with temporally-correlated noise and propagation embeddings
    """
    for epoch in range(n_epochs):
        for batch in training_loader:
            # Select appropriate fragment duration for CASCADE
            cascade_fragment_duration = select_fragment_duration(batch.snr)

            # Fetch time-correlated embedding pairs
            correlated_pairs = embedding_db.get_correlated_pairs(
                duration=cascade_fragment_duration,
                batch_size=batch.size
            )

            for pair in correlated_pairs:
                # Generate CASCADE signal
                cascade_signal = cascade_encoder(batch.data)

                # Apply frequency-matched noise from exact time window
                cascade_freq = batch.operating_frequency  # e.g., 3573 kHz
                noise_embedding = select_frequency_channel(
                    pair.noise_embeddings,
                    cascade_freq
                )

                # Apply propagation from SAME time window
                prop_embedding = pair.prop_embedding  # Temporally aligned

                # Both effects are from same moment in time
                cascade_signal = apply_propagation(cascade_signal, prop_embedding)
                cascade_signal = apply_noise(cascade_signal, noise_embedding)

                # Train with naturally correlated conditions
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

**Week 1**: Create curated diverse dataset (3-5TB from 40-50TB total)
- Apply rarity scoring to identify valuable training examples
- Ensure coverage of all propagation modes and conditions
- Balance rare events with baseline coverage

**Weeks 2-3**: Train both VAE models on curated dataset
- 3-5 days training per model on single GPU
- Validate reconstruction quality and embedding distributions
- Fine-tune hyperparameters based on validation metrics

**Weeks 3-4**: Generate embeddings for training dataset
- Process 3-5TB of IQ data through trained models
- Apply multi-scale processing:
  - Frequency channelization: 250-500 Hz channels
  - Temporal windowing: 0.5, 1.0, 2.0, 5.0 second slices
  - Preserve exact temporal correlation between QRN and FT8
- Generate 15-25GB of temporally-correlated embeddings
- Build indexes for efficient retrieval by duration and frequency

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

## Storage Implications of Multi-Scale Embeddings

The multi-scale approach to embeddings, while providing rich training data, has significant storage implications that must be carefully managed. Understanding these implications is crucial for planning infrastructure and optimizing the training pipeline.

### Storage Requirements Analysis

The move from simple single embeddings to multi-scale frequency-temporal embeddings dramatically increases storage needs:

**Original Simple Approach**:
- One 64D noise embedding per 10-minute recording
- One 128D propagation embedding per recording
- Total: ~0.8 KB per recording
- Full training set: 15-25 GB

**Full Multi-Scale Approach (Unoptimized)**:
- 48 frequency channels (250 Hz each across 12 kHz)
- 2,220 temporal windows (0.5, 1.0, 2.0, 5.0 second slices with 50% overlap)
- 106,560 QRN embeddings per recording
- 625 FT8 propagation embeddings (25 signals × 25 time windows)
- Total: ~27.6 MB per recording
- Full dataset: 41.4 TB (impractical)
- Training subset only (150,000 recordings): **4.14 TB**

The 4.14 TB figure for the training subset represents all possible frequency-time combinations without any optimization. While large, this is actually manageable with modern storage solutions, considering that the original raw IQ data is 3-5 TB. The embeddings provide a different representation of the same data, not a massive expansion.

### Storage Optimization Strategies

Several strategies can reduce storage while preserving training effectiveness:

**Hierarchical Frequency Sampling**: Dense sampling near CASCADE operating frequencies, sparse sampling elsewhere:
```python
def hierarchical_frequency_sampling(cascade_freq):
    """
    Generate more embeddings near CASCADE frequencies,
    fewer in distant spectrum
    """
    channels = []

    # Dense: Every 250 Hz within ±1 kHz of CASCADE
    for offset in range(-1000, 1000, 250):
        channels.append(cascade_freq + offset)

    # Sparse: Every 1 kHz elsewhere
    for offset in [-6000, -4000, -2000, 2000, 4000, 6000]:
        channels.append(cascade_freq + offset)

    return channels  # 15 channels instead of 48
```

**Selective Temporal Sampling**: Reduce overlap and sample strategically:
```python
def selective_temporal_sampling():
    """
    Sample time windows based on importance
    """
    windows = []

    # High resolution for short durations (better for transients)
    for start in np.arange(0, 600, 1.0):  # Every second for 0.5s windows
        windows.append((start, 0.5))

    # Lower resolution for longer durations (for slow fading)
    for start in np.arange(0, 600, 10.0):  # Every 10s for 5s windows
        windows.append((start, 5.0))

    return windows  # ~450 windows instead of 2,220
```

**Compression Techniques**:
- Use float16 instead of float32: 50% reduction with minimal precision loss
- Quantize to int8 for QRN embeddings: 75% reduction
- Apply zlib compression to embedding arrays: Additional 20-30% reduction

With these optimizations, storage reduces to approximately 150-200 GB for the training subset - a 20x reduction while preserving the most valuable information.

### Practical Storage Architecture

For training CASCADE, the recommended approach balances completeness with practicality:

```python
# Storage hierarchy for embeddings
/embeddings/
├── cascade_frequencies/     # Full resolution at CASCADE freqs (1 TB)
│   ├── 3573khz/            # 80m band
│   ├── 7074khz/            # 40m band
│   └── ...
├── sparse_spectrum/        # Reduced resolution elsewhere (500 GB)
│   ├── 1000khz_steps/
│   └── ...
├── rare_events/           # Full resolution for K≥5 storms (100 GB)
│   ├── k7_storms/
│   └── x_class_flares/
└── indices/               # Fast lookup structures (10 GB)
    ├── kdtree.idx
    └── correlation.idx
```

This hierarchical structure ensures detailed coverage where it matters most (CASCADE frequencies and rare events) while maintaining broader spectral awareness with reduced resolution.

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

## Propagation Augmentation

CASCADE leverages real-world propagation characteristics extracted from FT8/WSPR recordings to create realistic training conditions. Rather than relying on simplified channel models, the system applies authentic propagation effects captured during 18 months of global monitoring.

### Core Concept

The propagation augmentation system uses the trained Propagation VAE to transfer real channel characteristics from FT8/WSPR signals to synthetic CASCADE transmissions. This ensures the model trains on signals that have experienced genuine ionospheric propagation, multipath, fading, and noise conditions.

The key insight is that FT8 and WSPR signals, with their known transmitted content, allow precise extraction of channel transfer functions. These extracted characteristics can then be applied to CASCADE signals during training, providing authentic propagation without requiring CASCADE deployments during the data collection phase.

### Channel Transfer Learning Algorithm

The algorithm operates in two main phases:

**Phase 1: Channel Characteristic Extraction**

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

- **Multipath Profile**: By analyzing the impulse response, the system identifies multiple propagation paths with their respective delays and amplitudes
- **Fading Characteristics**: Comparing amplitude variations reveals fading depth, rate, and correlation time
- **Doppler Effects**: Frequency analysis shows both shifts (from ionospheric motion) and spreading (from multipath)
- **Phase Evolution**: Tracking phase changes across symbols reveals path length variations

**Phase 2: Application to CASCADE Signals**

During training, these extracted characteristics augment synthetic CASCADE signals:

```python
def apply_propagation_to_cascade(cascade_signal, propagation_embedding):
    """
    Apply real propagation characteristics to synthetic CASCADE signal
    """
    # Decode embedding to channel parameters
    channel = propagation_vae_decoder(propagation_embedding)

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

### Training Integration

The propagation embeddings are naturally [correlated with noise embeddings](data_pipeline.md#correlation-preservation) from the same recording, ensuring CASCADE never trains on impossible combinations like Arctic noise with tropical propagation or storm conditions with calm channel characteristics. See [Data Pipeline: Phase 4](data_pipeline.md#phase-4-embedding-generation-months-19-20) for the complete embedding generation workflow.

## Embedding Analytics and Storage

The embeddings form a rich analytical space that reveals propagation patterns, guides training strategies, and enables scientific discovery.

### Selective Dual Storage Strategy

Rather than choosing between database and filesystem storage, CASCADE uses selective dual storage that maximizes analytical value while minimizing costs. This approach stores ALL embeddings in files (cheap, complete) while selectively storing high-value embeddings in the database (expensive, queryable).

**Cost Analysis for 50 million embeddings (~48GB)**:
- PostgreSQL only: $6.10/month (full SQL, slow for bulk I/O)
- HDF5 only: $0.78/month (fast bulk I/O, no queries)
- Selective dual: $2-3/month (best of both worlds)

```python
class SelectiveDualStorage:
    """
    Complete archive in files, high-value subset in database
    """

    def is_high_value(self, metadata):
        """Criteria for database storage"""
        # Always store rare/interesting events
        return any([
            metadata.get('k_index', 0) >= 5,           # Geomagnetic storms
            metadata.get('x_ray_class') in ['M', 'X'], # Solar flares
            metadata.get('snr', 0) < -20,              # Extreme weak signals
            metadata.get('propagation_mode') in ['Aurora', 'TEP', 'MS'],
            metadata.get('is_anomaly', False),
            random.random() < 0.01  # 1% random sample
        ])
```

### Analytical Discoveries

**Propagation Mode Clustering**: Embeddings naturally cluster into distinct propagation modes (F2, Sporadic-E, Aurora). Some clusters potentially represent unknown propagation phenomena not yet documented in amateur radio literature.

**Anomaly Detection**: Outliers in embedding space identify rare propagation events deserving 10x training weight:
- Trans-equatorial propagation during "closed" conditions
- Anomalous daytime DX on supposedly "nighttime" bands
- Potentially unknown propagation modes

**Temporal Evolution**: Tracking embedding changes over time reveals propagation transitions (sunrise/sunset, mode changes) that challenge decoders and require focused training.

**Geographic Patterns**: Path-dependent clustering shows distinct embedding signatures for:
- Polar paths (auroral propagation)
- Oceanic paths (long-distance transoceanic)
- Mountain diffraction paths (terrain effects)

### Curriculum Learning from Clustering

Embeddings enable progressive training from typical to challenging conditions:

```python
def curriculum_from_embeddings(embeddings):
    """Build training curriculum from embedding clusters"""
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

## Future Enhancements

Several improvements could further enhance the embedding models:

**Multi-Band Awareness**: Currently separate models per band could be unified with band-aware architectures.

**Temporal Dynamics**: Incorporating time series of embeddings to capture channel evolution.

**Adversarial Robustness**: Training with adversarial examples to improve embedding stability.

**Few-Shot Learning**: Adapting quickly to new propagation modes not seen during training.

The embedding models form a critical bridge between raw radio recordings and efficient CASCADE training, transforming overwhelming amounts of IQ data into compact, meaningful representations that preserve the full richness of HF radio propagation.

## See Also

- **[Data Pipeline](data_pipeline.md)** - How embeddings are generated from collected IQ data
- **[Training README](README.md)** - Overall training strategy using embeddings
- **[Privacy Protection](../privacy.md)** - Anonymization of callsigns while preserving geographic data
- **[Long-Term Roadmap](long_term_roadmap.md)** - Future embedding model improvements and reprocessing plans