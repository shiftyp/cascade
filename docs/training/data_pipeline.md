# CASCADE Data Pipeline

The CASCADE training data pipeline transforms 18 months of raw HF radio recordings into efficient, diversity-preserving embeddings that enable realistic model training. This document describes the complete flow from data collection through embedding generation to final model training.

## Overview

The pipeline follows a carefully designed sequence that preserves the natural correlations in radio propagation while maximizing the diversity of conditions the model experiences. Rather than synthesizing artificial combinations, we leverage the vast scale of real-world data to capture authentic radio environments.

### Key Principles

1. **Collect Everything First**: The full 18-month collection period captures complete seasonal and solar cycles before any processing begins
2. **Preserve Natural Correlations**: Noise and propagation characteristics from the same time and place remain paired
3. **Bias Toward Rarity**: Uncommon events like geomagnetic storms receive disproportionate representation in training
4. **Efficient Representation**: Channel embeddings compress terabytes of IQ data into gigabytes of trainable features

## Timeline and Phases

### Phase 1: Data Collection (Months 1-18)

During this phase, the system continuously records IQ data from KiwiSDR receivers worldwide, accumulating 35-75TB of raw recordings. No processing occurs during collection - the focus is purely on gathering diverse propagation conditions.

The collection targets six HF amateur bands with 12 kHz bandwidth recordings centered on frequencies that capture both FT8/WSPR signals and adjacent quiet spectrum. Each recording includes full metadata about solar conditions, geomagnetic activity, time, and location.

Critical aspects of the collection phase:
- Continuous baseline collection using 6-10 simultaneous SDRs
- Dynamic scaling to 20-50 SDRs during geomagnetic storms or rare propagation events
- Geographic and temporal diversity through intelligent SDR rotation
- Complete archival of all data regardless of quality

### Phase 2: Dataset Curation (Month 19, Week 1)

After collection completes, the system analyzes the full dataset to create a carefully curated subset for training. This subset, approximately 3-5TB in size, oversamples rare events while maintaining baseline coverage of common conditions.

The curation process computes a rarity score for each recording based on multiple factors:

```python
def calculate_rarity_score(recording):
    """
    Higher scores indicate rarer, more valuable training examples
    """
    score = 1.0

    # Geomagnetic storms are exponentially rarer at higher K indices
    k_index = recording.k_index
    k_frequency_distribution = {
        0: 0.30,  # 30% of time
        1: 0.25,  # 25% of time
        2: 0.20,  # 20% of time
        3: 0.10,  # 10% of time
        4: 0.08,  # 8% of time
        5: 0.04,  # 4% of time
        6: 0.02,  # 2% of time
        7: 0.008, # 0.8% of time
        8: 0.002, # 0.2% of time
        9: 0.0001 # 0.01% of time
    }
    score *= (1.0 / k_frequency_distribution.get(k_index, 0.0001))

    # Solar flares weighted by class
    if recording.xray_class == 'X':
        score *= 500  # Extremely rare
    elif recording.xray_class == 'M':
        score *= 50   # Rare
    elif recording.xray_class == 'C':
        score *= 5    # Uncommon

    # Propagation modes have different natural frequencies
    if recording.propagation_mode == 'Aurora':
        score *= 100  # High-latitude, K-dependent
    elif recording.propagation_mode == 'TEP':
        score *= 50   # Trans-equatorial, rare
    elif recording.propagation_mode == 'Es':
        score *= 10   # Sporadic, seasonal

    return min(score, 10000)  # Cap to prevent single events dominating
```

The curation strategy ensures that every K=9 storm, every X-class flare, and every auroral opening is included in the training set, while common F2 propagation is sampled at just 1-10% to prevent the model from overfitting to the most frequent conditions.

### Phase 3: Embedding Model Training (Month 19, Weeks 2-3)

The system trains two specialized Variational Autoencoder (VAE) models on the curated dataset - one for noise characterization and one for propagation effects. Training on the diverse 5TB subset takes approximately 3-5 days on a single GPU.

The key insight is that the embedding models need diversity, not volume. By training on a carefully selected subset that includes all rare events, the models learn to represent the full space of possible radio conditions without processing all 75TB of mostly redundant data.

### Phase 4: Embedding Generation (Months 19-20)

Once trained, the VAE models process the curated 5TB dataset to generate embeddings. This transforms the raw IQ data into compact vector representations:
- Noise embeddings: 64-dimensional vectors capturing QRN/QRM characteristics
- Propagation embeddings: 128-dimensional vectors encoding channel distortion

Processing 5TB of IQ data yields approximately 25GB of embeddings - a 200x compression ratio while preserving the essential information needed for training.

### Phase 5: CASCADE Model Training (Months 20-21)

Finally, CASCADE trains using the generated embeddings. The model learns by:
1. Generating fresh CASCADE signals from training data
2. Fetching naturally correlated noise and propagation embedding pairs
3. Applying both embeddings to create realistic receive conditions
4. Training the decoder to recover the original data

This approach ensures CASCADE only experiences realistic combinations that actually occur in nature.

## Diversity-Biased Sampling

The sampling strategy deliberately overrepresents rare conditions to ensure the model learns to handle edge cases. The system groups recordings into rarity tiers and samples them at different rates:

**Ultra-Rare Events (Score > 1000)**
These include K≥8 storms, X-class flares, and exotic propagation modes. The system includes 100% of these recordings, typically totaling less than 100GB but representing the most challenging conditions CASCADE will face.

**Very Rare Events (Score 100-1000)**
This tier includes moderate storms (K=6-7), M-class flares, and auroral propagation. The system samples 80% of these recordings, ensuring strong representation of unusual but important conditions.

**Rare Events (Score 10-100)**
These include minor storms, sporadic-E openings, and extreme SNR conditions. The system samples 30% of these recordings, balancing diversity with data volume.

**Uncommon Events (Score 3-10)**
This tier includes enhanced propagation, moderate solar activity, and grayline effects. The system samples 10% to maintain coverage without domination.

**Common Conditions (Score 1-3)**
Regular F2 propagation and typical noise conditions. The system samples only 1% to provide baseline coverage while preventing the model from overfitting to the most frequent cases.

## Correlation Preservation

A critical aspect of the pipeline is maintaining the natural correlations between noise and propagation. When solar flux is high, both the noise floor and propagation characteristics change together. When a geomagnetic storm occurs, it simultaneously affects both atmospheric noise and signal propagation.

The system preserves these correlations by extracting both noise and propagation embeddings from the same recordings:

```python
def extract_correlated_embeddings(recording):
    """
    Extract paired embeddings that preserve natural correlations
    """
    # Context shared by both embeddings
    context = {
        'timestamp': recording.timestamp,
        'k_index': recording.k_index,
        'solar_flux': recording.solar_flux,
        'location': recording.grid_square
    }

    # Process the same time window
    embedding_pairs = []
    for window in recording.iterate_windows():
        ft8_signals = detect_ft8(window)
        quiet_periods = find_quiet_periods(window)

        if ft8_signals and quiet_periods:
            # Both found in same window - perfect correlation
            prop_embedding = propagation_vae.encode(ft8_signals[0])
            noise_embedding = noise_vae.encode(quiet_periods[0])

            embedding_pairs.append({
                'propagation': prop_embedding,
                'noise': noise_embedding,
                'context': context,  # Same conditions for both
                'correlation': 'simultaneous'
            })

    return embedding_pairs
```

This approach ensures CASCADE never trains on impossible combinations like Arctic noise with tropical propagation or storm conditions with calm channel characteristics.

## Storage Requirements

The pipeline manages data at multiple stages with different storage needs:

**Raw IQ Archive**: 35-75TB on cold storage/NAS. This complete dataset is preserved for potential reprocessing with improved algorithms but is not accessed during routine training.

**Curated Training Set**: 3-5TB on fast NVMe storage. This diversity-biased subset is actively used during embedding generation and can be quickly accessed for experimentation.

**Embedding Database**: 25GB in RAM/SSD. The compact embeddings enable rapid random access during CASCADE training, with KD-tree indexes for efficient similarity searches.

## Quality Validation

Throughout the pipeline, multiple validation steps ensure data quality:

1. **Collection Validation**: GPS lock verification, sample rate consistency checks, and SDR metadata capture
2. **Curation Validation**: Coverage metrics ensuring all condition categories are represented
3. **Embedding Validation**: Reconstruction quality tests, clustering analysis, and physics consistency checks
4. **Training Validation**: Natural correlation preservation, condition coverage tracking, and convergence monitoring

## Benefits of This Approach

This pipeline design provides several key advantages:

**Authentic Training Data**: By using only real recordings with preserved correlations, CASCADE learns genuine radio physics rather than artificial combinations.

**Efficient Processing**: The 5TB curated subset can be processed quickly while still capturing the full diversity of 18 months of propagation.

**Rare Event Coverage**: Every geomagnetic storm, solar flare, and unusual propagation mode is guaranteed representation in training.

**Scalable Architecture**: The embedding approach enables rapid experimentation - new CASCADE variants can be trained in days rather than months.

**Scientific Value**: The curated dataset with rare event bias becomes a valuable research resource beyond just CASCADE training.

The entire pipeline transforms an overwhelming 75TB of raw recordings into an elegant training system that captures the full complexity of HF radio propagation in just 25GB of embeddings, while ensuring CASCADE learns to handle both common daily operations and the rare but critical edge cases it will encounter in deployment.