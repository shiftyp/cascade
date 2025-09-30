# CASCADE Data Pipeline

The CASCADE training data pipeline transforms 18 months of raw HF radio recordings into efficient, diversity-preserving embeddings that enable realistic model training. This document describes the complete flow from data collection through embedding generation to final model training.

## Table of Contents

1. [Overview](#overview)
2. [Timeline and Phases](#timeline-and-phases)
3. [Natural Cycle Coverage Strategy](#natural-cycle-coverage-strategy)
4. [Solar Minimum Collection Strategy](#solar-minimum-collection-strategy-boost-vs-balance)
5. [Diversity-Biased Sampling](#diversity-biased-sampling)
6. [Correlation Preservation](#correlation-preservation)
7. [Storage Requirements](#storage-requirements)
8. [Quality Validation](#quality-validation)
9. [Geographic Bias Considerations](#geographic-bias-considerations)
10. [Benefits of This Approach](#benefits-of-this-approach)
11. [Collection Implementation Strategy](#collection-implementation-strategy)

## Executive Summary

The CASCADE data pipeline addresses a fundamental bootstrapping problem: training an adaptive HF radio system without existing CASCADE deployments. The solution is to learn from the amateur radio community's existing digital modes (FT8/WSPR) that already operate under diverse HF conditions.

**Core Strategy**: Collect massive quantities of real-world HF data, bias heavily toward rare events, compress via embeddings, train CASCADE on authentic propagation conditions.

**Collection Scale**:
- **Duration**: 18 months continuous (2025-2026, solar minimum)
- **Sources**: 600-800 KiwiSDRs + 200-300 WebSDRs globally distributed
- **Concurrent SDRs**: 50-100 baseline, scaling to 200+ during events
- **Target Volume**: 200,000-250,000 hours total (300-400 hours/day)
- **Raw Storage**: 40-50TB FLAC-compressed IQ data
- **Bands**: 6 HF amateur bands (80m, 40m, 20m, 15m, 10m, 6m)

**Aggressive Solar Minimum Strategy**:
- **Philosophy**: Hoard all rare events during limited 18-month window
- **K≥3 storms**: 100% capture rate (vs normal K≥5, 30% rate)
- **C-class flares**: 100% capture rate (vs normal ignore)
- **Boost multipliers**: 5x-10x for any activity during minimum
- **Rationale**: Solar maximum won't arrive until 2028-2030; maximize opportunity-limited window

**Data Curation**:
- Full 40-50TB archive preserved for reprocessing
- 3-5TB curated subset for training (diversity-biased)
- Ultra-rare events (K≥8, X-class): 100% inclusion
- Common conditions (K≤1, quiet sun): 1% sampling
- Result: 15-25GB embeddings (200x compression)

**Key Innovation**: Natural correlation preservation ensures CASCADE never trains on impossible combinations (e.g., Arctic noise + tropical propagation). All noise and propagation embeddings extracted from same recordings maintain authentic physics.

## Overview

The pipeline follows a carefully designed sequence that preserves the natural correlations in radio propagation while maximizing the diversity of conditions the model experiences. Rather than synthesizing artificial combinations, we leverage the vast scale of real-world data to capture authentic radio environments.

### Key Principles

1. **Collect Everything First**: The full 18-month collection period captures complete seasonal and solar cycles before any processing begins
2. **[Preserve Natural Correlations](#correlation-preservation)**: Noise and propagation characteristics from the same time and place remain paired
3. **[Bias Toward Rarity](#diversity-biased-sampling)**: Uncommon events like geomagnetic storms receive disproportionate representation in training
4. **Efficient Representation**: [Channel embeddings](embedding_models.md) compress terabytes of IQ data into gigabytes of trainable features

## Timeline and Phases

### Phase 1: Data Collection (Months 1-18)

During this phase, the system continuously records IQ data from both KiwiSDR and WebSDR receivers worldwide, accumulating 40-50TB of raw recordings while respecting usage constraints. The hybrid collection strategy leverages KiwiSDR's 30-90 minute daily limits per IP and WebSDR's institutional policies to maximize coverage. No processing occurs during collection - the focus is purely on gathering diverse propagation conditions.

The collection targets six HF amateur bands with 12 kHz bandwidth IQ recordings at 12 kHz sample rate, 16-bit depth, centered on strategic frequencies that capture both FT8/WSPR signals and adjacent quiet spectrum. Each recording is FLAC compressed for 45-55% size reduction while maintaining lossless quality.

Critical aspects of the collection phase:
- Continuous baseline collection using 50-100 concurrent SDRs rotating through 30-90 minute sessions
- Dynamic scaling to 100-200 SDRs during solar events (K≥3 storms, C-class flares and above)
- Target 300-400 hours/day collection rate to achieve 200,000-250,000 total hours over 18 months
- Geographic and temporal diversity through intelligent rotation across 800-1100 public receivers (600-800 KiwiSDRs + 200-300 WebSDRs)
- Complete archival of all data regardless of quality with correlation IDs linking noise and propagation

### Phase 2: Dataset Curation (Month 19, Week 1)

After collection completes, the system analyzes the full dataset to create a carefully curated subset for training. This subset, approximately 3-5TB in size (from 40-50TB raw collection), oversamples rare events while maintaining baseline coverage of common conditions.

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

The system trains two specialized [Variational Autoencoder (VAE) models](embedding_models.md#architecture-overview) on the curated dataset - one for noise characterization and one for propagation effects. Training on the diverse 3-5TB subset takes approximately 2-3 days on a single GPU.

The key insight is that the embedding models need diversity, not volume. By training on a carefully selected subset that includes all rare events, the models learn to represent the full space of possible radio conditions without processing all 40-50TB of mostly redundant data.

### Phase 4: Embedding Generation (Months 19-20)

Once trained, the [VAE models](embedding_models.md#quiet-channel-vae-architecture) process the curated 3-5TB dataset to generate embeddings. This is when multi-scale processing occurs:

- **Frequency Channelization**: The 12 kHz recordings are divided into overlapping 250-500 Hz channels
- **Temporal Windowing**: Sliding windows of 0.5-5 seconds are extracted to match CASCADE's adaptive fragments
- **Multi-Scale Generation**: Each frequency-time tile produces its own embedding

This transforms the raw IQ data into compact vector representations:
- Noise embeddings: 64-dimensional vectors per frequency-time tile
- Propagation embeddings: 128-dimensional vectors encoding channel distortion

Processing 3-5TB of IQ data yields approximately 15-25GB of embeddings - a 200x compression ratio while preserving the essential information needed for training.

### Phase 5: CASCADE Model Training (Months 20-21)

Finally, CASCADE trains using the generated embeddings. The model learns by:
1. Generating fresh CASCADE signals from training data
2. Fetching naturally correlated noise and propagation embedding pairs
3. Applying both embeddings to create realistic receive conditions
4. Training the decoder to recover the original data

This approach ensures CASCADE only experiences realistic combinations that actually occur in nature.

## Natural Cycle Coverage Strategy

The 18-month collection period spans multiple natural cycles that affect HF propagation. CASCADE implements systematic coverage strategies to ensure comprehensive temporal diversity despite the limited timeframe relative to longer cycles.

### Captured Natural Cycles

**Diurnal Cycles (24 hours)** ✅ Complete Coverage
- Full 24-hour UTC coverage for each geographic region
- Gray-line enhancement periods systematically captured
- Day/night terminator crossing events prioritized

**Lunar Cycles (27 days)** ✅ Complete Coverage
- 18 months captures ~20 complete lunar cycles
- New moon and full moon periods tracked for EME effects
- Lunar tidal influences on ionosphere documented

**Solar Rotation (27 days)** ✅ Complete Coverage
- 18 months captures ~24 solar rotations
- Coronal hole and active region rotation effects included
- Solar wind stream interaction regions captured

**Seasonal Cycles (365 days)** ✅ Systematic Coverage
- 1.5 complete seasonal cycles captured
- Systematic seasonal balancing ensures 25% ±5% per season
- Winter collection weighted 20% higher for rarer conditions
- Equinoctial enhancement periods (Mar 15-Apr 15, Sep 15-Oct 15) prioritized

### Partially Captured Cycles

**QBO (Quasi-Biennial Oscillation, 28 months)** ⚠️ Single Phase
- 18-month collection captures one QBO phase transition
- Enhanced equatorial propagation monitoring during transitions
- QBO index and phase tracked for correlation analysis

**Solar Cycle (11 years)** ⚠️ Solar Minimum Only - Aggressive Boost Strategy
- Collection occurs during Solar Cycle 25 minimum (2025-2026)
- Implements aggressive rare event maximization strategy:
  * K≥3 storm threshold with 100% capture rate
  * All C-class flares and above captured (100% rate)
  * 5x-10x rarity score multipliers for any activity
  * Opportunity-cost driven: maximize limited window value
  * Future [Phase 2 collection](long_term_roadmap.md#phase-2-solar-maximum-balance-2028-2030) planned for solar maximum (~2028-2030)

### Cycle-Aware Collection Weighting

```python
def calculate_cycle_aware_rarity_score(recording):
    """
    Enhanced rarity scoring that accounts for natural cycle phases
    """
    base_score = calculate_base_rarity_score(recording)

    # Solar minimum: AGGRESSIVE rare event boost strategy
    if recording.solar_cycle_phase == 'MINIMUM':
        # Aggressive multipliers - maximize diversity capture
        activity_multipliers = {
            3: 50,    # K=3: 50x vs normal 8x (6x increase)
            4: 100,   # K=4: 100x vs normal 25x (4x increase)
            5: 500,   # K≥5: 500x vs normal 50x (10x increase)
        }

        if recording.k_index >= 3:
            multiplier = activity_multipliers.get(recording.k_index, 500)
            base_score *= multiplier

        # Aggressive flare boosting - 100% capture strategy
        flare_multipliers = {
            'C': 25,   # NEW: Include C-class (was ignored)
            'M': 200,  # 4x increase from normal 50x
            'X': 2000  # 4x increase from normal 500x
        }

        if recording.xray_class in flare_multipliers:
            base_score *= flare_multipliers[recording.xray_class]

    # Seasonal weighting
    seasonal_factors = {
        'WINTER': 1.2,    # Rarer conditions, enhanced weighting
        'SPRING': 1.1,    # Equinoctial enhancement
        'SUMMER': 0.9,    # Common E-layer conditions
        'AUTUMN': 1.1     # Equinoctial enhancement
    }
    base_score *= seasonal_factors[recording.season]

    # Equinoctial period boost
    if recording.equinoctial_enhancement:
        base_score *= 1.3

    # QBO enhancement for equatorial paths
    if (recording.is_equatorial_path and
        recording.qbo_phase == 'TRANSITION'):
        base_score *= 1.4

    # Lunar enhancement for VHF considerations
    if recording.lunar_phase in [0.0, 0.5]:  # New or full moon
        base_score *= 1.1

    return min(base_score, 10000)  # Cap maximum score

def enhanced_rarity_scoring_example():
    """
    Example of cycle-aware rarity scoring in action
    """
    # Example 1: Storm during solar minimum winter
    storm_recording = {
        'k_index': 5,
        'xray_class': 'M',
        'solar_cycle_phase': 'MINIMUM',
        'season': 'WINTER',
        'equinoctial_enhancement': False,
        'qbo_phase': 'EASTERLY',
        'lunar_phase': 0.0,  # New moon
        'propagation_mode': 'Aurora',
        'snr': -20,
        'seasonal_balance_factor': 1.2
    }

    # Base K=5 score: 1/0.04 = 25
    # M-class during minimum: * 75 = 1,875
    # Solar minimum activity boost: * 1.5 = 2,812
    # Winter enhancement: * 1.3 = 3,656
    # New moon boost: * 1.1 = 4,022
    # Aurora in winter: * 150 = 603,300
    # Seasonal balance: * 1.2 = 723,960
    # Final score: 723,960 (very high priority)

    score1 = calculate_cycle_aware_rarity_score(storm_recording)
    print(f"Storm during solar minimum winter: {score1}")

    # Example 2: Common summer conditions
    common_recording = {
        'k_index': 1,
        'xray_class': None,
        'solar_cycle_phase': 'MINIMUM',
        'season': 'SUMMER',
        'equinoctial_enhancement': False,
        'qbo_phase': 'EASTERLY',
        'lunar_phase': 0.3,  # Waxing crescent
        'propagation_mode': 'F2',
        'snr': 15,
        'seasonal_balance_factor': 0.9
    }

    # Base K=1 score: 1/0.25 = 4
    # No flare: * 1 = 4
    # Summer reduction: * 0.9 = 3.6
    # F2 propagation: * 1 = 3.6
    # Seasonal balance: * 0.9 = 3.24
    # Final score: 3.24 (low priority)

    score2 = calculate_cycle_aware_rarity_score(common_recording)
    print(f"Common summer conditions: {score2}")

    # Ratio shows 220,000x preference for rare events
    print(f"Rare/common ratio: {score1/score2:.0f}x")
```

### Seasonal Balance Enforcement

```python
class SeasonalBalanceManager:
    """
    Ensures systematic 25% ±5% coverage per season
    """
    def __init__(self):
        self.target_hours_per_season = 62500  # 250k hours / 4 seasons
        self.tolerance = 0.05  # ±5%
        self.seasonal_quotas = {
            'WINTER': {'target': 62500, 'collected': 0, 'weight': 1.2},
            'SPRING': {'target': 62500, 'collected': 0, 'weight': 1.1},
            'SUMMER': {'target': 62500, 'collected': 0, 'weight': 0.9},
            'AUTUMN': {'target': 62500, 'collected': 0, 'weight': 1.1}
        }

    def get_seasonal_priority(self, current_season):
        """
        Return collection priority for current season
        """
        quota = self.seasonal_quotas[current_season]
        completion_rate = quota['collected'] / quota['target']

        # Increase priority if behind target
        if completion_rate < 0.8:
            return quota['weight'] * 1.5
        elif completion_rate < 0.9:
            return quota['weight'] * 1.2
        else:
            return quota['weight']

    def should_throttle_season(self, season):
        """
        Check if season is over-represented and should be throttled
        """
        quota = self.seasonal_quotas[season]
        completion_rate = quota['collected'] / quota['target']
        return completion_rate > 1.05  # Over 105% of target
```

### Long-Term Cycle Metadata

All recordings include comprehensive cycle context:

```python
cycle_metadata = {
    'solar_cycle': {
        'number': 25,
        'phase': 'MINIMUM',
        'days_since_minimum': 365,
        'expected_duration_years': 11
    },
    'qbo': {
        'index': -15.2,
        'phase': 'EASTERLY',
        'months_since_transition': 8
    },
    'seasonal': {
        'season': 'WINTER',
        'balance_factor': 1.2,
        'equinoctial_enhancement': False,
        'days_to_equinox': 45
    },
    'lunar': {
        'phase': 0.3,  # Waxing crescent
        'age_days': 7,
        'next_extreme': 'full_moon'
    }
}
```

This comprehensive cycle tracking enables:
1. **Training Data Understanding**: Know exactly which cycle phases are represented
2. **Bias Correction**: Compensate for solar minimum during training
3. **Scientific Analysis**: Correlate CASCADE performance with natural cycles
4. **Future Planning**: Design follow-up data collection for missing cycle phases

## Solar Minimum Collection Strategy: Boost vs Balance

### The Strategic Choice

CASCADE faces a fundamental decision during the 18-month solar minimum collection period (2025-2026): attempt to "balance" the dataset with mostly quiet conditions, or aggressively boost rare events to maximize diversity. We choose the **aggressive boost strategy** for compelling reasons:

#### Opportunity Cost Analysis

**Solar Cycle Reality:**
- Solar Cycle 25 minimum: 2025-2026 (our collection window)
- Solar Cycle 25 maximum: ~2028-2030 (3+ years away)
- K≥5 storms: ~1% frequency during minimum vs ~5% during maximum
- M/X-class flares: 10-20x rarer during minimum

**The Choice:**
- **Balance Strategy**: Collect mostly quiet conditions, wait years for activity
- **Boost Strategy**: Hoard every rare event we can get NOW

### Aggressive Boost Implementation

```python
def solar_minimum_opportunity_maximization():
    """
    Maximize rare event capture during limited opportunity window
    """
    strategy = {
        'philosophy': 'Capture everything rare - we only get one chance',
        'thresholds': {
            'storm_min': 3,        # K≥3 vs normal K≥5
            'flare_min': 'C',      # All C+ vs normal M+ only
            'capture_rate': 1.0    # 100% vs normal 10-30%
        },
        'multipliers': {
            'K=3_storms': 50,      # vs normal 8 (6x boost)
            'C_class_flares': 25,  # vs normal 0 (NEW)
            'M_class_flares': 200, # vs normal 50 (4x boost)
            'X_class_flares': 2000 # vs normal 500 (4x boost)
        },
        'rationale': {
            'limited_window': 'Only 18 months during solar minimum',
            'next_opportunity': '2028-2030 for high activity',
            'training_value': 'Rare events more valuable than common',
            'diversity_priority': 'Model needs edge case experience'
        }
    }
    return strategy
```

### Two-Phase Collection Strategy

**Phase 1 (2025-2026): Diversity Maximization**
- Aggressive boost of ALL activity during solar minimum
- 100% capture rate for K≥3 storms and C+ flares
- Accept dataset bias toward rare events
- Focus: Capture what we CAN get during minimum

**[Phase 2](long_term_roadmap.md#phase-2-solar-maximum-balance-2028-2030) (2028-2030): Balance Correction**
- Collect during solar maximum
- Balance dataset with high-activity periods
- Create representative long-term training set
- Focus: Complete the dataset with missing cycle phases

### Training Bias Management

```python
def handle_solar_minimum_bias():
    """
    Account for aggressive rare event boosting in training
    """
    # True natural frequencies
    natural_distribution = {
        'quiet_sun': 0.70,    # Actually most common
        'moderate': 0.25,     # Somewhat common
        'active': 0.05        # Actually rare
    }

    # Our biased collection (due to boost strategy)
    collected_distribution = {
        'quiet_sun': 0.30,    # Underrepresented
        'moderate': 0.40,     # Better represented
        'active': 0.30        # Overrepresented due to boost
    }

    # Training compensation weights
    compensation_weights = {
        'quiet_sun': 0.70 / 0.30,  # 2.33x upweight
        'moderate': 0.25 / 0.40,   # 0.625x downweight
        'active': 0.05 / 0.30      # 0.167x downweight
    }

    return compensation_weights
```

### Scientific Justification

1. **Maximum Value Extraction**: Extract every bit of diversity from limited window
2. **Training Efficacy**: Models benefit more from rare events than common ones
3. **Future Flexibility**: Easier to add common data later than recreate rare events
4. **Honest Bias Handling**: Explicit documentation and compensation for bias

This strategy acknowledges that we have one shot at solar minimum conditions and maximizes the scientific and training value of our opportunity-limited collection window.

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
def extract_correlated_embeddings(recording_10min):
    """
    Extract temporally-aligned embeddings preserving natural correlations
    """
    correlated_pairs = []

    # Detect all FT8 signals with precise timing
    ft8_signals = detect_ft8_with_timing(recording_10min)

    for ft8 in ft8_signals:
        # Process multiple temporal scales matching CASCADE fragments
        for duration in [0.5, 1.0, 2.0, 5.0]:
            stride = duration / 2  # 50% overlap

            for window_start in np.arange(0, ft8.duration - duration, stride):
                # Absolute timestamp in recording
                abs_time = ft8.start_time + window_start

                # Extract FT8 propagation slice
                ft8_slice = ft8.signal[window_start:window_start + duration]
                prop_embedding = propagation_vae.encode(ft8_slice)

                # Extract QRN from EXACT same time window
                qrn_slice = recording_10min[abs_time:abs_time + duration]

                # Multi-channel noise extraction (250 Hz channels)
                noise_embeddings = []
                for freq_channel in range(250, 12000, 250):
                    channel_qrn = bandpass_filter(qrn_slice, freq_channel, width=250)
                    noise_emb = noise_vae.encode(channel_qrn)
                    noise_embeddings.append({
                        'center_hz': freq_channel,
                        'embedding': noise_emb
                    })

                # Create unique correlation ID
                correlation_id = f"{recording_10min.session_id}_{abs_time}_{duration}"

                correlated_pairs.append({
                    'correlation_id': correlation_id,
                    'prop_embedding': prop_embedding,
                    'noise_embeddings': noise_embeddings,
                    'absolute_time': abs_time,
                    'duration': duration,
                    'k_index': recording_10min.k_index,
                    'solar_flux': recording_10min.solar_flux
                })

    return correlated_pairs
```

This approach ensures CASCADE never trains on impossible combinations like Arctic noise with tropical propagation or storm conditions with calm channel characteristics.

## Storage Requirements

The pipeline manages data at multiple stages with different storage needs:

### Raw Data Storage

**Raw IQ Archive**: 40-50TB on Tigris S3 cold storage. This complete dataset (200,000-250,000 hours) is preserved for potential reprocessing with improved algorithms but is not accessed during routine training. FLAC compression achieves 45-55% size reduction.

**Curated Training Set**: 3-5TB on fast NVMe storage. This diversity-biased subset is actively used during embedding generation and can be quickly accessed for experimentation.

### Embedding Storage Options

The multi-scale embedding approach generates substantial data that requires careful storage planning:

**Full Resolution (Unoptimized)**:
- 48 frequency channels × 2,220 temporal windows = 106,560 QRN embeddings per recording
- 25 FT8 signals × 25 temporal windows = 625 propagation embeddings per recording
- Total: ~27.6 MB per 10-minute recording
- For training subset (150,000 recordings): **4.14 TB total**

**Optimized Storage**:
- Selective temporal sampling (450 windows instead of 2,220)
- Hierarchical frequency sampling (15 channels near CASCADE frequencies)
- Float16 compression for noise embeddings
- Total: ~1 MB per recording, **150 GB for training subset**

### Recommended Storage Architecture

For practical training, a hybrid approach works best:

```python
# Active training storage (local NVMe)
/training/
├── embeddings/          # 4.14 TB (full resolution)
│   ├── qrn/            # Organized by frequency/duration
│   │   ├── 250hz/
│   │   └── ...
│   └── ft8/            # Organized by path
└── models/             # 100 GB for checkpoints

# Long-term storage (S3/Wasabi)
/archive/
├── raw_flac/           # 3-5 TB curated subset
└── embeddings_compressed/  # 150 GB optimized version
```

### Training Infrastructure Costs

**Lambda Labs GPU Cloud**:
- Storage: $0.20/GB/month (expensive for large datasets)
- Better approach: Use instance-local NVMe during training

**Cost-Optimized Training Workflow**:
1. Store embeddings in S3/Wasabi: ~$25/month for 4TB
2. Spin up H100 instance with 4TB+ local NVMe: $2.49/hr
3. Transfer embeddings to local storage (one-time, ~1 hour)
4. Train for 2-3 weeks on local NVMe
5. Save checkpoints to persistent storage
6. Total cost for 3-week training: ~$1,250

**Local Infrastructure Alternative**:
- 8TB NVMe SSD: ~$600 one-time cost
- Sufficient for all embeddings plus working space
- No recurring storage costs

## Quality Validation

Throughout the pipeline, multiple validation steps ensure data quality:

1. **Collection Validation**: GPS lock verification, sample rate consistency checks, and SDR metadata capture
2. **Curation Validation**: Coverage metrics ensuring all condition categories are represented
3. **Embedding Validation**: Reconstruction quality tests, clustering analysis, and physics consistency checks
4. **Training Validation**: Natural correlation preservation, condition coverage tracking, and convergence monitoring

## Geographic Bias Considerations

The global distribution of SDRs introduces significant geographic bias that must be addressed:

### Coverage Distribution
- **Well-covered** (80% of SDRs): North America, Europe, Japan
- **Moderate** (15% of SDRs): Australia, parts of Asia
- **Severely underrepresented** (<5% of SDRs): Africa, South America, Middle East, Central Asia, Pacific Islands
- **Absent**: Antarctica, Arctic regions, most ocean areas

### Propagation Impact
This clustering means CASCADE will undertrain on:
- **Equatorial phenomena**: Trans-equatorial propagation (TEP), equatorial spread-F
- **Tropical QRN**: Unique thunderstorm noise patterns
- **Southern auroral**: Southern hemisphere auroral effects
- **Long ocean paths**: Pacific/Indian Ocean propagation
- **Desert propagation**: Temperature inversion effects

### Mitigation Strategy

CASCADE implements a comprehensive geographic diversity strategy (documented in tasks T083-T092) to ensure balanced global coverage despite inherent SDR distribution bias.

#### Phase 1: Initial Training (Months 1-21)

**Latitude Band Quota System**: The system divides the globe into four latitude bands with minimum collection quotas:

| Latitude Band | Coverage | Minimum Quota | Priority Weight |
|--------------|----------|---------------|-----------------|
| Arctic | >66.5°N | 20% | 1.5x |
| Temperate | 23.5-66.5° | 20% | 1.0x |
| Tropical | ±23.5° | 20% | 1.2x |
| Antarctic | <-66.5°S | 20% | 5.0x |

**Hemispheric Balance Requirements**: Target distribution ensures global coverage:
- **Northern Hemisphere**: 40% (reduced from natural ~75%)
- **Southern Hemisphere**: 40% (increased from natural ~20%)
- **Equatorial Band (±10°)**: 20% (critical for trans-equatorial paths)

**Under-represented Region Boosting**: Dynamic scoring adjustments based on SDR density:
- Antarctica: 5x collection weight
- Southern Ocean: 3x collection weight
- Remote Pacific: 2.5x collection weight
- Standard regions: 1x collection weight

**Ocean Path Requirements**: Minimum 30% of collected paths must cross ocean to capture long-distance propagation, salt water enhancement effects, and transoceanic aviation/maritime paths.

**Reciprocal Path Inference**: For extremely sparse regions, the system infers bidirectional paths with 0.5x confidence weighting compared to direct observations.

**Implementation Components**:
1. **10x upweighting** for any recordings from underrepresented regions
2. **Reciprocal path assumption**: EU→Africa implies similar Africa→EU propagation
3. **Partner deployments**: 10-20 low-cost SDRs in critical gaps ($2-3k investment)
4. **Synthetic augmentation**: Physics-based models for missing regions
5. **Explicit bias documentation**: Model limitations clearly stated

#### Phase 2: Telemetry-Based Gap Closure (Post-Deployment)

[Telemetry](continuous_improvement.md#privacy-preserving-telemetry) transforms geographic bias from a permanent limitation into a temporary bootstrap problem. Once CASCADE deploys globally, each system becomes a data contributor, creating a virtuous cycle where deployment in underserved regions directly improves performance for all users in that region.

**The Telemetry Virtuous Cycle:**
1. **Bootstrap Phase (Month 0-6)**: Early adopters in underserved regions experience 40-50% performance but contribute valuable telemetry
2. **Rapid Improvement (Month 6-12)**: First telemetry-trained models boost regional performance to 75-85%
3. **Maturity (Month 12-18)**: Comprehensive geographic coverage achieves 85-95% globally
4. **Continuous Refinement (Month 18+)**: Ongoing telemetry maintains and improves edge cases

**Telemetry Collection Architecture:**
```python
class GeographicTelemetry:
    """
    Privacy-preserving telemetry from CASCADE deployments worldwide
    """
    def __init__(self):
        self.telemetry_schema = {
            'grid_square': 'FK29',           # 70x35 mile area only
            'timestamp': 1234567890,          # UTC, rounded to hour
            'propagation_mode': 'F2',        # Detected mode
            'noise_characteristics': {
                'qrn_level': -95,             # dBm noise floor
                'qrn_type': 'tropical',       # Classification
                'stability': 0.85             # Temporal stability
            },
            'channel_conditions': {
                'multipath_spread': 2.3,     # ms
                'doppler_spread': 1.2,       # Hz
                'coherence_bandwidth': 850   # Hz
            },
            'performance_metrics': {
                'adaptation_time': 450,      # ms to converge
                'achieved_efficiency': 0.87, # Shannon efficiency
                'decode_success': True        # Binary success
            }
        }

    def aggregate_by_region(self, telemetry_batch):
        """
        Aggregate telemetry to preserve privacy (K-anonymity = 10)
        """
        # Never report individual stations
        # Require minimum 10 reports per grid square per day
        return aggregated_regional_statistics

def prioritize_telemetry_regions():
    """
    Focus telemetry collection on underrepresented areas
    """
    telemetry_weights = {
        'africa': 5.0,              # Highest priority
        'south_america': 4.0,       # High priority
        'pacific_islands': 5.0,     # Highest priority
        'middle_east': 3.5,         # High priority
        'polar_regions': 4.5,       # Very high priority
        'north_america': 0.5,       # Low priority (well covered)
        'europe': 0.4,              # Lowest priority
    }

    # Only collect detailed telemetry from high-priority regions
    # Reduces bandwidth and storage for well-covered areas
    return telemetry_weights
```

**Progressive Model Improvement Timeline:**
```python
def telemetry_driven_improvement_schedule():
    """
    Systematic improvement through telemetry feedback
    """
    improvement_timeline = {
        'month_0': {
            'deployment': 'Initial CASCADE release',
            'telemetry_points': 0,
            'geographic_gaps': ['africa', 'pacific', 'polar', 'middle_east'],
            'worst_performance': '20-40%'
        },
        'month_6': {
            'milestone': 'First telemetry update',
            'telemetry_points': 50000,
            'focus': ['africa', 'south_america'],
            'improvement': '40-50% → 75-85%',
            'technique': 'Direct telemetry training on gap regions'
        },
        'month_12': {
            'milestone': 'Seasonal variation captured',
            'telemetry_points': 200000,
            'focus': ['pacific', 'middle_east', 'polar'],
            'improvement': '50-60% → 80-90%',
            'technique': 'Multi-season aggregation'
        },
        'month_18': {
            'milestone': 'Global coverage achieved',
            'telemetry_points': 500000,
            'coverage': '85% of inhabited land areas',
            'worst_performance': '85% minimum globally',
            'technique': 'Comprehensive retraining with full telemetry'
        }
    }
    return improvement_timeline

def telemetry_contribution_incentives():
    """
    Why users in underserved regions WANT to contribute telemetry
    """
    incentive_structure = {
        'direct_benefit': 'Your telemetry directly improves YOUR performance',
        'regional_improvement': 'Helps all users in your region',
        'rapid_updates': 'See improvements within 6 months',
        'privacy_preserved': 'No personal data ever shared',
        'community_contribution': 'Part of global amateur radio improvement'
    }

    # Regions with worst initial performance have highest incentive
    regional_incentives = {
        'africa': 'Transform from 40% to 85% performance',
        'pacific': 'Enable reliable emergency communications',
        'polar': 'Support scientific expeditions',
        'south_america': 'Improve cross-continent paths'
    }

    return incentive_structure, regional_incentives
```

**Privacy-Preserving Features:**
- Grid squares only (70x35 mile areas)
- No callsigns or exact locations
- Minimum K=10 anonymity per region
- Temporal aggregation to hourly
- Differential privacy (ε=1.0) on all metrics
- Opt-in with clear data usage policy

### Diversity Metrics and Monitoring

The system tracks multiple metrics to ensure geographic balance:

1. **Simpson's Diversity Index**: Measures geographic distribution (target: >0.7)
2. **Hemispheric Balance Score**: Ratio between hemispheres (target: 0.8-1.2)
3. **Continental Coverage**: Number of continents represented (target: 7/7)
4. **Latitude Distribution**: Chi-square test for uniformity
5. **Ocean Path Percentage**: Proportion of ocean-crossing paths (target: >30%)

**Progressive Quota Relaxation**: As collection progresses, quotas gradually relax to allow flexibility:

| Collection Progress | Underrepresented Weight | Overrepresented Weight |
|-------------------|------------------------|----------------------|
| 0-30% | 3.0x | 0.5x |
| 30-70% | 2.0x | 0.7x |
| 70-100% | 1.5x | 0.9x |

This ensures strict diversity early while allowing optimization later.

### Expected Performance Progression by Region

| Region | SDR-Only | +Partner SDRs | +6mo Telemetry | +12mo Telemetry | +18mo Telemetry |
|--------|----------|---------------|----------------|-----------------|-----------------|
| **North America** | 95%+ | 95%+ | 95%+ | 95%+ | 95%+ |
| **Europe** | 95%+ | 95%+ | 95%+ | 95%+ | 95%+ |
| **Japan/East Asia** | 90-95% | 90-95% | 95%+ | 95%+ | 95%+ |
| **South America** | 60-70% | 75-80% | 80-85% | 85-90% | 90-95% |
| **Southeast Asia** | 60-70% | 70-75% | 75-85% | 85-90% | 90-95% |
| **Africa** | 40-50% | 65-75% | 75-85% | 80-90% | 85-95% |
| **Middle East** | 40-50% | 65-70% | 70-80% | 80-90% | 85-95% |
| **Pacific Islands** | 35-45% | 60-70% | 70-80% | 80-85% | 85-95% |
| **Polar Regions** | 20-30% | 50-60% | 65-75% | 75-85% | 75-85% |

### The Telemetry Feedback Loop

```
┌─────────────────────────────────────────────────────────────┐
│  Underserved Region (e.g., Africa, Pacific)                 │
│  Initial Performance: 40-50%                                │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Early Adopters Deploy CASCADE                              │
│  • Experience suboptimal but usable performance             │
│  • Contribute 100-500 telemetry points/day                  │
│  • Strong incentive to improve local performance            │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Telemetry Aggregation (Privacy-Preserved)                  │
│  • K=10 anonymity per grid square                          │
│  • Differential privacy ε=1.0                              │
│  • No PII or message content                               │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Model Retraining (Every 6 Months)                         │
│  • Heavy weighting on underserved regions (5-10x)          │
│  • Direct training on local propagation characteristics    │
│  • Seasonal and diurnal pattern learning                   │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Performance Improvement Deployed                           │
│  • Regional performance: 40% → 75% → 85% → 95%            │
│  • Better performance attracts more users                   │
│  • More users = more telemetry = faster improvement        │
└─────────────┴───────────────────────────────────────────────┘
              │
              └──── Positive Feedback Loop ─────┘
```

### Telemetry Value Proposition

**For Underserved Regions:**
- Transform from barely functional (40%) to excellent (85-95%) performance
- Direct correlation: your telemetry improves your region's performance
- See measurable improvements within 6 months
- Enable reliable emergency and scientific communications

**For the CASCADE Project:**
- Zero infrastructure cost - users ARE the infrastructure
- Self-correcting geographic bias over time
- Community-driven continuous improvement
- Achieves truly global coverage within 18 months

**For the Amateur Radio Community:**
- Preserves privacy while enabling collective improvement
- Demonstrates value of collaborative data sharing
- Creates more equitable global communications capability
- Supports emergency communications in remote regions

## Benefits of This Approach

This pipeline design provides several key advantages:

**Authentic Training Data**: By using only real recordings with preserved correlations, CASCADE learns genuine radio physics rather than artificial combinations.

**Efficient Processing**: The 3-5TB curated subset can be processed quickly while still capturing the full diversity of 18 months of propagation.

**Rare Event Coverage**: Every geomagnetic storm, solar flare, and unusual propagation mode is guaranteed representation in training.

**Scalable Architecture**: The embedding approach enables rapid experimentation - new CASCADE variants can be trained in days rather than months.

**Scientific Value**: The curated dataset with rare event bias becomes a valuable research resource beyond just CASCADE training.

## Collection Implementation Strategy

### SDR Usage Management

The system implements intelligent strategies to maximize collection within usage constraints:

**KiwiSDR Management (600-800 receivers globally)**
- 30-90 minute daily limits per IP address
- Graceful session management with automatic disconnection before limits
- Database tracking of per-SDR usage with 24-hour rolling windows
- Prioritization of rare events when approaching daily limits
- Geographic distribution enables 24-hour coverage across time zones

**WebSDR Integration (200-300 receivers globally)**
- Leverage institutional WebSDR's higher user capacity for baseline collection
- Typically allow longer sessions than KiwiSDRs (no strict daily limits)
- Coordinate with university operators for research access agreements
- Primary use for continuous baseline collection
- Often provide better bandwidth and stability than KiwiSDRs

### Recording Center Frequencies

Strategic 12 kHz windows centered to capture both signals and quiet zones:
- **80m**: 3576 kHz (WSPR 3568.6, FT8 3573, quiet 3576-3582)
- **40m**: 7080 kHz (FT8 7074, quiet digital 7078-7086)
- **20m**: 14080 kHz (FT8 14074, quiet 14078-14086)
- **15m**: 21080 kHz (FT8 21074, quiet 21078-21086)
- **10m**: 28080 kHz (FT8 28074, quiet 28078-28086)
- **6m**: 50303 kHz (WSPR 50293, quiet 50297-50309)

### SDR Rotation Strategy

With 800-1100 total SDRs available, the rotation strategy enables continuous collection:

**Available SDR Pool:**
- 600-800 KiwiSDRs (30-90 minute daily limits per IP)
- 200-300 WebSDRs (typically no strict daily limits, institutional policies vary)

**Rotation Mathematics:**
- 50 concurrent connections needed continuously
- Each KiwiSDR provides 0.5-1.5 hours/day (average 1 hour)
- Each WebSDR can provide 4-8 hours/day (varies by institution)
- Daily rotation: ~200-300 KiwiSDRs + ~20-30 WebSDRs = 50 continuous streams
- Geographic distribution ensures 24/7 coverage across time zones

**Optimization Strategy:**
- WebSDRs prioritized for baseline collection (longer sessions, better stability)
- KiwiSDRs used for geographic diversity and specific band coverage
- Automatic failover when SDRs go offline or reach limits
- Priority queue for rare propagation events

### Distributed Architecture

The system uses Fly.io for scalable distributed collection:
- **2-10 worker machines** auto-scaling based on propagation conditions
- **Redis/KeyDB** message queue for SDR assignment coordination
- **Distributed locks** preventing multiple workers claiming same SDR
- **Health monitoring** with 30-second heartbeats and graceful shutdown
- **Centralized scheduler** monitoring conditions and publishing assignments

### Quality Assurance

Comprehensive QA system for data validation:
- **1% random sampling** stored in hot object storage for manual review
- **Daily QA reports** with metadata and quality metrics
- **Waterfall visualization** dashboard for spectrogram analysis
- **IQ sample replay** with time/frequency selection and measurements
- **Searchable interface** with filters for date, band, SDR, propagation mode

The entire pipeline transforms 40-50TB of raw recordings (200,000-250,000 hours) into an elegant training system that captures the full complexity of HF radio propagation in just 15-25GB of [embeddings](embedding_models.md), while ensuring CASCADE learns to handle both common daily operations and the rare but critical edge cases it will encounter in deployment.

## See Also

- **[Embedding Models](embedding_models.md)** - VAE architectures that compress IQ data into embeddings
- **[Privacy Protection](../privacy.md)** - Callsign anonymization and grid square handling
- **[Continuous Improvement](continuous_improvement.md)** - Post-deployment telemetry and model updates
- **[Long-Term Roadmap](long_term_roadmap.md)** - Multi-phase collection strategy through 2040
- **[Training README](README.md)** - Overall training strategy and expert training stages