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
  * Future Phase 2 collection planned for solar maximum (~2028-2030)

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

**Phase 2 (2028-2030): Balance Correction**
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