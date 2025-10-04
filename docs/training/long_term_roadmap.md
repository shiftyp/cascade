# CASCADE Long-Term Roadmap (2025-2040)

This document outlines CASCADE's 15-year strategic vision for continuous data collection, model improvement, and adaptation to changing propagation conditions. It addresses multi-phase solar cycle collection, climate change impacts on the ionosphere, technology evolution, and CASCADE's role as a long-term scientific resource.

## Table of Contents

1. [Multi-Phase Data Collection Strategy](#multi-phase-data-collection-strategy-2025-2040)
2. [Climate Change and Ionospheric Adaptation](#climate-change-and-ionospheric-adaptation)
3. [Technology Evolution Planning](#technology-evolution-planning)
4. [Continuous Model Improvement](#continuous-model-improvement-cycles)
5. [Scientific Contributions](#scientific-contributions-and-data-archival)
6. [Risk Mitigation](#risk-mitigation-and-contingencies)

---

## Multi-Phase Data Collection Strategy (2025-2040)

CASCADE implements a **continuous collection strategy** spanning multiple solar cycles, seasonal variations, and long-term climate trends. The collection never truly "ends" - instead, it transitions from intensive bootstrap phases to sustained baseline monitoring.

### Phase 1: Solar Minimum Bootstrap (2025-2026)

**Timeline**: 18 months during Solar Cycle 25 minimum
**Status**: Current/Active

**Objectives**:
- Collect baseline 200,000-300,000 hours from 800-1100 SDRs (see [Data Pipeline](data_pipeline.md))
- [Aggressive rare event capture](data_pipeline.md#aggressive-boost-implementation) (K≥3 storms, C+ flares at 100%)
- Establish initial CASCADE [training dataset](embedding_models.md)
- Accept intentional bias toward rare events

**Collection Parameters**:
- 50-100 SDRs concurrent (baseline)
- 200+ SDRs during any activity (K≥3)
- 300-400 hours/day collection rate
- 6 HF bands, 12 kHz bandwidth per band
- FLAC compression, 35-75TB total storage

**Deliverable**: Initial CASCADE v1.0 model trained on solar minimum + rare event diversity

### Phase 2: Solar Maximum Balance (2028-2030)

**Timeline**: 18-24 months during Solar Cycle 25 maximum
**Status**: Planned

**Objectives**:
- Balance the solar minimum bias with high-activity data
- Capture frequent K≥5 storms and M/X-class flares at natural rates
- Create representative full-cycle training dataset
- Validate Phase 1 model performance under active conditions

**Collection Parameters**:
- 50-100 SDRs concurrent (baseline remains same)
- Natural event rates: K≥5 storms occur 5-10x more frequently
- Standard sampling: K≥5 at 30%, M-class at 50%, X-class at 100%
- Additional 150k-200k hours (cumulative 350k-450k total)
- Combined storage: 75-100TB total

**Deliverable**: CASCADE v2.0 model balanced across full solar cycle

### Phase 3: Next Solar Minimum Validation (2033-2035)

**Timeline**: 12 months during Solar Cycle 26 minimum
**Status**: Future Planning

**Objectives**:
- Decadal validation of Phase 1 model
- Detect long-term ionospheric trends (climate-driven)
- Compare 2025 vs 2033 solar minimum characteristics
- Validate model generalization across solar cycles

**Collection Parameters**:
- Reduced intensity: 20-30 SDRs concurrent
- Focus on trend detection, not volume
- 50k-75k hours (decadal comparison baseline)
- Incremental 20-25TB storage

**Deliverable**: Decadal trend analysis, climate adaptation updates

### Continuous Baseline Monitoring (2026+)

**Timeline**: Indefinite, low-intensity
**Status**: Begins after Phase 1 completes

**Objectives**:
- Detect long-term propagation trends
- Monitor climate change effects on HF propagation
- Provide continuous training data for edge cases
- Maintain dataset freshness

**Collection Parameters**:
- 5-10 SDRs concurrent (permanent baseline)
- ~50 hours/day collection rate
- Prioritize rare events and geographic gaps
- ~18k hours/year, ~1.5TB/year storage
- 10-year accumulation: ~180k hours, ~15TB

**Deliverable**: Continuous model updates, trend detection, scientific dataset

### Collection Timeline Visualization

```
2025 ████████████████████ Phase 1: Solar Min Bootstrap (intensive)
2026 ████████████████████
2027 ▓▓▓▓▓▓▓▓▓▓          Continuous Baseline (low intensity)
2028 ████████████████████ Phase 2: Solar Max Balance (intensive)
2029 ████████████████████
2030 ████████████████
2031 ▓▓▓▓▓▓▓▓▓▓          Continuous Baseline
2032 ▓▓▓▓▓▓▓▓▓▓
2033 ████████████████     Phase 3: Next Solar Min Validation (medium)
2034 ████████████████
2035 ▓▓▓▓▓▓▓▓▓▓          Continuous Baseline
2036-2040 ▓▓▓▓▓▓▓▓▓▓     Continuous + Solar Cycle 26 Max (2037-2038)

Legend: ████ Intensive collection, ▓▓▓▓ Baseline monitoring
```

### Total 15-Year Projection

- **Total Hours**: 600k-800k hours (2025-2040)
- **Total Storage**: 150-200TB raw IQ data
- **Total Embeddings**: 50-75GB compressed
- **Collection Cost**: ~$500/month SDR coordination + storage
- **Scientific Value**: Unprecedented multi-decadal HF propagation dataset

---

## Climate Change and Ionospheric Adaptation

Climate change is altering the ionosphere in measurable ways that will affect HF propagation. CASCADE must adapt to these evolving conditions through continuous monitoring and periodic model updates.

### Expected Ionospheric Changes (2025-2040)

#### Upper Atmosphere Cooling

**Mechanism**: Increased CO2 in the upper atmosphere radiates heat to space, cooling the thermosphere and ionosphere.

**Measured Trends** (from scientific literature):
- Thermosphere density decreasing ~2-3% per decade
- F2 layer peak height increasing ~1-2 km per decade
- Critical frequencies (foF2) decreasing ~0.5-1% per decade
- E-layer less affected (different chemistry)

**Impact on HF Propagation**:
```python
expected_changes_by_2040 = {
    'f2_layer_height': +15_to_30_km,      # Higher reflection point
    'critical_frequency': -5_to_10_percent,  # Lower MUF
    'skip_distance': +50_to_100_km,       # Longer minimum distance
    'absorption': 'slightly_reduced',      # Thinner D-layer
    'propagation_modes': {
        'f2_single_hop': 'reduced_ceiling',  # Lower MUF limits
        'multi_hop': 'slightly_improved',     # Less absorption
        'nvis': 'minimal_change',             # E-layer stable
        'sporadic_e': 'unclear_trend'         # Complex dependencies
    }
}
```

**CASCADE Adaptation Strategy**:
1. **Continuous Baseline Monitoring**: 5-10 SDRs track MUF trends year-over-year
2. **Tri-annual Retraining**: Every 3 years, retrain with last 5 years of data
3. **Trend Injection**: Explicitly model MUF trends in training
4. **Performance Tracking**: Monitor if CASCADE efficiency degrades with time

#### Tropospheric Warming

**Mechanism**: Lower atmosphere warming affects D-layer absorption and ground conductivity.

**Expected Effects**:
- **D-layer absorption**: Slightly increased (more water vapor, higher collision rates)
- **Thunderstorm frequency**: Increased in tropics (more QRN)
- **Ground conductivity**: Changed soil moisture affects ground wave and NVIS

**CASCADE Adaptation**:
- Regional telemetry tracks changing noise characteristics
- Noise embeddings automatically capture evolving QRN patterns
- Model retraining incorporates new baseline noise floors

### Detection Strategy

CASCADE will serve dual purpose: communication system AND ionospheric monitoring network.

```python
class IonosphericTrendDetector:
    """
    Detect long-term propagation changes from continuous collection
    """

    def __init__(self):
        self.baseline_year = 2025
        self.annual_metrics = []

    def compute_annual_ionospheric_metrics(self, year_data):
        """
        Extract ionospheric indicators from year of FT8/WSPR data
        """
        metrics = {
            'year': year_data.year,

            # MUF trends (from highest successful frequencies)
            'muf_daytime_median': calculate_muf(year_data, time='day'),
            'muf_nighttime_median': calculate_muf(year_data, time='night'),
            'muf_trend_mhz_per_year': None,  # Computed from multi-year

            # Skip distance trends
            'min_skip_distance_km': calculate_min_skip(year_data),
            'skip_trend_km_per_year': None,  # Computed from multi-year

            # Absorption trends (D-layer)
            'd_layer_absorption_db': estimate_d_layer_loss(year_data),
            'absorption_trend_db_per_year': None,  # Computed from multi-year

            # Sporadic-E frequency (climate sensitive?)
            'es_occurrence_rate': count_sporadic_e(year_data) / total_hours,
            'es_trend_per_year': None,  # Computed from multi-year

            # Quality metrics
            'sample_count': len(year_data),
            'geographic_coverage': calculate_diversity(year_data),
            'seasonal_balance': check_seasonal_balance(year_data)
        }

        self.annual_metrics.append(metrics)
        return metrics

    def detect_decadal_trends(self):
        """
        Identify long-term changes requiring model adaptation
        """
        if len(self.annual_metrics) < 5:
            return None  # Need 5+ years for reliable trends

        trends = {
            'muf_trend': linear_regression(
                [m['year'] for m in self.annual_metrics],
                [m['muf_daytime_median'] for m in self.annual_metrics]
            ),
            'skip_trend': linear_regression(
                [m['year'] for m in self.annual_metrics],
                [m['min_skip_distance_km'] for m in self.annual_metrics]
            ),
            'absorption_trend': linear_regression(
                [m['year'] for m in self.annual_metrics],
                [m['d_layer_absorption_db'] for m in self.annual_metrics]
            )
        }

        # Flag significant trends
        if abs(trends['muf_trend'].slope) > 0.1:  # >0.1 MHz/year
            alert_ionospheric_change('MUF trending significantly')

        if abs(trends['skip_trend'].slope) > 10:  # >10 km/year
            alert_ionospheric_change('Skip distance changing')

        return trends
```

### Model Update Triggers

CASCADE will retrain when significant trends are detected:

```python
def should_retrain_for_climate(trends):
    """
    Decide if climate-driven changes warrant model update
    """
    # Trigger retraining if:
    triggers = {
        'muf_drift': abs(trends['muf_trend'].slope) > 0.15,  # >1.5 MHz/decade
        'skip_drift': abs(trends['skip_trend'].slope) > 20,   # >200 km/decade
        'absorption_drift': abs(trends['absorption_trend'].slope) > 0.5,  # >5 dB/decade
        'performance_degradation': cascade_efficiency_trend < -0.02  # >2%/year drop
    }

    if any(triggers.values()):
        return {
            'should_retrain': True,
            'urgency': 'high' if triggers['performance_degradation'] else 'medium',
            'focus_areas': [k for k, v in triggers.items() if v],
            'estimated_improvement': estimate_retraining_benefit(trends)
        }

    return {'should_retrain': False}
```

---

## Technology Evolution Planning

Amateur radio infrastructure and digital modes evolve over time. CASCADE must adapt to changing technology landscapes while maintaining backward compatibility.

### SDR Infrastructure Evolution

#### Current State (2025-2026)
- **KiwiSDR**: 600-800 units globally, open-source, well-maintained
- **WebSDR**: 200-300 institutional installations
- **Status**: Healthy ecosystem, active development

#### Risk Scenarios

**Scenario 1: KiwiSDR Decline (Probability: Medium)**
- **Cause**: Hardware obsolescence, developer retirement, cost increases
- **Timeline**: 2028-2032
- **Impact**: Primary data source reduces from 600-800 to 200-400 SDRs

**Mitigation**:
1. **WebSDR prioritization**: Shift to institutional SDRs (more stable, well-funded)
2. **User-contributed telemetry**: Deployed CASCADE becomes primary data source
3. **Low-cost SDR deployments**: 50-100 Raspberry Pi + RTL-SDR units in critical gaps (~$10k investment)
4. **Archive leverage**: Extensive Phase 1-2 archives reduce dependency on ongoing collection

**Scenario 2: New SDR Technology Emergence (Probability: High)**
- **Cause**: Better, cheaper SDR hardware (e.g., successor to KiwiSDR)
- **Timeline**: 2027-2030
- **Impact**: Opportunity to expand coverage

**Adaptation**:
1. **Collector abstraction**: Design collectors to support new SDR types
2. **Rapid integration**: Add new SDR type support within 3-6 months
3. **Hybrid operation**: Run old and new SDR types simultaneously
4. **Performance comparison**: Validate new SDRs against known baselines

### Digital Mode Evolution

#### Current Dominant Modes (2025)
- **FT8**: ~80% of HF digital activity, highly optimized for weak signals
- **WSPR**: ~15% of activity, excellent for propagation studies
- **Other modes**: ~5% (FT4, JS8Call, etc.)

#### Expected Evolution (2025-2035)

**Near-term (2025-2028)**: FT8/WSPR dominance continues
- Incremental FT8 improvements (FT8B, FT8C potential updates)
- WSPR remains stable for propagation beacon use
- CASCADE launch may inspire new adaptive modes

**Mid-term (2028-2032)**: Potential new mode emergence
- **Risk**: Entirely new digital mode gains adoption
- **Impact**: FT8/WSPR data becomes less representative
- **Mitigation**:
  1. Add decoders for new modes (3-6 month development)
  2. Leverage telemetry as primary data source (mode-agnostic)
  3. Maintain baseline FT8/WSPR monitoring for trends

**Long-term (2032-2040)**: Multi-mode diversity
- Multiple digital modes coexist on HF bands
- CASCADE operates alongside FT8, WSPR, and future modes
- Richer propagation dataset from mode diversity

### Decoder Flexibility Strategy

```python
class AdaptiveDecoderPipeline:
    """
    Support multiple digital modes for propagation extraction
    """

    def __init__(self):
        self.decoders = {
            'ft8': FT8Decoder(),
            'wspr': WSPRDecoder(),
            'ft4': FT4Decoder(),  # Ready for future
            'js8call': JS8Decoder(),  # Ready for future
            # Add new modes as they emerge
        }

    def extract_propagation_from_any_mode(self, iq_sample, band):
        """
        Attempt all decoders, use whichever succeeds
        """
        for mode_name, decoder in self.decoders.items():
            try:
                decoded = decoder.decode(iq_sample)
                if decoded.valid:
                    # Extract propagation regardless of mode
                    return {
                        'mode': mode_name,
                        'propagation': extract_channel_from_known_signal(
                            iq_sample,
                            decoded.reconstructed_signal
                        ),
                        'path_geometry': {
                            'tx_grid': decoded.tx_grid,
                            'rx_grid': decoded.rx_grid,
                            'distance_km': calculate_distance(
                                decoded.tx_grid,
                                decoded.rx_grid
                            )
                        }
                    }
            except DecoderError:
                continue

        return None  # No mode decoded successfully
```

### Transition to Telemetry-Primary (2027+)

**The Strategic Shift**: After CASCADE deploys (late 2026), **[telemetry](continuous_improvement.md) becomes the primary data source**, reducing dependency on external SDR networks.

```python
class DataSourceTransition:
    """
    Gradual shift from SDR collection to telemetry-driven
    """

    def __init__(self):
        self.phases = {
            '2025-2026': {
                'sdr_collection': '100%',
                'telemetry': '0%',
                'strategy': 'Pure SDR - pre-deployment'
            },
            '2027': {
                'sdr_collection': '80%',
                'telemetry': '20%',
                'strategy': 'Early deployment telemetry supplementation'
            },
            '2028-2030': {
                'sdr_collection': '60%',
                'telemetry': '40%',
                'strategy': 'Balanced - Phase 2 solar max + growing telemetry'
            },
            '2031-2035': {
                'sdr_collection': '30%',
                'telemetry': '70%',
                'strategy': 'Telemetry-primary with SDR baseline validation'
            },
            '2036-2040': {
                'sdr_collection': '10%',
                'telemetry': '90%',
                'strategy': 'Telemetry-driven with minimal SDR validation'
            }
        }

    def get_data_mix(self, year):
        """
        Determine optimal data source mix for given year
        """
        # SDR collection always maintains small baseline for:
        # 1. Independent validation of telemetry
        # 2. Bias detection in user deployments
        # 3. Geographic gap coverage
        # 4. Scientific research (SDR data is "ground truth")

        return self.phases.get(str(year), self.phases['2036-2040'])
```

**Benefits of Telemetry Transition**:
1. **Independence**: Not dependent on amateur radio infrastructure
2. **Global coverage**: Wherever CASCADE deploys, data flows
3. **Cost reduction**: No SDR coordination costs after 2035
4. **Freshness**: Real-time data from actual CASCADE usage
5. **Scale**: Thousands of deployments >> hundreds of SDRs

---

## Continuous Model Improvement Cycles

CASCADE employs **nested update cycles** at different time scales, from rapid telemetry updates to decadal retraining.

### Update Cycle Hierarchy

```
┌─ Monthly (Hotfix) ──────────────────────────────────────┐
│  - Critical bug fixes from telemetry                     │
│  - Emergency propagation condition updates               │
│  - Deployment: Immediate if >10% improvement            │
└──────────────────────────────────────────────────────────┘

┌─ Semi-Annual (Telemetry Refinement) ────────────────────┐
│  - Aggregate 6 months of telemetry                       │
│  - Focus on geographic gaps and edge cases               │
│  - Deployment: If >5% improvement                       │
│  - Timeline: January and July each year                  │
└──────────────────────────────────────────────────────────┘

┌─ Bi-Annual (Major Feature Update) ──────────────────────┐
│  - Incorporate new algorithms and expert improvements    │
│  - Add support for new amateur radio modes               │
│  - Deployment: Version increment (v1.X → v1.Y)          │
│  - Timeline: Every 2 years starting 2027                 │
└──────────────────────────────────────────────────────────┘

┌─ 5-Year (Solar Cycle Phase) ────────────────────────────┐
│  - Solar min (2025) → rising (2027) → max (2029)       │
│  - Major retraining with full cycle data                 │
│  - Deployment: Major version (v1.X → v2.0)              │
│  - Timeline: 2030, 2035, 2040                            │
└──────────────────────────────────────────────────────────┘

┌─ 10-Year (Decadal Climate) ─────────────────────────────┐
│  - Incorporate climate-driven ionospheric trends         │
│  - Compare 2025 vs 2035 solar minimum conditions         │
│  - Validate long-term model generalization               │
│  - Deployment: Major architecture update (v2.X → v3.0)  │
│  - Timeline: 2035, 2045                                  │
└──────────────────────────────────────────────────────────┘
```

### Retraining Decision Matrix

```python
def determine_retraining_schedule():
    """
    Multi-criteria decision for model updates
    """

    triggers = {
        'emergency': {
            'conditions': ['critical_bug', 'security_issue', 'regulatory_change'],
            'timeline': 'immediate',
            'data_requirement': 'minimal',
            'threshold': 'any_improvement'
        },

        'telemetry_driven': {
            'conditions': ['6_months_telemetry', 'geographic_gap', 'new_edge_case'],
            'timeline': 'semi_annual',
            'data_requirement': '50k-100k telemetry samples',
            'threshold': '>5% improvement'
        },

        'solar_cycle_phase': {
            'conditions': ['solar_min_to_max_transition', 'max_to_min_transition'],
            'timeline': '5_year',
            'data_requirement': 'full_solar_phase_data',
            'threshold': '>10% improvement'
        },

        'climate_adaptation': {
            'conditions': ['ionospheric_trend_detected', 'performance_degradation'],
            'timeline': '10_year',
            'data_requirement': 'decadal_comparison_data',
            'threshold': 'stop_degradation'
        },

        'architectural_upgrade': {
            'conditions': ['new_expert_network', 'algorithm_breakthrough'],
            'timeline': 'opportunistic',
            'data_requirement': 'full_historical_archive',
            'threshold': '>15% improvement'
        }
    }

    return triggers
```

### Data Accumulation Strategy

As data accumulates over 15 years, retraining windows expand:

| Timeline | Available Data | Retraining Approach | Focus |
|----------|---------------|---------------------|-------|
| 2026 | 250k hours (solar min) | Initial model | Bootstrap |
| 2028-2030 | 400k hours (+solar max) | Full cycle balance | Phase 2 |
| 2033-2035 | 500k hours (+next min) | Decadal comparison | Trends |
| 2036-2040 | 700k hours (+continuous) | Multi-cycle optimization | Maturity |

**Archive Growth**: The 40-50TB initial archive grows to 150-200TB by 2040, but most retraining uses curated subsets (3-5TB per phase) rather than full archives.

---

## Scientific Contributions and Data Archival

Beyond CASCADE's operational purpose, the multi-decadal dataset becomes an invaluable scientific resource for ionospheric physics and amateur radio propagation research.

### Scientific Research Applications

#### Ionospheric Climate Research

The CASCADE dataset provides unprecedented long-term HF propagation measurements:

**Unique Capabilities**:
- **15-year continuous baseline**: Longest continuous amateur band monitoring dataset
- **Geographic diversity**: Global coverage including underserved regions
- **Multi-band coherence**: Simultaneous 6-band measurements for MUF determination
- **High temporal resolution**: Minute-by-minute propagation tracking
- **Correlation with space weather**: Every recording tagged with K-index, solar flux, X-ray class

**Research Questions Enabled**:
1. How is CO2-driven ionospheric cooling affecting HF propagation? (2025-2040 comparison)
2. Are Sporadic-E occurrence rates changing with climate? (requires 10+ year baseline)
3. How does the ionosphere respond differently to solar cycle 25 vs 26? (decadal comparison)
4. Can we detect long-term trends in D-layer absorption? (climate sensitivity)
5. How do QBO transitions affect equatorial propagation? (requires multiple cycles)

#### Amateur Radio Propagation Science

**Applications**:
- **Propagation prediction improvements**: Better models for VOACAP, ASAPS, etc.
- **Antenna modeling validation**: Real-world ground conductivity and terrain effects
- **Contest planning**: Historical data for optimal band/time selection
- **Emergency communications**: Reliability statistics for disaster scenarios

### Data Archival Strategy

The CASCADE data archive becomes a **permanent scientific resource**, maintained beyond operational needs:

```python
class ScientificDataArchive:
    """
    Long-term archival for scientific research (beyond CASCADE operations)
    """

    def __init__(self):
        self.archive_tiers = {
            'hot_archive': {
                'data': 'Last 2 years of raw IQ + embeddings',
                'storage': 'NVMe SSD for fast access',
                'purpose': 'Active CASCADE training and research',
                'size': '~15TB',
                'retention': 'permanent'
            },

            'warm_archive': {
                'data': 'Years 3-10 raw IQ + embeddings',
                'storage': 'S3 standard tier',
                'purpose': 'Trend analysis, model validation',
                'size': '~80TB',
                'retention': 'permanent'
            },

            'cold_archive': {
                'data': 'Full 15-year raw IQ collection',
                'storage': 'Glacier/deep archive',
                'purpose': 'Scientific research, reprocessing with future algorithms',
                'size': '~150-200TB',
                'retention': 'permanent',
                'note': 'Can reprocess with improved embedding models'
            },

            'metadata_archive': {
                'data': 'All FT8/WSPR decodes, space weather, station fingerprints',
                'storage': 'PostgreSQL with annual exports',
                'purpose': 'Research queries, statistical analysis',
                'size': '~500GB',
                'retention': 'permanent'
            }
        }

    def archival_cost_projection(self):
        """
        15-year archival cost estimate
        """
        costs = {
            'hot_archive': 15_000 * 0.023 * 12 * 15,    # $62k over 15 years
            'warm_archive': 80_000 * 0.0125 * 12 * 15,  # $180k
            'cold_archive': 175_000 * 0.00099 * 12 * 15, # $31k
            'metadata': 500 * 0.115 * 12 * 15,           # $10k
            'total_15_year': 283_000  # $283k for 15-year archive
        }

        # Average: ~$19k/year for permanent scientific dataset
        return costs
```

### Open Data Commitment

**Phased Open Release Strategy**:

```python
open_data_timeline = {
    '2027': {
        'release': '2025-2026 solar minimum metadata',
        'content': 'FT8/WSPR decodes, space weather, anonymized stats',
        'format': 'PostgreSQL dump, CSV exports',
        'size': '~50GB',
        'purpose': 'Enable independent propagation research'
    },

    '2031': {
        'release': 'Full Phase 1-2 curated embeddings',
        'content': '3-5TB curated subset embeddings + metadata',
        'format': 'HDF5 + documentation',
        'size': '~20GB embeddings',
        'purpose': 'ML research, alternative training approaches'
    },

    '2035': {
        'release': 'Decadal comparison dataset (2025 vs 2035)',
        'content': 'Matched recordings for climate trend analysis',
        'format': 'Raw IQ samples from same SDRs/times 10 years apart',
        'size': '~5TB',
        'purpose': 'Ionospheric climate research'
    },

    '2040': {
        'release': 'Full 15-year archive (research access)',
        'content': 'Complete 150-200TB raw IQ archive',
        'format': 'FLAC on Glacier, request-based access',
        'size': '150-200TB',
        'purpose': 'Long-term ionospheric research, reprocessing'
    }
}
```

**Privacy Protection**: All releases maintain callsign anonymization and location generalization while preserving scientific utility.

---

## Risk Mitigation and Contingencies

Long-term projects face numerous risks. CASCADE includes explicit contingency plans for foreseeable challenges.

### Infrastructure Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **KiwiSDR network decline** | Medium | High | Shift to WebSDR + telemetry, deploy low-cost SDRs |
| **Storage cost increase** | Medium | Medium | Aggressive compression, tiered archival, grant funding |
| **Compute cost increase** | Low | Medium | Model efficiency improvements, quantization |
| **S3 service changes** | Low | High | Multi-cloud strategy (S3 + Wasabi + Backblaze) |

### Scientific Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Ionospheric changes invalidate model** | Medium | High | Continuous retraining, trend incorporation |
| **New propagation modes emerge** | Low | Medium | Anomaly detection, research collaboration |
| **Climate changes exceed models** | Medium | Medium | Decadal retraining, physics-informed constraints |
| **Solar cycle 26 behaves unusually** | Medium | Medium | Phase 3 validation (2033-2035) |

### Regulatory Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **HF band reallocation** | Low | Critical | Multi-band design, VHF/UHF capability |
| **Increased SDR restrictions** | Medium | Medium | Coordinate with institutions, research agreements |
| **Privacy regulation changes** | Medium | Medium | Already privacy-first design, adaptable |
| **Spectrum fees for ML training** | Low | High | Research exemption, institutional partnerships |

### Funding and Sustainability

**Cost Projections (2025-2040)** (Updated with Tigris zero egress):
```python
fifteen_year_costs = {
    'data_collection': {
        'sdr_coordination': 500 * 12 * 15,      # $90k (manual + tooling)
        'storage': 19_000 * 15,                  # $285k (tiered archival, Tigris + archives)
        # bandwidth: $0 (Tigris has zero egress fees!)
        'subtotal': 375_000
    },

    'compute': {
        'embedding_training': 5_000 * 3,         # $15k (3 retraining cycles)
        'cascade_retraining': 10_000 * 5,        # $50k (5 major versions)
        'telemetry_processing': 100 * 12 * 15,  # $18k (continuous)
        'subtotal': 83_000
    },

    'personnel': {
        'maintenance_engineer': 80_000 * 15,     # $1.2M (part-time, 0.5 FTE)
        'research_scientist': 100_000 * 5,       # $500k (intermittent, analysis)
        'subtotal': 1_700_000
    },

    'total_15_year': 2_158_000,  # ~$2.16M over 15 years ($36k savings from zero egress!)
    'annual_average': 144_000     # ~$144k/year
}
```

**Funding Strategy**:
1. **2025-2027**: Initial grant funding ($300k) + volunteer labor
2. **2027-2030**: Amateur radio foundation grants ($200k)
3. **2030-2035**: Scientific research grants (ionospheric climate, $500k)
4. **2035-2040**: University partnership (institutionalize archive, $500k)
5. **Community**: Amateur radio community contributions (equipment, hosting, coordination)

### Sustainability Model

```python
def long_term_sustainability_strategy():
    """
    Ensure CASCADE survives beyond initial funding
    """

    sustainability_pillars = {
        'technical': {
            'telemetry_transition': 'Reduce SDR dependency over time',
            'automation': 'Minimize manual intervention',
            'efficiency': 'Reduce compute/storage costs annually',
            'open_source': 'Community can maintain core infrastructure'
        },

        'financial': {
            'grant_rotation': 'New grants every 3-5 years from different sources',
            'institutional_partnership': 'University hosting after 2030',
            'community_support': 'Amateur radio club contributions',
            'commercial_spinoffs': 'Potential licensing for maritime/aviation'
        },

        'organizational': {
            'distributed_governance': 'Not dependent on single individual',
            'documentation': 'Comprehensive, enables continuity',
            'skill_transfer': 'Train next generation of maintainers',
            'community_engagement': 'Active amateur radio involvement'
        },

        'scientific': {
            'research_value': 'Dataset justifies long-term preservation',
            'publications': 'Academic recognition sustains interest',
            'collaboration': 'Partner with ionospheric research groups',
            'education': 'Use in university propagation courses'
        }
    }

    return sustainability_pillars
```

---

## Long-Term Vision: CASCADE as Living Infrastructure

### 2025-2030: Bootstrap and Deployment

- **Data Collection**: Intensive SDR collection (Phase 1-2)
- **Model Maturity**: v1.0 → v2.0 (solar cycle complete)
- **User Adoption**: 1,000-10,000 deployments globally
- **Geographic Coverage**: 85-95% globally via telemetry
- **Scientific Output**: 5-10 publications on propagation and ML

### 2030-2035: Maturity and Scientific Integration

- **Data Collection**: Telemetry-primary (70%), SDR validation (30%)
- **Model Stability**: v2.X incremental updates
- **User Adoption**: 10,000-50,000 deployments
- **Research Integration**: Dataset used in 20+ ionospheric studies
- **Climate Detection**: First decadal trends identified (2025-2035)

### 2035-2040: Long-Term Scientific Resource

- **Data Collection**: Telemetry-dominant (90%), minimal SDR baseline
- **Model Evolution**: v3.0 with climate adaptation
- **User Adoption**: Mature amateur radio infrastructure
- **Scientific Legacy**: Recognized ionospheric monitoring network
- **Archive**: 150-200TB permanent research dataset

### Beyond 2040: Generational Handoff

CASCADE transitions from **project** to **infrastructure**:

- **Technical**: Fully automated, minimal maintenance
- **Financial**: Institutionally supported, sustainable funding
- **Scientific**: Core resource for HF propagation research
- **Community**: Amateur radio standard, self-maintaining
- **Climate**: 20+ year baseline for ionospheric trend detection

---

## Summary

CASCADE's long-term vision extends far beyond the initial 18-month data collection. The roadmap encompasses:

**Multi-Decadal Data Collection**:
- Intensive phases during key solar cycle periods (2025-2030)
- Continuous baseline monitoring (2026+)
- Decadal validation (2033-2035)
- Permanent low-level collection for trend detection

**Climate Adaptation**:
- Continuous monitoring of ionospheric cooling effects
- Tri-annual retraining to capture evolving conditions
- Decadal comparison for trend validation
- Model updates triggered by performance degradation

**Technology Evolution**:
- Transition from SDR-primary to telemetry-primary (2025→2040)
- Decoder flexibility for new amateur radio modes
- Infrastructure independence through distributed telemetry
- Architectural upgrades as ML advances

**Scientific Legacy**:
- 150-200TB permanent ionospheric propagation archive
- Multi-decadal baseline for climate change detection
- Open data releases for research community
- CASCADE as long-term monitoring infrastructure

**Sustainability**:
- Rotating grant funding strategy
- Institutional partnerships for long-term hosting
- Community engagement for distributed maintenance
- Scientific value justifies preservation

The roadmap ensures CASCADE evolves from a bootstrap project into a **permanent, self-sustaining component of amateur radio infrastructure** that continuously adapts to changing propagation conditions while serving as a critical dataset for ionospheric climate research.

## See Also

- **[Data Pipeline](data_pipeline.md)** - Phase 1 collection strategy and solar minimum boost
- **[Continuous Improvement](continuous_improvement.md)** - Telemetry and federated learning for ongoing updates
- **[Embedding Models](embedding_models.md)** - How embeddings enable efficient multi-phase retraining
- **[Training README](README.md)** - Near-term training strategy (2025-2027)