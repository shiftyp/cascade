# CASCADE Privacy Protection Approach

## Overview

CASCADE implements privacy protection for amateur radio data collection while preserving scientific value for HF propagation research. This document details our privacy-preserving approach, particularly regarding the handling of callsigns and grid squares.

## Core Privacy Principles

1. **Callsign Anonymization**: All amateur radio callsigns are anonymized using one-way cryptographic hashing
2. **Grid Square Preservation**: Maidenhead grid squares are preserved in cleartext for propagation analysis
3. **No Personal Data Storage**: No personally identifiable information is stored or transmitted
4. **Irreversible Transformation**: Anonymization uses salted SHA-256 hashing that cannot be reversed

## Why Grid Squares Are Preserved

Grid squares are preserved in cleartext for critical scientific reasons:

### 1. Propagation Distance Calculations
- Great circle distance calculations require precise geographic coordinates
- Grid squares provide ~5km resolution (6-character) or ~111km resolution (4-character)
- Essential for understanding HF propagation characteristics vs distance

### 2. Path Analysis
- Terminator crossings (day/night boundary) affect propagation
- Geographic features (ocean/land paths) influence signal behavior
- Magnetic latitude calculations for auroral effects

### 3. Scientific Value
- Grid squares alone do not identify individuals
- They represent geographic regions, not specific locations
- Multiple operators typically share the same grid square

## Implementation Details

### Callsign Hashing

```python
# One-way salted SHA-256 hash
hash = SHA256(callsign + salt)
anonymized = "ANON_" + hash[:8]  # First 8 chars of hash
```

- **Salt**: Cryptographically secure random salt prevents rainbow table attacks
- **Truncation**: 8-character prefix provides sufficient uniqueness while saving storage
- **Prefix**: "ANON_" clearly identifies anonymized data

### Grid Square Validation

```python
# Maidenhead grid square format
# Field: [A-R][A-R] (18x18 = 324 fields)
# Square: [0-9][0-9] (10x10 = 100 squares per field)
# Subsquare: [A-X][A-X] (24x24 = 576 subsquares per square)

Examples:
- FN42: 4-character grid (~111km resolution)
- FN42ab: 6-character grid (~5km resolution)
```

### Message Processing Example

**Original FT8 Message:**
```
W1ABC K2DEF FN42 -06
```

**After Anonymization:**
```
ANON_A3F2B91C ANON_7E4D8C2A FN42 -06
```

- Callsigns W1ABC and K2DEF are hashed
- Grid square FN42 is preserved
- Signal report -06 is preserved

## Privacy Analysis

### What Is Protected
- **Individual Callsigns**: Replaced with anonymous identifiers
- **Station Identity**: Cannot determine who transmitted signals
- **Activity Patterns**: Cannot track individual operator behavior

### What Is Preserved
- **Grid Squares**: Geographic regions for propagation analysis
- **Signal Reports**: SNR, power levels for propagation study
- **Timestamps**: UTC time for temporal analysis
- **Frequency**: Operating frequency for band analysis

### Privacy Risk Assessment

| Data Element | Risk Level | Mitigation |
|--------------|------------|------------|
| Callsign | High | One-way hash with salt |
| Grid Square | Low | Regional data only, multiple operators per grid |
| Frequency | None | No personal information |
| Signal Report | None | Technical data only |
| Timestamp | Low | UTC time, no correlation to individuals |
| **Neural State** | **Low-Medium** | **See Neural State Privacy below** |

## Neural State Privacy (Telemetry)

CASCADE [telemetry](training/continuous_improvement.md#telemetry-data-structure) captures internal neural network activations (3581-D vectors) from deployed radios. This section analyzes privacy implications of transmitting model internal state.

### What Neural Activations Contain

**Neural network activations are abstract feature representations**, not raw data:

```python
# Example: Shared encoder output (1024-D)
shared_features = [0.234, -0.891, 0.456, 0.123, ...]

# These values represent:
# - Learned features (not interpretable as raw IQ)
# - Statistical patterns (multi-path, fading rates)
# - Channel characteristics (Doppler, coherence bandwidth)
```

**What activations DO NOT contain:**
- Message content (already decoded/consumed before telemetry)
- Raw IQ samples (processed through multiple non-linear layers)
- Exact locations (only 4-char grid squares in metadata)
- Callsigns (not present in received signals, anonymized in metadata)

### Threat Model Analysis

**Can neural activations leak private information?**

**1. Message Content Reconstruction**
- **Risk**: Low
- **Analysis**: Activations are computed AFTER message decoding completes
- **Telemetry timing**: Captured when message already processed and cleared
- **Impossibility**: Multiple transmissions overwrite same activation space

**2. Location Inference**
- **Risk**: Low-Medium
- **Analysis**: Propagation features might correlate with specific paths
- **Mitigation**: Grid squares limited to 4-char (70×35 km area)
- **Additional protection**: Differential privacy noise on activations (optional)

**3. Station Re-identification**
- **Risk**: Low
- **Analysis**: Equipment signatures in Station Fingerprint encoder (16-D)
- **Mitigation**:
  - K-anonymity (minimum 10 samples per grid/band combination)
  - No temporal correlation (batched randomly)
  - Station fingerprints aggregated across multiple operators

**4. Behavioral Tracking**
- **Risk**: Low
- **Analysis**: Metadata includes usage patterns (queue depth, relay depth)
- **Mitigation**:
  - Timestamps rounded to hour
  - No session identifiers
  - No callsign linkage

### Privacy Protections

CASCADE implements multiple layers of protection for neural state telemetry:

**1. Differential Privacy on Activations (Optional)**
```python
def add_dp_noise_to_activations(activations, epsilon=1.0):
    """Add Laplace noise to neural activations"""
    sensitivity = 1.0  # L2 norm of activations typically ~1.0
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=activations.shape)
    return activations + noise
```

**2. K-Anonymity Enforcement**
- Minimum 10 samples per (grid_square, band, time_bin) before transmission
- Rare combinations discarded locally
- Never transmitted to server

**3. Temporal Decorrelation**
- Random delays (0-1 hour) before transmission
- Batch mixing across users
- No sequence numbers or session IDs

**4. No Raw Data Transmission**
- Activations only (not IQ samples)
- Already compressed representations
- Cannot reconstruct original signals

### Neural State vs Traditional Telemetry

Compared to traditional amateur radio telemetry (PSKReporter, WSPR):

| Feature | PSKReporter | CASCADE Neural Telemetry |
|---------|-------------|--------------------------|
| Callsigns | Cleartext | Not included (anonymized in metadata only) |
| Locations | Exact grid (6-char) | Truncated (4-char) |
| Message content | Mode/SNR only | Not included (activations post-decode) |
| Equipment info | Not captured | 16-D abstract fingerprint |
| Raw signals | Not captured | Not captured (only learned features) |
| **Privacy level** | **Moderate** | **High** |

### Conclusion

Neural state telemetry provides **stronger privacy** than traditional amateur radio reporting systems:
- No callsigns in neural activations
- No message content (already consumed)
- Abstract learned features (not raw signals)
- Multiple layers of anonymization

**Risk level**: Low-Medium (primarily location inference from propagation patterns)
**Mitigation**: K-anonymity, differential privacy, and temporal decorrelation

## Compliance

### Amateur Radio Regulations
- Complies with amateur radio privacy expectations
- Respects operator privacy while enabling research
- No transmission of protected information

### Data Protection
- No PII (Personally Identifiable Information) stored
- Irreversible anonymization process
- Secure salt management

## Technical Implementation

### Anonymizer Module
- Location: `modules/data/src/processors/anonymizer.py`
- Tests: `modules/data/tests/unit/test_anonymizer.py`

### Key Features
1. Automatic detection of grid squares vs callsigns
2. Consistent hashing (same callsign → same hash)
3. Batch processing support
4. Validation of anonymization completeness

### API Usage

```python
from modules.data.src.processors.anonymizer import CallsignAnonymizer

anonymizer = CallsignAnonymizer(salt="your-secure-salt")

# Process FT8 message
result = anonymizer.anonymize_message("W1ABC K2DEF FN42")
# Result: {
#   "anonymized_message": "ANON_A3F2B91C ANON_7E4D8C2A FN42",
#   "callsign_hashes": ["SHA256....", "SHA256...."],
#   "grid_squares": ["FN42"],
#   "privacy_method": "SHA256_SALTED_GRID_PRESERVED"
# }
```

## Verification

Unit tests verify:
- Grid squares are never hashed
- Callsigns are always hashed
- 4 and 6 character grids are handled correctly
- Grid square validation works properly
- Consistent hashing for same callsigns

## Future Considerations

### Potential Enhancements
1. **[Differential Privacy](training/continuous_improvement.md#differential-privacy-ε10)**: Add noise to aggregate statistics (implemented in telemetry)
2. **[K-Anonymity](training/continuous_improvement.md#k-anonymity-k10)**: Ensure minimum group sizes for analysis (implemented in telemetry)
3. **Temporal Aggregation**: Aggregate data over time windows

### Research Ethics
- Regular review of privacy measures
- Transparent documentation of methods
- Community feedback incorporation
- Ethical review board consultation if needed

## Contact

For privacy concerns or questions about our anonymization approach:
- GitHub Issues: https://github.com/anthropics/cascade/issues
- Documentation: https://cascade.example.com/privacy

---

*Last Updated: 2025-09-30*
*Version: 1.0.0*