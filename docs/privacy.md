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