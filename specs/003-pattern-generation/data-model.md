# Data Model: Pattern Generation

**Feature**: 003-pattern-generation

## Entities

### PatternSet
- pattern_count: int (64 or 128)
- beacon_count: int (always 48)
- message_count: int (16 for 64-set, 80 for 128-set)
- patterns: List[Pattern]
- min_correlation_db: float
- max_correlation_db: float
- generation_seed: int

### Pattern
- pattern_id: int (0-127)
- pattern_type: str ("beacon" | "message")
- freq_sequence: ndarray[32] (uint8, indices 0-3)
- iq_trajectory: ndarray[32] (complex64)
- iq_complexity_lambda: float (0.0-0.9)
- complexity_pool: str ("emergency" | "typical_dx" | "good_prop" | "nvis")

### CorrelationResult
- pattern_i_id: int
- pattern_j_id: int
- correlation_db: float
- passes: bool (<-37.5 dB)

### ValidationReport
- pattern_file: str
- pattern_count: int
- all_pairs_pass: bool
- min_correlation_db: float
- max_correlation_db: float
- failed_pairs: List[CorrelationResult]
