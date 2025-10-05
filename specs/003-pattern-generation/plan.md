# Implementation Plan: Pattern Generation - 64 and 128 Pattern Sets

**Branch**: `003-pattern-generation` | **Date**: 2025-10-04 | **Spec**: [spec.md](spec.md)

## Summary

Generate CASCADE's two orthogonal 4D pattern sets (64-pattern for testing/vetting, 128-pattern for production) using Zadoff-Chu sequences + simulated annealing to achieve -37.5 dB cross-correlation. This is the foundational prerequisite for all CASCADE training and operation.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: NumPy, SciPy (optimization), struct (binary I/O)
**Storage**: Local filesystem - outputs cascade_patterns_64.bin (19 KB) and cascade_patterns_128.bin (38 KB)
**Testing**: PyTest for validation tests
**Target Platform**: Linux/macOS with 8-core CPU
**Project Type**: Monorepo - Training module (pattern infrastructure)
**Performance Goals**: 64-pattern in 8-12 hours, 128-pattern in 18-24 hours, <-37.5 dB orthogonality
**Constraints**: <16 GB RAM, deterministic with seeds, one-time generation (patterns never change)
**Scale/Scope**: Two pattern sets, 2,016 pairs (64-pattern) + 8,128 pairs (128-pattern) to validate

## Constitution Check

- [x] **Data-First Development**: Patterns are prerequisite infrastructure, generated BEFORE any training
- [x] **Monorepo Module Architecture**: Output to `modules/training/patterns/` directory
- [x] **Clean Separation**: Pure mathematical generation, no protocol/model layer involvement
- [x] **Test-Driven Development**: Validation tests verify orthogonality before accepting patterns
- [x] **Real-World Data Priority**: N/A (patterns are mathematical constructs, not trained)
- [x] **Privacy-Preserving**: N/A (no user data involved)
- [x] **Reproducible Research**: Deterministic with fixed seeds, versioned pattern files

**No violations** - Pattern generation is pure infrastructure aligned with constitutional principles

## Project Structure

### Training Module - Pattern Infrastructure
```
modules/training/
├── patterns/                    # NEW: Pattern generation and storage
│   ├── generator.py            # Main generation orchestrator
│   ├── zadoff_chu.py           # Zadoff-Chu sequence generation
│   ├── iq_trajectories.py      # IQ curve generation (hierarchical)
│   ├── optimizer.py            # Simulated annealing for -37.5 dB
│   ├── correlation.py          # 4D cross-correlation calculation
│   ├── binary_format.py        # CASCADE binary file I/O
│   └── validator.py            # Pattern validation and reports
├── data/                        # NEW: Generated pattern files
│   ├── cascade_patterns_64.bin   # 64-pattern set (19 KB)
│   └── cascade_patterns_128.bin  # 128-pattern set (38 KB)
└── tests/
    └── patterns/               # NEW: Pattern validation tests
        ├── test_zadoff_chu.py
        ├── test_iq_generation.py
        ├── test_optimizer.py
        ├── test_correlation.py
        ├── test_binary_format.py
        └── test_pattern_validation.py
```

**Note**: This is foundational infrastructure for the Training Module, used by all subsequent features.

---

## Phase 0: Research

### Research Completed

Based on docs/implementation/pattern_generation_spec.md (714 lines), all technical decisions are documented:

1. **Zadoff-Chu Sequences**: LTE-proven, patent-free, provides base orthogonality (~-15 dB)
2. **Simulated Annealing**: Optimize to exactly -37.5 dB (better than -30 dB target)
3. **4-tone Selection**: Each pattern selects 4 from 78-tone grid (adaptive)
4. **IQ Hierarchical**: Baked-in complexity (λ) determined by pattern ID
5. **Binary Format**: Magic bytes b'CASC', version 2, pattern count, checksums

### Key Decisions

- **Algorithm**: Zadoff-Chu (base) + simulated annealing (optimization)
- **Orthogonality target**: -37.5 dB (exceeds -30 dB specification)
- **IQ complexity**: Hierarchical pools (emergency λ=0.0, typical λ=0.4, NVIS λ=0.8)
- **Storage format**: Custom binary (efficient loading, 292 bytes per pattern)
- **Validation**: Exhaustive pairwise correlation check

**No unknowns remain** - Implementation spec provides complete algorithm

---

## Phase 1: Design & Contracts

### Data Model

**PatternSet** (Container for pattern collection)
- pattern_count: int (64 or 128)
- beacon_patterns: List[Pattern] (48 patterns, IDs 0-47)
- message_patterns: List[Pattern] (16 or 80 patterns, IDs 48-63 or 48-127)
- min_correlation_db: float (worst-case orthogonality)
- generation_time_hours: float

**Pattern** (Single 4D orthogonal pattern)
- pattern_id: int (0-63 for 64-set, 0-127 for 128-set)
- pattern_type: str ("beacon" or "message")
- freq_sequence: ndarray (32 × uint8, tone indices 0-3)
- iq_trajectory: ndarray (32 × complex64)
- iq_complexity_lambda: float (0.0 to 0.9)
- complexity_pool: str ("emergency", "typical_dx", "good_prop", "nvis")

**CorrelationMatrix** (Orthogonality validation)
- pattern_set_size: int
- pair_correlations: Dict[(int, int), float] (all pattern pairs)
- min_correlation_db: float
- max_correlation_db: float
- failed_pairs: List[(int, int, float)] (if any >-37.5 dB)

**BinaryPatternFile** (Serialized output)
- filename: str (cascade_patterns_64.bin or cascade_patterns_128.bin)
- magic_bytes: bytes (b'CASC')
- version: int (2 for 128-pattern chaos)
- pattern_count: int
- file_size_bytes: int
- checksum: bytes (CRC16 per pattern)

### API Contracts

Since this is a one-time generation script (not a service), contracts are **command-line interface + validation functions**:

#### Contract 1: Pattern Generation Command
```bash
python -m modules.training.patterns.generator \
    --output cascade_patterns_64.bin \
    --count 64 \
    --beacon 48 \
    --message 16 \
    --seed 42

# Expected output:
# Generating 64 patterns (48 beacon + 16 message)...
# [Progress bar 0-64]
# Validating orthogonality...
# Min correlation: -42.3 dB ✓
# Max correlation: -37.8 dB ✓
# Output: cascade_patterns_64.bin (19,234 bytes)
```

#### Contract 2: Pattern Validation Function
```python
def validate_pattern_set(pattern_file: str) -> ValidationReport:
    """
    Validate pattern file orthogonality and format

    Returns: ValidationReport with pass/fail
    """
    pass
```

#### Contract 3: 4D Correlation Calculation
```python
def compute_4d_correlation(
    pattern_i: Pattern,
    pattern_j: Pattern
) -> float:
    """
    Compute cross-correlation in Time × Freq × IQ space

    Returns: Correlation in dB (must be < -37.5)
    """
    pass
```

### Quickstart

**Generate and validate 64-pattern set**:
```bash
# Generate
python -m modules.training.patterns.generator --count 64 --seed 42

# Validate
python -m modules.training.patterns.validator cascade_patterns_64.bin

# Expected: All pairs <-37.5 dB ✓
```

**Generate 128-pattern set**:
```bash
python -m modules.training.patterns.generator --count 128 --seed 42
python -m modules.training.patterns.validator cascade_patterns_128.bin
```

---

## Phase 2: Task Planning Approach

**Tasks will include**:

1. **Setup**: Module structure, dependencies
2. **Contract tests**: Validation functions, correlation checks (TDD)
3. **Core generation**:
   - Zadoff-Chu sequence generation
   - IQ trajectory generation (adaptive λ minimization)
   - 4D correlation calculation (with phase distortion testing)
   - Simulated annealing optimizer (direct IQ mutation)
   - **Two-phase optimization** (frequency-first, then IQ refinement)
   - **Phase-aware cost function** (optimize under HF propagation conditions)
   - Binary file I/O
4. **Multi-trial generation**: Parallel execution with checkpointing
5. **Platform adaptation**: CPU auto-detection, hybrid architecture support
6. **Visualization**: IQ plots, λ histograms, correlation matrices
7. **Distributed execution** (optional): Fly.io worker infrastructure
8. **CLI**: Progress tracking, auto-tuning, distributed mode
9. **Validation**: Exhaustive pair checking, phase robustness, format validation
10. **Integration**: End-to-end tests (local + distributed)

**Estimated**: 47 tasks (41 core + 6 optimization enhancements)

**Timeline**:
- Development: 22-27 hours (includes two-phase + phase-aware + 2-FSK implementation)
- Testing: 3-5 hours
- **Production generation (optimal)**: 72-96 hours local (8 trials × 400K, 2-FSK architecture)
  - Expected: -42.6 to -43 dB separation, **λ=0.08-0.10**, $0 cost
  - BPSK patterns: 75 (59%) - exceptional low-SNR robustness
  - Best for: Core Ultra 7 265K, Ryzen 9, similar high-end CPUs
  - Equipment throughput: QMX @ 200 sym/s = 44-175 bps (1× to 4× 2-FSK)
- **Production generation (fast)**: 30-40 hours Fly.io (32 workers × 100K, 2-FSK)
  - Expected: -40.7 dB, λ=0.14-0.16, $9.60 cost
  - Best for: Users without capable local hardware or need faster results

---

## Complexity Tracking

No constitutional violations. Pattern generation is foundational infrastructure.

---

## Progress Tracking

- [x] Phase 0: Research (existing implementation spec provides all details)
- [x] Phase 1: Design complete
- [x] Phase 2: Task approach described
- [ ] Phase 3: Tasks (via /tasks command)

---

*Based on Constitution v1.0.0*
*Prerequisite for Feature 002 (Phase 0 Training)*
