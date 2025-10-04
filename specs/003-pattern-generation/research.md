# Research: Pattern Generation

**Feature**: 003-pattern-generation
**Date**: 2025-10-04

## Research Summary

**Primary Reference**: All technical details documented in `docs/implementation/pattern_generation_spec.md` (714 lines)

This research document consolidates key decisions for implementation.

---

## Key Technical Decisions

### 1. Pattern Generation Algorithm

**Decision**: Zadoff-Chu sequences + simulated annealing optimization

**Components**:
- **Base generation**: Zadoff-Chu sequences (31 patterns from u=0 to u=30)
- **Remaining patterns**: Random initialization
- **Optimization**: Simulated annealing to achieve -37.5 dB
- **Iterations**: Up to 100,000 per pattern

**Rationale**:
- Zadoff-Chu provides good starting orthogonality (~-15 dB)
- Simulated annealing proven for combinatorial optimization
- Can achieve exactly -37.5 dB (better than -30 dB target)
- Patent-free (1970s mathematics)

**Reference**: docs/implementation/pattern_generation_spec.md lines 23-76

---

### 2. 4D Correlation Calculation

**Decision**: Exhaustive pairwise check in Time × Freq × IQ space

**Algorithm**:
```python
correlation = 0
for t in range(32):  # Time dimension
    tone_i = pattern_i.freq_sequence[t]  # Tone index (0-3)
    tone_j = pattern_j.freq_sequence[t]

    if tone_i != tone_j:
        continue  # Different tones → orthogonal in frequency

    # Same tone → check IQ orthogonality
    iq_i = pattern_i.iq_trajectory[t]
    iq_j = pattern_j.iq_trajectory[t]
    correlation += abs(iq_i * conj(iq_j))

correlation_db = 20 * log10(correlation / 32)
```

**Rationale**:
- Must validate ALL pairs (2,016 for 64-pattern, 8,128 for 128-pattern)
- Correlation in dB allows threshold checking (<-37.5 dB)
- Time dimension implicit in loop
- Frequency uses tone INDEX (0-3), not actual frequency

**Reference**: docs/model/tfiq_dimensions.md lines 118-150

---

### 3. IQ Complexity Optimization (UPDATED 2025-10-04)

**Decision**: Minimize λ during optimization rather than pre-assigning by pattern ID

**Optimization Strategy**:
- **Primary objective**: Achieve -37.5 dB orthogonality (hard constraint)
- **Secondary objective**: **Minimize λ** (prefer simplest IQ complexity)
- All patterns start with λ = 0.0 (BPSK line on I-axis)
- Simulated annealing optimizes BOTH tone sequence AND λ value
- Only increase λ if needed to achieve orthogonality
- Result: Empirically discover minimum IQ complexity for 128 orthogonal patterns

**Rationale**:
- Simpler IQ = easier decoding, lower SNR threshold, better robustness
- Complex IQ should only be used if REQUIRED for orthogonality
- Let optimization discover true constraint rather than pre-assigning pools
- Maximizes system robustness under poor HF propagation conditions

---

### 4. Two-Phase Optimization (FINAL - 2025-10-04)

**Decision**: Separate frequency and IQ optimization into two phases

**Approach**:
- **Phase 1** (80% of iterations): Frequency-only with BPSK (λ=0)
  - Optimize tone sequence trying to achieve -37.5 dB with simplest IQ
  - If successful, pattern stays at λ=0 (maximum robustness)
  - Default: 320K iterations out of 400K budget
- **Phase 2** (20% of iterations): IQ refinement if Phase 1 insufficient
  - Add minimum IQ complexity to achieve orthogonality target
  - Direct IQ mutation with adaptive λ discovery
  - Default: 80K iterations for refinement

**Rationale**:
- Focuses optimization effort on BPSK solutions first
- Only adds IQ complexity when proven necessary
- Expected: 20-30% of patterns achieve λ=0 (vs 10-15% single-phase)
- Lower average λ across entire pattern set (0.17 vs 0.22)

**Comparison**:
| Approach | Avg λ | Patterns at λ=0 | Separation |
|----------|-------|-----------------|------------|
| Single-phase | 0.22 | 10-15% | -40.7 dB |
| Two-phase | 0.17 | 20-30% | -42.6 dB |

**Reference**: Cost-benefit analysis and optimization strategy discussion (2025-10-04)

---

### 5. Phase-Aware Optimization (FINAL - 2025-10-04)

**Decision**: Include phase distortion in cost function during optimization

**Approach**:
- Test each candidate pattern under 3-5 random phase scenarios
- Random phase per tone: ±π radians (frequency-dependent distortion)
- Random phase per symbol: ±0.2 radians (time-varying channel)
- Cost function uses **worst-case** correlation across all scenarios

**Rationale**:
- HF channels introduce random phase rotation (non-negotiable physics)
- Optimizing for ideal orthogonality may produce phase-fragile patterns
- Phase-aware optimization ensures patterns robust from the start
- May naturally guide toward lower λ (simpler IQ less phase-sensitive)

**Trade-offs**:
- Performance: ~2-3x slower per iteration (5 phase tests vs 1)
- Quality: Phase-robust separation within 5-6 dB of ideal (vs 7-8 dB)
- Total time: Same (fewer total iterations needed for robust solutions)

**Expected outcomes**:
- Ideal separation: -42.6 dB
- Phase-robust separation: -36 to -37 dB (acceptable for HF)
- Average λ: 0.17 (potentially lower than non-phase-aware)

**Reference**: HF phase distortion analysis and robustness requirements (2025-10-04)

---

### 6. Execution Mode Selection (FINAL - 2025-10-04)

**Decision**: 8 trials × 400K iterations (depth strategy) for local high-end CPUs

**Cost-benefit analysis**:

**Local (Core Ultra 7 265K)**:
- 8 trials × 400K iterations
- Time: 72-96 hours (3-4 days)
- Cost: $0 (free, electricity ~$2)
- Quality: -42.6 dB, λ=0.17
- $/dB: $0/dB (optimal)

**Cloud (Fly.io)**:
- 32 trials × 100K iterations (breadth strategy)
- Time: 30-40 hours
- Cost: $9.60
- Quality: -40.7 dB, λ=0.22
- $/dB: $4.94/dB

**Optimal stopping point**: 8 trials at 400K iterations
- Marginal cost beyond 8: $15+/dB (diminishing returns)
- Depth convergence better than diversity at 400K iteration level
- Local execution 1.9x faster per core than cloud vCPUs

**Recommendation**:
- Primary: Local 8×400K for production (best quality, free)
- Alternative: Cloud 32×100K for users without capable hardware

**Reference**: Platform performance comparison and ROI analysis (2025-10-04)