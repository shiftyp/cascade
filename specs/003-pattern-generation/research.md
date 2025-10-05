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

---

### 7. FSK Order Selection: 2-FSK vs 4-FSK (FINAL - 2025-10-04)

**Decision**: Use 2-FSK (2 adjacent tones per pattern) for optimal λ minimization

**Capacity with 150-tone grid (3 kHz SSB channel)**:

| FSK Order | Tones/Pattern | Non-Overlap Capacity | Avg λ | BPSK % |
|-----------|---------------|---------------------|-------|---------|
| **2-FSK** | 2 (64 Hz) | **75 patterns** | **0.08-0.10** | **59%** ✅ |
| 4-FSK | 4 (128 Hz) | 37 patterns | 0.17 | 29% |
| 6-FSK | 6 (192 Hz) | 25 patterns | 0.22-0.25 | 20% |

**Analysis**:
```
To achieve 128 patterns:
- 2-FSK: 75 frequency-orthogonal (λ=0) + 53 IQ-orthogonal (λ=0.15-0.30)
- 4-FSK: 37 frequency-orthogonal (λ=0) + 91 IQ-orthogonal (λ=0.15-0.35)
- 6-FSK: 25 frequency-orthogonal (λ=0) + 103 IQ-orthogonal (λ=0.20-0.40)

More tones per pattern → less frequency diversity → more IQ complexity needed
```

**Why 2-FSK for emergency communications**:
1. **Low-SNR performance**: λ=0.09 avg vs 0.17 (47% reduction)
   - Can decode at 2-3 dB lower SNR
   - Critical for weak/emergency paths
2. **BPSK majority**: 59% patterns at λ=0 (vs 29% with 4-FSK)
   - Maximum robustness for emergencies
   - Better phase distortion tolerance
3. **Equipment scalability**: Modular transmissions (1×, 2×, 4×, 8× 2-FSK)
   - QRP: 1 pattern @ 44 bps
   - Modern: 4 patterns @ 175 bps
   - Premium: 8 patterns @ 350 bps
4. **Backward compatible**: Legacy and SDR coexist naturally

**Trade-off**: Reduced selective fading diversity (64 Hz vs 128 Hz span)
- Compensated by +2-3 dB SNR margin from lower λ
- Net effect: 2-FSK superior overall

**Reference**: FSK order analysis, equipment scalability discussion (2025-10-04)

---

### 8. Equipment Targeting: QMX as Baseline (FINAL - 2025-10-04)

**Decision**: Design CASCADE for QMX-class SDR ($150) as primary target, support legacy as fallback

**Equipment cost trends**:
```
Entry-level SDR:
2020: $550 (Hermes Lite 2)
2025: $150 (QMX assembled)
2030: $75-100 (projected)

Traditional HF:
Flat ($600-1400) or rising

Crossover: SDR now cheaper than legacy for new purchases
```

**Market projections**:
```
SDR adoption (digital mode users):
2025: 25-30%
2027: 40-50%
2030: 60-70%

By 2030: SDR is majority among active digital mode operators
```

**QMX advantages for CASCADE** ($180 with GPS):
1. 200 sym/s capable (NN-enhanced) → 44-175 bps
2. GPS-disciplined (±0.1 Hz) → enables 20 Hz tone spacing
3. IQ mode (15-20 kHz BW) → multi-band monitoring
4. Cost ($180) < legacy maintenance
5. Modular 2-FSK scaling (1×, 2×, 4× transmissions)

**Dual-mode strategy**:
- **SDR mode** (primary): 200 sym/s, 20 Hz spacing, full 3 kHz channel
  - Equipment: QMX ($180), modern SDR
  - Market: 30% today → 60% by 2030
  - Throughput: 44-350 bps

- **Legacy mode** (fallback): 40 sym/s, legacy-compatible
  - Equipment: ANY SSB radio since 1970
  - Market: 70% today → 30% by 2030 (still important!)
  - Throughput: 8.75-17.5 bps
  - Use case: Emergency fallback, universal participation

**Cognitive coexistence**: Legacy and SDR share same 3 kHz channel
- Kernel detects equipment type
- Coordinates assignments to avoid collisions
- Network capacity improves as SDR adoption grows

**Reference**: QMX specification analysis, market research, cost-benefit for EMCOMM deployment (2025-10-04)