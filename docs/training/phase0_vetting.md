# Phase 0: Ideal Conditions Vetting

**Purpose:** Validate CASCADE's 128-pattern chaos architecture in ideal conditions (AWGN only) before investing 18 months in real HF data collection.

**Duration:** 2.5-3 days (60 hours training on 1x RTX 4090)

**Status:** Pre-implementation planning phase

---

## Table of Contents

1. [Overview](#overview)
2. [Ideal Conditions Definition](#ideal-conditions-definition)
3. [Test Progression](#test-progression)
4. [Success Criteria](#success-criteria)
5. [Expected Outcomes](#expected-outcomes)
6. [Synthetic Training Strategy](#synthetic-training-strategy)
7. [V0.5 Beta Deployment Path](#v05-beta-deployment-path)
8. [Risk-Benefit Analysis](#risk-benefit-analysis)

---

## Overview

### The Problem

CASCADE's architecture is complex:
- 128 orthogonal patterns with -37.5 dB correlation
- Frequency reuse (same pattern on different tones)
- Time reuse (asynchronous chaos)
- Kernel-driven emergent coordination
- RS(32,20) aligned structure

**Question:** Will this actually work?

**Traditional answer:** Collect 150K hours of real HF data, train for 3-4 weeks, hope it works.

**Risk:** 18 months + $5K wasted if architecture is flawed.

### The Solution

**Phase 0 Vetting:** Train in ideal conditions (AWGN only) to validate architecture before data collection.

**Logic:**
```
If model CANNOT achieve 85%+ Shannon in AWGN (ideal),
  it DEFINITELY won't achieve 78-85% in real HF (harsh).

If model CAN achieve 85%+ Shannon in AWGN,
  then 78-85% in real HF is realistic (10-15% degradation expected).
```

**Time investment:** 2.5 days vs 18 months

**Risk reduction:** Massive (validate before committing to data collection)

---

## Ideal Conditions Definition

### Environment Specification

**Channel model:**
```python
def ideal_channel(signal, snr_db):
    """
    Ideal AWGN channel (no HF impairments)

    This is the SIMPLEST channel - if CASCADE can't handle this,
    it definitely can't handle real HF.
    """
    # Just add white Gaussian noise (no propagation effects)
    noise_power = signal_power / (10 ** (snr_db / 10))
    awgn = np.random.normal(0, sqrt(noise_power), len(signal))

    return signal + awgn
```

**What's removed:**
- ❌ Atmospheric QRN (lightning crashes, solar noise)
- ❌ Man-made QRM (powerlines, radar, broadcast stations)
- ❌ Multipath (multiple ionospheric reflections, 1-10ms delay spread)
- ❌ Fading (Rayleigh/Rician signal strength variation)
- ❌ Doppler (ionospheric motion causing frequency shifts)
- ❌ Ionospheric flutter and phase distortion
- ❌ Selective fading (frequency-dependent nulls)

**What's kept (essential for CASCADE):**
- ✅ AWGN at specified SNR levels (-22 to +15 dB)
- ✅ Multi-user interference (45 overlapping CASCADE signals)
- ✅ Chaos overlaps (random start times, collisions)
- ✅ Asynchronous timing (no coordination, users start whenever)
- ✅ Clock drift (±50 Hz per user for fingerprinting)
- ✅ RS symbol erasures (test 37.5% tolerance)
- ✅ Pattern reuse via frequency diversity
- ✅ Pattern reuse via time diversity
- ✅ Kernel/antikernel coordination

### Why This Environment Tests Core Architecture

**The hard problems CASCADE must solve:**

1. **Pattern separation** - Can -37.5 dB orthogonality separate 45 users?
2. **Frequency reuse** - Can same pattern work on different tones without collision?
3. **Time reuse** - Can handle arbitrary async starts and partial overlaps?
4. **Chaos handling** - Can decode random overlapping transmissions?
5. **RS decoding** - Can recover with 37.5% symbol erasures?
6. **Kernel utilization** - Do kernels actually help performance?
7. **Multi-pattern** - Can decode 1-4 patterns simultaneously?
8. **Emergent coordination** - Do prokernels/antikernels improve efficiency?

**None of these depend on realistic HF propagation!**

If model can't do these in clean AWGN, adding multipath/fading won't help.

---

## Test Progression

### Test 1: Single User Baseline

**Training time:** 1 hour
**Compute:** 1x RTX 4090

**Setup:**
```python
training_config = {
    'num_users': 1,
    'patterns': [64],  # Single pattern from Typical DX pool
    'snr_db': 15,
    'channel': 'AWGN',
    'num_samples': 10000,
}

# Generate training data
for sample in range(10000):
    # Single CASCADE user
    data = random_bytes(18)  # 144 bits
    pattern_id = 64

    # Generate RS pattern transmission
    signal = generate_rs_pattern(pattern_id, data)

    # Add AWGN
    received = signal + awgn(snr_db=15)

    # Train model to decode
    decoded = model.decode(received)
    loss = cross_entropy(decoded, data)
```

**Targets:**
- Decode accuracy: **99.9%**
- Shannon efficiency: **95%+** (~12,000 bps for single user)
- Pattern recognition: **99.9%+**

**Validates:**
- Basic pattern encode/decode works
- RS(32,20) structure functions correctly
- Model can learn at all

---

### Test 2: Pattern Orthogonality (10 users)

**Training time:** 4 hours

**Setup:**
```python
training_config = {
    'num_users': 10,
    'patterns': [64, 70, 76, 82, 88, 94, 100, 106, 112, 118],  # 10 different patterns
    'snr_db': 15,
    'channel': 'AWGN',
    'start_times': 'random',  # Each user starts at random time
}

# All 10 users transmitting simultaneously
# Separated by pattern orthogonality (-37.5 dB)
```

**Targets:**
- Decode accuracy: **98%** (all 10 users correctly separated)
- Shannon efficiency: **92%** (~11,500 bps shared by 10 users)
- Per-user throughput: ~1,150 bps

**Validates:**
- -37.5 dB orthogonality sufficient for separation
- Model can separate multiple users
- Pattern correlation works

---

### Test 3: Frequency Reuse (20 users)

**Training time:** 6 hours

**Setup:**
```python
training_config = {
    'num_users': 20,
    'patterns': 10,  # Only 10 unique patterns (2 users per pattern)
    'tone_reuse': True,  # Same pattern on different tone selections
    'snr_db': 15,
}

# Example: Both use Pattern 64
users = [
    {'pattern': 64, 'tones': [12, 35, 51, 65]},  # User A
    {'pattern': 64, 'tones': [8, 29, 47, 63]},   # User B (different tones!)
    # ... 18 more users
]

# Users with same pattern but different tones
# Should be separated by FREQUENCY, not pattern orthogonality
```

**Targets:**
- Decode accuracy: **95%**
- Shannon efficiency: **90%** (~11,300 bps)
- Interference: Users with disjoint tones should have ~0 dB interference

**Validates:**
- **Critical test**: Frequency reuse mechanism works
- Same pattern on different tone selections → FDMA-like separation
- This is KEY to 1,024 user capacity claim

---

### Test 4: Time Reuse (30 users)

**Training time:** 8 hours

**Setup:**
```python
training_config = {
    'num_users': 30,
    'patterns': 20,
    'start_time_offsets': 'random(0, 1.6s)',  # Asynchronous starts
    'some_overlap': True,  # Intentional partial time overlap
}

# Users start at different times
# Pattern duration: 1.6s
# Some users overlap partially (e.g., User A at t=0s, User B at t=0.8s)
```

**Targets:**
- Decode accuracy: **93%**
- Shannon efficiency: **88%** (~11,000 bps)
- Partial overlap handling: RS(32,20) tolerates collisions

**Validates:**
- Time reuse via asynchronous starts works
- Partial overlaps handled by RS erasure tolerance
- Chaos operation feasible

---

### Test 5: Full Chaos (45 users) - CRITICAL TEST

**Training time:** 12 hours

**Setup:**
```python
training_config = {
    'num_users': 45,
    'patterns': 128,  # All patterns available
    'frequency_reuse': True,  # Same patterns on different tones
    'time_reuse': True,  # Asynchronous starts
    'chaos_mode': True,  # No coordination, pure chaos
    'snr_db': 15,
}

# Generate 45 users with:
for user in range(45):
    user_config = {
        'pattern': random.choice(128),  # Random pattern
        'tones': select_4_from_78_random(),  # Random tone selection
        'start_time': random.uniform(0, 10),  # Random start (10s window)
        'clock_drift': random.uniform(-50, 50),  # Hz
        'data': random_bytes(18),
    }
```

**Targets:**
- Decode accuracy: **90%** (accept 10% loss in full chaos)
- Shannon efficiency: **85%+** (~10,600+ bps)
- Active users: All 45 decoded simultaneously

**Validates:**
- **MOST CRITICAL TEST**: Full chaos mode achieves target efficiency
- If this passes, architecture is sound
- If this fails (<80%), architecture needs revision

**Why 85% in AWGN supports 78-85% in real HF:**
- Real HF adds 10-15% overhead (multipath, fading, real noise)
- 85% - 10% = 75% (low end of target range) ✓
- 85% - 5% = 80% (if propagation is gentle) ✓

---

### Test 6: Kernel Coordination (45 users)

**Training time:** 16 hours

**Setup:**
```python
# Train 3-round kernel exchange
for round in [1, 2, 3]:
    # Round 1: Random kernels (baseline)
    if round == 1:
        kernels = [random_kernel() for _ in range(45)]

    # Round 2: Prokernels (announce available_tones, capabilities)
    elif round == 2:
        kernels = [generate_prokernel(user) for user in users]

    # Round 3: Antikernels (request interference reduction)
    elif round == 3:
        kernels = [adapt_kernel_with_antikernels(user, interferers) for user in users]

    # Train with kernels
    decoded = model.decode_with_kernels(signal, kernels)
```

**Targets:**
- Round 1 (no kernels): 85% Shannon baseline
- Round 2 (prokernels): 86% Shannon (+1%)
- Round 3 (antikernels): 87% Shannon (+2% total)
- Steady state (cached): 88-90% Shannon

**Validates:**
- Kernel coordination provides measurable improvement
- Prokernels help (frequency guidance)
- Antikernels help (interference coordination)
- Emergent coordination mechanism works

---

### Test 7: SNR Degradation Sweep

**Training time:** 12 hours

**Setup:**
```python
# Train across full SNR range
snr_levels = [15, 10, 5, 0, -5, -10, -15, -22]

for snr_db in snr_levels:
    # 45 users at this SNR
    # Expected capacity reduces as SNR drops

    train_at_snr(snr_db, num_users=45)
```

**Targets:**
- +15 dB: 85%+ Shannon, 45 active users
- +10 dB: 80-85% Shannon, 45 users
- +5 dB: 75-80% Shannon, 40 users
- 0 dB: 70-75% Shannon, 35 users
- -5 dB: 60-65% Shannon, 25 users
- -10 dB: 50-55% Shannon, 15 users
- -15 dB: 40-45% Shannon, 8 users
- -22 dB: 25-30% Shannon, 3-5 users

**Validates:**
- Graceful degradation across SNR range
- Pattern pool selection mechanism
- Adaptive capacity reduction
- Emergency patterns work at -22 dB

---

## Success Criteria

### VETTING PASSES (Architecture Valid) ✓

**If Tests 1-7 achieve:**
- ✅ Test 5: ≥85% Shannon with 45 users in AWGN
- ✅ Test 6: Kernel coordination provides ≥2% improvement
- ✅ Test 3: Frequency reuse works (same pattern, different tones)
- ✅ Test 7: Graceful degradation from +15 to -22 dB
- ✅ All accuracy targets met

**Implications:**
- Architecture validated ✓
- 78-85% Shannon in real HF is realistic (real impairments cost 10-15%)
- Safe to proceed with data collection OR synthetic training
- Confidence in deployment readiness

**Next steps (3 options):**
1. **Wait for real data** (18 months, 78-85% quality at launch)
2. **Train on synthetic** (3 weeks, 65-70% quality, fast beta)
3. **Hybrid approach** (2-3 months, 70-75% quality, balanced)

### VETTING FAILS (Architecture Issues) ✗

**If Tests achieve <80% Shannon or other failures:**

**Possible issues:**
- Pattern orthogonality insufficient (-37.5 dB not enough)
- Chaos overlap tolerance too optimistic
- Frequency reuse causes unexpected interference
- RS erasure tolerance inadequate
- Kernel coordination doesn't help or makes worse
- Multi-pattern decoding fails

**Next steps:**
- ❌ STOP data collection planning
- 🔧 Fix architecture:
  - Increase patterns? (128 → 256)
  - Improve orthogonality? (-37.5 dB → -40 dB)
  - Add coordination? (reduce chaos, add time slots)
  - Adjust RS rate? (32,20 → 32,24 for more FEC)
- 🔁 Re-vet with fixes
- ✅ Only proceed when vetting passes

---

## Expected Outcomes

### Best Case: 85-90% Shannon in AWGN

**Result:** Architecture exceeds expectations

**Implications:**
- Could achieve 80-85% in real HF (upper end of target)
- Frequency + time reuse working very well
- Kernel coordination highly effective
- May be room for even higher efficiency

**Decision:** Safe to pursue any path (real data, synthetic, or hybrid)

### Expected Case: 85-87% Shannon in AWGN

**Result:** Architecture meets targets

**Implications:**
- Will achieve 78-85% in real HF (middle of target range)
- Frequency reuse working as designed
- Kernel coordination providing expected boost
- Architecture is sound

**Decision:** Proceed with confidence

### Marginal Case: 80-84% Shannon in AWGN

**Result:** Architecture below expectations

**Implications:**
- Will achieve 70-78% in real HF (lower end or below target)
- Something not working as well as designed
- May need optimization before full deployment

**Decision:**
- Investigate what's limiting performance
- Consider architectural tweaks
- Proceed cautiously or with modifications

### Failure Case: <80% Shannon in AWGN

**Result:** Architecture has fundamental problems

**Implications:**
- Cannot achieve targets in real HF
- Major issues need fixing
- Design assumptions incorrect

**Decision:** Fix architecture before any data collection

---

## Synthetic Training Strategy

### If Vetting Passes: Option to Train on Synthetic Models

**Motivation:** Deploy in 1-2 months instead of 18 months

**Synthetic impairment models:**

1. **QRN (Atmospheric Noise):**
   - ITU-R P.372 models (lightning, galactic, solar, man-made)
   - Statistical characteristics by time of day, season, solar cycle
   - **Gap from real:** Misses actual temporal patterns and rare events

2. **QRM (Man-Made Interference):**
   - Powerline harmonics (60/50 Hz and harmonics)
   - Radar pulses (OTH radar, weather radar)
   - Broadcast station bleed-through
   - **Gap from real:** Doesn't capture specific local interference

3. **Propagation (Ionospheric):**
   - Watterson HF channel model (multipath, fading, Doppler)
   - ITU-R P.533 (propagation prediction)
   - Statistical multipath delay spread (1-10ms)
   - Rayleigh/Rician fading models
   - **Gap from real:** Simplified ionosphere, misses irregularities

**Expected performance with synthetic training:**
```
Vetting (AWGN): 85-90% Shannon

Add synthetic QRN: 82-85% Shannon (-3-5% from noise)
Add synthetic propagation: 70-75% Shannon (-10-12% from multipath/fading)
Add synthetic QRM: 65-70% Shannon (-5% from interference)

Final: 65-70% Shannon efficiency (vs 78-85% target)
```

**Why 15-20% degradation from AWGN to synthetic full:**
- Synthetic models don't match real complexity
- Missing edge cases and rare conditions
- Geographic biases in models
- Oversimplified ionospheric behavior

**Is 65-70% usable?**
- ✓ Still 13× better than FT8 (~5% Shannon)
- ✓ Still 20× better than voice (~3% Shannon)
- ✓ Functional for emergency communications
- ✓ Proves architecture works
- ✓ Enables telemetry collection
- ⚠ Not production quality, needs "beta" label

---

## V0.5 Beta Deployment Path

### Fast Deployment with Continuous Improvement

**Timeline:**
```
Week 0: Phase 0 vetting (AWGN)
  → Validate architecture: 85%+ Shannon ✓

Week 1-4: Train on synthetic models
  → QRN: ITU-R P.372
  → Propagation: Watterson + ITU-R P.533
  → QRM: Powerline + radar models
  → Expected: 65-70% Shannon

Week 4: Deploy V0.5 beta
  → 20-50 early adopters
  → Clear "beta" labeling
  → Collect telemetry from real usage

Month 2-12: Continuous improvement via telemetry
  → Monthly fine-tuning with real usage data
  → 65% → 68% → 72% → 75% → 78%+ progression
  → Users see visible monthly improvements

Month 6-9: Optionally collect 150K hours real data
  → While V0.5 operates
  → Parallel data collection + beta operation

Month 9-12: Full retrain with real data + telemetry
  → V1.0 release: 78-85% Shannon
  → Trained on both KiwiSDR data AND real usage telemetry
  → Better than pure KiwiSDR training (has actual usage patterns)
```

### V0.5 Beta Performance Profile

**Initial deployment (synthetic training):**
- Shannon efficiency: 65-70%
- Per-user throughput: 180-200 bps (1 pattern)
- Active users: 35-40 (vs 45 target)
- Decode accuracy: 75-85%
- Multi-pattern: 1-2 patterns typical (vs 1-4 target)

**After 3 months (fine-tuning with telemetry):**
- Shannon efficiency: 72-75%
- Per-user: 195-210 bps
- Active users: 40-43
- Decode accuracy: 82-88%

**After 6 months (continued fine-tuning):**
- Shannon efficiency: 75-78%
- Per-user: 210-220 bps
- Active users: 43-45
- Decode accuracy: 88-92%
- Approaching target performance!

**Month 9-12: V1.0 (full retrain):**
- Shannon efficiency: 78-85% ✓ Target achieved
- Trained on real data + telemetry
- Better than waiting 18 months (has usage patterns)

---

## Comparison: Three Paths

### Path A: Wait for Real Data (Conservative)

**Timeline:**
```
Month 0-6: Collect 24K-36K hours KiwiSDR (V1 MVP dataset)
Month 6-18: Collect 150K-300K hours full dataset
Month 18-19: Pattern generation + model training
Month 19: Deploy V1.0 production
```

**V1.0 at launch:**
- Shannon: 78-85%
- Quality: High
- Time to deploy: 19 months
- Cost: $5,000 (data collection + compute)

**Pros:**
- Optimal quality at launch
- No beta period
- Professional first impression

**Cons:**
- Very long wait (19 months)
- No validation until end
- High risk if architecture wrong
- No user feedback until production

---

### Path B: Synthetic Beta (Aggressive)

**Timeline:**
```
Week 0: Phase 0 vetting (validate architecture)
Week 1-4: Train on synthetic models
Week 4: Deploy V0.5 beta
Month 2-12: Continuous improvement via telemetry
Month 12: V1.0 with telemetry data
```

**V0.5 at launch (Month 1):**
- Shannon: 65-70%
- Quality: Medium
- Time to deploy: 1 month
- Cost: $1,500

**V1.0 at Month 12:**
- Shannon: 78-85% (via telemetry fine-tuning + optional real data)
- Quality: High
- Total time: 12 months (vs 19)

**Pros:**
- Fast deployment (1 month)
- Early user feedback
- Telemetry from actual usage (better than passive)
- Proves demand
- Architecture validated early

**Cons:**
- Lower initial quality (may frustrate users)
- Requires careful "beta" messaging
- Geographic bias from synthetic models
- May miss edge cases

---

### Path C: Hybrid (Balanced)

**Timeline:**
```
Week 0: Phase 0 vetting
Month 0-2: Quick collection of 5K hours real HF (from public SDRs)
Month 2-3: Train on synthetic + 5K real hybrid
Month 3: Deploy V0.8 limited release
Month 3-12: Collect telemetry + optional additional real data
Month 12: V1.0 with full dataset
```

**V0.8 at launch (Month 3):**
- Shannon: 70-75%
- Quality: Good
- Time to deploy: 3 months
- Cost: $3,000

**Pros:**
- Faster than Path A (3 months vs 19 months)
- Better quality than Path B (70-75% vs 65-70%)
- Moderate risk
- Some real data benefits

**Cons:**
- Still requires some data collection
- Not as fast as Path B
- Not as high quality as Path A

---

## Risk-Benefit Analysis

### Phase 0 Vetting (2.5 days)

**Investment:**
- Time: 2.5 days
- Cost: ~$200 (GPU rental)
- Effort: Minimal

**Return:**
- Validates 18-month investment decision
- Catches architecture flaws early
- Provides baseline performance data
- De-risks entire project

**Risk/Reward:** **Extremely favorable** (tiny cost, huge risk reduction)

**Recommendation:** **ALWAYS run Phase 0 vetting** before any major data collection

---

### Synthetic Training for Beta (3 weeks after vetting)

**Investment:**
- Time: 3 weeks training
- Cost: ~$1,500
- Effort: Moderate

**Return:**
- Deploy in Month 1 (vs Month 19)
- Get real user feedback
- Collect telemetry (better than passive)
- Validate demand
- Monthly visible improvements

**Risk/Reward:** **High risk, high reward**

**Risks:**
- Initial quality lower (65-70%)
- May frustrate early adopters
- Reputation risk if "beta" not clear
- Geographic bias

**Benefits:**
- 18× faster deployment
- Real usage telemetry
- Validates architecture in field
- Continuous improvement visible to users

**Recommendation:** **Consider if:**
- Have early adopters willing to beta test
- Can tolerate lower initial quality
- Want fast validation and telemetry
- Timeline matters (conferences, demos, funding)

**Skip if:**
- First impression critical
- No beta tolerance
- Can wait 18 months
- Want optimal launch quality

---

## Implementation Requirements

### Phase 0 Vetting Implementation

**Code needed:**
```python
# Training script
def run_vetting_phase():
    """
    Run all 7 vetting tests
    Generates report with pass/fail for each
    """

    tests = [
        test_single_user(),
        test_pattern_orthogonality(),
        test_frequency_reuse(),
        test_time_reuse(),
        test_full_chaos(),
        test_kernel_coordination(),
        test_snr_degradation(),
    ]

    generate_vetting_report(tests)

    if all_tests_pass(tests):
        print("✓ VETTING PASSED - Architecture validated")
        print("  Safe to proceed with data collection or synthetic training")
    else:
        print("✗ VETTING FAILED - Architecture needs revision")
        print("  Fix issues before proceeding")
```

**Data generation:**
- Pure Python (no real data needed)
- Synthetic CASCADE signal generation
- AWGN noise addition
- Multi-user mixing
- Ground truth labeling

**Compute requirements:**
- 1x RTX 4090 GPU
- 64 GB RAM
- 500 GB NVMe storage
- 60 hours GPU time (~$200 rental)

---

## Recommendations

### Primary Recommendation: Run Phase 0 Vetting

**Always do this before major data collection:**
- ✓ Low cost (2.5 days, $200)
- ✓ High value (validates 18-month decision)
- ✓ Fast feedback (know if architecture works)
- ✓ Risk reduction (catch issues early)

### Secondary Recommendation: Path Depends on Constraints

**If timeline critical (demos, funding, conferences):**
→ **Path B**: Vetting → Synthetic beta → Telemetry improvement

**If quality critical (professional launch, reputation):**
→ **Path A**: Vetting → Wait for real data → Optimal V1.0

**If balanced (faster than A, better than B):**
→ **Path C**: Vetting → Quick 5K hours → Hybrid training → V0.8

### My Overall Recommendation

**Phase 0 Vetting → Path C (Hybrid)**

**Rationale:**
1. Vetting validates architecture (2.5 days, essential)
2. 5K hours real data collectible quickly (2 months from public SDRs)
3. Hybrid training yields 70-75% Shannon (acceptable beta quality)
4. Deploy V0.8 at Month 3 (much faster than 19 months)
5. Telemetry improves to 78%+ by Month 6-9
6. V1.0 at Month 12 with full dataset if desired

**This balances:**
- ✓ Fast deployment (3 months)
- ✓ Acceptable quality (70-75%, not 65%)
- ✓ Risk reduction (vetting + small real data)
- ✓ Continuous improvement (telemetry-driven)

---

## Next Steps

1. **Create vetting test specification** (see `/docs/implementation/vetting_test_spec.md`)
2. **Implement data generation** (synthetic CASCADE + AWGN)
3. **Run Phase 0 vetting** (2.5 days)
4. **Review results** and decide path forward
5. **Either:**
   - Proceed with chosen path (A, B, or C)
   - Fix architecture and re-vet

---

## See Also

### Related Documentation
- **[Training Strategy](README.md)** - Overall training pipeline and phases
- **[Model Architecture](../model/README.md)** - What's being trained
- **[Pattern Architecture](../model/pattern_architecture.md)** - 128-pattern chaos system
- **[Data Pipeline](data_pipeline.md)** - Real HF data collection (Path A)
- **[Continuous Improvement](continuous_improvement.md)** - Telemetry-driven improvement (Paths B & C)

### Implementation Specifications
- **[Pattern Generation Spec](../implementation/pattern_generation_spec.md)** - Generate 128 patterns (needed for vetting)
- **[Training Data Pipeline](data_pipeline.md)** - How to generate synthetic CASCADE signals

### Architecture References
- **[Architecture Overview](../../architecture.md)** - Executive summary of what's being validated
- **[Signal Specification](../protocol/signal_specification.md)** - Physical layer parameters

---

*Document created: 2025-10-04*
*Status: Planning phase - vetting not yet implemented*
