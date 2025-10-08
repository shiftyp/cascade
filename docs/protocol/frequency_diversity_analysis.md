# Frequency Diversity Analysis: Multi-Carrier Repetition

**Question:** What if we transmit the SAME pattern with SAME IQ data simultaneously across multiple frequency pairs (2, 4, or 8 pairs)?

**Key insight:** CASCADE patterns have flip-orthogonality, so adjacent/nearby carriers don't interfere even when using same pattern!

---

## Current System (Single Pair)

**Configuration:**
- 1 frequency pair (e.g., tones at 1500 Hz + 1520 Hz)
- Symbol rate: 200 symbols/second (5ms per symbol)
- Bandwidth per pair: ~40 Hz (with GMSK BT=0.3)
- Capacity: 8 patterns × 67 pairs = 536 logical channels

**Shannon capacity (per pair):**
```
C_single = B * log2(1 + SNR)

At SNR = -21 dB (0.0079 linear):
C = 40 * log2(1.0079) ≈ 0.45 bits/second

At SNR = 0 dB (1.0 linear):
C = 40 * log2(2) = 40 bits/second

At SNR = +10 dB (10 linear):
C = 40 * log2(11) ≈ 138 bits/second
```

**Current throughput (512s pattern @ QPSK + Polar 2/3):**
```
512 symbols × 2 bits/symbol × 2/3 = 683 bits in 2.56s = 267 bps
Symbol rate: 200 sym/s = 200 bits/s raw (BPSK)
```

---

## Multi-Carrier Diversity (N-Pair Simultaneous Transmission)

**Configuration:**
- Transmit SAME pattern + SAME IQ data on N frequency pairs **simultaneously**
- All pairs transmit in parallel (not time-multiplexed)
- Symbol rate: 200 symbols/second (unchanged)
- Receiver combines signals from all N pairs
- Total bandwidth: N × 40 Hz

### Key Advantage: Flip-Orthogonal Patterns

**CASCADE patterns are orthogonal even when flipped!**
- Adjacent frequency pairs can use same pattern without interference
- Example: Pairs 20, 21, 22, 23 transmit same pattern simultaneously
- Flip-orthogonality handles inter-carrier interference
- No guard bands needed between diversity carriers

### Shannon Capacity with Maximal Ratio Combining (MRC)

**For diversity combining (same information on N independent fading channels):**
```
C_diversity = B * log2(1 + N × SNR)

Where:
- B = 40 Hz (single pair bandwidth, not N×B!)
- N × SNR = combined SNR from N diversity branches
- Information bandwidth is B (not N×B) because same data on all carriers
```

**Wait, this needs clarification:**

Shannon's theorem for diversity combining:
- If we have N parallel channels with independent fading
- Each with bandwidth B and average SNR
- Transmitting the SAME signal on all N
- Optimal combining gives: C = B × log2(1 + N×SNR)

But we're using N×B total RF bandwidth for B of information bandwidth.

**Effective Shannon limit (accounting for bandwidth usage):**
```
C_effective = (N×B) × log2(1 + N×SNR) / N
            = B × log2(1 + N×SNR)

So spectral efficiency = C / (N×B) = log2(1 + N×SNR) / (N×B)

Compare to single channel:
Spectral efficiency_single = log2(1 + SNR) / B

Diversity uses N× bandwidth but only increases capacity logarithmically.
```

### Numerical Analysis: SNR Gain vs Bandwidth Cost

**2-pair diversity (N=2):**
```
SNR gain: +3 dB (2×)
Bandwidth: 80 Hz (2× cost)

At SNR_single = -21 dB (0.0079):
C = 40 * log2(1 + 2×0.0079) = 40 * log2(1.0158) ≈ 0.91 bps
Spectral eff: 0.91/80 = 0.011 bits/s/Hz

Single pair for comparison:
C = 40 * log2(1.0079) ≈ 0.45 bps
Spectral eff: 0.45/40 = 0.011 bits/s/Hz (SAME!)

At SNR_single = 0 dB (1.0):
C = 40 * log2(1 + 2×1.0) = 40 * log2(3) ≈ 63 bps
Spectral eff: 63/80 = 0.79 bits/s/Hz

Single: 40/40 = 1.0 bits/s/Hz (BETTER!)
```

**4-pair diversity (N=4):**
```
SNR gain: +6 dB (4×)
Bandwidth: 160 Hz (4× cost)

At SNR_single = -21 dB (0.0079):
C = 40 * log2(1 + 4×0.0079) = 40 * log2(1.0316) ≈ 1.82 bps
Spectral eff: 1.82/160 = 0.011 bits/s/Hz (SAME as single!)

At SNR_single = 0 dB:
C = 40 * log2(1 + 4×1.0) = 40 * log2(5) ≈ 93 bps
Spectral eff: 93/160 = 0.58 bits/s/Hz (WORSE than single!)
```

**8-pair diversity (N=8):**
```
SNR gain: +9 dB (8×)
Bandwidth: 320 Hz (8× cost)

At SNR_single = -21 dB (0.0079):
C = 40 * log2(1 + 8×0.0079) = 40 * log2(1.0632) ≈ 3.62 bps
Spectral eff: 3.62/320 = 0.011 bits/s/Hz (SAME!)

At SNR_single = 0 dB:
C = 40 * log2(1 + 8×1.0) = 40 * log2(9) ≈ 127 bps
Spectral eff: 127/320 = 0.40 bits/s/Hz (MUCH WORSE!)
```

**Key finding:** Spectral efficiency is constant at very low SNR, but degrades at higher SNR!

---

## Symbol Rate Implications

### Could we increase symbol rate with diversity gain?

**Current: 200 symbols/second (5ms per symbol)**

**Hypothesis:** With +6 dB SNR from 4× diversity, could we double symbol rate to 400 sym/s?

**Limiting factors:**
1. **HF multipath delay spread: 1-5ms typical**
   - Symbol duration currently 5ms (matches upper limit)
   - At 400 sym/s: 2.5ms symbol duration
   - **Risk**: ISI from multipath, would need equalization
   - SNR gain doesn't help with ISI (different problem!)

2. **Tone spacing: 20 Hz (fixed by 135-tone grid)**
   - At 400 sym/s: GMSK occupies ~80 Hz per pair
   - Need 80 Hz minimum spacing between pairs
   - Only 33 pairs fit in 2.7 kHz (vs 67 currently)
   - **Lost capacity from fewer pairs!**

3. **Network capacity calculation:**
   ```
   Current (200 sym/s, no diversity):
   8 patterns × 67 pairs = 536 channels

   With 400 sym/s + 4× diversity:
   8 patterns × (33 pairs / 4 diversity) = 66 channels

   Net: 8× capacity loss!
   Active users: 5-6 (vs 40-45 currently)
   ```

**Verdict:** Increasing symbol rate with diversity is **counterproductive**. Multipath and tone spacing are hard limits, not SNR limits.

---

## Practical Application: Adaptive Diversity Mode

**Keep symbol rate at 200 sym/s, use diversity adaptively:**

### Mode 1: Normal (Current - No Diversity)
```
Pairs: 1
Symbol rate: 200 sym/s
Bandwidth: 40 Hz
Channels: 8 × 67 = 536
Active users: 40-45
Min SNR: -21 dB (BPSK + Polar 1/2 + 2048s)
Shannon limit: 0.45 bps @ -21 dB
```

### Mode 2: 2× Diversity
```
Pairs: 2 adjacent (e.g., pairs 20-21)
Symbol rate: 200 sym/s (unchanged)
Bandwidth: 80 Hz total
Channels: 8 × (67/2) ≈ 268
Active users: 20-22
Min SNR: -24 dB (+3 dB gain)
Shannon limit: 0.91 bps @ -21 dB SNR_single
Use: Moderate fading, DX paths
```

### Mode 3: 4× Diversity
```
Pairs: 4 adjacent (e.g., pairs 20-23)
Symbol rate: 200 sym/s (unchanged)
Bandwidth: 160 Hz total
Channels: 8 × (67/4) ≈ 134
Active users: 10-12
Min SNR: -27 dB (+6 dB gain)
Shannon limit: 1.82 bps @ -21 dB SNR_single
Use: Severe fading, emergency traffic
```

### Mode 4: 8× Diversity (Extreme)
```
Pairs: 8 adjacent (e.g., pairs 20-27)
Symbol rate: 200 sym/s (unchanged)
Bandwidth: 320 Hz total
Channels: 8 × (67/8) ≈ 67
Active users: 5-6
Min SNR: -30 dB (+9 dB gain)
Shannon limit: 3.62 bps @ -21 dB SNR_single
Use: Extremely poor conditions, last resort
```

**Flip-orthogonality enables adjacent carriers:** No guard bands needed, patterns handle inter-carrier interference!

---

## Recommendations

### When to Use Diversity

**Normal operations (SNR > -21 dB): NO DIVERSITY**
- Use independent frequency pairs
- Maximize user capacity (40-45 users)
- Shannon capacity adequate

**Marginal conditions (SNR -24 to -21 dB): 2× DIVERSITY**
- 50% capacity reduction (20-22 users)
- +3 dB SNR improvement
- Improves link reliability

**Poor conditions (SNR -27 to -24 dB): 4× DIVERSITY**
- 75% capacity reduction (10-12 users)
- +6 dB SNR improvement
- Emergency/critical messages

**Extreme conditions (SNR < -27 dB): 8× DIVERSITY**
- 87% capacity reduction (5-6 users)
- +9 dB SNR improvement
- Last-resort communications

### Adaptive Protocol

**Kernel negotiates diversity mode:**
```python
rx_kernel = {
    'pattern_id': 3,
    'frequency_pair': 20,  # Base pair
    'diversity_mode': 4,    # Use 4× (pairs 20-23)
    'modulation': 'BPSK',
    'polar_rate': 1/2,
    ...
}
```

**Automatic escalation:**
1. Try single pair (normal mode)
2. If fails: Retry with 2× diversity
3. If fails: Retry with 4× diversity
4. If fails: 8× diversity or abort

---

## Summary

**Shannon Limit Analysis:**
- Diversity trades bandwidth for SNR robustness
- At very low SNR (< -20 dB): Spectral efficiency constant regardless of N
- At moderate SNR (> -10 dB): Independent channels far more efficient
- **Conclusion:** Use diversity only when SNR is critically low

**Symbol Rate:**
- Cannot increase beyond 200 sym/s (HF multipath limit)
- SNR gain from diversity doesn't help with ISI
- Faster symbols would reduce frequency pairs, negating capacity gains

**Best Use Case:**
- **Adaptive diversity for emergency/poor conditions only**
- Normal: 1 pair, 536 channels, -21 dB min SNR
- Emergency: 4 pairs, 134 channels, **-27 dB min SNR** (+6 dB)
- Kernel-negotiated, automatic fallback

**Key Insight:** CASCADE's flip-orthogonal patterns make adjacent-carrier diversity practical without guard bands!

---

*Analysis completed: 2025-10-07*
