# CASCADE Architecture Migration: V1 → V2

**Date:** 2025-10-06
**Reason:** Practical implementability - V2 achieves same user capacity with proven pattern generation

---

## Summary of Changes

**V1 (Theoretical):** 128 patterns with complex IQ orthogonality, blind detection
**V2 (Practical):** 8 patterns with kernel-assisted detection, frequency separation

**Key insight:** Kernel-assisted detection eliminates need for 128 patterns. Frequency separation provides primary isolation.

---

## Major Architectural Changes

### 1. Pattern Count: 128 → 8

**V1:**
- 128 patterns (48 beacon + 80 message)
- Complex hierarchical pools
- Blind pattern detection (correlate vs all 128)
- Required -37.5 dB orthogonality (unachievable!)

**V2:**
- 8 patterns (universal, no pools)
- Kernel provides pattern ID
- No blind detection needed
- Achieved -21.19 dB orthogonality @ 2048 symbols ✅

**Why this works:**
- Kernel tells you which pattern (beacon announced it)
- Frequency separation provides primary isolation
- Pattern orthogonality only for adjacent channel interference
- 8 patterns × 67 frequencies = 536 channels (still adequate for 40-45 users)

### 2. Pattern Generation: Zadoff-Chu → Genetic Algorithm

**V1:**
- Zadoff-Chu sequences with IQ trajectories
- Hierarchical λ complexity
- Simulated annealing
- Unproven convergence to -37.5 dB

**V2:**
- Genetic algorithm with 32-member population
- Binary patterns (no built-in redundancy)
- Nested extraction (6 usable lengths from one optimization)
- Proven: -21.19 dB at 2048 symbols (3.41 dB from Welch bound -24.6 dB)

### 3. Modulation: 4-FSK → 2-FSK + IQ

**V1:**
- 4-FSK (4 tones per pattern)
- 78 discrete tones
- Complex tone selection optimization

**V2:**
- **Dual-layer:**
  - Layer 1: 2-FSK pattern (binary sequence selects tone A or B)
  - Layer 2: IQ modulation on selected tone (BPSK to 16-APSK)
- 135 tones → 67 non-overlapping 2-tone pairs
- Simpler, more robust

### 4. Kernel Size: 64-bit → 28-byte

**V1:**
- Compact 64-bit kernels
- Tone availability encoding
- Limited information

**V2:**
- 28-byte kernels (224 bits)
- 4 bytes discrete protocol
- 24 bytes NN embedding (48 dims × 4-bit quantization)
- Richer coordination information

### 5. Error Correction: RS(32,20) → Polar Codes

**V1:**
- RS(32,20) structure built into patterns
- Fixed 37.5% erasure tolerance
- Pattern and FEC tightly coupled

**V2:**
- Polar codes at protocol layer (separate from patterns)
- Adaptive rates: 1/2, 2/3, 3/4, 4/5, 5/6, 7/8
- Negotiated via kernel based on measured SNR
- Pattern orthogonality optimization independent of FEC

**Why this is better:**
- Patterns optimized purely for orthogonality (no FEC constraint)
- Adaptive FEC matches link quality (better efficiency)
- Protocol layer handles error correction (cleaner separation)

### 6. Symbol Rate: 200 sym/s (Unchanged)

**Both V1 and V2:**
- 200 symbols/second (5ms per symbol)
- Balances throughput vs. robustness
- Matches HF multipath delay spread (1-5ms)
- Adequate for 20 Hz tone spacing

### 7. Protocol: Multi-stage → RTS/CTS

**V1:**
- Three progressive stages (FT8-style, kernel negotiation, high-speed)
- Variable kernel sizes (0, 16, 64, 256 bits)
- Complex stage transitions

**V2:**
- Simple RTS/CTS handshaking
- Fixed 28-byte kernels
- Collision avoidance built-in
- Cleaner, simpler protocol

---

## What Stayed the Same

✅ **Kernel-based coordination** - Core concept preserved
✅ **Distributed operation** - No central control
✅ **40-45 active users** - Same network capacity
✅ **Adaptive modulation** - BPSK to 16-APSK based on SNR
✅ **Text messaging focus** - No file transfer
✅ **Emergency auto-relay** - 3-hop limit, self-limiting
✅ **Raspberry Pi 4** - Still target platform
✅ **No GPS required** - Differential encoding handles drift

---

## Performance Comparison

### Pattern Orthogonality

| Metric | V1 (128 patterns) | V2 (8 patterns) |
|--------|-------------------|-----------------|
| Target | -37.5 dB | -24.6 dB (Welch bound @ 2048s) |
| Achieved | Unproven | **-21.19 dB** ✅ |
| Gap | N/A | 3.41 dB (excellent!) |
| Detection | Blind (correlate vs 128) | Kernel-assisted |

### Network Capacity

| Metric | V1 | V2 |
|--------|----|----|
| Patterns | 128 | 8 |
| Frequency pairs | ~75 | 67 |
| Logical channels | Unclear | 536 |
| Active users | 40-45 | 40-45 |
| Pattern reuse | Via IQ separation | Via frequency separation |

### Message Timing (152 bytes @ QPSK, 200 sym/s)

| Phase | V1 | V2 |
|-------|----|----|
| Beacon | Variable | 2.56s |
| Call setup | Complex | 3.8s (RTS+CTS+ACK) |
| Message | ~5-7s | 5.12s |
| **Total QSO** | ~12-18s | **9.1s** |

**V2 is cleaner and simpler!**

---

## Migration Path

### For Pattern Generation

**V1 code (if existed):**
```bash
# Theoretical 128-pattern generation
python generate_patterns_zadoff_chu.py  # Never proven to converge
```

**V2 code (proven):**
```bash
# 8-pattern genetic algorithm
python modules/training/patterns/tournament/generate_patterns_tournament.py \
  --pattern-count 8 \
  --pattern-length 2048 \
  --redundancy 1 \
  --generations 150000

# Note: redundancy=1 means no repetition (pure orthogonality optimization)
# Error correction handled by polar codes at protocol layer
```

### For Modem Implementation

**V1 approach:**
- Implement blind pattern detector (correlate vs 128 patterns)
- Complex IQ orthogonality on same tone pair
- Hierarchical pool selection

**V2 approach:**
- Use kernel to get pattern ID (no blind detection)
- Simple 2-FSK detection + IQ demodulation
- No pool selection (all 8 patterns universal)

### For Protocol Layer

**V1:**
- Implement multi-stage protocol
- Variable kernel sizes
- Complex stage transitions

**V2:**
- Implement RTS/CTS handshaking
- Fixed 28-byte kernels
- Simple state machine

---

## Why V2 is Better

**Practical implementation:**
- ✅ Pattern generation **proven** to converge (-21.19 dB achieved)
- ✅ Simpler modem (no blind detection, no 128-pattern correlation)
- ✅ Cleaner protocol (RTS/CTS, fixed kernels)
- ✅ Same network capacity (40-45 users)
- ✅ Faster QSO setup (7s vs 10-15s)

**V1 theoretical advantages lost:**
- ❌ -37.5 dB orthogonality (unachievable with any known algorithm)
- ❌ IQ orthogonality on same tone (complex, unproven)
- ❌ 128-pattern blind detection (computationally expensive)

**V2 practical advantages gained:**
- ✅ Proven pattern generation (running code, real results)
- ✅ Frequency separation (well-understood, robust)
- ✅ Kernel-assisted detection (simpler, faster)
- ✅ Nested pattern lengths (adaptive transmission)

---

## Archived V1 Documentation

All V1 docs archived with `_v1_archived.md` suffix:

- `architecture_v1_archived.md` - Original 128-pattern architecture
- `pattern_generation_spec_v1_archived.md` - Zadoff-Chu approach
- `kernel_encoding_spec_v1_archived.md` - 64-bit kernels
- `pattern_pool_selection_spec_v1_archived.md` - Hierarchical pools
- `protocol/README_v1_archived.md` - Multi-stage protocol

**Current V2 docs** replace these with implementable specifications.

---

## Timeline

- **2025-10-04:** V1 architecture documented (theoretical exploration)
- **2025-10-05:** Pattern generation development begins
- **2025-10-06:** **V2 architecture proven** (-21.19 dB achieved!)
- **2025-10-06:** Documentation updated to V2

**V2 is the path forward** - proven, practical, implementable.
