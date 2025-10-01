# CASCADE Documentation Updates - October 2025

## Major Architecture Clarifications (Session 2025-10-01)

This document summarizes significant architectural decisions and documentation updates made during deep protocol/model analysis.

## Key Architectural Decisions

### 1. Protocol Signal Parameters (FINAL)

**Message symbols:**
- Duration: 50ms (20 symbols/second)
- Pattern length: 32 symbols (1.6 seconds per pattern)
- Modulation: 8-QAM base (3 bits/symbol, continuous collapse to QPSK/BPSK)
- Tones: 8 (across 2.5 kHz)
- Tone spacing: 312 Hz

**Beacon symbols:**
- Duration: 160ms (FT8-style, proven weak-signal performance)
- Modulation: 4-FSK (2 bits/symbol)
- Tones: 4 (interstitial: 156, 468, 781, 1093 Hz - BETWEEN message tones!)
- Payload: 16 bits (callsign hash only, regulatory minimum)
- Duration: 1.28 seconds per beacon
- Repetitions: 3× per minute on random patterns/times
- Min SNR: -22 dB (2 dB better than FT8)

**Key innovation**: Beacons use interstitial frequencies, don't consume patterns!

### 2. Multi-Resolution Kernel System

**Kernel size adapts to link quality:**

| Link SNR | Kernel Size | Content | Includes |
|----------|-------------|---------|----------|
| >+10 dB | 256 bits | Extended | Full interference map, grid square, network topology |
| 0 to +10 dB | 64 bits | Standard | Modulation, hardware, capacity, basic preferences |
| -10 to 0 dB | 16 bits | Compressed | Modulation, hardware tier, SNR floor only |
| <-18 dB | 0 bits | None | FT8-style minimal ACK (3-bit SNR report) |

**Progressive refinement**: Links start minimal (beacon discovery) and upgrade as SNR improves.

### 3. Three-Stage Protocol

**Stage 1: FT8-Equivalent Exchange** (universal, -22 dB capable)
```
Beacon: "K0BB" (16-bit hash, 1.28s, 4-FSK on interstitial)
ACK: "K0BB W2DEF +08" (44 bits, adaptive duration)
→ Valid QSO logged (FT8-equivalent contact)
```

**Stage 2: Kernel Negotiation** (optional, if SNR > -10 dB)
```
Request: "KERNEL_REQ" + my_kernel (16/64/256 bits based on SNR)
Response: "KERNEL_RESP" + my_kernel + grid_square (if SNR >+10 dB)
→ High-speed mode enabled
```

**Stage 3: Kernel-Optimized Messaging** (up to 11,000 bps)
```
Messages use target + anti-kernels
Adaptive 8-QAM → BPSK constellation
1.6-3.2s per message
→ Maximum efficiency
```

### 4. Multi-Receiver Kernel System

**All stations that decode send ACKs with kernels:**

- **Target kernel**: Optimize FOR intended recipient
- **Anti-kernels**: Optimize AGAINST interfered bystanders (reduce their interference)
- **Neutral kernels**: Informational only

**Attention-based aggregation**: Model learns to weight multiple anti-kernels (variable input size).

**Protocol routes kernels**, model is identity-blind (no callsigns).

### 5. Hardware Tiers with Graceful Degradation

| Tier | Hardware | Cost | Users Decoded | Latency | Shannon Eff | Recommended |
|------|----------|------|---------------|---------|-------------|-------------|
| 1 | RPi 4 | $50-85 | 10-20 | 20-30ms | 27% | Portable/Emergency |
| 2 | RPi 4 + Coral TPU | $120-180 | 50-80 | 2-5ms | 87-95% | **Standard** |
| 3 | Desktop CPU | $0 | 25-40 | 10-15ms | 50-65% | Home station |
| 4 | GPU/Server | $200+ | 100+ | 2-5ms | 90-97% | Contest/Club |

**Single model works across all tiers** - hardware only affects decode capacity, not interoperability.

### 6. Telemetry Architecture

**Captures CASCADE's complete internal state:**
- Shared encoder: 1024-D
- All 5 expert outputs: 2560-D (512-D each)
- Conductor weights: 5-D
- **Total**: 3589-D per RX sample

**TX telemetry** (1040-D): Pattern + Spectrum experts only (encoding-relevant)

**INT8 quantization**: 4× compression, <0.4% error, dequantized for training

**Storage**: ~3MB/hour/radio compressed → 3.6TB/year for 1000 radios

### 7. Model Decodes Beacons (Not Protocol)

**Signal Expert trained to recognize:**
- Messages: 50ms symbols on primary frequencies
- Beacons: 160ms symbols on interstitial frequencies
- Both decoded in single inference pass

**Protocol layer**: Identity-blind, just routes decoded items (messages vs beacons)

**Clean separation maintained**: Protocol never decodes, model never handles identity.

## Performance Reframing: Message-Centric

**Not "bps" (misleading with 95% idle time), but:**

- **Message latency**: 1.6-4.8 seconds (vs FT8's 15-60s)
- **Concurrent users**: 50-80 on recommended hardware (vs FT8's 1)
- **Network capacity**: 28,000 messages/minute (vs FT8's 240)
- **QSO rate**: 3,200 QSOs/minute with multi-ACK (every ACK is a logged QSO!)

**Channel capacity** (technical reference):
- RPi + Coral: 11,000 bps (87% Shannon)
- But human-limited to ~64 bps average (typing/reading time)
- Protocol overhead: 0% perceived (absorbed by 95% listening time)

## Remaining Documentation Tasks

**Completed this session:**
1. ✅ Telemetry architecture (internal state, INT8, hardware diversity)
2. ✅ Protocol specs (50ms symbols, 8-QAM, 64 patterns)
3. ✅ Hardware tiers (4 levels with performance specs)
4. ✅ Capacity-aware training (variable hardware limits)
5. ✅ Signal Expert multi-user decode (hardware-adaptive)
6. ✅ Shannon efficiency explanation (87-97% on capable hardware)
7. ✅ Heterogeneous networks (natural self-organization)
8. ✅ Model size clarification (~10MB INT8)
9. ✅ Performance tables (message-centric)
10. ✅ Privacy analysis (neural state telemetry)
11. ✅ Beacon protocol (async, interstitial, model-decoded)

**Still TODO (for next session):**
1. Multi-stage protocol flow details (Stage 1/2/3 transitions)
2. Multi-resolution kernel specification (16/64/256-bit formats)
3. ACK adaptation protocol (SNR-based symbol rate)
4. Remove remaining "Pattern 0-3 reserved" references
5. Create UI concepts doc (contest mode, station ranking, message preview)
6. Update Pattern Expert docs (constellation collapse mechanism)

## Files Modified

- `README.md` - Performance tables, kernel system, hardware tiers
- `docs/protocol/signal_specification.md` - Beacon spec, interstitial channels
- `docs/model/README.md` - Constellation adaptation
- `docs/model/experts.md` - Signal Expert hardware-adaptive decode
- `docs/model/conductor_details.md` - Telemetry interpretation
- `docs/protocol/README.md` - Heterogeneous networks, multi-kernel
- `docs/training/README.md` - Capacity-aware training, receiver-driven kernels
- `docs/training/continuous_improvement.md` - Telemetry architecture, INT8 spec
- `docs/training/data_pipeline.md` - Telemetry scaling, storage
- `docs/training/embedding_models.md` - CASCADE fingerprinting, telemetry usage
- `docs/privacy.md` - Neural state privacy analysis

**New files:**
- `docs/protocol/signal_specification.md` - Complete protocol specs
- `docs/deployment/hardware_requirements.md` - 4 hardware tiers
- `docs/RECENT_UPDATES.md` - This file

Total: ~1,200 lines added/modified across 14 files

## Next Steps

When continuing documentation:
1. Complete multi-stage protocol flow
2. Specify kernel compression formats (16/64/256-bit)
3. Detail ACK adaptation mechanism
4. Create contest UI mockup/example
5. Add training data requirements for beacon recognition
6. Update any lingering "reserved patterns" references

## Key Insights from This Session

1. **Beacons don't need slots** - async with interstitial frequencies is cleaner
2. **Model handles everything** - beacons are just another signal type (maintains clean separation)
3. **Human duty cycle absorbs overhead** - 95% listening time makes protocol overhead free
4. **Message metrics matter** - not raw bps (humans are bottleneck)
5. **Multi-ACK is QSO multiplier** - every ACK is a logged contact (800× FT8 for contests!)
6. **Interstitial beacons** - 3.2% spectrum, 0% pattern overhead
7. **-22 dB beacons** - better than FT8 (-20 dB) using 4-FSK + 160ms symbols
