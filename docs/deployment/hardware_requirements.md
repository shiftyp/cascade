# CASCADE Hardware Requirements (V2)

CASCADE V2 is designed to run on **minimal hardware** thanks to kernel-assisted detection eliminating the need for blind pattern correlation.

**Key simplification:** Kernel provides pattern ID, so no expensive 128-pattern correlation needed. Raspberry Pi 4 CPU-only is sufficient!

---

## Recommended Configuration

### Raspberry Pi 4 (CPU-Only)

**Hardware:**
- Raspberry Pi 4 Model B (2GB+ RAM sufficient)
- USB sound card (44.1-48 kHz sampling)
- Cost: $50 (RPi) + $15 (sound card) = **$65 total**

**Performance:**
- **Decoding**: Kernel-assisted (correlate vs 1 pattern, not 8)
- **Encoding**: <5ms (select pattern, encode IQ data)
- **User capacity**: 40-45 simultaneous users
- **Power**: 8W (ultra-portable, 12+ hours on 100Wh battery)

**Use cases:**
- Full CASCADE experience
- Portable/emergency operations
- Nets and contests
- All message types (beacon to large messages)

**Why CPU-only works:**
- Kernel tells you which pattern (no blind search)
- Simple 2-FSK detection + IQ demodulation
- No expensive correlation matrix
- 8 patterns (not 128) if fallback needed

---

## Alternative Platforms

### Desktop/Laptop

**Hardware:**
- Modern x86 CPU (4+ cores)
- Built-in or USB sound card
- Cost: $0 (use existing computer)

**Performance:**
- Faster than RPi4 (more cores, higher clock)
- Can handle 100+ users if needed
- Overkill for CASCADE but works great

**Use cases:**
- Development
- High-capacity nets
- Contest operations
- Base station

### Embedded Linux (Advanced)

**Hardware:**
- Orange Pi, Rock Pi, or similar SBC
- 1GB+ RAM
- Sound interface
- Cost: $30-60

**Performance:**
- Varies by platform
- Most ARM64 boards adequate
- Test before deployment

---

## Radio Requirements

### SSB Transceiver

**Requirements:**
- **SSB mode** (USB or LSB, 2.7 kHz bandwidth)
- **Sound card interface** (CAT control optional but helpful)
- **Power**: 5W minimum (QRP works!), 50-100W recommended

**Examples:**
- QRP: QMX, (tr)uSDX ($50-200, 5W)
- Modern: IC-7300, FT-991A ($800-1200, 100W)
- Classic: FT-857D, IC-706 ($400-600 used, 100W)
- Any SSB-capable rig works!

**No GPS required:**
- Differential encoding handles ±10 Hz drift
- Sound card clock sufficient
- GPS optional for precision timing

### Sound Card Interface

**Minimum specs:**
- 44.1 kHz sampling (48 kHz also fine)
- 16-bit resolution
- Stereo (I/Q if SDR, or dual mono for rig interface)
- <50ms latency

**Options:**
- USB sound card: $10-30 (Behringer UCA202, etc.)
- SignaLink USB: $120 (integrated interface, plug-and-play)
- DigiRig: $80 (modern, compact)
- Built-in sound card: $0 (works for testing)

---

## Software Requirements

### Operating System

**Recommended:**
- Raspberry Pi OS (64-bit) on RPi4
- Ubuntu 22.04+ on desktop
- Any modern Linux works

**Also supported:**
- macOS (development)
- Windows via WSL2 (development)

### Dependencies

**Python 3.11+:**
```bash
pip install numpy scipy psutil rich pyyaml
```

**Optional:**
```bash
pip install sounddevice  # Audio I/O
pip install pyaudio      # Alternative audio
```

**No deep learning framework needed!**
- No TensorFlow/PyTorch at runtime
- Pattern selection via lookup table
- IQ encoding via simple math
- Decoding via kernel-assisted correlation

---

## Performance by Hardware

### Pattern Detection (Kernel-Assisted)

| Hardware | Detection Time | Users Supported | Power |
|----------|----------------|-----------------|-------|
| RPi 4 | <5ms | 40-45 | 8W |
| Desktop i5 | <2ms | 100+ | 35W |
| Desktop i7 | <1ms | 200+ | 65W |

**All adequate** - V2 is not compute-intensive!

### Encoding

**All platforms:** <1ms
- Select pattern from kernel
- Encode IQ data (simple constellation mapping)
- No NN inference needed

---

## Network Scaling

### Single Station

**With RPi4:**
- Decode: 40-45 users
- Encode: Unlimited (trivial computation)
- Transmit: 1 pattern typical, up to 8 for high-power stations

### Multiple Stations (Mesh)

**Network capacity scales with stations:**
- 10 stations: 40-45 active users total (each decodes all)
- 50 stations: 40-45 active users total (distributed decode)
- 100+ stations: 40-45 active users total (high redundancy)

**Decoding is distributed:**
- Each station decodes what it can hear
- Strong signals decoded by all
- Weak signals decoded by nearby/powerful stations
- Mesh routing finds paths

---

## Cost Comparison

| Configuration | Cost | Power | Users | Use Case |
|--------------|------|-------|-------|----------|
| **RPi4 + Sound** | **$65** | **8W** | **40-45** | **Recommended** |
| QMX + RPi4 | $250 | 13W | 40-45 | Portable QRP |
| Desktop (existing) | $15 | 35W | 100+ | Base station |
| IC-7300 + RPi4 | $865 | 28W | 40-45 | Modern shack |

**V2 dramatically reduces hardware requirements vs V1:**
- V1: Required Coral TPU ($60) for 128-pattern blind detection
- V2: CPU-only sufficient with kernel-assisted detection
- Savings: $60 + simpler software

---

## Storage Requirements

### Pattern Files

**Generated once, used forever:**
```
patterns_p8_l2048_r2x.pkl: ~50 KB
```

Negligible storage - patterns fit in RAM.

### Message Database

**For 1 year of operation:**
- 100 messages/day × 365 days × 200 bytes avg = 7.3 MB
- SQLite database adequate
- No large storage needed

### Logs (Optional)

**IQ recordings for debugging:**
- 1 hour @ 12 kHz = ~90 MB (compressed)
- Typically not needed in production
- Optional for troubleshooting

---

## Power Consumption

### Portable Operation

| Component | Power | Battery Life (100Wh) |
|-----------|-------|---------------------|
| RPi 4 | 8W | 12.5 hours |
| USB sound | 0.5W | - |
| QMX (5W TX, 50% duty) | 2.5W avg | - |
| **Total** | **11W** | **9 hours** |

### Base Station

| Component | Power | Notes |
|-----------|-------|-------|
| Desktop PC | 35W | Idle, spikes during TX |
| IC-7300 (50W TX) | 25W avg | 50% duty cycle |
| **Total** | **60W** | Easily powered from mains |

---

## Deployment Scenarios

### Emergency/Portable (RPi4 + QMX)

**Configuration:**
- RPi4 + QMX + battery
- Weight: <2 lbs
- Power: 11W (9 hours on 100Wh)
- Cost: $250

**Performance:**
- 5W transmit (QRP)
- 40-45 user network
- Full protocol support

### Home Station (Desktop + Modern Rig)

**Configuration:**
- Desktop PC (existing) + IC-7300
- Always-on base station
- AC powered

**Performance:**
- 100W transmit
- 100+ user decode capacity
- Net control capable

### Mobile (Laptop + FT-857D)

**Configuration:**
- Laptop + FT-857D in vehicle
- 12V powered
- Mobile operations

**Performance:**
- 50-100W transmit
- Full CASCADE capability
- Contest/field day ready

---

## No Special Hardware Needed!

**V2 eliminates:**
- ❌ Coral Edge TPU ($60)
- ❌ GPU ($200-800)
- ❌ GPS receiver ($30-80)
- ❌ High-end CPU
- ❌ Large RAM

**V2 runs on:**
- ✅ Raspberry Pi 4 CPU ($ 50)
- ✅ Any SSB transceiver
- ✅ Basic USB sound card ($15)
- ✅ **Total: $65** for full implementation!

---

## Archived Documentation

**V1 (Coral TPU required):** See `hardware_requirements_v1_archived.md`

V1 required Coral TPU for blind 128-pattern detection.
V2 uses kernel-assisted detection, runs on CPU-only.
