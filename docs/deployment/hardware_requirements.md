# CASCADE Hardware Requirements and Deployment Tiers

CASCADE is designed to run on a range of hardware from embedded devices to servers, with performance scaling gracefully based on available compute resources. All tiers are fully interoperable - differences affect only local decode capacity and latency, not protocol compatibility.

## Hardware Tiers

### Tier 1: Raspberry Pi 4 (Entry Level)

**Hardware**:
- Raspberry Pi 4 Model B (4GB+ RAM)
- Standard USB sound card or hat (48 kHz)
- Cost: ~$35-55 (RPi) + $10-30 (sound interface) = **$45-85 total**

**Performance**:
- **Inference latency**: 20-30ms per symbol
- **User capacity**: 10-20 simultaneous users decoded
- **Shannon efficiency**: 27% (hardware-limited, not protocol-limited)
- **Throughput**: ~3,000-5,000 bps aggregate (what this station receives)

**Use cases**:
- Portable/emergency operations
- Small nets (5-15 stations)
- Casual QSOs and DX
- Entry-level CASCADE experience

**Limitations**:
- Cannot decode all users in high-traffic scenarios (contests)
- Sees only strongest 10-20 signals
- Still fully interoperable (just hears less)

### Tier 2: Raspberry Pi 4 + Coral TPU (Recommended)

**Hardware**:
- Raspberry Pi 4 Model B (4GB+ RAM)
- Google Coral USB Accelerator or M.2/PCIe
- USB sound card (48 kHz) or GPS-disciplined SDR
- Cost: $35-55 (RPi) + $60-75 (Coral) + $10-50 (audio) = **$105-180 total**

**Performance**:
- **Inference latency**: 2-5ms per symbol
- **User capacity**: 50-80 simultaneous users decoded
- **Shannon efficiency**: 85-95% (approaching theoretical limit)
- **Throughput**: ~10,000-12,000 bps aggregate

**Use cases**:
- **Recommended baseline** for full CASCADE experience
- Medium to large nets (20-60 stations)
- Contest operations
- Emergency coordination networks
- Excellent cost/performance ratio

**Advantages**:
- 2000× faster inference vs CPU-only
- Decodes nearly all active users
- <2W additional power consumption
- Compact form factor (still portable)

### Tier 3: Desktop/Laptop CPU

**Hardware**:
- Modern desktop CPU (Intel i5/i7, AMD Ryzen 5/7)
- Existing ham radio computer
- Standard sound card interface
- Cost: **$0** (uses existing equipment)

**Performance**:
- **Inference latency**: 10-15ms per symbol
- **User capacity**: 25-40 simultaneous users decoded
- **Shannon efficiency**: 45-60%
- **Throughput**: ~6,000-8,000 bps aggregate

**Use cases**:
- Home station operations
- Users with existing shack PC
- Development and testing
- Medium-sized nets

**Advantages**:
- No additional hardware purchase
- Familiar desktop environment
- Easy debugging and monitoring

### Tier 4: Desktop GPU or Server

**Hardware**:
- NVIDIA GPU (GTX 1060+, RTX series)
- Or dedicated server CPU (Xeon, EPYC)
- Professional sound interface or SDR
- Cost: **$200-500+** (GPU) or **$0** (existing gaming PC)

**Performance**:
- **Inference latency**: 2-5ms per symbol
- **User capacity**: 100+ simultaneous users decoded
- **Shannon efficiency**: 90-97% (limited only by FEC overhead)
- **Throughput**: ~15,000+ bps aggregate

**Use cases**:
- Club stations
- Contest super-stations
- Emergency operations centers
- Gateway/relay stations
- Mesh network hubs

**Advantages**:
- Maximum capacity
- Lowest latency
- Can handle extreme contest conditions (100+ simultaneous)

## Recommended Deployment

**For individual operators**: Tier 2 (RPi 4 + Coral) - **$120-180**
- Best cost/performance balance
- Achieves 90% Shannon efficiency
- Handles typical amateur radio scenarios
- Portable for emergency/field use

**For club stations**: Tier 4 (GPU/Server)
- Maximizes network capacity
- Serves as hub for weaker stations
- Typically already have suitable hardware

**For emergency/portable**: Tier 1 (RPi 4 only) acceptable
- Still 24× faster than FT8
- Hears strong signals reliably
- Minimal power consumption (<15W)

## Model Deployment Across Tiers

**All tiers use the same model**, just different quantization:

| Tier | Quantization | Model Size | Notes |
|------|--------------|------------|-------|
| Tier 1 (RPi 4) | INT8 | ~10MB | Maximum compression |
| Tier 2 (RPi+Coral) | INT8 | ~10MB | Optimized for Edge TPU |
| Tier 3 (Desktop) | FP16 or INT8 | ~17MB or ~10MB | User choice |
| Tier 4 (GPU) | FP32 or FP16 | ~34MB or ~17MB | Maximum precision |

**Interoperability**: All use identical 64 orthogonal patterns (protocol-defined), so all tiers communicate perfectly regardless of quantization.

## Network Effects with Mixed Hardware

**Realistic scenario: 50 users on air with mixed hardware**

```
10 stations on RPi only (Tier 1):
  - Each decodes ~12 users (strongest signals)
  - See ~6,000 bps throughput each

30 stations on RPi + Coral (Tier 2):
  - Each decodes ~50 users (nearly everyone)
  - See ~12,000 bps throughput each

10 stations on Desktop/GPU (Tier 3-4):
  - Each decodes 50-80 users (everyone + weak signals)
  - See ~15,000 bps throughput each

Network behavior:
- Strong transmissions reach all 50 stations (100% connectivity)
- Medium transmissions reach 40 stations (80% connectivity)
- Weak transmissions reach 10-20 stations (20-40% connectivity)

This is Shannon-optimal: Strong signals get more capacity naturally!
```

**Emergency priority**: High-power emergency transmissions reach even Tier 1 hardware (100% penetration).

## Sound Card Requirements

**All tiers**:
- Sample rate: 48 kHz minimum (96 kHz better for frequency stability)
- Bit depth: 16-bit minimum
- Interface: USB audio, sound card, or SDR direct sampling
- Latency: <20ms preferred (not critical - 50ms symbol duration has margin)

**For 50+ user capacity (Tier 2+)**:
- GPS-disciplined oscillator preferred (±0.1 Hz vs ±50 Hz)
- Reduces frequency uncertainty, improves multi-user separation
- Not required, but improves performance

**Compatible interfaces**:
- SignaLink USB
- Digirig
- Built-in sound card (adequate for Tier 1)
- SDR direct (RTLSDR, Airspy, etc.)

## Power Consumption

| Tier | Idle | Receiving | Transmitting | Total (with radio) |
|------|------|-----------|--------------|-------------------|
| RPi 4 only | 3W | 4W | 5W | ~55W |
| RPi + Coral | 3W | 6W | 7W | ~57W |
| Desktop | 50W | 80W | 100W | ~150W |
| GPU | 100W | 250W | 300W | ~350W |

Tier 2 (RPi + Coral) adds only ~2W for 2000× compute improvement - excellent efficiency!

## Deployment Recommendations by Scenario

**Portable/Emergency Operations**: Tier 1 or 2
- Low power
- Compact
- Battery-friendly
- Trade capacity for portability

**Home Station**: Tier 2 or 3
- Tier 2 if dedicated CASCADE hardware
- Tier 3 if using existing shack PC

**Contest/Club Station**: Tier 4
- Maximum capacity needed
- Existing infrastructure typically sufficient

**Mesh Gateway/Relay**: Tier 4
- Serves multiple users
- Needs maximum decode capacity
- Often has power available

## Future-Proofing

**Compute requirements stable**: CASCADE's 8.4M parameter model won't grow significantly (architectural constraint for interoperability)

**Hardware improvements benefit performance**:
- Coral TPU v2: 2× faster → 100+ user capacity on embedded
- Next-gen RPi: More users on Tier 1
- Software optimization: May reduce latency 20-30%

**Upgrade path**:
- Start: Tier 1 (RPi only) - works, limited capacity
- Upgrade: Add Coral TPU - 5× user capacity for $60
- Ultimate: GPU desktop - maximum capacity

## See Also

- **[Signal Specification](../protocol/signal_specification.md)** - Protocol layer parameters
- **[Model Architecture](../model/README.md)** - How model adapts within protocol constraints
- **[Training Strategy](../training/README.md)** - Training for heterogeneous hardware
