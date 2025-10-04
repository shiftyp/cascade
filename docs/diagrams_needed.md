# CASCADE Visualization & Diagram Specifications

**Purpose:** Specifications for generating visual diagrams to accompany CASCADE documentation

**Status:** Updated for 128-pattern chaos architecture (Oct 4, 2025)

---

## High Priority Diagrams

### 1. 4D Pattern Trajectory (Time × Frequency × I × Q)

**Placement:** `docs/model/pattern_architecture.md` - Section "4D Mathematical Foundation"

**Purpose:** Illustrate CASCADE's core 4-dimensional pattern concept

**Specification:**
```
Three-panel technical diagram showing how patterns exist in 4D space:

PANEL 1 - "Time × Frequency (Discrete Hopping)":
- X-axis: Time (0 to 1.6s), labeled every 0.4s
- Y-axis: Frequency (300-2800 Hz)
- Show 32 discrete tone hops as connected dots
- Example pattern hopping: [684, 1388, 460, 1932, 556, ...]
- Annotation: "Discrete FHSS (not continuous CSS)"
- Show 78 reference tone grid as light gray horizontal lines
- 4 selected tones highlighted (pattern uses 4 from 78)

PANEL 2 - "I × Q (NVIS Exceptional, λ=0.7)":
- IQ plane: I and Q axes -1.5 to +1.5
- Moderate Lissajous curve (realistic for HF NVIS)
- ~8 constellation points marked
- Label: "Pattern 115 (NVIS pool)"
- Annotation: "Patterns 112-127: Complex IQ"

PANEL 3 - "I × Q (Typical DX, λ=0.4)":
- Same IQ axes
- Simple ellipse (MOST COMMON on HF)
- ~4 constellation points marked
- Label: "Pattern 72 (Typical DX pool)"
- Annotation: "Patterns 64-95: Simple-moderate IQ (MOST HF OPERATION)"

Title: "CASCADE 4D Patterns: Discrete Frequency × Continuous IQ"
Subtitle: "128 patterns, each with baked-in IQ complexity (λ parameter)"
```

---

### 2. 78-Tone Reference Grid

**Placement:** `docs/protocol/adaptive_tone_grid.md` - Section "Reference Tone Specification"

**Purpose:** Show complete 78-tone grid with pattern-based allocation

**Specification:**
```
Frequency spectrum visualization:

X-axis: Frequency 300-2800 Hz
- 78 vertical bars at 32 Hz spacing
- Bars numbered 0-77
- Show example: Tones [300, 332, 364, ..., 2732, 2764] Hz

Pattern Tone Selection Examples (overlaid):
- Pattern 5: Uses tones [12, 35, 51, 65] - shown in BLUE
- Pattern 12: Uses tones [15, 35, 54, 68] - shown in GREEN
- Pattern 25: Uses tones [8, 29, 47, 63] - shown in ORANGE
- Highlight tone 35: Used by BOTH patterns 5 and 12 (overlap allowed)

Annotations:
- "78 tones, 32 Hz spacing (HF-optimized)"
- "Each pattern selects 4 from 78 (adaptive)"
- "C(78,4) = 1,426,425 possible combinations"
- "Patterns can share tones (Time × IQ separation)"
- "96.7% spectrum efficiency (2464/2500 Hz)"

Title: "78-Tone Reference Grid with Adaptive Pattern Selection"
```

---

### 3. Hierarchical Pattern Pools by IQ Complexity

**Placement:** `docs/model/pattern_architecture.md` - Section "Message Patterns - Propagation-Matched Pools"

**Purpose:** Show how patterns are organized into HF-realistic complexity pools

**Specification:**
```
Four IQ constellation panels showing pattern pools:

PANEL 1 - "Emergency Pool (IDs 48-63, λ=0.0-0.1)":
- IQ axes: -1.5 to +1.5
- BPSK line (nearly collapsed)
- 2 constellation points
- Label: "16 patterns, minimal IQ"
- Use case: "Disturbed propagation, emergency traffic"

PANEL 2 - "Typical DX Pool (IDs 64-95, λ=0.3-0.5)" [HIGHLIGHTED]:
- IQ axes: -1.5 to +1.5
- Simple circle or ellipse
- 4-6 constellation points
- Label: "32 patterns - MOST COMMON"
- Use case: "Multi-hop DX, 70% of HF operation"
- Border: Thick green (most important)

PANEL 3 - "Good Propagation (IDs 96-111, λ=0.5-0.7)":
- IQ axes: -1.5 to +1.5
- Moderate ellipse
- 8 constellation points
- Label: "16 patterns, moderate IQ"
- Use case: "Single-hop F2, good conditions"

PANEL 4 - "NVIS Exceptional (IDs 112-127, λ=0.7-0.9)":
- IQ axes: -1.5 to +1.5
- Moderate Lissajous (realistic for HF NVIS, NOT full complexity)
- 8-10 constellation points
- Label: "16 patterns, complex IQ"
- Use case: "80m/40m NVIS only, rarely used"

Annotations:
- "Pattern ID indicates baked-in IQ complexity"
- "Model selects pool based on measured multipath"
- "HF multipath limits practical λ to 0.3-0.6 for DX"

Title: "Hierarchical Pattern Organization by HF Propagation"
Subtitle: "80 message patterns in 4 pools (baked-in complexity)"
```

---

### 4. 128-Pattern Chaos Mode Performance

**Placement:** `docs/architecture.md` - After "Core Parameters" section

**Purpose:** Show optimization trade-offs that led to 128 patterns

**Specification:**
```
Comparison chart showing pattern count trade-offs:

Bar chart with 4 groups (32, 64, 128, 256 patterns):

Each group shows:
- Correlation Time (ms): [0.75, 1.5, 3.0, 6.0]
- Per-User Throughput (bps): [168, 179, 218, 261]
- RPi4 Compatible: [✓, ✓, ✓, ✗]

Highlight 128-pattern bar (green):
- "OPTIMAL: 218 bps/user, 3ms correlation"
- "Fits RPi4 with 1.5ms margin"

256-pattern bar (red):
- "11.5ms > 10ms budget"
- "Doesn't fit RPi4"

Annotations:
- "128 maximizes per-user throughput while fitting RPi4"
- "14.5× improvement over 256-pattern coordinated (15 bps)"
- "78% Shannon efficiency with chaos + micro-tuning"

Title: "Why 128 Patterns is Optimal"
```

---

### 5. Emergency Relay Network Topology

**Placement:** `docs/protocol/emergency_relay_network.md` - Section "Ad-Hoc Relay Network in Action"

**Purpose:** Visualize Hurricane Maria scenario with multi-tier relay

**Specification:**
```
Geographic relay network map:

CENTER: Puerto Rico (Emergency Origin)
- KP4XXX (San Juan, FK68)
- Large RED circle
- Label: "EMERGENCY: All infrastructure down"

TIER 1 (Direct to emergency, 1600-5400km):
- K4XXX (Miami, FL) - 1600km
  * Role: "Coast Guard liaison"
  * Green circle, solid line to KP4XXX
- W2XXX (New York, NY) - 2600km
  * Role: "FEMA Region 2"
  * Green circle, solid line to KP4XXX
- K5XXX (Houston, TX) - 3200km
  * Role: "Military coordination"
  * Green circle, solid line to KP4XXX
- W6XXX (Los Angeles, CA) - 5400km
  * Green circle, dashed line to KP4XXX (weak signal)

TIER 2 (Via Tier 1 relay):
- Connected to Tier 1 stations with dotted lines
- Blue circles (smaller)
- Examples: Atlanta, Boston, San Antonio

Map background: North America + Caribbean
Signal strength shown by line style: solid (strong) / dashed (weak)

Annotations:
- "Pattern-based emergency (IDs 0-15, 48-63)"
- "Multi-pattern for redundancy (4× throughput)"
- "-25 dB sensitivity critical for weak PR → mainland signals"
- "Self-organizing: No infrastructure needed"

Title: "CASCADE Emergency Relay Network - Hurricane Maria 2017"
Subtitle: "HF was only communication for 2-3 weeks"
```

---

---

### 6. Multi-User 4D Separation

**Placement:** `docs/model/tfiq_dimensions.md` - Section "Multi-User Separation Strategy"

**Purpose:** Show how 45 users coexist in same 2.5 kHz bandwidth

**Specification:**
```
3D visualization showing user separation:

AXES:
- X: Time (0-3.2s, showing 2 pattern durations)
- Y: Frequency (300-2800 Hz, 78-tone grid)
- Z: IQ complexity (represented by color intensity)

ELEMENTS:
Show 12 users transmitting simultaneously:
- User 1: Pattern 5, tones [12,35,51,65], blue path
- User 2: Pattern 12, tones [15,35,54,68], green path (SHARES tone 35!)
- User 3: Pattern 25, tones [8,29,47,63], orange path
- ... (continue for 12 users)

Overlaps highlighted:
- Red circles where users share tones at same time
- Label: "RS(32,20) tolerates overlaps"

Legend:
- Different colors = different patterns
- Line thickness = IQ complexity
- Overlapping tones = shared frequency (Time × IQ separates)

Annotations:
- "45 active users, 1,024 total capacity"
- "Pattern orthogonality: -37.5 dB"
- "78-85% Shannon efficiency (kernel-coordinated)"
- "Kernels provide emergent coordination"

Title: "45 Simultaneous Users in 2.5 kHz"
Subtitle: "4D separation: Time × Tone Selection × I × Q"
```

---

## Medium Priority Diagrams

### 7. Shannon Efficiency Breakdown

**Placement:** `docs/protocol/signal_specification.md` - Section "Shannon capacity at various SNRs"

**Purpose:** Explain the 22% gap from Shannon limit

**Specification:**
```
Waterfall chart showing Shannon efficiency breakdown:

Starting capacity: 12,570 bps (Shannon @ +15 dB, 2.5 kHz)

Losses (waterfall):
1. Channel estimation: -8-12% → 11,065 bps
2. Chaos overlaps: -4-6% → 10,410 bps
3. Multi-user interference: -5-7% → 9,950 bps
4. Pattern correlation: -3% → 9,805 bps

Final: 9,805 bps = 78% efficiency

Annotation: "Eliminated overheads: Guard intervals (0%), Timing coordination (0%)"

Each bar shows:
- % lost
- Absolute bps remaining
- Justification for overhead

Title: "CASCADE Shannon Efficiency: 78% Target"
Subtitle: "128-pattern chaos with ±2 Hz micro-tuning"
```

---

### 8. Pattern Count vs Hardware Compatibility

**Placement:** `docs/architecture.md` - Section "Improvement History"

**Purpose:** Show RPi4 compatibility trade-off

**Specification:**
```
Scatter plot:

X-axis: Pattern Count (32, 64, 128, 256, 512)
Y-axis 1 (left): Inference Time (ms)
Y-axis 2 (right): Per-User Throughput (bps)

Plot 1 (bars): Inference time
- 32: 0.75ms
- 64: 1.5ms
- 128: 3.0ms (GREEN - optimal)
- 256: 6.0ms
- 512: 12ms

Horizontal line at 10ms: "RPi4 Budget Limit"

Plot 2 (line graph): Per-user throughput
- 32: 168 bps
- 64: 179 bps
- 128: 218 bps (PEAK, GREEN dot)
- 256: 261 bps (theoretical, marked with X - doesn't fit)
- 512: 285 bps (theoretical, marked with X)

Annotation: "128 patterns maximizes throughput while fitting RPi4"

Title: "Pattern Count Optimization for Raspberry Pi 4"
```

---

### 9. Expert Network Architecture

**Placement:** `docs/model/README.md` - Section "Expert Networks"

**Purpose:** Show mixture-of-experts architecture

**Specification:**
```
Neural network data flow diagram:

INPUT (left): "Raw IQ Samples"

SHARED ENCODER:
- Box: "Shared Encoder"
- "1024D universal features"
- Outputs to all 5 experts

EXPERTS (middle, vertical stack):
1. Noise Expert: "~1M params, QRN suppression, 512D output"
2. Signal Expert: "~1.2M params, Multi-user separation, 512D"
3. Propagation Expert: "~900K params, Channel equalization, 512D"
4. Pattern Complexity Expert: "~500K params, Pool selection, 512D"
5. Spectrum Allocation Expert: "~800K params, Frequency optimization, 512D"

CONDUCTOR (center-right):
- "Attention-Based Conductor"
- Shows dynamic weights: [w₁, w₂, w₃, w₄, w₅]
- Arrows showing weighting

DECODER (right):
- "Decoder Network"
- Input: Weighted combination
- Output: "Decoded data + kernel"

EXAMPLE WEIGHTINGS (bottom):
- Low SNR: [Noise: 0.6, Signal: 0.1, Prop: 0.2, Pattern: 0.05, Spectrum: 0.05]
- Multi-user: [0.1, 0.5, 0.2, 0.1, 0.1]
- Multipath: [0.1, 0.2, 0.5, 0.1, 0.1]

Title: "CASCADE Mixture-of-Experts Architecture"
Subtitle: "~9.2M total parameters, 8.5ms inference on RPi4"
```

---

### 10. Emergency Pattern Detection (4D Correlation)

**Placement:** `docs/protocol/emergency_relay_network.md` - Section "Emergency Detection"

**Purpose:** Show pattern-based emergency detection (not frequency-based)

**Specification:**
```
Side-by-side comparison:

LEFT PANEL - "Traditional Approach":
- Frequency spectrum showing dedicated 1550 Hz monitoring
- Label: "Reserved frequency = wasted spectrum"
- Efficiency: "95.8% (100 Hz reserved / 2500 Hz)"

RIGHT PANEL - "CASCADE Pattern-Based":
- Same frequency spectrum, all 78 tones shown
- Pattern 0 correlation detector overlay
- Show Pattern 0 can use ANY 4 tones from 78
- Example: Pattern 0 using [5, 23, 47, 61] today
- Different conditions: Pattern 0 using [7, 25, 49, 63] tomorrow
- Label: "Zero spectrum reservation"
- Efficiency: "96.7% (all tones shared)"

Detection mechanism:
- "Model correlates all 128 patterns anyway"
- "Pattern 0-15 correlation > threshold = emergency"
- "Zero additional CPU overhead"

Title: "Pattern-Based Emergency Detection"
Subtitle: "No frequency reservation, 96.7% spectrum efficiency"
```

---

## Medium Priority Diagrams

### 10. Protocol-Model Layer Separation

**Placement:** `README.md` - Section "Core Philosophy" or `docs/interface/README.md`

**Purpose:** Illustrate the clean boundary between discrete and continuous decisions

**Specification:**
```
Split diagram showing two layers:

LEFT SIDE - "Protocol Layer (Discrete)":
Box containing:
- WHO: Callsign-based pattern assignment
- WHETHER: Binary relay decisions (approve/reject)
- WHAT: Priority classification (Emergency/High/Normal/Low)
- WHEN: Beacon scheduling timing decisions

Characteristics:
- Verifiable rules
- No neural networks
- Deterministic
- Comprehensible to operators

RIGHT SIDE - "Model Layer (Continuous)":
Box containing:
- HOW: Encoding optimization, pattern selection, FEC strength
- WHEN: Fragment duration, ACK window detection
- HOW MUCH: Bandwidth allocation, power distribution, redundancy

Characteristics:
- Neural network optimization
- Gradient descent
- Adaptive
- Improves via telemetry

BOUNDARY (center):
- Clear interface with bidirectional arrows
- Protocol → Model: "Constraints (assigned patterns, priority, time limits)"
- Model → Protocol: "Optimizations (encoding params, timing)"

Benefits listed:
- Protocol correctness remains verifiable
- Model improvements don't break compatibility
- System behavior comprehensible
- NNs optimize within safe boundaries

Title: "Clean Protocol-Model Separation"
Subtitle: "Discrete decisions + Continuous optimization"
```

---

---

### 11. Kernel Lifecycle: Three-Round Exchange

**Placement:** `docs/protocol/kernel_lifecycle.md` - After "Table of Contents"

**Purpose:** Show how kernels evolve through prokernel → antikernel → adaptation cycle

**Specification:**
```
Three-stage flow diagram:

ROUND 1 - "Initial Transmission":
Station A → Station B
- A transmits using default encoding
- B measures: SNR -8 dB, decode SUCCESS
- B sends prokernel: "Use QPSK, heavy FEC, I'm RPi4"
- A stores B's RX kernel

ROUND 2 - "Interference Detected":
Station A → Station B (using B's kernel)
Station C (bystander) also decodes A's transmission
- C detects interference with ongoing QSO
- C sends antikernel: "You're interfering, shift +50 Hz or reduce power"
- A receives antikernel feedback

ROUND 3 - "Adapted Transmission":
Station A → Station B (using adapted kernel)
- A incorporates C's antikernel
- A adapts: Frequency +50 Hz, Power -3 dB
- B: Still decodes well (kernel maintained)
- C: Interference reduced 66% (0.35 → 0.12)
- Network converged

Arrows showing:
- Prokernel (green): B → A ("optimize FOR me")
- Antikernel (orange): C → A ("optimize AGAINST interfering me")
- Adapted kernel (blue): A updates transmission

Timeline: ~5 minutes total for convergence

Title: "Kernel Lifecycle: Emergent Network Cooperation"
Subtitle: "Self-organizing interference avoidance"
```

---

---

### 12. Heterogeneous Hardware Network

**Placement:** `docs/deployment/hardware_requirements.md` - Section "Heterogeneous Hardware Networks"

**Purpose:** Show how different hardware tiers coexist naturally

**Specification:**
```
Network topology showing 50 active users:

VISUALIZATION:
- 50 user icons scattered across diagram
- Color-coded by hardware tier
- Size represents decode capacity

Hardware Tiers:
- 20 × RPi only (small red circles): Each sees 12-15 users
- 20 × RPi+Coral (medium green circles): Each sees 40-45 users
- 7 × Desktop (large blue circles): Each sees 30-35 users
- 3 × GPU (extra-large purple circles): Each sees 45+ users

Signal Overlays:
- Emergency transmission (thick red lines): Reaches ALL stations (100% penetration)
- Strong signal (thick lines): Reaches 90%+ of stations
- Medium signal (medium lines): Reaches Coral/Desktop/GPU (60-80%)
- Weak signal (thin lines): Reaches GPU only (20-30%)

Legend:
- Circle size = decode capacity
- Line thickness = signal strength
- "Shannon-optimal: Limited hardware prioritizes strong signals"

Annotations:
- "No central coordination"
- "Weak receivers see strong signals"
- "Powerful receivers see everything"
- "Natural upgrade incentive"

Title: "Heterogeneous Hardware Network"
Subtitle: "50 users, mixed hardware, self-organizing"
```

---

---

### 13. SNR-Based Adaptive Degradation

**Placement:** `docs/model/pattern_architecture.md` - Section "Multi-Pattern Transmission"

**Purpose:** Show how receiver capability drives throughput

**Specification:**
```
Side-by-side comparison of 4 scenarios:

SCENARIO 1 - "Weak Link (RPi → RPi)":
- Transmitter: 1 pattern allocated
- Receiver: Can decode 1 pattern
- Throughput: 218 bps
- Duration: 1.6s per message
- Visual: Single blue bar

SCENARIO 2 - "Medium Link (RPi → Coral)":
- Transmitter: 2 patterns allocated
- Receiver: Can decode 2 patterns
- Throughput: 436 bps
- Duration: 1.6s (parallel patterns)
- Visual: Two blue bars (parallel)

SCENARIO 3 - "Strong Link (Desktop → Coral)":
- Transmitter: 4 patterns allocated
- Receiver: Can decode 4 patterns
- Throughput: 872 bps
- Duration: 1.6s (parallel patterns)
- Visual: Four blue bars (parallel)

SCENARIO 4 - "Optimal Link (GPU → GPU)":
- Transmitter: 4 patterns allocated
- Receiver: Can decode 4 patterns easily
- Throughput: 872 bps
- Duration: 1.6s
- Visual: Four blue bars (thick, strong)

Annotations:
- "Receiver's kernel specifies max_patterns_simultaneous"
- "Transmitter adapts to receiver capability"
- "4× throughput on strong links"
- "Same 1.6s duration (parallel transmission)"

Title: "Kernel-Driven Multi-Pattern Transmission"
Subtitle: "1× to 4× throughput based on receiver capability"
```

---

---

### 14. FHSS vs CSS Patent Safety

**Placement:** `docs/protocol/link_adaptation.md` or `docs/model/README.md`

**Purpose:** Show that SNR is pairwise and often asymmetric

**Specification:**
```
Network diagram with 4 stations showing bidirectional SNR:

STATIONS:
- Station A (West): RPi + Coral, 100W
- Station B (East): RPi only, 10W
- Station C (North): Desktop, 100W
- Station D (South): GPU, 100W

LINKS (bidirectional arrows with SNR labels):
A ↔ B:
- A→B: +10 dB (A has high power)
- B→A: -5 dB (B has low power, A has good RX)
- ASYMMETRIC: 15 dB difference!

A ↔ C:
- A→C: +8 dB
- C→A: +12 dB (C has better RX than A)
- ASYMMETRIC: 4 dB difference

B ↔ D:
- B→D: +5 dB (B low power, but D great RX)
- D→B: +15 dB (D high power, B weak RX)
- ASYMMETRIC: 10 dB difference

Annotations:
- "Each direction has unique kernel"
- "A→B uses B's RX kernel"
- "B→A uses A's RX kernel"
- "Power + antenna + hardware + propagation all factor in"
- "Pairwise optimization critical"

Title: "Pairwise Asymmetric Link Quality"
Subtitle: "Each direction optimized independently"
```

---

### 15. Real-Time Adaptation During QSOs (MAML)

**Placement:** `docs/protocol/message_validation.md` or `docs/telemetry_research.md`

**Purpose:** Show dual-layer validation preventing false positives

**Specification:**
```
Flow diagram showing validation process:

TRANSMITTED:
- Message: "Hello W1ABC" (payload)
- CRC32: 0xABCD1234 (4 bytes)
- xxHash32: 0x5678CDEF (4 bytes)
Total: 136 bytes

RECEIVED @ -18 dB (very noisy):
Neural network decodes...

TWO OUTCOMES:

OUTCOME 1 - "Valid Message":
- NN predicts: "Hello W1ABC"
- NN computes CRC32: 0xABCD1234 ✓ MATCH
- Compute xxHash32: 0x5678CDEF ✓ MATCH
- Result: ACCEPT (valid message)

OUTCOME 2 - "Hallucination Caught":
- NN predicts: "Hello W2DEF" (WRONG, but plausible)
- NN computes CRC32: 0x9ABC5432 ✓ MATCH (NN learned CRC!)
- Compute xxHash32: 0x1234ABCD ✗ FAIL
- Result: REJECT (hallucination detected)

Statistics:
- Without xxHash: 2% hallucination rate @ low SNR
- With xxHash: 0% hallucinations pass validation
- Overhead: 6.25% (8 bytes)

Annotations:
- "Layer 1 (CRC): NN learns - improves training"
- "Layer 2 (xxHash): NN cannot forge - prevents false positives"
- "Like checksums in QR codes"

Title: "Dual-Layer Validation Prevents Hallucinations"
Subtitle: "CRC for training, xxHash for verification"
```

---

---

## Low Priority Diagrams

### 16. Federated Learning Flow

**Placement:** `docs/training/data_pipeline.md` or `docs/model/README.md`

**Purpose:** Show how CASCADE trains with synthetic signals + real propagation

**Specification:**
```
Data pipeline flow:

STAGE 1 - "Generate Synthetic CASCADE":
- 5-50 users (random count)
- Each user:
  * Random pattern (48-127)
  * Random modulation (BPSK/QPSK/64-QAM)
  * Random SNR (-25 to +15 dB)
  * Random clock drift (±50 Hz per user)
  * Random start time (asynchronous)
- Output: Clean multi-user CASCADE signal

STAGE 2 - "Apply Real HF Channel":
- Load real KiwiSDR recording (150K-300K hours)
- Extract channel impulse response:
  * Multipath delay spread (1-10ms measured)
  * Doppler spread (±2 Hz measured)
  * Fading coefficients (Rayleigh/Rician measured)
  * Ionospheric flutter (real)
- Apply to synthetic CASCADE
- Output: Realistically propagated signal

STAGE 3 - "Add Real Noise":
- Load noise segment from KiwiSDR
- Real atmospheric QRN
- Real man-made QRM
- Real solar conditions
- Mix with propagated signal
- Output: Training sample with ground truth

Ground Truth Labels:
- User 1: Pattern 5, "Hello", SNR -12 dB, drift +30 Hz
- User 2: Pattern 12, "CQ", SNR +5 dB, drift -15 Hz
- ...

Benefits:
- ✓ Can't record real CASCADE (doesn't exist yet)
- ✓ Physics is universal (real propagation applies)
- ✓ Real noise characteristics critical
- ✓ Perfect ground truth for training

Title: "Training Data Pipeline"
Subtitle: "Synthetic CASCADE + Real HF Propagation/Noise"
```

---

---

### 17. Geographic Coverage Evolution

**Placement:** `docs/protocol/chaos_transmission.md` - Opening section

**Purpose:** Contrast chaos vs time-slotted operation

**Specification:**
```
Two timeline diagrams:

COORDINATED MODE (top):
Time: 0s -------- 10s -------- 20s -------- 30s
Users in slots:
- User 1: [TX] .... [TX] .... [TX] ....
- User 2: .... [TX] .... [TX] .... [TX]
- User 3: [TX] .... [TX] .... [TX] ....
Guard intervals shown (wasted time)
Label: "30 users need 30s (low efficiency)"

CHAOS MODE (bottom):
Time: 0s ---------- 10s ----------
All 45 users transmitting randomly (overlapping bars)
No gaps, continuous activity
RS tolerance handles overlaps
Label: "45 users active in 10s (high efficiency)"

Comparison metrics:
- Coordinated: 60% Shannon, 15 bps/user, needs timing
- Chaos: 78% Shannon, 218 bps/user, zero coordination

Title: "Chaos Mode: 14.5× Throughput Improvement"
```

---

### 12. Multi-Hop Emergency Relay Timeline

**Placement:** `docs/protocol/emergency_relay_network.md` - Section "Timeline Example"

**Purpose:** Show 6-phase emergency protocol with timing

**Specification:**
```
Horizontal timeline diagram:

T=0s: Emergency Alert
- Pattern 0-15 correlation detection
- Duration: ~24s
- Shown as RED block

T=24s: Network Clearing
- All stations stop beacons, transmit "CLEARING"
- Duration: ~10s
- Light orange block

T=34s: Emergency Negotiation
- 4-FSK on beacon tones
- Emergency details + full callsign + grid
- Duration: ~38s
- Yellow block

T=72s: Prokernel Responses (staggered)
- Relay stations announce capabilities
- Duration: ~48s (overlap allowed)
- Blue blocks (staggered)

T=120s: Final Kernel
- Emergency station sends relay network map
- Duration: ~40s
- Purple block

T=160s: Emergency Traffic Begins
- Message patterns (48-63) with multi-pattern
- Continuous operation
- Green ongoing bar →

Total setup: 160s (~2.7 minutes) to relay network operational

Title: "Emergency Protocol: 6-Phase Timeline"
Subtitle: "From alert to full relay network in <3 minutes"
```

---

### 14. FHSS vs CSS Patent Safety

**Placement:** `docs/protocol/adaptive_tone_grid.md` - Section "Discrete Tone Hopping (FHSS)"

**Purpose:** Demonstrate CASCADE's patent-safe frequency hopping vs LoRa CSS

**Specification:**
```
Side-by-side waveform comparison:

TOP - "LoRa CSS (Patented)":
- Time axis: 0 to 100ms
- Frequency axis: 300-2800 Hz
- Show continuous upward frequency sweep (chirp)
- Smooth curve from 300 → 2800 Hz
- Label: "Continuous chirp (CSS)"
- Patent status: "⚠️ Semtech patents"

BOTTOM - "CASCADE FHSS (Patent-Free)":
- Same time/frequency axes
- Show discrete frequency hops
- Example: 684 Hz (50ms) → 1388 Hz (instant hop) → 460 Hz (50ms) → 1932 Hz
- Horizontal line segments at each tone (constant frequency)
- Vertical jumps between tones (instantaneous)
- Label: "Discrete hops (FHSS)"
- Patent status: "✓ Public domain"

Key Differences Table:
| Aspect | LoRa CSS | CASCADE FHSS |
|--------|----------|--------------|
| Within symbol | Sweeps | Constant frequency |
| Data encoding | Chirp time shift | IQ modulation |
| Patent | Semtech | Public domain |
| Hardware | Specialized | Sound cards |

Title: "FHSS vs CSS: Patent-Safe Architecture"
Subtitle: "Discrete frequency hopping like Bluetooth, not continuous chirps like LoRa"
```

---

### 15. Raspberry Pi Hardware Tiers

**Placement:** `docs/deployment/hardware_requirements.md` - Opening section

**Purpose:** Show deployment options and performance

**Specification:**
```
Four hardware configuration cards:

CARD 1 - "RPi 4 Only":
- Icon: Raspberry Pi
- Cost: $50-85
- Inference: 8.5ms
- Shannon: 40-50%
- Users decoded: 15-25
- Use case: "Portable, Emergency"
- Status: "Works but limited"

CARD 2 - "RPi 4 + Coral TPU" [HIGHLIGHTED]:
- Icon: RPi + TPU module
- Cost: $120-180
- Inference: 2-5ms
- Shannon: 70-75%
- Users decoded: 40-45
- Use case: "RECOMMENDED - Standard deployment"
- Status: "Optimal price/performance"

CARD 3 - "Desktop x86":
- Icon: Desktop computer
- Cost: $0 (existing hardware)
- Inference: 5-8ms
- Shannon: 60-70%
- Users decoded: 30-40
- Use case: "Home station"

CARD 4 - "GPU Server":
- Icon: Server
- Cost: $200-500+
- Inference: 2-5ms
- Shannon: 75-78%
- Users decoded: 45+
- Use case: "Contest, Club station"

Common specs box:
- "All tiers: 128 patterns, 78-tone grid, identical protocol"
- "Perfect interoperability across all tiers"

Title: "CASCADE Deployment Tiers"
```

---

---

### 13. SNR-Based Adaptive Degradation

**Placement:** `README.md` - Section "Adaptive Capacity (SNR-Based)" or `docs/protocol/signal_specification.md`

**Purpose:** Show how CASCADE gracefully degrades from excellent to poor conditions

**Specification:**
```
Stacked diagram showing 6 SNR levels:

LEVEL 1 - ">+10 dB (Excellent)":
- Patterns active: 80 message
- IQ complexity: λ=0.4-0.6 (typical HF DX)
- Active users: 45
- Per-user: 218 bps (1p), 872 bps (4p)
- Mode: Full chaos
- Visual: Dense activity, many overlapping transmissions

LEVEL 2 - "+5 to +10 dB (Good)":
- Patterns: 64 message
- IQ complexity: λ=0.3-0.5
- Active users: 35-40
- Per-user: 180-220 bps
- Mode: Full chaos

LEVEL 3 - "0 to +5 dB (Fair)":
- Patterns: 48 message
- IQ complexity: λ=0.2-0.4
- Active users: 25-30
- Per-user: 140-180 bps
- Mode: Partial chaos

LEVEL 4 - "-5 to 0 dB (Weak)":
- Patterns: 32 message
- IQ complexity: λ=0.1-0.3
- Active users: 15-20
- Per-user: 110-140 bps
- Mode: Limited chaos

LEVEL 5 - "-10 to -5 dB (Very Weak)":
- Patterns: 16 emergency
- IQ complexity: λ=0.0-0.1 (collapsed)
- Active users: 8-12
- Per-user: 90-110 bps
- Mode: Coordinated

LEVEL 6 - "<-10 dB (Extreme)":
- Patterns: 16 emergency
- IQ complexity: λ=0.0 (BPSK)
- Active users: 3-5
- Per-user: 50-90 bps
- Mode: FT8-like

Annotations:
- "Smooth transitions (2-4 dB hysteresis)"
- "Emergency patterns work at all SNR levels"
- "Graceful degradation: 872 → 50 bps range"

Title: "SNR-Based Adaptive Degradation"
Subtitle: "From +15 dB (872 bps) to -22 dB (50 bps)"
```

---

### 14. Beacon Pattern Pools (Emergency vs Normal)

**Placement:** `docs/model/pattern_architecture.md` - Section "Beacon Patterns"

**Purpose:** Show how 48 beacon patterns are organized

**Specification:**
```
Two groups of beacon patterns:

GROUP 1 - "Emergency Beacons (0-15)":
- 16 patterns in grid layout (4×4)
- Each pattern ID shown: 0, 1, 2, ... 15
- IQ complexity: λ=0.0 (BPSK line)
- Color: RED
- Label: "Maximum robustness"
- Sub-allocation:
  * Emergency 1: Patterns 0-3
  * Emergency 2: Patterns 4-7
  * Emergency 3: Patterns 8-11
  * Emergency 4: Patterns 12-15
- Use case: "Emergency detection via correlation"

GROUP 2 - "Normal Beacons (16-47)":
- 32 patterns in grid layout (8×4)
- Each pattern ID shown: 16, 17, 18, ... 47
- IQ complexity: λ=0.1-0.3 (simple circles/ellipses)
- Color: BLUE
- Label: "Anti-kernel resilient pool"
- Use case: "Kernel exchange, network discovery"

Properties shown:
- Each pattern selects 4 tones from 78-tone grid
- Adaptive selection based on local interference
- Stations pick pattern with clearest tones
- Pattern orthogonality: -37.5 dB

Title: "48 Beacon Patterns: Emergency + Normal"
Subtitle: "Pattern-based separation, zero spectrum reservation"
```

---

### 15. Real-Time Adaptation During QSOs (MAML)

**Placement:** `docs/telemetry_research.md` - Section "Real-Time Adaptation During QSOs"

**Purpose:** Show how model adapts during active conversation

**Specification:**
```
Timeline showing model performance during 10-minute QSO:

X-axis: Time (0-10 minutes)
Y-axis: Decode Success Rate (%)

BASELINE (dotted line at 85%):
- "Base model (no adaptation)"

ADAPTATION CURVE (solid blue line):
- T=0: 85% (base model)
- T=1min: 89% (+4%, 5 samples collected)
- T=2min: 92% (+7%, 10 samples, MAML kicks in)
- T=3min: 94% (+9%)
- T=5min: 95% (+10%, converged)
- T=10min: 95% (stable)

QSO END (T=10min):
- Red dotted line showing "Model reverts to base"
- Back to 85% baseline

NEXT QSO (if cached):
- Starts at 91% if partner cached
- Faster convergence

Annotations:
- "Meta-learning (MAML): Learn to learn fast"
- "5-15% improvement within first minute"
- "QSO-specific adaptation"
- "Reverts after QSO ends"
- "Per-station cache for frequent partners"

Hardware comparison:
- RPi4: Kernel refinement only (+3-5%)
- Coral: MAML enabled (+10-15%)
- x86: Full online learning (+15-20%)

Title: "Real-Time Adaptation During Active QSOs"
Subtitle: "10% improvement in first minute via meta-learning"
```

---

## Low Priority Diagrams

### 16. Federated Learning Flow

**Placement:** `docs/training/continuous_improvement.md` - Section "Monthly fine-tuning"

**Purpose:** Show privacy-preserving telemetry aggregation

**Specification:**
```
Data flow diagram:

STATIONS (bottom, 20+ icons):
- Each transmits encrypted telemetry
- Local differential privacy (ε=1.0)
- No raw IQ data transmitted

AGGREGATION SERVER (middle):
- Byzantine-robust aggregation
- Detects malicious updates
- Combines gradients

MODEL UPDATE (top):
- New model version
- Distributed back to network
- Backward compatible

Timeline: Monthly cycle
Metrics:
- "1-5% improvement per month"
- "No PII collected"
- "Geographic bias reduces over time"

Title: "Privacy-Preserving Continuous Improvement"
```

---

---

### 17. Geographic Coverage Evolution

**Placement:** `docs/training/continuous_improvement.md` - Section "Long-Term Performance Targets"

**Purpose:** Show how telemetry fills geographic gaps

**Specification:**
```
World map with heatmap overlay, 4 time snapshots:

MONTH 0 (V1.0):
- NA/EU/Japan: Dark green (92%)
- Africa/Pacific: Light yellow (42%)
- Polar: Red (25%)
- Legend: "KiwiSDR training bias"

MONTH 9 (V1.1):
- NA/EU: Dark green (93%)
- Africa/Pacific: Yellow (58-60%)
- Improvement arrows

MONTH 18 (V2.0 Full Retrain):
- Much more uniform
- Africa/Pacific: Light green (87-89%)
- Variance: 0.35 → 0.08

MONTH 30+ (V3.0+):
- Nearly uniform coverage
- All regions: 91-95%
- Variance: <0.05

Title: "Geographic Bias Reduction via Telemetry"
Subtitle: "V1 → V3: 92% NA to 93% global (uniform)"
```

---

## Summary

### Diagram Priorities

**Must Have (Critical for understanding):**
1. 4D Pattern Trajectory - Core 4D concept
2. 78-Tone Reference Grid - Spectrum allocation
3. Hierarchical Pattern Pools - IQ organization by propagation
4. 128-Pattern Optimization - Why this architecture
5. Emergency Relay Network - Real-world HF emergency
6. Multi-User 4D Separation - How 45 users coexist
7. Shannon Efficiency Breakdown - 78% target explained
8. Pattern Count vs Hardware - RPi4 compatibility trade-off
9. Expert Network Architecture - Mixture-of-experts model

**Should Have (Key Mechanisms):**
10. Protocol-Model Separation - Architecture philosophy
11. Kernel Lifecycle - 3-round exchange
12. Heterogeneous Network - Mixed hardware coexistence
13. SNR-Based Degradation - Graceful adaptation
14. FHSS vs CSS - Patent safety
15. Real-Time MAML Adaptation - QSO-specific learning

**Nice to Have (Supporting Context):**
16. Federated Learning - Privacy-preserving improvement
17. Geographic Evolution - Long-term vision

**Total:** 17 diagrams

### Priority Breakdown:
- **HIGH (Must Have)**: 9 diagrams - Core technical concepts (1-9)
- **MEDIUM (Should Have)**: 6 diagrams - Key mechanisms (10-15)
- **LOW (Nice to Have)**: 2 diagrams - Supporting context (16-17)

### Placement Summary

| Subdirectory | Diagrams | IDs |
|--------------|----------|-----|
| `docs/` | Architecture optimization | 4 |
| `docs/model/` | Patterns, experts, separation | 1, 3, 9 |
| `docs/protocol/` | Tone grid, emergency, chaos, kernel | 2, 5, 7, 10, 13, 14 |
| `docs/deployment/` | Hardware tiers, heterogeneous | 8, 12 |
| `docs/training/` | Federated, geographic, MAML | 15, 16, 17 |
| `docs/interface/` | Protocol-model boundary | 10 |
| `README.md` | Adaptive degradation | 13 |

**Total: 17 diagrams** across all documentation

### File Organization

**Recommended:**
```
docs/
├── diagrams/              # Shared diagram storage
│   ├── architecture/      # Architecture diagrams (1, 4)
│   ├── protocol/          # Protocol diagrams (2, 5, 6, 7, 12)
│   ├── model/             # Model diagrams (3, 9)
│   ├── deployment/        # Hardware diagrams (8)
│   └── training/          # Training diagrams (10, 11)
```

Or embed in respective doc directories:
```
docs/
├── model/diagrams/        # Model-specific
├── protocol/diagrams/     # Protocol-specific
└── ...
```

---

## Updated Architecture References

All diagrams updated to reflect **128-pattern chaos architecture:**

### Core Specifications (use in all diagrams):
- ✅ 128 total patterns (48 beacon: 0-47, 80 message: 48-127)
- ✅ 78 reference tones (300-2764 Hz, 32 Hz spacing)
- ✅ 4 tones per pattern (adaptive selection from 78-tone grid)
- ✅ 78-85% Shannon efficiency (kernel-coordinated: 78% initial, 85% steady state)
- ✅ 45 active users, 1,024 total capacity (frequency + time reuse)
- ✅ 218 bps/user (1 pattern), 872 bps (4 patterns)
- ✅ 38 KB pattern storage
- ✅ -37.5 dB orthogonality achieved
- ✅ RS(32,20) aligned structure (37.5% erasure tolerance)
- ✅ 8.5ms RPi4 inference

### Emergency Patterns:
- Beacon: 0-15 (16 patterns)
- Message: 48-63 (16 patterns)
- Detection: Via pattern correlation (zero overhead)
- **No frequency reservation** (pattern-based)

### Message Pattern Pools:
- Emergency/disturbed: 48-63 (16 patterns, λ=0.0-0.1)
- Typical DX: 64-95 (32 patterns, λ=0.3-0.5) ← **MOST COMMON**
- Good propagation: 96-111 (16 patterns, λ=0.5-0.7)
- NVIS exceptional: 112-127 (16 patterns, λ=0.7-0.9)

---

*Last updated: 2025-10-04*
*Aligned with docs/architecture.md and docs/model/pattern_architecture.md*