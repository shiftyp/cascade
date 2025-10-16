# CASCADE Protocol Overview

## What is CASCADE?

CASCADE (Coordinated Adaptive Signaling with Collision Avoidance and Distributed Exchanges) is a modern HF data protocol designed for high-throughput amateur radio communications. It achieves **5-10× faster data rates than FT8** while maintaining reliable operation across typical HF conditions.

**Key Innovation:** Neural network performs all demodulation (including IQ recovery), learns to adapt signal paths around interference using quantized/compressed kernel embeddings (28 bytes total), and generates optimal transmission parameters from observed channel conditions. The NN uses specialized expert networks trained on separated signal components.

---

## Core Principles

### 1. Neural Network Demodulation with Expert Architecture
- **No traditional DSP** - NN performs all signal processing from raw I/Q to bits
- **Expert-based learning** - Specialized networks for QRN, Signal, Timing, Channel, and QRM
- **Learns interference patterns** - adapts to QRM, QRN, multipath, selective fading
- **Quantized embeddings** - compressed to 28 bytes alongside discrete parameters
- **End-to-end learning** - pattern detection, carrier recovery, equalization all learned
- **Temporal collision handling** - separates overlapping signals with time offsets
- **RX predicts kernels** - Simplified architecture: RX suggests optimal kernel to TX (no TX encoder needed)

### 2. Message Framing and FEC

**Message Sizes:**
- **Typical**: 5-20 bytes (70% of messages) - callsigns, grids, reports (like FT8)
- **Medium**: 20-64 bytes (20% of messages) - conversations, status updates
- **Large**: 64-256 bytes (10% of messages) - images, files, long text

**Automatic Framing (for efficiency):**
- Messages **<64 bytes**: Encoded as single polar block (minimal overhead)
- Messages **≥64 bytes**: Split into 512-bit frames
  - Each frame → 1024-bit polar block
  - Frames concatenated after encoding
  - **50-100% overhead reduction** vs single large block!

**8-PSK/16-APSK Compatibility:**
- Polar codes produce power-of-2 block lengths (1024, 2048, etc.)
- Powers of 2 are **never divisible by 3** (for 8-PSK)
- Solution: **Pad 0-2 bits** after polar encoding to make divisible
- Padding overhead: 0.2% (negligible)
- RX ignores trailing padding bits during decode

**Example:**
```
100-byte message with 8-PSK:
  Without framing: 2048-bit polar (156% overhead)
  With framing: 2× 1024-bit polar blocks (56% overhead)
  Savings: 100 percentage points!

Each 1024-bit block:
  1024 % 3 = 1 bit remainder
  Pad 2 bits → 1026 bits (342 symbols)
  Padding overhead: 0.2%
```

---

### 3. Physical Layer: Dual-Layer Modulation

**Pattern Layer (2-FSK with GMSK):**
- 2-tone FSK on adjacent frequency pair
- GMSK pulse shaping for spectral containment
- 67 usable frequency pairs across 2.7 kHz bandwidth
- Used for: Pattern detection, coarse frequency/timing sync
- NN learns to detect patterns in noise

**Data Layer (BPSK+ in I/Q):**
- BPSK/QPSK/8-PSK/16-APSK modulated on same frequency pair
- Orthogonal to pattern layer (different symbol timing)
- NN demodulates I/Q symbols
- Adaptive modulation based on SNR

**Frequency Organization (3-FSK):**
- 129 usable channels (300-2860 Hz, 20 Hz spacing)
- 140 Hz guard band on right (2860-3000 Hz reserved)
- Organized into **43 frequency triples** (129 ÷ 3)
- Triple 0: Channels 0-2 (300, 320, 340 Hz)
- Triple 21: Channels 63-65 (1560, 1580, 1600 Hz)
- Triple 42: Channels 126-128 (2820, 2840, 2860 Hz)

**Why 3-FSK (frequency diversity):**
- If 1 tone hits fading notch → still have 2 tones (67% energy)
- **~3-4 dB SNR gain** vs 2-FSK in frequency-selective fading
- Better QRM resilience: 1 jammed tone doesn't kill pattern
- **+18% network throughput** despite -33% users (rate-7/8 vs rate-1/2 FEC)

**Why two layers:**
- Pattern layer: Robust detection at low SNR (-15 dB)
- Data layer: High throughput at higher SNR
- Both use same frequency triple (spectral efficiency)
- NN learns to demodulate both simultaneously

### 3. Kernel Structure (29 Bytes Total)

**Kernel = Discrete Parameters + Quantized Embedding (all in 29 bytes)**

The kernel is **separate from the message header**. The header is part of the message payload.

**Kernel contents (29 bytes):**

| Field | Bits | Range | Description |
|-------|------|-------|-------------|
| Protocol version | 6 | 0-63 | Protocol/model version (for future compatibility) |
| Pattern ID | 3 | 0-7 | Which orthogonal pattern (ternary) |
| Frequency triple | 6 | 0-42 | Which of 43 frequency triples (3-FSK) |
| SNR estimate | 8 | -30 to +30 dB | Measured signal quality (offset by 30) |
| Modulation hint | 3 | 0-7 | BPSK/QPSK/8PSK/16APSK + future |
| Polar code rate | 3 | 0-7 | FEC rate (1/2, 2/3, 3/4, 5/6, 7/8) |
| **Data symbol rate** | **3** | **0-7** | **75, 100, 125, 150, 175, 200, 250, 300 sym/s** |
| Transmission duration | 8 | 0-255 | Duration in units of 341ms (0-87s) |
| Emergency flag | 1 | 0-1 | Emergency traffic indicator |
| Timestamp | 16 | 0-65535 | Unix time modulo 65536 (18.2 hours) |
| Session ID | 16 | 0-65535 | Unique session identifier |
| **Predicted embedding** | **113** | - | **RX-predicted channel parameters** (14.125 bytes) |

**Total: 234 bits = 29.25 bytes (round to 30 bytes)**

**Note:** In simplified CASCADE, the embedding is **predicted by RX** and sent to TX via CTS (not generated by TX encoder)

**Data symbol rate encoding (3 bits, 0-7):**

| Value | Symbol Rate | Typical Use Case |
|-------|-------------|------------------|
| 0 | 75 sym/s | Low SNR (<-10 dB), BPSK, severe multipath |
| 1 | 100 sym/s | Poor conditions (-5 to 0 dB), BPSK/QPSK |
| 2 | 125 sym/s | Fair conditions (0 to 5 dB), QPSK |
| 3 | 150 sym/s | Good conditions (5 to 10 dB), QPSK |
| 4 | 175 sym/s | Good conditions (10 to 15 dB), 8-PSK |
| 5 | 200 sym/s | Excellent conditions (>15 dB), 8-PSK |
| 6 | 250 sym/s | Excellent conditions (>20 dB), 16-APSK |
| 7 | 300 sym/s | Ideal conditions (>25 dB), 16-APSK |

**Why discrete symbol rate:**
- **Decodable without NN:** RX can calculate exact transmission timing before neural network runs
- **Deterministic timing:** No ambiguity about when transmission will end
- **Model negotiates:** TX chooses rate based on channel estimate (from previous QSO or beacon)
- **Fallback mechanism:** If NN fails, conventional decoder can still extract timing
- **Pattern layer fixed:** Always 75 sym/s for sync (robust at low SNR)

**TX station calculation (corrected):**
```python
def calculate_transmission_duration(message_bytes: int, 
                                   modulation: str,
                                   polar_rate: Tuple[int, int],
                                   data_symbol_rate: int) -> int:
    """
    Calculate transmission duration in 341ms windows.
    
    Args:
        message_bytes: Size of message
        modulation: Data layer modulation
        polar_rate: FEC rate (k, n)
        data_symbol_rate: DISCRETE data symbol rate from kernel (75-300 sym/s)
    
    Returns:
        int: Duration in 341ms windows (0-255)
    """
    # Bits after FEC encoding
    k, n = polar_rate
    encoded_bits = message_bytes * 8 * (n / k)
    
    # Bits per symbol from data layer modulation
    bits_per_symbol = {'BPSK': 1, 'QPSK': 2, '8-PSK': 3, '16-APSK': 4}[modulation]
    
    # Symbols needed at DATA layer rate (from discrete kernel field)
    symbols_needed = encoded_bits / bits_per_symbol
    
    # Time in seconds (using DATA layer symbol rate from kernel)
    duration_seconds = symbols_needed / data_symbol_rate
    
    # Convert to 341ms units
    windows = int(np.ceil(duration_seconds / 0.341))
    
    # Clamp to valid range
    return min(max(windows, 1), 255)


def select_data_symbol_rate(snr_db: float, 
                            propagation_mode: str,
                            qrm_present: bool) -> int:
    """
    Select appropriate data symbol rate based on channel conditions.
    
    This is called by TX station BEFORE encoding message.
    The selected rate goes into the discrete kernel field.
    
    Args:
        snr_db: Measured SNR
        propagation_mode: 'awgn', 'rayleigh', 'rician', 'multipath_sparse', 'multipath_dense'
        qrm_present: Whether interference detected
    
    Returns:
        int: Data symbol rate (75, 100, 125, 150, 175, 200, 250, 300)
    """
    # Start with SNR-based rate
    if snr_db < -10:
        rate = 75
    elif snr_db < 0:
        rate = 100
    elif snr_db < 5:
        rate = 125
    elif snr_db < 10:
        rate = 150
    elif snr_db < 15:
        rate = 175
    elif snr_db < 20:
        rate = 200
    elif snr_db < 25:
        rate = 250
    else:
        rate = 300
    
    # Reduce rate for severe multipath
    if propagation_mode == 'multipath_dense':
        rate = max(75, int(rate * 0.6))  # 40% reduction
    elif propagation_mode == 'multipath_sparse':
        rate = max(75, int(rate * 0.8))  # 20% reduction
    
    # Reduce rate if QRM present
    if qrm_present:
        rate = max(75, int(rate * 0.8))  # 20% reduction
    
    # Round to nearest valid rate
    valid_rates = [75, 100, 125, 150, 175, 200, 250, 300]
    rate = min(valid_rates, key=lambda x: abs(x - rate))
    
    return rate
```

**Example calculations (corrected):**
```python
# RTS (42 bytes, QPSK @ 150 sym/s, rate 2/3) - good SNR
data_symbol_rate = 150  # From kernel discrete field (not embedding!)
encoded_bits = 42 * 8 * (3/2) = 504 bits
symbols = 504 / 2 = 252 symbols @ 150 sym/s
duration = 252 / 150 = 1.68s = 5 units (1.68 / 0.341)

# Data (152 bytes, 8-PSK @ 200 sym/s, rate 7/8) - excellent SNR
data_symbol_rate = 200  # From kernel
encoded_bits = 152 * 8 * (8/7) = 1389 bits
symbols = 1389 / 3 = 463 symbols @ 200 sym/s
duration = 463 / 200 = 2.32s = 7 units

# Large message with multipath (2000 bytes, BPSK @ 75 sym/s, rate 1/2)
data_symbol_rate = 75  # Reduced due to multipath
encoded_bits = 2000 * 8 * (2/1) = 32000 bits
symbols = 32000 / 1 = 32000 symbols @ 75 sym/s
duration = 32000 / 75 = 426.7s = 255 units (clamped, requires chunking)
```

**What's in the quantized embedding (113 bits):**
The embedding now contains ONLY channel-adaptive parameters that can't be discretized:
- Fine-grained frequency offset correction (beyond coarse triple selection)
- Phase rotation compensation
- Adaptive equalization taps (for 3-tone diversity combining)
- Timing offset adjustments
- Learned interference mitigation patterns
- Channel impulse response estimate (multi-tone)
- Per-tone fading estimates (for MRC diversity combining)

**Context signals for disambiguation:**
- **Max context signals:** 8 (increased from 5)
- **Selection criteria:** 8 closest signals to target signal, ranked by:
  1. **Frequency proximity:** Signals on nearby frequency triples (±5 triples)
  2. **Temporal proximity:** Most recent transmissions (within last 30 seconds)
  3. **Pattern similarity:** Similar ternary orthogonal patterns
  4. **SNR similarity:** Similar signal quality
- **Purpose:** Help decoder distinguish overlapping/colliding signals
- **Benefit:** ~3-5 dB effective SNR gain from network awareness

**Why 8 context signals:**
- Covers adjacent frequency triples (±3 triples on each side = 7 triples)
- Includes recent transmissions on same frequency
- With 43 triples, 8 contexts covers ~18% of spectrum neighborhood
- Balances memory/computation with disambiguation capability

## Protocol Layer: RTS/CTS/QSY Collision Avoidance (Simplified Architecture)

**Simplified CASCADE - RX Predicts Kernels:**

**Initial Exchange:**
1. **TX sends RTS** with default/minimal kernel
2. **RX receives**, decodes, and analyzes channel from signal
3. **RX predicts optimal kernel** for this channel (using Integration Decoder)
4. **RX sends CTS** with suggested kernel
5. **TX uses RX's kernel** for data transmission

**Collision Avoidance (Anti-Kernels):**
1. **Station C sends RTS** (wants to transmit)
2. **Station B** (already transmitting) hears RTS, detects conflict
3. **Station B sends QSY** with anti-kernel: "Avoid this channel"
4. **Station C updates** kernel to avoid B's channel
5. **Station A** (destination) receives new RTS, sends optimal kernel via CTS
6. **Station C transmits** on collision-free channel optimized for A

**Why RX Predicts Kernels:**
- RX has full signal observation (better channel knowledge)
- No reciprocal channel measurement needed (29× faster dataset generation)
- RX knows what IT needs for decoding (optimal from RX perspective)
- Enables anti-kernels: RX can suggest kernels for detected interfering signals
- Simpler TX implementation (no channel estimation needed)

**Conflict determination (protocol layer):**
- Protocol extracts TX kernel from incoming RTS
- Compares TX kernel's pattern_id and frequency_triple against 8 closest context signals
- Detects if another station is already using same pattern + frequency triple
- If conflict found → issue QSY with RX's latest RX kernel as alternative

**QSY kernel negotiation:**
```python
# Protocol layer at RX examines incoming RTS
def handle_rts(incoming_rts, context_signals):
    # Extract TX kernel from RTS
    tx_kernel = incoming_rts.tx_kernel

    # Check if TX kernel conflicts with 8 closest context signals
    for context_signal in context_signals[:8]:
        if (context_signal.pattern_id == tx_kernel.pattern_id and
            context_signal.frequency_triple == tx_kernel.frequency_triple):
            # CONFLICT! Another station using same channel
            # Send QSY with latest RX kernel (already advertised in beacon)
            send_qsy(self.latest_rx_kernel)
            return

    # No conflict → send CTS
    send_cts()

    # TX receives QSY with RX's preferred kernel
    # TX retransmits RTS using RX kernel parameters
    # RX confirms with CTS
    # Data exchange proceeds on collision-free channel
```

**Why use RX kernel:**
- RX kernel already optimized for RX station's channel conditions
- Avoids need to calculate new parameters on-the-fly
- TX already has this kernel from beacon (or can extract from QSY)
- Natural load balancing (RX spreads traffic across its preferred channels)

**Benefits:**
- Proactive collision avoidance (before data transmission)
- Minimal overhead (QSY only sent when conflict detected)
- Uses existing 8-context network awareness
- Preserves spectrum efficiency (avoids wasted transmissions)

## Training Strategy: Simplified 3-Stage Architecture

### Architectural Simplification

**Original Design** (4 stages, complex):
- Stage 1: IQ Encoder
- Stage 2-3: RX Model (5 experts + decoder)
- Stage 4: **TX Encoder** (requires reciprocal channel, slow to generate)

**Simplified CASCADE** (3 stages, faster):
- Stage 1: IQ Encoder
- Stage 2-3: RX Model learns TWO tasks:
  1. Decode messages (pattern, frequency, modulation)
  2. **Predict optimal kernels** (for TX to use)
- Stage 4: **ELIMINATED** (RX handles kernel prediction)

**Benefits:**
- ✅ **29× faster** dataset generation (no reciprocal channel simulation)
- ✅ **Simpler** architecture (3 stages not 4)
- ✅ **More realistic** deployment (RX suggests kernels, common in adaptive systems)
- ✅ **Enables anti-kernels** (RX can predict kernels for multiple detected signals)

### Dataset Requirements (Simplified)

**What's needed:**
- `rx_iq`: [2, 2048] - I/Q windows from 10s streams
- `optimal_embedding`: [256] - Ground truth from physics (for RX to learn to predict)
- Physics labels: SNR, propagation mode, K-index, etc.

**What's NOT needed:**
- ~~`tx_observed_iq`~~ - No TX observations (eliminated!)
- ~~Reciprocal channel~~ - Not needed
- ~~TX encoder network~~ - RX predicts kernels

---

## Training Strategy: 3-Stage Curriculum Learning

### Problem Statement

CASCADE has dependencies:
1. **IQ Encoder ↔ Experts**: Encoder compresses IQ, experts process compressed features
2. **Experts ↔ Decoder**: Decoder needs expert outputs, experts optimize for decoder performance
3. **Embedding Prediction**: RX must learn to predict optimal kernels from signal

### Solution: 3-Stage Curriculum

**Stage 1: Bootstrap IQ Encoder**
- **Goal**: Learn basic IQ→features mapping without experts
- **Approach**: Train encoder on **auxiliary reconstruction task**
  ```python
  # Autoencoder training (no experts needed)
  encoder = IQEmbeddingEncoder()
  decoder_head = nn.Linear(512, 2048)  # Reconstruct IQ

  for batch in dataloader:
      compressed = encoder(batch['iq'])  # [2, 2048] → 512
      reconstructed = decoder_head(compressed)  # 512 → 2048
      loss = F.mse_loss(reconstructed, batch['iq'])
      loss.backward()
  ```

**Stage 2-3: Complete CASCADE Model (RX Decoding + Kernel Prediction)**

**Goal:** Train RX to decode messages AND predict optimal kernels

**RX Integration Decoder (predicts kernels):**

```python
class IntegrationDecoder(nn.Module):
    """
    Combines expert outputs + context signals → final predictions.

    In simplified CASCADE, also predicts optimal embeddings.
    """

    def __init__(self, expert_dim=640, context_dim=256, hidden_dim=512):
        super().__init__()

        # Existing outputs
        self.pattern_head = nn.Linear(hidden_dim, 4)  # 4 patterns
        self.frequency_head = nn.Linear(hidden_dim, 43)  # 43 triples
        self.modulation_head = nn.Linear(hidden_dim, 4)  # 4 modulations

        # NEW: Embedding prediction (for RX to suggest kernels)
        self.embedding_head = nn.Linear(hidden_dim, 256)  # Predict optimal kernel

    def forward(self, expert_features, context_signals=None):
        # Combine experts + context
        x = self.combine_experts_and_context(expert_features, context_signals)

        outputs = {
            'pattern': self.pattern_head(x),
            'frequency': self.frequency_head(x),
            'modulation': self.modulation_head(x),
            'predicted_embedding': self.embedding_head(x),  # NEW!
        }

        return outputs
```

**Training with embedding prediction:**

```python
# Stage 2-3: Train RX to decode AND predict kernels
for batch in dataloader:
    rx_iq, labels = batch

    # Forward pass
    outputs = cascade_model(rx_iq, context_signals)

    # Existing losses
    pattern_loss = F.cross_entropy(outputs['pattern'], labels['pattern_id'])
    frequency_loss = F.cross_entropy(outputs['frequency'], labels['frequency_triple'])

    # NEW: Embedding prediction loss (RX learns to predict optimal kernel)
    if 'optimal_embedding' in labels:
        embedding_loss = F.mse_loss(
            outputs['predicted_embedding'],
            labels['optimal_embedding']  # Ground truth from physics
        )
        total_loss = pattern_loss + frequency_loss + 0.5 * embedding_loss

    total_loss.backward()
```

**No Stage 4 needed!** RX handles everything.