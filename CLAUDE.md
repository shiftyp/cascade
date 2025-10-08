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

### 2. Physical Layer: Dual-Layer Modulation

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
- 120 usable channels (300-2680 Hz, 20 Hz spacing)
- 300 Hz guard band on right (2680-3000 Hz reserved)
- Organized into **40 frequency triples** (120 ÷ 3)
- Triple 0: Channels 0-2 (300, 320, 340 Hz)
- Triple 19: Channels 57-59 (1440, 1460, 1480 Hz)
- Triple 39: Channels 117-119 (2640, 2660, 2680 Hz)

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
| Frequency triple | 6 | 0-39 | Which of 40 frequency triples (3-FSK) |
| SNR estimate | 8 | -30 to +30 dB | Measured signal quality (offset by 30) |
| Modulation hint | 3 | 0-7 | BPSK/QPSK/8PSK/16APSK + future |
| Polar code rate | 3 | 0-7 | FEC rate (1/2, 2/3, 3/4, 5/6, 7/8) |
| **Data symbol rate** | **3** | **0-7** | **75, 100, 125, 150, 175, 200, 250, 300 sym/s** |
| Transmission duration | 8 | 0-255 | Duration in units of 341ms (0-87s) |
| Emergency flag | 1 | 0-1 | Emergency traffic indicator |
| Timestamp | 16 | 0-65535 | Unix time modulo 65536 (18.2 hours) |
| Session ID | 16 | 0-65535 | Unique session identifier |
| Quantized embedding | 113 | - | Compressed NN parameters (14.125 bytes) |

**Total: 234 bits = 29.25 bytes (round to 30 bytes)**

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
- With 40 triples, 8 contexts covers 20% of spectrum neighborhood
- Balances memory/computation with disambiguation capability

## Training Strategy: Breaking the Circular Dependencies

### Problem Statement

CASCADE has multiple circular dependencies:
1. **IQ Encoder ↔ Experts**: Encoder compresses IQ, experts process compressed features
2. **Experts ↔ Decoder**: Decoder needs expert outputs, experts optimize for decoder performance  
3. **Context ↔ Decoder**: Decoder produces kernels for context, but needs context to decode
4. **Quantized Embeddings**: TX needs channel estimate, RX needs embedding to decode

### Solution: 5-Stage Curriculum Learning

**Stage 1: Bootstrap IQ Encoder (Weeks 7-10)**
- **Goal**: Learn basic IQ→features mapping without experts
- **Approach**: Train encoder on **auxiliary reconstruction task**
  ```python
  # Autoencoder training (no experts needed)
  encoder = IQEmbeddingEncoder()
  decoder_head = nn.Linear(512, 2048)  # Reconstruct IQ
  
  for batch in dataloader:
      compressed = encoder(batch['iq'])  # 2048 → 512
      reconstructed = decoder_head(compressed)  # 512 → 2048
      loss = F.mse_loss(reconstructed, batch['iq'])
      loss.backward()
  ```

### Stage 2-4: Embedding Autoencoder Training (Parallel to Expert Training)

**Goal:** Learn to compress/decompress channel adaptation parameters

**Embedding Autoencoder Components:**

```python
class EmbeddingEncoder(nn.Module):
    """TX: Encode channel observations → continuous embedding → quantize to 14 bytes."""
    
    def forward(self, channel_features: torch.Tensor) -> torch.Tensor:
        # channel_features: From Channel Expert (128 dims)
        # Output: Continuous embedding (256 floats)
        x = self.fc1(channel_features)  # 128 → 512
        x = F.relu(self.bn1(x))
        x = self.fc2(x)  # 512 → 256
        return x  # Continuous embedding


class LearnedQuantizer(nn.Module):
    """Quantize 256 floats → 112 bits (14 bytes) using learned codebook."""
    
    def __init__(self):
        super().__init__()
        # Learned codebook vectors
        self.coarse_codebook = nn.Parameter(torch.randn(256, 32))  # 8 bits
        self.fine_codebook = nn.Parameter(torch.randn(2**10, 32))  # 104 bits (residual)
    
    def forward(self, continuous_embedding: torch.Tensor) -> torch.Tensor:
        # Quantize using vector quantization
        # Step 1: Coarse quantization (8 bits)
        coarse_indices = self.find_nearest_codebook(continuous_embedding, self.coarse_codebook)
        coarse_vectors = self.coarse_codebook[coarse_indices]
        
        # Step 2: Residual (fine) quantization (104 bits)
        residual = continuous_embedding - coarse_vectors
        fine_indices = self.find_nearest_codebook(residual, self.fine_codebook)
        
        # Total: 8 + 104 = 112 bits
        quantized_bits = torch.cat([
            coarse_indices.unsqueeze(1),  # 8 bits
            fine_indices.unsqueeze(1)      # 104 bits (stored as indices)
        ], dim=1)
        
        return quantized_bits  # 112 bits total


class EmbeddingDecoder(nn.Module):
    """RX/TX: Dequantize 112 bits (14 bytes) → reconstructed embedding (256 floats)."""
    
    def forward(self, quantized_bits: torch.Tensor, 
                coarse_codebook: torch.Tensor, fine_codebook: torch.Tensor) -> torch.Tensor:
        # Extract indices
        coarse_idx = quantized_bits[:, 0]
        fine_idx = quantized_bits[:, 1]
        
        # Look up codebook vectors
        coarse_vectors = coarse_codebook[coarse_idx]
        fine_vectors = fine_codebook[fine_idx]
        
        # Reconstruct
        reconstructed = coarse_vectors + fine_vectors  # 256 floats
        
        # Project to final embedding space
        x = self.fc1(reconstructed)  # 256 → 512
        x = F.relu(self.bn1(x))
        x = self.fc2(x)  # 512 → 256
        return x  # Reconstructed embedding (256 floats)
```

**Training the Embedding Autoencoder:**

```python
class EmbeddingAutoencoderTrainer:
    """Train embedding encoder/quantizer/decoder end-to-end."""
    
    def __init__(self, channel_expert, device='cuda'):
        self.channel_expert = channel_expert
        self.device = device
        
        # Freeze channel expert (use pre-trained from Stage 2)
        self.channel_expert.eval()
        for param in self.channel_expert.parameters():
            param.requires_grad = False
        
        # Embedding autoencoder components
        self.encoder = EmbeddingEncoder().to(device)
        self.quantizer = LearnedQuantizer().to(device)
        self.decoder = EmbeddingDecoder().to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.quantizer.parameters()) + 
            list(self.decoder.parameters()),
            lr=1e-3
        )
    
    def train_epoch(self, dataloader):
        """Train embedding autoencoder end-to-end."""
        total_loss = 0.0
        
        for batch_iq, labels in dataloader:
            batch_iq = batch_iq.to(self.device)
            
            # Get channel features from frozen Channel Expert
            with torch.no_grad():
                compressed_iq = iq_encoder(batch_iq)
                channel_features = self.channel_expert(compressed_iq)  # 128 dims
            
            # Embedding autoencoder forward pass
            continuous_embedding = self.encoder(channel_features)  # 256 floats
            quantized_bits = self.quantizer(continuous_embedding)  # 112 bits
            reconstructed = self.decoder(
                quantized_bits, 
                self.quantizer.coarse_codebook, 
                self.quantizer.fine_codebook
            )  # 256 floats
            
            # Loss: Reconstruction + Rate constraint
            recon_loss = F.mse_loss(reconstructed, continuous_embedding)
            
            # Rate loss: Penalize if embedding doesn't fit in 112 bits
            # (This is implicit in VQ - codebook size limits bits)
            
            # Task loss: Test if reconstructed embedding helps demodulation
            # Apply embedding to signal processing and measure BER
            task_loss = self.test_demod_with_embedding(
                batch_iq, reconstructed, labels
            )
            
            total_loss_batch = recon_loss + 0.5 * task_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        return total_loss / len(dataloader)
    
    def test_demod_with_embedding(self, iq_samples, embedding, labels):
        """Test if embedding helps demodulation (surrogate task for TX/RX)."""
        # Apply embedding mutations to signal
        # Measure if it improves decode accuracy
        # This is a proxy for "does the embedding contain useful channel info?"
        # ...implementation details...
        pass
```

**When to train:** Weeks 15-19 (parallel to Stage 3-4)