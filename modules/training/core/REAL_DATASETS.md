# Real-World HF Datasets for CASCADE Training

## Public HF Radio Datasets

### 1. RadioML Datasets (DeepSig)
**URL**: https://www.deepsig.ai/datasets
- **RadioML 2016.10a**: 220,000 samples, 11 modulations, -20 to +18 dB SNR
- **RadioML 2018.01a**: 2.5M samples, 24 modulations, variable SNR
- **Format**: Complex I/Q samples (2×128 samples per frame)
- **License**: Creative Commons
- **Good for**: Modulation classification pre-training
- **Limitation**: VHF/UHF focused, not HF-specific

```python
# Download and use RadioML
import h5py
import requests

def download_radioml():
    url = "https://www.deepsig.ai/datasets/radioml-2018.01a.tar.gz"
    # Download and extract...
    pass

def load_radioml_sample():
    with h5py.File('RML2016.10a_dict.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        # data[(mod, snr)] = array of IQ samples
        qpsk_samples = data[('QPSK', 10)]  # QPSK at 10 dB SNR
    return qpsk_samples
```

### 2. SIGMOID (Signal Metadata Format)
**URL**: https://github.com/gnuradio/gr-sigmf
- **Format**: SigMF standard for recording/sharing RF data
- **Community**: Many users share HF recordings
- **Good for**: Real propagation conditions
- **Access**: Various archives, check sigmf.org

### 3. WebSDR Recordings
**URL**: http://websdr.org
- **Coverage**: 100+ SDRs worldwide covering HF bands
- **Access**: Can record directly from web interface
- **Bands**: All amateur HF bands (160m - 10m)
- **Good for**: Real-time channel conditions
- **License**: Record for research (follow local laws)

```python
# Example: Record from WebSDR
def record_from_websdr(frequency_khz=14070, duration_sec=30):
    """
    Record real HF signals from WebSDR.
    Note: Requires manual browser interaction or selenium automation.
    """
    # 1. Navigate to http://websdr.ewi.utwente.nl:8901/
    # 2. Set frequency to 14.070 MHz (FT8 frequency)
    # 3. Set mode to USB
    # 4. Record audio output
    # 5. Convert to I/Q samples

    # For legal automation, use official API if available
    pass
```

### 4. KiwiSDR Network
**URL**: http://kiwisdr.com/public/
- **Network**: 600+ receivers worldwide
- **API**: Direct I/Q sample access via waterfall extension
- **Format**: Real-time streaming or file export
- **Good for**: Diverse propagation paths
- **License**: Individual SDR operators set terms

```python
# KiwiSDR Python client
def record_from_kiwisdr(host='kiwisdr.example.com', freq_khz=14070, duration_sec=60):
    """
    Record from KiwiSDR using kiwiclient library.

    Install: pip install kiwiclient
    """
    from kiwiclient import KiwiSDRStream

    # Connect to KiwiSDR
    sdr = KiwiSDRStream(host=host, port=8073)
    sdr.connect()

    # Set frequency and mode
    sdr.set_freq(freq_khz)
    sdr.set_mod('iq')  # Get I/Q samples

    # Record
    samples = sdr.record(duration_sec)

    sdr.disconnect()

    return samples
```

### 5. Signal Identification Wiki (sigidwiki.com)
**URL**: https://www.sigidwiki.com
- **Database**: 1000+ signal types with examples
- **Format**: Audio recordings (.wav), waterfall images
- **Coverage**: HF through microwave
- **Good for**: Learning real interference patterns
- **Access**: Free, community-contributed

### 6. GNU Radio Datasets
**URL**: https://github.com/gnuradio/gnuradio/wiki/Datasets
- **Various**: Community-shared recordings
- **Format**: GNU Radio .cfile (complex float32)
- **Good for**: Diverse real-world signals
- **License**: Varies by contributor

## Best Approach: Hybrid Training

### Strategy 1: Transfer Learning from RadioML

```python
# 1. Pre-train on RadioML (clean, labeled)
# 2. Fine-tune on CASCADE synthetic data
# 3. Final tuning on real HF recordings

class TransferLearningPipeline:
    def stage1_radioml_pretraining(self):
        """Pre-train IQ encoder on RadioML dataset."""
        # Load RadioML
        radioml_dataset = RadioMLDataset('RML2016.10a_dict.pkl')

        # Train IQ encoder on modulation classification
        encoder = IQEmbeddingEncoder()
        classifier = nn.Linear(512, 11)  # 11 RadioML modulations

        # Train...
        # This gives encoder good feature extraction

    def stage2_cascade_synthetic(self):
        """Train on CASCADE synthetic data."""
        # Use pre-trained encoder weights
        # Train experts and decoder
        pass

    def stage3_real_hf_finetuning(self):
        """Fine-tune on real HF recordings."""
        # Use real WebSDR/KiwiSDR recordings
        # Fine-tune all components
        pass
```

### Strategy 2: Data Augmentation with Real Noise

```python
class RealNoiseAugmentation:
    """Mix real HF noise with synthetic signals."""

    def __init__(self, real_noise_path='real_hf_noise.npy'):
        # Load real noise recordings from quiet HF band
        self.noise_library = np.load(real_noise_path)

    def augment(self, clean_signal):
        # Pick random noise segment
        noise_start = np.random.randint(0, len(self.noise_library) - len(clean_signal))
        noise = self.noise_library[noise_start:noise_start + len(clean_signal)]

        # Mix with SNR control
        snr_db = np.random.uniform(-15, 20)
        signal_power = np.mean(np.abs(clean_signal)**2)
        noise_power = np.mean(np.abs(noise)**2)

        target_noise_power = signal_power / (10**(snr_db/10))
        noise_scaled = noise * np.sqrt(target_noise_power / noise_power)

        return clean_signal + noise_scaled
```

### Strategy 3: Domain Adaptation

```python
class DomainAdaptationTrainer:
    """
    Train with both synthetic and real data simultaneously.
    Use domain adversarial training to make features domain-invariant.
    """

    def __init__(self, encoder, decoder, discriminator):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator  # Tells synthetic vs real

    def train_step(self, synthetic_batch, real_batch):
        # 1. Train decoder on labeled synthetic data
        synthetic_features = self.encoder(synthetic_batch['iq'])
        predictions = self.decoder(synthetic_features)
        task_loss = compute_loss(predictions, synthetic_batch['labels'])

        # 2. Train discriminator to classify domain
        real_features = self.encoder(real_batch['iq'])
        domain_pred_synthetic = self.discriminator(synthetic_features)
        domain_pred_real = self.discriminator(real_features)

        domain_loss = bce_loss(domain_pred_synthetic, 0) + \
                     bce_loss(domain_pred_real, 1)

        # 3. Train encoder to fool discriminator (domain confusion)
        adversarial_loss = bce_loss(domain_pred_synthetic, 1)

        # Combined loss
        total_loss = task_loss + 0.1 * adversarial_loss
        total_loss.backward()
```

## Recommended Data Collection Plan

### Week 1-2: Build Dataset Infrastructure
```bash
# 1. Set up automated WebSDR recording
python scripts/record_websdr.py --freq 14070 --duration 3600 --output data/real/

# 2. Download RadioML
wget https://www.deepsig.ai/datasets/radioml-2018.01a.tar.gz

# 3. Record from local SDR (if available)
rtl_sdr -f 14070000 -s 2400000 -n 28800000 real_recording.iq
```

### Week 3-4: Collect Diverse Conditions
- **Different times of day**: Dawn, noon, dusk, midnight (4× per day)
- **Different bands**: 40m, 20m, 15m (3 bands)
- **Different locations**: Use multiple KiwiSDRs worldwide (5-10 locations)
- **Target**: 100 hours of recordings = ~200 GB

### Week 5-6: Process and Label
```python
def process_real_recordings():
    """
    Extract useful segments from recordings.
    """
    # 1. Detect signals (energy detection)
    # 2. Classify signals (FT8, CW, SSB, etc.)
    # 3. Extract clean segments
    # 4. Label propagation conditions (manual or automatic)
    # 5. Create dataset with metadata

    return processed_dataset
```

## Quick Start: Easiest Real Data Source

**Immediate action** (can do in 1 hour):

```python
# Use KiwiSDR to record real HF conditions
!pip install kiwiclient

from kiwiclient import KiwiSDRStream
import numpy as np

# Record 10 minutes from University of Twente WebSDR
sdr = KiwiSDRStream(host='websdr.ewi.utwente.nl', port=8073)
sdr.connect()
sdr.set_freq(14070)  # FT8 frequency
sdr.set_mod('iq')

# Record
real_samples = sdr.record(600)  # 10 minutes

# Save
np.save('data/real/websdr_14070_600s.npy', real_samples)

print(f"Recorded {len(real_samples)} real HF samples")
print(f"Use these for: noise augmentation, domain adaptation, fine-tuning")
```

## Summary

**Best hybrid approach:**
1. **Generate 200K synthetic CASCADE samples** (base training)
2. **Download RadioML 2018** (30 GB, pre-training)
3. **Record 10-20 hours from KiwiSDR/WebSDR** (~20 GB, real conditions)
4. **Mix real noise into synthetic signals** (augmentation)
5. **Fine-tune on real recordings** (domain adaptation)

This gives you:
- Strong base from synthetic data (controlled, complete coverage)
- Real-world robustness from actual HF recordings
- Best of both worlds!
