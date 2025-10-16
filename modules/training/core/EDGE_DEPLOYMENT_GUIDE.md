# CASCADE Edge Deployment Guide

Complete guide for deploying CASCADE RX decoder on edge devices (Raspberry Pi + AI accelerator).

---

## Table of Contents

1. [Hardware Options](#hardware-options)
2. [Model Export (PyTorch → ONNX)](#model-export)
3. [Hailo-8 Deployment (Recommended)](#hailo-8-deployment)
4. [Coral TPU Deployment (Alternative)](#coral-tpu-deployment)
5. [Raspberry Pi Integration](#raspberry-pi-integration)
6. [Performance Benchmarks](#performance-benchmarks)

---

## Hardware Options

### Option 1: Hailo-8 (Recommended) 🌟

**Hardware:**
- Raspberry Pi 5 (8GB)
- Raspberry Pi AI HAT+ (includes Hailo-8)
- ~$150 total

**Specs:**
- 26 TOPS INT8
- Direct ONNX support
- ~3-5ms inference latency
- ~8-10W total power

**Pros:**
- ✅ Direct PyTorch/ONNX conversion (simple)
- ✅ 6.5× more powerful than Coral
- ✅ Official Raspberry Pi support
- ✅ Better for complex models like CASCADE

**Cons:**
- ❌ More expensive ($70 vs $25)
- ❌ Newer ecosystem (less mature)

### Option 2: Coral Edge TPU

**Hardware:**
- Raspberry Pi 4/5
- Coral USB Accelerator or M.2 module
- ~$80-100 total

**Specs:**
- 4 TOPS INT8
- TFLite models only
- ~5-15ms inference latency
- ~7-8W total power

**Pros:**
- ✅ Cheaper ($25)
- ✅ Mature ecosystem
- ✅ Well-documented

**Cons:**
- ❌ Requires TFLite conversion (complex)
- ❌ Less TOPS (may limit throughput)

---

## Model Export (PyTorch → ONNX)

### Step 1: Prepare Trained Model

```python
import torch
from phase3_model_training import IQEmbeddingEncoder, IntegrationDecoder
# Import all expert networks

# Load trained checkpoint
checkpoint = torch.load('cascade_model.pth', map_location='cpu')

# Create model (same architecture as training)
iq_encoder = IQEmbeddingEncoder(input_size=2048, output_size=512)
experts = [
    QRNExpert(512, 128),
    SignalExpert(512, 128),
    TimingExpert(512, 128),
    ChannelExpert(512, 128),
    QRMExpert(512, 128)
]
decoder = IntegrationDecoder(640, 512)

# Load weights
iq_encoder.load_state_dict(checkpoint['iq_encoder'])
for i, expert in enumerate(experts):
    expert.load_state_dict(checkpoint[f'expert_{i}'])
decoder.load_state_dict(checkpoint['decoder'])

# Set to eval mode
iq_encoder.eval()
for expert in experts:
    expert.eval()
decoder.eval()
```

### Step 2: Create Inference-Only Wrapper

```python
class CASCADEInference(torch.nn.Module):
    """
    Single-module wrapper for CASCADE inference.
    Combines encoder, experts, and decoder for easier export.
    """

    def __init__(self, iq_encoder, experts, decoder):
        super().__init__()
        self.iq_encoder = iq_encoder
        self.experts = torch.nn.ModuleList(experts)
        self.decoder = decoder

    def forward(self, iq_input):
        """
        Forward pass for inference.

        Args:
            iq_input: [batch, 2, 2048] - I/Q samples

        Returns:
            Dict with pattern_logits, frequency_logits, etc.
        """
        # Encode I/Q
        encoded = self.iq_encoder(iq_input)

        # Expert processing
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(encoded)
            expert_outputs.append(expert_out)

        # Concatenate expert outputs
        expert_features = torch.cat(expert_outputs, dim=-1)

        # Decode
        outputs = self.decoder(expert_features)

        return outputs

# Create inference model
cascade_inference = CASCADEInference(iq_encoder, experts, decoder)
cascade_inference.eval()
```

### Step 3: Export to ONNX

```python
import torch.onnx

# Create dummy input (batch_size=1 for edge deployment)
dummy_input = torch.randn(1, 2, 2048)

# Export to ONNX
torch.onnx.export(
    cascade_inference,
    dummy_input,
    "cascade_rx_decoder.onnx",
    export_params=True,
    opset_version=13,  # Use ONNX opset 13 or higher
    do_constant_folding=True,  # Optimize constant ops
    input_names=['iq_input'],
    output_names=['pattern_logits', 'frequency_logits', 'modulation_logits',
                  'data_rate_logits', 'duration'],
    dynamic_axes={
        'iq_input': {0: 'batch_size'},  # Allow dynamic batch if needed
    }
)

print("✅ Exported to cascade_rx_decoder.onnx")
```

### Step 4: Verify ONNX Model

```python
import onnx
import onnxruntime as ort

# Load and check ONNX model
onnx_model = onnx.load("cascade_rx_decoder.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX model is valid")

# Test inference with ONNX Runtime
ort_session = ort.InferenceSession("cascade_rx_decoder.onnx")

# Run inference
dummy_input_np = dummy_input.numpy()
outputs = ort_session.run(None, {'iq_input': dummy_input_np})

print(f"✅ ONNX inference works")
print(f"   Pattern logits shape: {outputs[0].shape}")
print(f"   Frequency logits shape: {outputs[1].shape}")
```

---

## Hailo-8 Deployment

### Step 1: Install Hailo Tools

On development machine (Ubuntu/Linux):

```bash
# Download Hailo Dataflow Compiler
# https://hailo.ai/developer-zone/

# Install Hailo SDK
pip install hailo-sdk-client

# Install Hailo Model Zoo (optional, for examples)
git clone https://github.com/hailo-ai/hailo_model_zoo.git
```

### Step 2: Compile Model for Hailo-8

```bash
# Parse ONNX model
hailo parser onnx cascade_rx_decoder.onnx \
    --hw-arch hailo8 \
    --output cascade_rx.har

# Optimize model
hailo optimize \
    --har cascade_rx.har \
    --hw-arch hailo8 \
    --output cascade_rx_optimized.har

# Compile for Hailo-8
hailo compiler \
    --har cascade_rx_optimized.har \
    --hw-arch hailo8 \
    --output cascade_rx.hef

# Result: cascade_rx.hef (Hailo Executable Format)
```

### Step 3: Quantize Model (INT8)

```python
from hailo_sdk_client import ClientRunner

# Create calibration dataset (representative inputs)
calibration_data = []
for i in range(100):
    # Load real I/Q samples from validation set
    iq_sample = load_validation_sample(i)  # [2, 2048]
    calibration_data.append(iq_sample)

# Run quantization
runner = ClientRunner(har="cascade_rx_optimized.har")
runner.optimize(calibration_data)
runner.compile(output_file="cascade_rx_quantized.hef")

print("✅ Quantized model ready for Hailo-8")
```

### Step 4: Deploy on Raspberry Pi

**Install HailoRT on Raspberry Pi:**

```bash
# On Raspberry Pi
wget https://hailo.ai/downloads/hailort/hailort-4.x.x-arm64.deb
sudo dpkg -i hailort-4.x.x-arm64.deb

# Install Python bindings
pip install hailort
```

**Inference Code:**

```python
from hailo_platform import HEF, VDevice, ConfigureParams, InferVStreams

# Load model
hef = HEF("cascade_rx_quantized.hef")

# Create virtual device
vdevice_params = VDevice.create_params()
vdevice = VDevice(vdevice_params)

# Configure network
network_group = vdevice.configure(hef)[0]
network_group_params = network_group.create_params()

# Run inference
with InferVStreams(network_group, network_group_params) as infer_pipeline:
    # Prepare input (from SDR)
    iq_input = get_iq_from_sdr()  # [2, 2048]
    iq_input_uint8 = quantize_input(iq_input)  # Convert to INT8

    # Infer
    outputs = infer_pipeline.infer({
        'iq_input': iq_input_uint8
    })

    # Decode outputs
    pattern_logits = outputs['pattern_logits']
    frequency_logits = outputs['frequency_logits']

    # Post-process (argmax, etc.)
    pattern_id = np.argmax(pattern_logits)
    frequency_triple = np.argmax(frequency_logits)

    print(f"Detected: Pattern {pattern_id}, Frequency triple {frequency_triple}")
```

---

## Coral TPU Deployment

### Step 1: Convert ONNX → TensorFlow

```bash
# Install converter
pip install onnx-tf

# Convert
python -c "
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load('cascade_rx_decoder.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('cascade_rx_tf')
"
```

### Step 2: Convert TF → TFLite

```python
import tensorflow as tf

# Load TF model
converter = tf.lite.TFLiteConverter.from_saved_model('cascade_rx_tf')

# Quantization settings
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# Representative dataset for quantization
def representative_dataset():
    for i in range(100):
        # Load calibration samples
        iq_sample = load_validation_sample(i)
        yield [iq_sample.astype(np.float32)]

converter.representative_dataset = representative_dataset

# Convert
tflite_model = converter.convert()

# Save
with open('cascade_rx.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ TFLite model created")
```

### Step 3: Compile for Edge TPU

```bash
# Install Edge TPU compiler
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update
sudo apt install edgetpu-compiler

# Compile for Edge TPU
edgetpu_compiler cascade_rx.tflite

# Result: cascade_rx_edgetpu.tflite
```

### Step 4: Run on Raspberry Pi

```python
from pycoral.utils import edgetpu
from pycoral.adapters import common
import numpy as np

# Initialize TPU
interpreter = edgetpu.make_interpreter('cascade_rx_edgetpu.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input
iq_input = get_iq_from_sdr()  # [2, 2048]
iq_input_quantized = quantize_input(iq_input, input_details[0])

# Set input
interpreter.set_tensor(input_details[0]['index'], iq_input_quantized)

# Run inference
interpreter.invoke()

# Get outputs
pattern_logits = interpreter.get_tensor(output_details[0]['index'])
frequency_logits = interpreter.get_tensor(output_details[1]['index'])

# Decode
pattern_id = np.argmax(pattern_logits)
frequency_triple = np.argmax(frequency_logits)

print(f"Detected: Pattern {pattern_id}, Frequency triple {frequency_triple}")
```

---

## Raspberry Pi Integration

### Complete RX Pipeline

```python
#!/usr/bin/env python3
"""
CASCADE RX Decoder - Raspberry Pi + Hailo-8/Coral TPU
"""

import numpy as np
from scipy import signal
import sounddevice as sd  # For audio from SDR

class CASCADEReceiver:
    """Real-time CASCADE receiver for Raspberry Pi."""

    def __init__(self, model_path, device_type='hailo'):
        self.device_type = device_type

        if device_type == 'hailo':
            from hailo_platform import HEF, VDevice
            # Load Hailo model
            self.hef = HEF(model_path)
            self.device = VDevice()
            self.network = self.device.configure(self.hef)[0]
        elif device_type == 'coral':
            from pycoral.utils import edgetpu
            # Load Coral model
            self.interpreter = edgetpu.make_interpreter(model_path)
            self.interpreter.allocate_tensors()

        # Circular buffer for continuous reception
        self.buffer_size = 4096  # 85ms at 48kHz
        self.buffer = np.zeros((2, self.buffer_size))
        self.write_idx = 0

    def process_audio_chunk(self, audio_chunk):
        """
        Process incoming I/Q audio from SDR.

        Args:
            audio_chunk: [2, chunk_size] - I and Q samples
        """
        chunk_size = audio_chunk.shape[1]

        # Add to circular buffer
        end_idx = self.write_idx + chunk_size
        if end_idx <= self.buffer_size:
            self.buffer[:, self.write_idx:end_idx] = audio_chunk
        else:
            # Wrap around
            first_part = self.buffer_size - self.write_idx
            self.buffer[:, self.write_idx:] = audio_chunk[:, :first_part]
            self.buffer[:, :chunk_size - first_part] = audio_chunk[:, first_part:]

        self.write_idx = (self.write_idx + chunk_size) % self.buffer_size

        # Check if we have a complete 2048-sample window
        if self.write_idx % 2048 == 0:
            # Extract window for inference
            read_idx = (self.write_idx - 2048) % self.buffer_size

            if read_idx + 2048 <= self.buffer_size:
                window = self.buffer[:, read_idx:read_idx + 2048]
            else:
                # Wrapped window
                first = self.buffer[:, read_idx:]
                second = self.buffer[:, :(read_idx + 2048) % self.buffer_size]
                window = np.concatenate([first, second], axis=1)

            # Run inference
            self.run_inference(window)

    def run_inference(self, iq_window):
        """Run CASCADE inference on 2048-sample window."""
        # Normalize input
        iq_normalized = iq_window / (np.std(iq_window) + 1e-10)

        if self.device_type == 'hailo':
            # Hailo-8 inference
            with InferVStreams(self.network) as infer:
                output = infer.infer({'iq_input': iq_normalized.astype(np.float32)})
        elif self.device_type == 'coral':
            # Coral TPU inference
            self.interpreter.set_tensor(
                self.interpreter.get_input_details()[0]['index'],
                iq_normalized.astype(np.float32)
            )
            self.interpreter.invoke()
            output = {}
            for out in self.interpreter.get_output_details():
                output[out['name']] = self.interpreter.get_tensor(out['index'])

        # Decode outputs
        pattern_id = np.argmax(output['pattern_logits'])
        frequency_triple = np.argmax(output['frequency_logits'])
        modulation = np.argmax(output['modulation_logits'])

        # Check if detection threshold met
        pattern_confidence = np.max(output['pattern_logits'])

        if pattern_confidence > 0.7:  # Threshold
            print(f"✓ Message detected:")
            print(f"  Pattern: {pattern_id}")
            print(f"  Frequency: {frequency_triple}")
            print(f"  Modulation: {modulation}")
            print(f"  Confidence: {pattern_confidence:.2f}")

    def start_reception(self, sdr_device='rtlsdr'):
        """Start continuous reception from SDR."""
        print(f"Starting CASCADE receiver on {self.device_type.upper()}...")

        # Open audio stream from SDR
        # Assumes SDR outputs I/Q audio at 48kHz
        stream = sd.InputStream(
            channels=2,  # Stereo (I and Q)
            samplerate=48000,
            blocksize=512,  # Process in 512-sample chunks
            callback=self.audio_callback
        )

        with stream:
            print("Listening for CASCADE signals...")
            input("Press Enter to stop\n")

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio error: {status}")

        # indata: [frames, 2] (interleaved I/Q)
        iq_chunk = indata.T  # [2, frames]
        self.process_audio_chunk(iq_chunk)

# Usage:
receiver = CASCADEReceiver('cascade_rx_quantized.hef', device_type='hailo')
receiver.start_reception()
```

---

## Performance Benchmarks

### Expected Performance (Raspberry Pi 5)

| Metric | Hailo-8 | Coral TPU | Pi CPU Only |
|--------|---------|-----------|-------------|
| **Inference Time** | 3-5ms | 5-15ms | 50-200ms |
| **Throughput** | 200-300 msgs/sec | 70-200 msgs/sec | 5-20 msgs/sec |
| **Power (total)** | 8-10W | 7-8W | 5-6W |
| **Cost** | $150 | $80-100 | $80 |
| **Conversion Effort** | Low (direct ONNX) | High (via TFLite) | None |

### Real-Time Capability

For CASCADE real-time decoding:
- **Window size**: 2048 samples = 42.7ms @ 48kHz
- **Must process in**: <43ms for real-time
- **Hailo-8**: ✅ 3-5ms (8-14× margin)
- **Coral TPU**: ✅ 5-15ms (3-8× margin)
- **Pi CPU**: ❌ 50-200ms (too slow)

**Both accelerators support real-time CASCADE decoding!**

---

## Model Size Analysis

### Trained Model (FP32):
```
IQ Encoder:     ~220K params = 0.88 MB
5 Experts:      ~1000K params = 4.0 MB
Integration:    ~280K params = 1.12 MB
--------------------------------
Total:          ~1.5M params = 6.0 MB FP32
```

### Quantized (INT8):
```
Total:          ~1.5M params = 1.5 MB INT8
```

**Fits easily** in:
- Hailo-8: Up to 50MB on-chip
- Coral TPU: Up to 8MB on-chip (✅ 1.5MB fits)

---

## Deployment Checklist

### Before Deployment:

- [ ] Train CASCADE model on GPU
- [ ] Test on validation set (verify accuracy)
- [ ] Export to ONNX using export script
- [ ] Verify ONNX with ONNX Runtime
- [ ] Test with sample I/Q data

### For Hailo-8:

- [ ] Install Hailo SDK on dev machine
- [ ] Compile ONNX → HEF
- [ ] Create INT8 calibration dataset
- [ ] Quantize and test accuracy
- [ ] Deploy HEF to Raspberry Pi with HailoRT
- [ ] Integrate with SDR audio input
- [ ] Test real-time performance

### For Coral TPU:

- [ ] Convert ONNX → TF → TFLite
- [ ] Compile for Edge TPU
- [ ] Test TFLite model accuracy
- [ ] Deploy to Raspberry Pi with pycoral
- [ ] Integrate with SDR audio input
- [ ] Test real-time performance

---

## Recommended Configuration

### For Portable HF Operation:

```
Hardware Stack:
├── Raspberry Pi 5 (8GB)
├── Raspberry Pi AI HAT+ (Hailo-8)
├── RTL-SDR or similar (HF upconverter)
└── Power: 12V battery or USB-PD

Software Stack:
├── Raspberry Pi OS (64-bit)
├── HailoRT runtime
├── CASCADE RX model (.hef)
├── SDR driver (rtl-sdr, SoapySDR)
└── Custom receiver app (Python or C++)

Power Budget:
├── Pi 5: ~5-6W
├── Hailo-8: ~2-3W
├── SDR: ~1-2W
└── Total: ~8-11W (2-3 hours on 5000mAh USB battery)
```

### Performance Goals:
- ✅ Decode CASCADE messages in real-time
- ✅ Handle 30+ simultaneous users (4-center design)
- ✅ Operate at -15 dB SNR (pattern detection)
- ✅ Operate at -6 to +20 dB SNR (data decoding)
- ✅ <10W power consumption
- ✅ Portable (battery-powered)

---

## Next Steps

1. **Complete training** on GPU (100M samples)
2. **Export best checkpoint** to ONNX
3. **Order hardware**: Raspberry Pi 5 + AI HAT+ (Hailo-8)
4. **Set up deployment pipeline** (compile, quantize, test)
5. **Integrate with SDR** (audio input pipeline)
6. **Field test** portable HF operation

---

## Troubleshooting

### Model won't compile for Hailo/Coral:
- Check ONNX opset version (use 11-13)
- Verify all operations are supported (run onnx.checker)
- Try simplifying problematic layers

### Accuracy drops after quantization:
- Increase calibration dataset size (100 → 1000 samples)
- Use post-training quantization (PTQ)
- Consider quantization-aware training (QAT)

### Inference too slow:
- Check batch size (use 1 for lowest latency)
- Verify model compiled for correct hardware
- Profile to find bottleneck layers

### Integration issues:
- Verify I/Q sample format (float32, normalized)
- Check input shape exactly matches export
- Test with known-good sample first

---

## Conclusion

✅ **CASCADE is fully compatible with edge deployment**

**Recommended**: Hailo-8 on Raspberry Pi 5
- Direct ONNX support (simple)
- 26 TOPS (fast)
- <5ms latency (real-time capable)
- <10W power (portable)

**Alternative**: Coral TPU
- Cheaper ($25 vs $70)
- Works but requires TFLite conversion
- Adequate performance for CASCADE

Both options enable **portable, battery-powered HF CASCADE receivers** suitable for field operation, emergency communications, and amateur radio use.
