# CASCADE Model → Coral Edge TPU Compatibility Analysis

## Summary

**Overall Assessment**: ✅ **Highly Compatible** with minor modifications needed

**Model Size**: ~1.4 MB quantized (INT8) - Easily fits in Coral's 8 MB on-chip memory

---

## Layer-by-Layer Compatibility

### ✅ Fully Compatible Operations

| Layer Type | Count | TFLite Support | Notes |
|------------|-------|----------------|-------|
| Conv1d | 8+ | ✅ Full | Standard convolution |
| Linear/Dense | 45+ | ✅ Full | Fully connected layers |
| BatchNorm1d | 20+ | ✅ Full | Native support |
| LayerNorm | 3 | ✅ Full | Native support |
| ReLU | Many | ✅ Full | Standard activation |
| Dropout | Many | ✅ N/A | Disabled at inference |
| GRU | 1 | ✅ Full | Recurrent layer supported |

### ⚠️ Needs Attention

| Operation | Location | Issue | Solution |
|-----------|----------|-------|----------|
| **MultiheadAttention** | TimingExpert (line 224)<br>IntegrationDecoder (lines 353-354) | Partial TFLite support;<br>may not compile for TPU | Replace with manual attention<br>implementation or use<br>TF-compatible version |
| **AdaptiveAvgPool1d** | IQEncoder (line 95) | Dynamic output size | Replace with fixed-size<br>`AvgPool1d` or `GlobalAvgPool` |
| **squeeze(-1)** | IQEncoder (line 114) | Minor: dimension ops | Replace with `reshape` |

---

## Potential Incompatibilities by Component

### IQ Embedding Encoder
```python
✅ Conv1d layers (standard)
✅ BatchNorm1d (standard)
✅ ReLU activation (standard)
⚠️  AdaptiveAvgPool1d → Replace with AvgPool1d(kernel_size=calculated)
⚠️  squeeze(-1) → Replace with reshape
✅ Linear layer (standard)
```

### Expert Networks

**QRN Expert, Signal Expert, QRM Expert:**
```python
✅ Conv1d layers (standard)
✅ BatchNorm/LayerNorm (standard)
✅ ReLU/GELU (standard)
✅ All compatible
```

**Timing Expert:**
```python
✅ Conv1d layers (standard)
⚠️  MultiheadAttention → May need custom implementation
✅ Otherwise compatible
```

**Channel Expert:**
```python
✅ Conv1d layers (standard)
✅ GRU (supported in TFLite)
✅ Compatible
```

### Integration Decoder
```python
✅ Linear layers (standard)
✅ LayerNorm (standard)
⚠️  MultiheadAttention (self + cross) → Need workaround
✅ Otherwise compatible
```

---

## Recommended Modifications for Coral TPU

### 1. Replace MultiheadAttention (Critical)

**Option A: Manual Implementation**
```python
class ManualAttention(nn.Module):
    """TFLite-compatible attention."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # Implement attention manually using matmul operations
        # TFLite supports matmul, softmax, reshape
        ...
```

**Option B: Remove Attention**
- Replace with Conv1d layers (faster on Coral TPU)
- Slight accuracy trade-off but much better compatibility

### 2. Fix AdaptiveAvgPool1d

**Current:**
```python
self.global_pool = nn.AdaptiveAvgPool1d(1)
```

**TFLite-Compatible:**
```python
# Calculate input size after convolutions (fixed at 2048 input)
# After 4 stride-2 convs: 2048 → 1024 → 512 → 256 → 128
self.global_pool = nn.AvgPool1d(kernel_size=128, stride=1)
# Or just use: torch.mean(x, dim=-1, keepdim=True)
```

### 3. Replace squeeze/unsqueeze

**Current:**
```python
x = x.squeeze(-1)
```

**TFLite-Compatible:**
```python
x = x.reshape(batch_size, -1)  # Explicit reshape
```

---

## Conversion Pipeline

### Recommended Steps:

1. **Modify model for TFLite**
   - Replace MultiheadAttention with manual implementation
   - Fix AdaptivePooling → fixed-size pooling
   - Replace squeeze with reshape

2. **Export to ONNX**
   ```python
   torch.onnx.export(
       model,
       dummy_input,
       "cascade_model.onnx",
       input_names=['iq_input'],
       output_names=['pattern_logits', 'frequency_logits', ...],
       dynamic_axes=None  # Fixed batch size for TPU
   )
   ```

3. **Convert ONNX → TensorFlow**
   ```python
   import onnx
   from onnx_tf.backend import prepare

   onnx_model = onnx.load("cascade_model.onnx")
   tf_rep = prepare(onnx_model)
   tf_rep.export_graph("cascade_model_tf")
   ```

4. **Convert TF → TFLite**
   ```python
   converter = tf.lite.TFLiteConverter.from_saved_model("cascade_model_tf")
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.target_spec.supported_ops = [
       tf.lite.OpsSet.TFLITE_BUILTINS_INT8
   ]
   tflite_model = converter.convert()
   ```

5. **Compile for Coral TPU**
   ```bash
   edgetpu_compiler cascade_model.tflite
   ```

---

## Expected Performance on Coral TPU

**Inference Time Estimate:**
- Input: 2048 samples (42ms of audio)
- Model: ~1.4M params, mostly Conv1d + attention
- **Expected latency: 5-15ms** (well within real-time budget)

**Throughput:**
- **70-200 inferences/sec** on single Coral TPU
- Sufficient for real-time decoding

---

## Deployment Architecture

```
Raspberry Pi 5 + Coral TPU
├── Audio Input (48kHz SDR)
├── Preprocessing (CPU): Windowing, normalization
├── CASCADE RX Model (Coral TPU): Decode messages
│   ├── IQ Encoder
│   ├── 5 Experts (parallel inference)
│   └── Integration Decoder
├── Postprocessing (CPU): Message assembly, error correction
└── Output: Decoded messages
```

**Power Budget:**
- Raspberry Pi 5: ~5-8W
- Coral TPU: ~2-3W
- **Total: <10W** for full CASCADE receiver

---

## Critical Modifications Needed

### High Priority (Must Fix):
1. ✅ **Replace `AdaptiveAvgPool1d` with fixed-size pooling**
2. ⚠️ **Replace or simplify `MultiheadAttention`**
   - Option 1: Manual implementation with matmul ops
   - Option 2: Replace with Conv1d layers
3. ✅ **Use reshape instead of squeeze/unsqueeze**

### Medium Priority (Should Fix):
4. ✅ **Set fixed batch size** (no dynamic shapes)
5. ✅ **Test quantization accuracy** (INT8 vs FP32)

### Low Priority (Nice to Have):
6. ✅ **Optimize for Coral TPU compiler** (specific op fusion)

---

## Recommendation

**✅ CASCADE is Coral TPU compatible** with 2-3 days of conversion work:

**Day 1**: Modify PyTorch model
- Replace MultiheadAttention with manual implementation
- Fix AdaptivePooling
- Test modified model maintains accuracy

**Day 2**: Export and convert
- PyTorch → ONNX
- ONNX → TensorFlow
- TensorFlow → TFLite
- Debug conversion issues

**Day 3**: Coral TPU compilation and testing
- Compile for Edge TPU
- Benchmark on Raspberry Pi
- Verify accuracy and latency

**Expected Result**: Functional CASCADE receiver running at <10ms latency, <10W power, suitable for portable HF operation.

---

## Alternative: Use Coral's TF Models Directly

Instead of converting, consider:
- **Train in TensorFlow/Keras from the start** (future models)
- Easier Coral TPU deployment
- But loses PyTorch ecosystem benefits

For this model: **Conversion is feasible** - proceed with PyTorch → TFLite pipeline.
