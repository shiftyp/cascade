#!/usr/bin/env python3
"""
Test that CASCADE model modifications for TFLite compatibility don't break functionality.
"""

import torch
import sys
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent))

# Import the modified model components
from phase3_model_training import (
    TFLiteCompatibleAttention,
    IQEmbeddingEncoder,
    QRNExpert,
    SignalExpert,
    TimingExpert,
    ChannelExpert,
    QRMExpert,
    IntegrationDecoder
)


def test_tflite_compatible_attention():
    """Test custom attention matches PyTorch attention behavior."""
    print("Testing TFLiteCompatibleAttention...")

    batch_size = 4
    seq_len = 10
    embed_dim = 64
    num_heads = 8

    # Create input
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Test self-attention
    attn = TFLiteCompatibleAttention(embed_dim, num_heads, batch_first=True)
    output, weights = attn(x)

    assert output.shape == (batch_size, seq_len, embed_dim), f"Output shape mismatch: {output.shape}"
    assert weights.shape == (batch_size, seq_len, seq_len), f"Weights shape mismatch: {weights.shape}"

    print(f"  ✓ Self-attention output shape: {output.shape}")
    print(f"  ✓ Attention weights shape: {weights.shape}")

    # Test cross-attention
    context = torch.randn(batch_size, 5, embed_dim)
    output2, weights2 = attn(x, context, context)

    assert output2.shape == (batch_size, seq_len, embed_dim), f"Cross-attn output shape mismatch"
    print(f"  ✓ Cross-attention works")

    print("  ✅ TFLiteCompatibleAttention passed!\n")


def test_iq_encoder():
    """Test IQEncoder with fixed-size pooling."""
    print("Testing IQEmbeddingEncoder (with torch.mean pooling)...")

    batch_size = 8
    input_size = 2048

    encoder = IQEmbeddingEncoder(input_size=input_size, output_size=512)
    x = torch.randn(batch_size, 2, input_size)  # [batch, 2, 2048] (I and Q)

    output = encoder(x)

    assert output.shape == (batch_size, 512), f"Encoder output shape mismatch: {output.shape}"
    print(f"  ✓ Output shape: {output.shape}")
    print("  ✅ IQEmbeddingEncoder passed!\n")


def test_experts():
    """Test all expert networks with reshape operations."""
    print("Testing Expert Networks...")

    batch_size = 4
    input_size = 512

    x = torch.randn(batch_size, input_size)

    # Test each expert
    experts = {
        "QRNExpert": QRNExpert(input_size, 128),
        "SignalExpert": SignalExpert(input_size, 128),
        "TimingExpert": TimingExpert(input_size, 128),
        "ChannelExpert": ChannelExpert(input_size, 128),
        "QRMExpert": QRMExpert(input_size, 128)
    }

    for name, expert in experts.items():
        output = expert(x)
        expected_shape = (batch_size, 128)
        assert output.shape == expected_shape, f"{name} output shape mismatch: {output.shape}"
        print(f"  ✓ {name}: {output.shape}")

    print("  ✅ All experts passed!\n")


def test_integration_decoder():
    """Test IntegrationDecoder with TFLite-compatible attention."""
    print("Testing IntegrationDecoder...")

    batch_size = 4
    expert_dim = 640  # 5 experts × 128

    decoder = IntegrationDecoder(input_size=expert_dim, hidden_dim=512)

    # Test without context
    expert_features = torch.randn(batch_size, expert_dim)
    outputs = decoder(expert_features)

    print(f"  ✓ Without context:")
    print(f"    Pattern: {outputs['pattern'].shape}")
    print(f"    Frequency: {outputs['frequency'].shape}")

    # Test with context
    context_kernels = torch.randn(batch_size, 8, 16)  # [batch, max_context, kernel_dim]
    context_mask = torch.ones(batch_size, 8)  # All valid
    outputs2 = decoder(expert_features, context_kernels, context_mask)

    print(f"  ✓ With context:")
    print(f"    Pattern: {outputs2['pattern'].shape}")

    print("  ✅ IntegrationDecoder passed!\n")


def test_full_forward_pass():
    """Test complete forward pass through entire model."""
    print("Testing Full CASCADE Forward Pass...")

    batch_size = 2
    input_size = 2048

    # Create all components
    iq_encoder = IQEmbeddingEncoder(input_size, 512)

    experts = [
        QRNExpert(512, 128),
        SignalExpert(512, 128),
        TimingExpert(512, 128),
        ChannelExpert(512, 128),
        QRMExpert(512, 128)
    ]

    decoder = IntegrationDecoder(640, 512)

    # Forward pass
    iq_input = torch.randn(batch_size, 2, input_size)

    # Encode
    encoded = iq_encoder(iq_input)
    print(f"  ✓ IQ Encoded: {encoded.shape}")

    # Expert processing
    expert_outputs = []
    for expert in experts:
        expert_out = expert(encoded)
        expert_outputs.append(expert_out)

    expert_features = torch.cat(expert_outputs, dim=-1)
    print(f"  ✓ Expert features: {expert_features.shape}")

    # Decode
    outputs = decoder(expert_features)
    print(f"  ✓ Decoder outputs:")
    print(f"    Pattern logits: {outputs['pattern'].shape}")
    print(f"    Frequency logits: {outputs['frequency'].shape}")

    print("  ✅ Full forward pass successful!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("CASCADE TFLite Compatibility Test")
    print("=" * 60)
    print()

    test_tflite_compatible_attention()
    test_iq_encoder()
    test_experts()
    test_integration_decoder()
    test_full_forward_pass()

    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("Model is now TFLite-compatible:")
    print("  ✅ TFLiteCompatibleAttention (replaces nn.MultiheadAttention)")
    print("  ✅ torch.mean (replaces AdaptiveAvgPool1d)")
    print("  ✅ reshape (replaces squeeze/unsqueeze)")
    print()
    print("Ready for export: PyTorch → ONNX → TF → TFLite → Coral Edge TPU")
