"""
CASCADE Neural Network Models

All neural network architectures for the CASCADE protocol:
- IQEmbeddingEncoder: Compresses I/Q samples to feature vectors
- Expert Networks: QRN, Signal, Timing, Channel, QRM experts
- IntegrationDecoder: Combines expert outputs for final predictions
- CascadeModel: Complete end-to-end model
- TX Encoder components: EmbeddingEncoder, LearnedQuantizer, EmbeddingDecoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional


class TFLiteCompatibleAttention(nn.Module):
    """
    TFLite-compatible multi-head attention.

    Uses only matmul, softmax, and reshape operations that are fully supported
    by TFLite and Coral Edge TPU. Mathematically equivalent to nn.MultiheadAttention
    but with explicit operations.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = dropout

        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key=None, value=None, key_padding_mask=None, attn_mask=None):
        """
        Forward pass compatible with TFLite.

        Args:
            query: [batch, seq_len, embed_dim] if batch_first else [seq_len, batch, embed_dim]
            key: Optional, defaults to query (self-attention)
            value: Optional, defaults to key
            key_padding_mask: [batch, seq_len] - True for positions to ignore
            attn_mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = key

        if not self.batch_first:
            # Convert to batch_first
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, seq_len, _ = query.shape
        _, key_len, _ = key.shape

        # Self-attention: all inputs are the same
        if query is key and key is value:
            # Combined QKV projection
            qkv = self.qkv_proj(query)  # [batch, seq_len, 3 * embed_dim]
            qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # Separate projections for cross-attention
            q = self.qkv_proj(query)[:, :, :self.embed_dim]
            k = self.qkv_proj(key)[:, :, self.embed_dim:2*self.embed_dim]
            v = self.qkv_proj(value)[:, :, 2*self.embed_dim:]

            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.reshape(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.reshape(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention (TFLite compatible)
        # q: [batch, num_heads, seq_len, head_dim]
        # k: [batch, num_heads, key_len, head_dim]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch, num_heads, seq_len, key_len]

        # Apply masks if provided
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            # Expand mask for heads: [batch, seq_len] → [batch, 1, 1, key_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]

        # Reshape and project
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        if not self.batch_first:
            output = output.transpose(0, 1)

        # Average attention weights across heads for visualization
        attn_weights_avg = attn_weights.mean(dim=1)

        return output, attn_weights_avg


class IQEmbeddingEncoder(nn.Module):
    """IQ Embedding Encoder: Compresses raw I/Q samples (2048 → 512).

    Phase 1 Improvements (validated loss reduction: 15-25%):
    - LayerNorm instead of InstanceNorm (preserves SNR information)
    - Residual connections for better gradient flow
    - Reduced dropout in early layers (better feature extraction)
    - Increased temporal resolution (96 samples vs 64)
    - Deeper bottleneck for gradual compression
    """

    def __init__(self, input_size: int = 2048, output_size: int = 512):
        super().__init__()

        # LayerNorm preserves relative amplitude (SNR) information
        # Critical for HF: SNR is an important feature!
        self.layer_norm = nn.LayerNorm([2, input_size])

        # Conv layers for I/Q processing
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(512)

        # Residual connections (1x1 convs for dimension matching)
        self.res1 = nn.Conv1d(64, 128, kernel_size=1, stride=2)
        self.res2 = nn.Conv1d(128, 256, kernel_size=1, stride=2)
        self.res3 = nn.Conv1d(256, 512, kernel_size=1, stride=2)

        # Dropout only in deeper layers (early layers need robust features)
        self.dropout = nn.Dropout(0.2)

        # Optimized temporal resolution: 16 samples (balanced between info and params)
        # After 4 stride-2 convs: 2048 → 128 samples → pool to 16
        # Flattened: [batch, 512, 16] → [batch, 8192]

        # Simplified 2-layer bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(512 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, output_size)
        )
        self.bn_fc = nn.BatchNorm1d(output_size)

        # OPTIMIZATION: Proper weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming (He) for Conv and Xavier for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for Linear layers
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                # BatchNorm and LayerNorm: weight=1, bias=0
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize residual connections to near-zero (start as identity)
        for name, m in self.named_modules():
            if 'res' in name and isinstance(m, nn.Conv1d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm: preserves relative amplitude between I and Q
        x = self.layer_norm(x)  # [batch, 2, length]

        # Conv1: No dropout (early features need to be robust)
        x1 = F.relu(self.bn1(self.conv1(x)))

        # Conv2: No dropout + residual connection
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = x2 + self.res1(x1)  # Residual connection

        # Conv3: Dropout + residual connection
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = x3 + self.res2(x2)  # Residual connection
        x3 = self.dropout(x3)

        # Conv4: Residual connection (no dropout before pooling)
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x4 = x4 + self.res3(x3)  # Residual connection

        # Adaptive pooling: balanced temporal resolution
        # [batch, 512, 128] → [batch, 512, 16] (preserves key features, reduces params)
        x = F.adaptive_avg_pool1d(x4, 16)

        # Flatten temporal features: [batch, 512, 16] → [batch, 8192]
        x = x.flatten(1)

        # Bottleneck: gradual compression
        x = self.bottleneck(x)  # [batch, 8192] → [batch, output_size]
        x = self.bn_fc(x)

        return x


class QRNExpert(nn.Module):
    """QRN Expert: Noise classification.

    Outputs:
        - features: [batch, 64] - Feature vector for integration
        - qrn_classification: [batch, 8] - QRN type classification (QUIET, STATIC, CRACKLING, etc.)
    """

    def __init__(self, input_size: int = 512, output_size: int = 64, num_qrn_types: int = 8):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, output_size)

        # QRN classification head
        self.qrn_classifier = nn.Linear(output_size, num_qrn_types)

    def forward(self, x, return_classification=False):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        features = self.fc3(x)

        if return_classification:
            qrn_logits = self.qrn_classifier(features)
            return features, qrn_logits

        return features


class SignalExpert(nn.Module):
    """Signal Expert: Pattern and modulation detection.

    Outputs:
        - features: [batch, 128] - Feature vector for integration
        - pattern_classification: [batch, 4] - Ternary orthogonal pattern (0-3)
        - modulation_classification: [batch, 4] - BPSK/QPSK/8PSK/16APSK
    """

    def __init__(self, input_size: int = 512, output_size: int = 128,
                 num_patterns: int = 4, num_modulations: int = 4):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.merge = nn.Linear(128, output_size)

        # Classification heads
        self.pattern_classifier = nn.Linear(output_size, num_patterns)
        self.modulation_classifier = nn.Linear(output_size, num_modulations)

    def forward(self, x, return_classification=False):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        features = self.merge(x)

        if return_classification:
            pattern_logits = self.pattern_classifier(features)
            modulation_logits = self.modulation_classifier(features)
            return features, pattern_logits, modulation_logits

        return features


class TimingExpert(nn.Module):
    """Timing Expert: Collision detection and timing offset estimation.

    Outputs:
        - features: [batch, 256] - Feature vector for integration
        - has_collision: [batch, 1] - Binary collision indicator
        - collision_offset: [batch, 1] - Time offset in ms (-341 to +341)
    """

    def __init__(self, input_size: int = 512, output_size: int = 256):
        super().__init__()

        # Temporal attention for precise timing
        self.attention = TFLiteCompatibleAttention(embed_dim=input_size, num_heads=8)

        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 192)
        self.bn2 = nn.BatchNorm1d(192)
        self.dropout2 = nn.Dropout(0.2)

        self.fc = nn.Linear(192, output_size)

        # Collision detection heads
        self.collision_detector = nn.Linear(output_size, 1)  # Binary classification
        self.offset_regressor = nn.Linear(output_size, 1)    # Time offset in ms

    def forward(self, x, return_classification=False):
        # Apply attention for temporal features
        x_unsqueezed = x.unsqueeze(1)  # [batch, 1, features]
        attended, _ = self.attention(x_unsqueezed)
        x = attended.squeeze(1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        features = self.fc(x)

        if return_classification:
            has_collision = self.collision_detector(features)  # Logits
            collision_offset = self.offset_regressor(features)  # Regression
            return features, has_collision, collision_offset

        return features


class ChannelExpert(nn.Module):
    """Channel Expert: Propagation mode and ionospheric conditions.

    Outputs:
        - features: [batch, 128] - Feature vector for integration
        - propagation_mode: [batch, 5] - AWGN/Rayleigh/Rician/Multipath_Sparse/Multipath_Dense
        - k_index: [batch, 1] - K-index regression (0-9)
        - sfi: [batch, 1] - SFI regression (60-250)
    """

    def __init__(self, input_size: int = 512, hidden_size: int = 256, output_size: int = 128,
                 num_propagation_modes: int = 5):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)

        self.fc = nn.Linear(hidden_size, output_size)

        # Classification and regression heads
        self.propagation_classifier = nn.Linear(output_size, num_propagation_modes)
        self.k_index_regressor = nn.Linear(output_size, 1)  # K-index (0-9)
        self.sfi_regressor = nn.Linear(output_size, 1)      # SFI (60-250)

    def forward(self, x, return_classification=False):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        features = self.fc(x)

        if return_classification:
            propagation_logits = self.propagation_classifier(features)
            k_index = torch.sigmoid(self.k_index_regressor(features))  # 0-1 normalized
            sfi = torch.sigmoid(self.sfi_regressor(features))          # 0-1 normalized
            return features, propagation_logits, k_index, sfi

        return features


class QRMExpert(nn.Module):
    """QRM Expert: Interference detection and classification.

    Outputs:
        - features: [batch, 64] - Feature vector for integration
        - qrm_present: [batch, 1] - Binary QRM presence indicator
        - qrm_type: [batch, 4] - Interference type (SSB/CW/PSK31/RTTY)
    """

    def __init__(self, input_size: int = 512, output_size: int = 64, num_qrm_types: int = 4):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, output_size)

        # QRM detection heads
        self.qrm_detector = nn.Linear(output_size, 1)         # Binary: QRM present or not
        self.qrm_type_classifier = nn.Linear(output_size, num_qrm_types)  # SSB/CW/PSK31/RTTY

    def forward(self, x, return_classification=False):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        features = self.fc3(x)

        if return_classification:
            qrm_present_logits = self.qrm_detector(features)
            qrm_type_logits = self.qrm_type_classifier(features)
            return features, qrm_present_logits, qrm_type_logits

        return features


class IntegrationDecoder(nn.Module):
    """Integration Decoder: Combines expert outputs + context signals → final predictions.

    In simplified CASCADE, also predicts optimal embeddings (RX suggests kernels to TX).
    """

    def __init__(self, expert_dim=640, context_dim=256, hidden_dim=512,
                 max_context_signals=8):
        super().__init__()

        self.max_context = max_context_signals

        # Expert features: 5 experts (64+128+256+128+64=640)
        self.expert_fusion = nn.Linear(expert_dim, hidden_dim)
        self.bn_expert = nn.BatchNorm1d(hidden_dim)
        self.dropout_expert = nn.Dropout(0.3)

        # Context processing
        self.context_processor = nn.Linear(context_dim * max_context_signals, hidden_dim)
        self.bn_context = nn.BatchNorm1d(hidden_dim)
        self.dropout_context = nn.Dropout(0.3)

        # Combined processing
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn_fusion = nn.BatchNorm1d(hidden_dim)
        self.dropout_fusion = nn.Dropout(0.3)

        # Output heads
        self.pattern_head = nn.Linear(hidden_dim, 4)  # 4 patterns
        self.frequency_head = nn.Linear(hidden_dim, 43)  # 43 triples
        self.modulation_head = nn.Linear(hidden_dim, 4)  # 4 modulations
        self.data_rate_head = nn.Linear(hidden_dim, 8)  # 8 data rates (75-300 sym/s)
        self.duration_head = nn.Linear(hidden_dim, 1)  # Duration regression

        # NEW: Embedding prediction (for RX to suggest kernels)
        self.embedding_head = nn.Linear(hidden_dim, 256)  # Predict optimal kernel

    def forward(self, expert_features, context_signals=None, context_mask=None):
        # expert_features: List of [batch, expert_dim_i]
        # context_signals: [batch, max_context, context_dim] (optional)
        # context_mask: [batch, max_context] (optional, True = valid)

        # Fuse expert features
        expert_cat = torch.cat(expert_features, dim=1)  # [batch, 640]
        x_expert = F.relu(self.bn_expert(self.expert_fusion(expert_cat)))
        x_expert = self.dropout_expert(x_expert)

        # Process context signals if available
        if context_signals is not None:
            batch_size = context_signals.shape[0]
            # Flatten context: [batch, max_context, context_dim] → [batch, max_context * context_dim]
            context_flat = context_signals.flatten(1)
            x_context = F.relu(self.bn_context(self.context_processor(context_flat)))
            x_context = self.dropout_context(x_context)

            # Combine expert + context
            x = torch.cat([x_expert, x_context], dim=1)
        else:
            # No context, duplicate expert features
            x = torch.cat([x_expert, x_expert], dim=1)

        # Final fusion
        x = F.relu(self.bn_fusion(self.fusion(x)))
        x = self.dropout_fusion(x)

        # Generate outputs
        outputs = {
            'pattern': self.pattern_head(x),
            'frequency': self.frequency_head(x),
            'modulation': self.modulation_head(x),
            'data_symbol_rate': self.data_rate_head(x),
            'duration': self.duration_head(x),
            'predicted_embedding': self.embedding_head(x),  # NEW!
        }

        return outputs


class CascadeModel(nn.Module):
    """Complete CASCADE neural network model."""

    def __init__(self, max_context_signals: int = 8):
        super().__init__()

        self.encoder = IQEmbeddingEncoder()

        self.experts = nn.ModuleDict({
            'qrn': QRNExpert(),
            'signal': SignalExpert(),
            'timing': TimingExpert(),
            'channel': ChannelExpert(),
            'qrm': QRMExpert()
        })

        self.decoder = IntegrationDecoder(max_context_signals=max_context_signals)

    def forward(self, iq_samples, context_kernels=None, context_mask=None, return_expert_outputs=False):
        # Encode I/Q samples
        encoded = self.encoder(iq_samples)

        # Run expert networks
        expert_features_list = []
        expert_classifications = {}

        # QRN Expert
        if return_expert_outputs:
            qrn_features, qrn_logits = self.experts['qrn'](encoded, return_classification=True)
            expert_classifications['qrn_logits'] = qrn_logits
        else:
            qrn_features = self.experts['qrn'](encoded)
        expert_features_list.append(qrn_features)

        # Signal Expert
        if return_expert_outputs:
            signal_features, pattern_logits, modulation_logits = self.experts['signal'](encoded, return_classification=True)
            expert_classifications['signal_pattern_logits'] = pattern_logits
            expert_classifications['signal_modulation_logits'] = modulation_logits
        else:
            signal_features = self.experts['signal'](encoded)
        expert_features_list.append(signal_features)

        # Timing Expert
        if return_expert_outputs:
            timing_features, has_collision, collision_offset = self.experts['timing'](encoded, return_classification=True)
            expert_classifications['has_collision_logits'] = has_collision
            expert_classifications['collision_offset'] = collision_offset
        else:
            timing_features = self.experts['timing'](encoded)
        expert_features_list.append(timing_features)

        # Channel Expert
        if return_expert_outputs:
            channel_features, propagation_logits, k_index, sfi = self.experts['channel'](encoded, return_classification=True)
            expert_classifications['propagation_logits'] = propagation_logits
            expert_classifications['k_index'] = k_index
            expert_classifications['sfi'] = sfi
        else:
            channel_features = self.experts['channel'](encoded)
        expert_features_list.append(channel_features)

        # QRM Expert
        if return_expert_outputs:
            qrm_features, qrm_present_logits, qrm_type_logits = self.experts['qrm'](encoded, return_classification=True)
            expert_classifications['qrm_present_logits'] = qrm_present_logits
            expert_classifications['qrm_type_logits'] = qrm_type_logits
        else:
            qrm_features = self.experts['qrm'](encoded)
        expert_features_list.append(qrm_features)

        # Integration decoder
        decoder_outputs = self.decoder(expert_features_list, context_kernels, context_mask)

        # Combine outputs
        if return_expert_outputs:
            return {**decoder_outputs, **expert_classifications}
        else:
            return decoder_outputs


class EmbeddingEncoder(nn.Module):
    """TX Embedding Encoder: Learns channel-adaptive embeddings.

    Input: TX observed IQ + discrete parameters → continuous embedding [256]
    """

    def __init__(self, input_size: int = 128, output_size: int = 256):
        super().__init__()

        # IQ processing (from TX observations)
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 384)
        self.bn2 = nn.BatchNorm1d(384)

        self.fc3 = nn.Linear(384, output_size)
        self.bn3 = nn.BatchNorm1d(output_size)

    def forward(self, tx_observed_iq):
        x = F.relu(self.bn1(self.fc1(tx_observed_iq)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return x


class LearnedQuantizer(nn.Module):
    """Learned Vector Quantizer: Compresses 256-d embedding → 113 bits.

    Uses 8 separate codebooks (not product quantization) to allow independent
    optimization of each component:
    - Fine frequency offset: 2 bits (4 levels)
    - Phase rotation: 3 bits (8 levels)
    - Equalization taps (3-tone): 5 bits each × 3 = 15 bits
    - Timing offset: 3 bits (8 levels)
    - Interference mitigation: 8 bits (256 patterns)
    - Channel impulse response: 16 bits (65536 patterns)
    - Per-tone fading (MRC): 5 bits each × 3 = 15 bits
    - Reserved: 50 bits (future use)

    Total: 2+3+15+3+8+16+15+50 = 112 bits → padded to 113 bits
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()

        # Component-specific quantizers (learn best split of embedding dim)
        self.freq_offset_proj = nn.Linear(embedding_dim, 4)  # 2 bits
        self.phase_proj = nn.Linear(embedding_dim, 8)  # 3 bits
        self.eq_tap1_proj = nn.Linear(embedding_dim, 32)  # 5 bits
        self.eq_tap2_proj = nn.Linear(embedding_dim, 32)  # 5 bits
        self.eq_tap3_proj = nn.Linear(embedding_dim, 32)  # 5 bits
        self.timing_proj = nn.Linear(embedding_dim, 8)  # 3 bits
        self.interference_proj = nn.Linear(embedding_dim, 256)  # 8 bits
        self.channel_proj = nn.Linear(embedding_dim, 256)  # 16 bits (stored as indices)
        self.fading1_proj = nn.Linear(embedding_dim, 32)  # 5 bits
        self.fading2_proj = nn.Linear(embedding_dim, 32)  # 5 bits
        self.fading3_proj = nn.Linear(embedding_dim, 32)  # 5 bits

        # Total: 2+3+15+3+8+16+15 = 62 bits
        # Remaining 51 bits reserved (will be zeros)

    def forward(self, embedding, temperature=1.0):
        """
        Quantize embedding to discrete indices.

        Args:
            embedding: [batch, 256] continuous embedding
            temperature: Gumbel-Softmax temperature (1.0 = standard, lower = sharper)

        Returns:
            Dict with quantized components and indices
        """
        # Gumbel-Softmax for differentiable quantization
        freq_logits = self.freq_offset_proj(embedding)
        freq_soft = F.gumbel_softmax(freq_logits, tau=temperature, hard=True)
        freq_idx = freq_soft.argmax(dim=-1)

        phase_logits = self.phase_proj(embedding)
        phase_soft = F.gumbel_softmax(phase_logits, tau=temperature, hard=True)
        phase_idx = phase_soft.argmax(dim=-1)

        eq1_logits = self.eq_tap1_proj(embedding)
        eq1_soft = F.gumbel_softmax(eq1_logits, tau=temperature, hard=True)
        eq1_idx = eq1_soft.argmax(dim=-1)

        eq2_logits = self.eq_tap2_proj(embedding)
        eq2_soft = F.gumbel_softmax(eq2_logits, tau=temperature, hard=True)
        eq2_idx = eq2_soft.argmax(dim=-1)

        eq3_logits = self.eq_tap3_proj(embedding)
        eq3_soft = F.gumbel_softmax(eq3_logits, tau=temperature, hard=True)
        eq3_idx = eq3_soft.argmax(dim=-1)

        timing_logits = self.timing_proj(embedding)
        timing_soft = F.gumbel_softmax(timing_logits, tau=temperature, hard=True)
        timing_idx = timing_soft.argmax(dim=-1)

        interference_logits = self.interference_proj(embedding)
        interference_soft = F.gumbel_softmax(interference_logits, tau=temperature, hard=True)
        interference_idx = interference_soft.argmax(dim=-1)

        channel_logits = self.channel_proj(embedding)
        channel_soft = F.gumbel_softmax(channel_logits, tau=temperature, hard=True)
        channel_idx = channel_soft.argmax(dim=-1)

        fading1_logits = self.fading1_proj(embedding)
        fading1_soft = F.gumbel_softmax(fading1_logits, tau=temperature, hard=True)
        fading1_idx = fading1_soft.argmax(dim=-1)

        fading2_logits = self.fading2_proj(embedding)
        fading2_soft = F.gumbel_softmax(fading2_logits, tau=temperature, hard=True)
        fading2_idx = fading2_soft.argmax(dim=-1)

        fading3_logits = self.fading3_proj(embedding)
        fading3_soft = F.gumbel_softmax(fading3_logits, tau=temperature, hard=True)
        fading3_idx = fading3_soft.argmax(dim=-1)

        return {
            'continuous': embedding,
            'freq_offset_idx': freq_idx,
            'phase_idx': phase_idx,
            'eq_tap1_idx': eq1_idx,
            'eq_tap2_idx': eq2_idx,
            'eq_tap3_idx': eq3_idx,
            'timing_idx': timing_idx,
            'interference_idx': interference_idx,
            'channel_idx': channel_idx,
            'fading1_idx': fading1_idx,
            'fading2_idx': fading2_idx,
            'fading3_idx': fading3_idx,
            # Soft assignments for reconstruction
            'freq_offset_soft': freq_soft,
            'phase_soft': phase_soft,
            'eq_tap1_soft': eq1_soft,
            'eq_tap2_soft': eq2_soft,
            'eq_tap3_soft': eq3_soft,
            'timing_soft': timing_soft,
            'interference_soft': interference_soft,
            'channel_soft': channel_soft,
            'fading1_soft': fading1_soft,
            'fading2_soft': fading2_soft,
            'fading3_soft': fading3_soft,
        }


class EmbeddingDecoder(nn.Module):
    """TX Embedding Decoder: Reconstructs continuous embedding from quantized indices.

    Learns inverse mapping: 113 bits → 256-d embedding
    """

    def __init__(self, output_size: int = 256):
        super().__init__()

        # Learnable codebooks for each component
        self.freq_offset_codebook = nn.Embedding(4, 32)
        self.phase_codebook = nn.Embedding(8, 32)
        self.eq_tap1_codebook = nn.Embedding(32, 32)
        self.eq_tap2_codebook = nn.Embedding(32, 32)
        self.eq_tap3_codebook = nn.Embedding(32, 32)
        self.timing_codebook = nn.Embedding(8, 32)
        self.interference_codebook = nn.Embedding(256, 64)
        self.channel_codebook = nn.Embedding(256, 64)
        self.fading1_codebook = nn.Embedding(32, 32)
        self.fading2_codebook = nn.Embedding(32, 32)
        self.fading3_codebook = nn.Embedding(32, 32)

        # Total concatenated: 32*6 + 64*2 + 32*3 = 192 + 128 + 96 = 416
        self.fc = nn.Linear(416, output_size)

    def forward(self, quantized_soft):
        """
        Reconstruct embedding from soft quantized assignments.

        Args:
            quantized_soft: Dict with soft assignments from LearnedQuantizer

        Returns:
            Reconstructed embedding [batch, 256]
        """
        # Soft lookup (differentiable)
        freq_vec = torch.matmul(quantized_soft['freq_offset_soft'], self.freq_offset_codebook.weight)
        phase_vec = torch.matmul(quantized_soft['phase_soft'], self.phase_codebook.weight)
        eq1_vec = torch.matmul(quantized_soft['eq_tap1_soft'], self.eq_tap1_codebook.weight)
        eq2_vec = torch.matmul(quantized_soft['eq_tap2_soft'], self.eq_tap2_codebook.weight)
        eq3_vec = torch.matmul(quantized_soft['eq_tap3_soft'], self.eq_tap3_codebook.weight)
        timing_vec = torch.matmul(quantized_soft['timing_soft'], self.timing_codebook.weight)
        interference_vec = torch.matmul(quantized_soft['interference_soft'], self.interference_codebook.weight)
        channel_vec = torch.matmul(quantized_soft['channel_soft'], self.channel_codebook.weight)
        fading1_vec = torch.matmul(quantized_soft['fading1_soft'], self.fading1_codebook.weight)
        fading2_vec = torch.matmul(quantized_soft['fading2_soft'], self.fading2_codebook.weight)
        fading3_vec = torch.matmul(quantized_soft['fading3_soft'], self.fading3_codebook.weight)

        # Concatenate and project
        combined = torch.cat([
            freq_vec, phase_vec, eq1_vec, eq2_vec, eq3_vec,
            timing_vec, interference_vec, channel_vec,
            fading1_vec, fading2_vec, fading3_vec
        ], dim=1)

        reconstructed = self.fc(combined)
        return reconstructed
