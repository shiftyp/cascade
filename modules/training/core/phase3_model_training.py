#!/usr/bin/env python3
"""
CASCADE Training Pipeline - Phase 3: Model Training

This script trains the CASCADE neural network model using a multi-stage curriculum:
- Stage 1: IQ Encoder bootstrap training
- Stage 2: Expert network training (5 experts)
- Stage 3: Integration decoder training

Outputs:
    - artifacts/phase3/checkpoints/*.pth - Model checkpoints
    - artifacts/phase3/training_curves.png - Training/validation curves
    - artifacts/phase3/training_log.json - Training history
    - artifacts/phase3/expert_*.png - Individual expert training plots
"""

import sys
import os
from pathlib import Path
import json
import argparse

# Add CASCADE root to Python path
cascade_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if cascade_root not in sys.path:
    sys.path.insert(0, cascade_root)

# CASCADE Training - Phase 3: Model Training

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Tuple, Dict, Optional, List

# Import CASCADE components
from modules.training.src.signal_generator.generator import SignalGenerator
from modules.training.core.physics_constrained_dataset import PhysicsConstrainedDataset
from modules.training.core.collision_dataset import CollisionAwareDataset

# Import optimized model architectures
from modules.training.core.cascade_models import (
    IQEmbeddingEncoder,
    QRNExpert,
    SignalExpert,
    TimingExpert,
    ChannelExpert,
    QRMExpert,
    IntegrationDecoder,
    CascadeModel,
    EmbeddingEncoder,
    LearnedQuantizer,
    EmbeddingDecoder,
    TFLiteCompatibleAttention
)

# GPU-accelerated dataset (if available)
try:
    from modules.training.core.enhanced_physics_dataset import EnhancedPhysicsDataset
    GPU_DATASET_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_DATASET_AVAILABLE = False

# Streaming dataset (FAST - 11× faster than EnhancedPhysicsDataset)
try:
    from modules.training.core.streaming_cascade_dataset import StreamingCascadeDataset, cascade_collate_fn
    STREAMING_DATASET_AVAILABLE = torch.cuda.is_available()
except ImportError:
    STREAMING_DATASET_AVAILABLE = False

# Reciprocal channel dataset (for TX encoder training)
try:
    from modules.training.core.reciprocal_channel_dataset import ReciprocalChannelDataset
    RECIPROCAL_DATASET_AVAILABLE = torch.cuda.is_available()
except ImportError:
    RECIPROCAL_DATASET_AVAILABLE = False


def create_artifacts_dir():
    """Create artifacts directory for phase 3 outputs."""
    artifacts_dir = Path(__file__).parent / 'artifacts' / 'phase3'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / 'checkpoints').mkdir(exist_ok=True)
    return artifacts_dir


# Modulation encoding for classification
MODULATION_TO_IDX = {
    'BPSK': 0,
    'QPSK': 1,
    '8-PSK': 2,
    '16-APSK': 3
}

IDX_TO_MODULATION = {v: k for k, v in MODULATION_TO_IDX.items()}


def encode_modulation_labels(modulation_strings):
    """
    Convert modulation string labels to integer indices.

    Args:
        modulation_strings: List or tensor of modulation strings

    Returns:
        torch.Tensor: Integer labels [0-3]
    """
    if isinstance(modulation_strings, str):
        return MODULATION_TO_IDX.get(modulation_strings, 0)

    if isinstance(modulation_strings, torch.Tensor):
        # Already encoded
        return modulation_strings

    # List/array of strings
    return torch.tensor([MODULATION_TO_IDX.get(m, 0) for m in modulation_strings])


# ============================================================================
# Model architectures imported from cascade_models.py
# (Old inline definitions removed to avoid conflicts)
# ============================================================================
# ============================================================================
# Loss Computation Functions
# ============================================================================

# QRN type string to index mapping (handle both cases)
QRN_TYPE_MAP = {
    'QUIET': 0, 'STATIC': 1, 'CRACKLING': 2, 'HISS': 3,
    'POPCORN': 4, 'FLUTTERY': 5, 'THUNDERSTORM': 6, 'AURORAL': 7,
    # Lowercase variants (from physics_coupling enums)
    'quiet': 0, 'static': 1, 'crackling': 2, 'hiss': 3,
    'popcorn': 4, 'fluttery': 5, 'thunderstorm': 6, 'auroral': 7
}

# Propagation mode string to index mapping (handle both cases)
PROPAGATION_MODE_MAP = {
    'AWGN': 0, 'RAYLEIGH': 1, 'RICIAN': 2,
    'MULTIPATH_SPARSE': 3, 'MULTIPATH_DENSE': 4,
    # Lowercase variants (from physics_coupling enums)
    'awgn': 0, 'rayleigh': 1, 'rician': 2,
    'multipath_sparse': 3, 'multipath_dense': 4
}

# Modulation string to index mapping
MODULATION_MAP = {
    'BPSK': 0, 'QPSK': 1, '8-PSK': 2, '16-APSK': 3
}

# QRM type string to index mapping (handle both uppercase and lowercase)
QRM_TYPE_MAP = {
    'SSB': 0, 'CW': 1, 'PSK31': 2, 'RTTY': 3,
    'ssb': 0, 'cw': 1, 'psk31': 2, 'rtty': 3  # Lowercase variants
}


def compute_expert_losses(outputs, batch_labels, device, loss_weights=None, debug=False):
    """
    Compute expert-specific losses with proper weighting.

    Args:
        outputs: Model outputs (dict)
        batch_labels: Batch labels (dict)
        device: Device to move tensors to
        loss_weights: Dict of loss weights (default: uniform)
        debug: If True, print detailed loss info

    Returns:
        Dict of individual losses and total weighted loss
    """
    if loss_weights is None:
        loss_weights = {
            'decoder': 1.0,
            'qrn': 0.5,
            'signal': 0.8,
            'timing': 0.6,
            'channel': 0.7,
            'qrm': 0.6
        }

    # Debug logging removed for cleaner output

    losses = {}

    # ============================================================================
    # DECODER LOSSES (from Integration Decoder)
    # ============================================================================
    if 'pattern_id' in batch_labels:
        pattern_ids = batch_labels['pattern_id'].to(device)
        losses['pattern_loss'] = F.cross_entropy(outputs['pattern'], pattern_ids)

    if 'frequency_triple' in batch_labels:
        frequency_triples = batch_labels['frequency_triple'].to(device)
        losses['frequency_loss'] = F.cross_entropy(outputs['frequency'], frequency_triples)

    if 'modulation' in batch_labels:
        # Convert modulation strings to indices
        modulation_strs = batch_labels['modulation']
        try:
            modulation_indices = torch.tensor([MODULATION_MAP[m] for m in modulation_strs], device=device)
            losses['modulation_loss'] = F.cross_entropy(outputs['modulation'], modulation_indices)
        except KeyError:
            pass  # Skip unknown modulations silently

    if 'data_symbol_rate' in batch_labels:
        # Data symbol rate indices: 75, 100, 125, 150, 175, 200, 250, 300 → 0-7
        data_rates = batch_labels['data_symbol_rate'].to(device)
        rate_to_idx = {75: 0, 100: 1, 125: 2, 150: 3, 175: 4, 200: 5, 250: 6, 300: 7}
        rate_indices = torch.tensor([rate_to_idx.get(int(r), 3) for r in data_rates.cpu()], device=device)
        losses['data_rate_loss'] = F.cross_entropy(outputs['data_symbol_rate'], rate_indices)

    if 'duration_windows' in batch_labels and 'duration' in outputs:
        duration_targets = batch_labels['duration_windows'].to(device).float().unsqueeze(1)
        # Normalize duration: assuming max ~255 windows (from 8-bit field in kernel)
        duration_targets_norm = duration_targets / 255.0
        losses['duration_loss'] = F.mse_loss(outputs['duration'], duration_targets_norm)

    # ============================================================================
    # QRN EXPERT LOSSES
    # ============================================================================
    if 'qrn_type' in batch_labels and 'qrn_logits' in outputs:
        qrn_strs = batch_labels['qrn_type']
        try:
            qrn_indices = torch.tensor([QRN_TYPE_MAP[q] for q in qrn_strs], device=device)
            losses['qrn_classification_loss'] = F.cross_entropy(outputs['qrn_logits'], qrn_indices)
        except KeyError:
            pass  # Skip unknown QRN types silently

    # ============================================================================
    # SIGNAL EXPERT LOSSES
    # ============================================================================
    if 'signal_pattern_logits' in outputs and 'pattern_id' in batch_labels:
        pattern_ids = batch_labels['pattern_id'].to(device)
        losses['signal_pattern_loss'] = F.cross_entropy(outputs['signal_pattern_logits'], pattern_ids)

    if 'signal_modulation_logits' in outputs and 'modulation' in batch_labels:
        modulation_strs = batch_labels['modulation']
        try:
            modulation_indices = torch.tensor([MODULATION_MAP[m] for m in modulation_strs], device=device)
            losses['signal_modulation_loss'] = F.cross_entropy(outputs['signal_modulation_logits'], modulation_indices)
        except KeyError:
            pass  # Skip unknown modulations silently

    # ============================================================================
    # TIMING EXPERT LOSSES (conditional on collision presence)
    # ============================================================================
    # Support both 'has_collision' (GPU dataset) and 'has_collisions' (legacy CPU dataset)
    collision_key = 'has_collision' if 'has_collision' in batch_labels else 'has_collisions'

    if collision_key in batch_labels and 'has_collision_logits' in outputs:
        has_collisions = batch_labels[collision_key].to(device).float().unsqueeze(1)
        losses['collision_detection_loss'] = F.binary_cross_entropy_with_logits(
            outputs['has_collision_logits'], has_collisions
        )

        # Offset regression loss (only on samples with collisions)
        # Support multiple field names from different datasets
        offset_key = None
        if 'time_offsets_ms' in batch_labels:  # EnhancedPhysicsDataset
            offset_key = 'time_offsets_ms'
        elif 'collision_offset_ms' in batch_labels:  # Alternative name
            offset_key = 'collision_offset_ms'
        elif 'collision_offsets_ms' in batch_labels:  # Legacy CPU dataset
            offset_key = 'collision_offsets_ms'

        if offset_key is not None and 'collision_offset' in outputs:
            collision_mask = has_collisions.squeeze(1) > 0.5
            if collision_mask.sum() > 0:
                # Get first collision offset for each sample
                offsets = batch_labels[offset_key]
                # Handle list of lists (multiple collisions per sample)
                offset_targets = []
                for offset_list in offsets:
                    if isinstance(offset_list, (list, tuple)) and len(offset_list) > 0:
                        offset_targets.append(float(offset_list[0]))  # Use first collision
                    else:
                        offset_targets.append(0.0)
                offset_targets = torch.tensor(offset_targets, device=device).unsqueeze(1)

                # Normalize collision offsets: assuming max ~100ms offset
                offset_targets_norm = offset_targets / 100.0

                pred_offsets = outputs['collision_offset'][collision_mask]
                target_offsets_norm = offset_targets_norm[collision_mask]
                losses['collision_offset_loss'] = F.mse_loss(pred_offsets, target_offsets_norm)

    # ============================================================================
    # CHANNEL EXPERT LOSSES
    # ============================================================================
    if 'propagation_mode' in batch_labels and 'propagation_logits' in outputs:
        prop_strs = batch_labels['propagation_mode']
        try:
            prop_indices = torch.tensor([PROPAGATION_MODE_MAP[p] for p in prop_strs], device=device)
            losses['propagation_loss'] = F.cross_entropy(outputs['propagation_logits'], prop_indices)
        except KeyError:
            pass  # Skip unknown propagation modes silently

    if 'k_index' in batch_labels and 'k_index' in outputs:
        k_targets = batch_labels['k_index'].to(device).float().unsqueeze(1)
        # Normalize K-index: 0-9 → 0-1
        k_targets_norm = k_targets / 9.0
        losses['k_index_loss'] = F.mse_loss(outputs['k_index'], k_targets_norm)

    if 'sfi' in batch_labels and 'sfi' in outputs:
        sfi_targets = batch_labels['sfi'].to(device).float().unsqueeze(1)
        # Normalize SFI: 60-250 → 0-1
        sfi_targets_norm = (sfi_targets - 60.0) / 190.0
        losses['sfi_loss'] = F.mse_loss(outputs['sfi'], sfi_targets_norm)

    # ============================================================================
    # QRM EXPERT LOSSES (conditional on QRM presence)
    # ============================================================================
    if 'has_qrm' in batch_labels and 'has_qrm_logits' in outputs:
        has_qrm = batch_labels['has_qrm'].to(device).float().unsqueeze(1)
        losses['qrm_detection_loss'] = F.binary_cross_entropy_with_logits(
            outputs['has_qrm_logits'], has_qrm
        )

        # QRM type classification loss (only on samples with QRM)
        if 'qrm_type' in batch_labels and 'qrm_type_logits' in outputs:
            qrm_mask = has_qrm.squeeze(1) > 0.5
            if qrm_mask.sum() > 0:
                qrm_strs = [batch_labels['qrm_type'][i] for i in range(len(qrm_mask)) if qrm_mask[i]]
                qrm_indices = torch.tensor([QRM_TYPE_MAP.get(q, 0) for q in qrm_strs], device=device)
                qrm_logits = outputs['qrm_type_logits'][qrm_mask]
                losses['qrm_type_loss'] = F.cross_entropy(qrm_logits, qrm_indices)

    # ============================================================================
    # COMPUTE TOTAL WEIGHTED LOSS
    # ============================================================================
    total_loss = 0.0

    # Decoder losses (weighted)
    decoder_loss = 0.0
    for key in ['pattern_loss', 'frequency_loss', 'modulation_loss', 'data_rate_loss', 'duration_loss']:
        if key in losses:
            decoder_loss += losses[key]
    if decoder_loss > 0:
        total_loss += loss_weights['decoder'] * decoder_loss

    # QRN Expert loss
    if 'qrn_classification_loss' in losses:
        total_loss += loss_weights['qrn'] * losses['qrn_classification_loss']

    # Signal Expert losses
    signal_loss = 0.0
    for key in ['signal_pattern_loss', 'signal_modulation_loss']:
        if key in losses:
            signal_loss += losses[key]
    if signal_loss > 0:
        total_loss += loss_weights['signal'] * signal_loss

    # Timing Expert losses
    timing_loss = 0.0
    for key in ['collision_detection_loss', 'collision_offset_loss']:
        if key in losses:
            timing_loss += losses[key]
    if timing_loss > 0:
        total_loss += loss_weights['timing'] * timing_loss

    # Channel Expert losses
    channel_loss = 0.0
    for key in ['propagation_loss', 'k_index_loss', 'sfi_loss']:
        if key in losses:
            channel_loss += losses[key]
    if channel_loss > 0:
        total_loss += loss_weights['channel'] * channel_loss

    # QRM Expert losses
    qrm_loss = 0.0
    for key in ['qrm_detection_loss', 'qrm_type_loss']:
        if key in losses:
            qrm_loss += losses[key]
    if qrm_loss > 0:
        total_loss += loss_weights['qrm'] * qrm_loss

    losses['total_loss'] = total_loss

    # Loss validation happens silently to avoid console spam

    return losses


def compute_joint_rxtx_losses(rx_outputs, tx_embedding_outputs, batch_labels, device, loss_weights=None):
    """
    Compute joint RX/TX losses for coupled training.

    Args:
        rx_outputs: RX model outputs (dict) including 'predicted_embedding'
        tx_embedding_outputs: TX embedding outputs (dict) with:
            - 'continuous': Original continuous embedding from TX encoder
            - 'reconstructed': Reconstructed after quantization
            - 'quantized_indices': Quantized bit indices
        batch_labels: Batch labels (dict) including 'optimal_embedding'
        device: Device to move tensors to
        loss_weights: Dict of loss weights

    Returns:
        Dict of individual losses and total weighted loss
    """
    if loss_weights is None:
        loss_weights = {
            'rx_decoder': 1.0,
            'rx_experts': 0.6,
            'tx_compression': 0.8,
            'rx_tx_consistency': 0.5,
            'embedding_task': 0.7
        }

    # Start with RX expert losses
    losses = compute_expert_losses(rx_outputs, batch_labels, device)

    # ============================================================================
    # TX EMBEDDING LOSSES
    # ============================================================================

    # TX compression loss: Can we reconstruct after quantization?
    if 'continuous' in tx_embedding_outputs and 'reconstructed' in tx_embedding_outputs:
        tx_continuous = tx_embedding_outputs['continuous']
        tx_reconstructed = tx_embedding_outputs['reconstructed']
        losses['tx_compression_loss'] = F.mse_loss(tx_reconstructed, tx_continuous)

    # RX-TX consistency loss: Does RX predict same embedding as TX generates?
    if 'predicted_embedding' in rx_outputs and 'continuous' in tx_embedding_outputs:
        rx_predicted = rx_outputs['predicted_embedding']
        tx_continuous = tx_embedding_outputs['continuous'].detach()  # Don't backprop through TX
        losses['rx_tx_consistency_loss'] = F.mse_loss(rx_predicted, tx_continuous)

    # Optimal embedding supervision: Does TX match ground truth?
    if 'optimal_embedding' in batch_labels and 'continuous' in tx_embedding_outputs:
        optimal = batch_labels['optimal_embedding'].to(device)
        tx_continuous = tx_embedding_outputs['continuous']
        losses['optimal_embedding_loss'] = F.mse_loss(tx_continuous, optimal)

    # ============================================================================
    # COMPUTE JOINT TOTAL LOSS
    # ============================================================================
    total_joint_loss = 0.0

    # RX decoder loss (pattern, freq, modulation, etc.)
    rx_decoder_loss = losses.get('total_loss', 0.0)  # Existing RX loss
    total_joint_loss += loss_weights['rx_decoder'] * rx_decoder_loss

    # TX embedding losses
    if 'tx_compression_loss' in losses:
        total_joint_loss += loss_weights['tx_compression'] * losses['tx_compression_loss']

    if 'rx_tx_consistency_loss' in losses:
        total_joint_loss += loss_weights['rx_tx_consistency'] * losses['rx_tx_consistency_loss']

    if 'optimal_embedding_loss' in losses:
        total_joint_loss += loss_weights['embedding_task'] * losses['optimal_embedding_loss']

    losses['total_joint_loss'] = total_joint_loss

    return losses


# ============================================================================
# Early Stopping Utility
# ============================================================================

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Tracks validation metric over epochs and stops training when the metric
    doesn't improve for a specified number of consecutive epochs (patience).
    """

    def __init__(self, patience=7, min_delta=0.0001, mode='min', verbose=True,
                 restore_best_weights=True, warmup_epochs=0):
        """
        Args:
            patience: Number of epochs to wait before stopping (default: 7)
            min_delta: Minimum change to qualify as improvement (default: 0.0001)
            mode: 'min' for loss (default), 'max' for accuracy
            verbose: Print messages when triggered (default: True)
            restore_best_weights: Restore model to best epoch when stopping (default: True)
            warmup_epochs: Minimum epochs before early stopping can trigger (default: 0)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.warmup_epochs = warmup_epochs

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.current_epoch = 0

        # For 'min' mode (loss), we want lower values
        # For 'max' mode (accuracy), we want higher values
        if mode == 'min':
            self.monitor_op = lambda score, best: score < (best - min_delta)
            self.best_score = float('inf')
        elif mode == 'max':
            self.monitor_op = lambda score, best: score > (best + min_delta)
            self.best_score = float('-inf')
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def __call__(self, score, epoch):
        """
        Check if training should stop.

        Args:
            score: Current validation metric (loss or accuracy)
            epoch: Current epoch number

        Returns:
            bool: True if training should stop, False otherwise
        """
        self.current_epoch = epoch

        # Skip early stopping during warmup period
        if epoch < self.warmup_epochs:
            if self.verbose and epoch == 0:
                print(f"Early stopping warmup: will not trigger before epoch {self.warmup_epochs}")
            # Still update best score during warmup
            if self.monitor_op(score, self.best_score):
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            return False

        # Check if score improved
        if self.monitor_op(score, self.best_score):
            if self.verbose:
                improvement = abs(score - self.best_score)
                print(f"  → Validation {self.mode} improved by {improvement:.6f} "
                      f"(patience counter reset: {self.counter} → 0)")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"  → No improvement in validation {self.mode} "
                      f"(patience: {self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n{'='*80}")
                    print(f"EARLY STOPPING TRIGGERED")
                    print(f"{'='*80}")
                    print(f"  Validation {self.mode} has not improved for {self.patience} epochs")
                    print(f"  Best {self.mode}: {self.best_score:.6f} at epoch {self.best_epoch + 1}")
                    print(f"  Current epoch: {epoch + 1}")
                    if self.restore_best_weights:
                        print(f"  Restoring model weights from epoch {self.best_epoch + 1}")
                    print(f"{'='*80}\n")
                return True

            return False


# ============================================================================
# Training Classes
# ============================================================================

class CascadeModelTrainer:
    """Stage 2-3: Train complete CASCADE model with experts and decoder.

    Supports progressive encoder unfreezing for better generalization:
    - Stage 2: Experts train with frozen encoder (3-5 epochs)
    - Stage 3: Joint E2E training with unfrozen encoder (20-25 epochs)
    """

    def __init__(self, pretrained_encoder, device='cuda', use_amp=True, freeze_encoder=True):
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.encoder_frozen = freeze_encoder

        # Use pretrained IQ encoder (frozen or fine-tunable)
        self.model = CascadeModel(max_context_signals=8).to(device)

        # Load pretrained encoder weights
        if pretrained_encoder is not None:
            self.model.encoder.load_state_dict(pretrained_encoder.state_dict())

        # Freeze encoder if requested (for expert warmup)
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen for expert warmup")

        # Only optimize trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=1e-4,
            weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Mixed precision training (2-3× faster on modern GPUs)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Enable TF32 for even faster training on A100/H100/GH200
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # Auto-tune kernels

        # torch.compile disabled (version incompatibility)

        self.train_losses = []
        self.val_losses = []
        self.train_stats = []  # Comprehensive training statistics

        # Confusion matrix tracker for detailed performance analysis
        self.confusion_tracker = ConfusionMatrixTracker()

    def unfreeze_encoder(self, encoder_lr=1e-5):
        """
        Unfreeze encoder and set up differential learning rates.

        Args:
            encoder_lr: Learning rate for encoder (default: 1e-5, 10× lower than experts)
        """
        if not self.encoder_frozen:
            print("Encoder already unfrozen")
            return

        # Unfreeze all encoder parameters
        for param in self.model.encoder.parameters():
            param.requires_grad = True

        # Set up differential learning rates
        param_groups = [
            {
                'params': self.model.encoder.parameters(),
                'lr': encoder_lr,
                'name': 'encoder'
            },
            {
                'params': self.model.experts.parameters(),
                'lr': 1e-4,
                'name': 'experts'
            },
            {
                'params': self.model.decoder.parameters(),
                'lr': 1e-4,
                'name': 'decoder'
            }
        ]

        # Recreate optimizer with parameter groups
        self.optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)

        # Recreate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Recreate scaler for mixed precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.encoder_frozen = False
        print(f"Encoder unfrozen: encoder_lr={encoder_lr}, expert/decoder_lr=1e-4")

    def get_regeneration_interval(self):
        """
        Get appropriate dataset regeneration interval based on encoder state.

        Returns:
            int: Regeneration interval in epochs
                - 0 if encoder frozen (Stage 2: no regen during warmup)
                - 5 if encoder unfrozen (Stage 3: standard interval)
        """
        if self.encoder_frozen:
            # Stage 2: No regeneration during expert warmup (short 3-5 epoch period)
            # Experts need stable features to learn basic patterns
            return 0
        else:
            # Stage 3: Standard regeneration (encoder changes provide natural diversity)
            return 5

    def train_epoch(self, train_loader):
        import time
        self.model.train()

        epoch_loss = 0.0
        epoch_pattern_acc = 0.0
        epoch_freq_acc = 0.0
        epoch_mod_acc = 0.0
        num_batches = 0
        total_samples = 0
        epoch_start = time.time()

        # Track GPU utilization
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for batch_idx, (batch_iq, batch_labels) in enumerate(tqdm(train_loader, desc="Training CASCADE Model")):
            batch_start = time.time()
            data_load_time = batch_start  # Will measure from iterator

            batch_iq = batch_iq.to(self.device, non_blocking=True)
            transfer_time = time.time()  # Time for data transfer

            # Extract labels
            pattern_ids = batch_labels['pattern_id'].to(self.device, non_blocking=True)
            frequency_triples = batch_labels['frequency_triple'].to(self.device, non_blocking=True)

            # Extract context (if available)
            context_kernels = None
            context_mask = None
            if 'context_kernels' in batch_labels and 'context_mask' in batch_labels:
                context_kernels = batch_labels['context_kernels'].to(self.device, non_blocking=True)
                context_mask = batch_labels['context_mask'].to(self.device, non_blocking=True)

            # Mixed precision forward pass
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch_iq, context_kernels, context_mask, return_expert_outputs=True)
                    loss_dict = compute_expert_losses(outputs, batch_labels, self.device, debug=False)
                    loss = loss_dict['total_loss']
            else:
                outputs = self.model(batch_iq, context_kernels, context_mask, return_expert_outputs=True)
                loss_dict = compute_expert_losses(outputs, batch_labels, self.device, debug=False)
                loss = loss_dict['total_loss']

            # Mixed precision backward pass
            self.optimizer.zero_grad(set_to_none=True)  # set_to_none=True for better performance
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Per-batch debug logging removed for cleaner output

            # Track metrics
            epoch_loss += loss.item()

            pattern_preds = outputs['pattern'].argmax(dim=1)
            epoch_pattern_acc += (pattern_preds == pattern_ids).float().mean().item()

            freq_preds = outputs['frequency'].argmax(dim=1)
            epoch_freq_acc += (freq_preds == frequency_triples).float().mean().item()

            # Track modulation accuracy if available
            if 'modulation' in outputs and 'modulation' in batch_labels:
                mod_targets = batch_labels['modulation'].to(self.device, non_blocking=True)
                mod_preds = outputs['modulation'].argmax(dim=1)
                epoch_mod_acc += (mod_preds == mod_targets).float().mean().item()

            num_batches += 1
            total_samples += len(batch_iq)

            # Batch statistics logged via progress bar only

        if num_batches == 0:
            raise RuntimeError("No batches processed in training epoch! Check dataset size and batch size.")

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        avg_pattern_acc = epoch_pattern_acc / num_batches
        avg_freq_acc = epoch_freq_acc / num_batches
        avg_mod_acc = epoch_mod_acc / num_batches if epoch_mod_acc > 0 else 0
        throughput = total_samples / epoch_time

        self.train_losses.append(avg_loss)
        self.train_stats.append({
            'loss': avg_loss,
            'pattern_acc': avg_pattern_acc,
            'freq_acc': avg_freq_acc,
            'mod_acc': avg_mod_acc,
            'throughput': throughput,
            'epoch_time': epoch_time
        })

        # Print concise epoch summary
        print(f"\nEpoch Summary: Loss={avg_loss:.4f}, Pattern={avg_pattern_acc*100:.1f}%, Freq={avg_freq_acc*100:.1f}%, {throughput:.0f} samples/s, {epoch_time:.0f}s")

        return avg_loss, avg_pattern_acc, avg_freq_acc

    def validate(self, val_loader, artifacts_dir=None, epoch=None, enable_confusion_matrix=True):
        self.model.eval()

        val_loss = 0.0
        val_pattern_acc = 0.0
        val_freq_acc = 0.0
        num_batches = 0

        # Reset confusion tracker for this validation pass
        if enable_confusion_matrix:
            self.confusion_tracker.reset()

        with torch.no_grad():
            for batch_iq, batch_labels in tqdm(val_loader, desc="Validating CASCADE Model"):
                batch_iq = batch_iq.to(self.device)

                # Extract labels
                pattern_ids = batch_labels['pattern_id'].to(self.device)
                frequency_triples = batch_labels['frequency_triple'].to(self.device)

                # Extract context (if available)
                context_kernels = None
                context_mask = None
                if 'context_kernels' in batch_labels and 'context_mask' in batch_labels:
                    context_kernels = batch_labels['context_kernels'].to(self.device)
                    context_mask = batch_labels['context_mask'].to(self.device)

                # Forward pass with context AND expert outputs
                outputs = self.model(batch_iq, context_kernels, context_mask, return_expert_outputs=True)

                # Compute expert-specific losses
                loss_dict = compute_expert_losses(outputs, batch_labels, self.device)
                loss = loss_dict['total_loss']

                # Track metrics
                val_loss += loss.item()

                pattern_preds = outputs['pattern'].argmax(dim=1)
                val_pattern_acc += (pattern_preds == pattern_ids).float().mean().item()

                freq_preds = outputs['frequency'].argmax(dim=1)
                val_freq_acc += (freq_preds == frequency_triples).float().mean().item()

                # Track confusion matrix data
                if enable_confusion_matrix:
                    # Encode modulation strings to integers for comparison
                    if 'modulation' in batch_labels:
                        mod_strings = batch_labels['modulation']
                        batch_labels['modulation_encoded'] = encode_modulation_labels(mod_strings).to(self.device)

                    ground_truth = {
                        'pattern_id': pattern_ids,
                        'frequency_triple': frequency_triples,
                        'modulation': batch_labels.get('modulation_encoded', torch.zeros_like(pattern_ids))
                    }

                    self.confusion_tracker.add_batch(outputs, ground_truth, batch_labels)

                num_batches += 1

        if num_batches == 0:
            raise RuntimeError("No batches processed in validation! Check dataset size and batch size.")

        avg_val_loss = val_loss / num_batches
        avg_pattern_acc = val_pattern_acc / num_batches
        avg_freq_acc = val_freq_acc / num_batches

        self.val_losses.append(avg_val_loss)

        # Generate confusion matrices
        if enable_confusion_matrix and artifacts_dir is not None and epoch is not None:
            self.confusion_tracker.compute_and_display_matrices(artifacts_dir, epoch)

        # Return basic metrics (extended validation for better tracking)
        return avg_val_loss, avg_pattern_acc, avg_freq_acc

    def train(self, train_loader, val_loader, num_epochs=30, save_path=None,
              early_stop_patience=None, early_stop_delta=0.0001,
              early_stop_warmup=0, enable_early_stop=True,
              resume_from_checkpoint=True):
        """
        Train CASCADE model with optional early stopping and auto-resume.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            save_path: Path to save model checkpoints
            early_stop_patience: Epochs to wait before stopping (default from env or 7)
            early_stop_delta: Minimum improvement threshold (default: 0.0001)
            early_stop_warmup: Minimum epochs before early stopping (default: 0)
            enable_early_stop: Enable early stopping (default: True)
            resume_from_checkpoint: Auto-resume from checkpoint if exists (default: True)
        """
        if save_path is None:
            save_path = 'checkpoints/cascade_model.pth'

        # Check for existing checkpoint and resume if requested
        start_epoch = 0
        best_val_loss = float('inf')

        if resume_from_checkpoint and os.path.exists(save_path):
            try:
                checkpoint = torch.load(save_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint['val_loss']
                self.train_losses = checkpoint.get('train_losses', [])
                self.val_losses = checkpoint.get('val_losses', [])
                print(f"Resuming from epoch {start_epoch}")
            except Exception as e:
                start_epoch = 0
                best_val_loss = float('inf')

        # Initialize early stopping if enabled
        early_stopper = None
        if enable_early_stop:
            if early_stop_patience is None:
                early_stop_patience = 7  # Default patience
            early_stopper = EarlyStopping(
                patience=early_stop_patience,
                min_delta=early_stop_delta,
                mode='min',
                verbose=True,
                restore_best_weights=True,
                warmup_epochs=early_stop_warmup
            )

        print("Stage 2-3: CASCADE Model Training")

        # Get artifacts directory for confusion matrices
        artifacts_dir_path = Path(save_path).parent if save_path else None

        for epoch in range(start_epoch, num_epochs):
            train_loss, train_pattern_acc, train_freq_acc = self.train_epoch(train_loader)
            val_loss, val_pattern_acc, val_freq_acc = self.validate(
                val_loader,
                artifacts_dir=artifacts_dir_path,
                epoch=epoch+1,
                enable_confusion_matrix=True
            )

            self.scheduler.step(val_loss)

            # Save checkpoint if improved
            saved_marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_pattern_acc': train_pattern_acc,
                    'val_pattern_acc': val_pattern_acc,
                }, save_path)
                saved_marker = " [saved]"

            # Concise epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}: Train={train_loss:.4f}, Val={val_loss:.4f} (P:{val_pattern_acc:.1%}, F:{val_freq_acc:.1%}){saved_marker}")

            # Check early stopping
            if early_stopper is not None and early_stopper(val_loss, epoch):
                print(f"Training stopped early at epoch {epoch + 1}")
                break

        print(f"Training complete. Best val_loss: {best_val_loss:.4f}")

        # Load best model
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        return self.model


class ConfusionMatrixTracker:
    """
    Track prediction accuracy across multiple dimensions for CASCADE decoding.

    Generates confusion matrices showing where the model succeeds/fails based on:
    - SNR (binned: <-10, -10 to -5, -5 to 0, 0 to 5, 5 to 10, 10 to 15, >15 dB)
    - Modulation (BPSK, QPSK, 8-PSK, 16-APSK)
    - Symbol rate (75, 100, 125, 150, 175, 200, 250, 300 sym/s)
    - QRN type (galactic, atmospheric, impulsive, quiet)
    - Propagation mode (AWGN, Rayleigh, Rician, multipath_sparse, multipath_dense)
    - Number of colliding messages (1, 2, 3, 4+)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracking for new epoch."""
        # Store per-sample results for aggregation
        self.results = []

    def add_batch(self, predictions, ground_truth, labels_dict):
        """
        Add a batch of predictions for tracking.

        Args:
            predictions: Dict with 'pattern', 'frequency', 'modulation' logits
            ground_truth: Dict with target values
            labels_dict: Full label dict with SNR, QRN, propagation_mode, etc.
        """
        batch_size = predictions['pattern'].shape[0]

        # Get predictions (argmax)
        pattern_preds = predictions['pattern'].argmax(dim=1).cpu().numpy()
        freq_preds = predictions['frequency'].argmax(dim=1).cpu().numpy()
        mod_preds = predictions['modulation'].argmax(dim=1).cpu().numpy() if 'modulation' in predictions else None

        # Get ground truth
        pattern_targets = ground_truth['pattern_id'].cpu().numpy()
        freq_targets = ground_truth['frequency_triple'].cpu().numpy()
        mod_targets = ground_truth.get('modulation', torch.zeros(batch_size)).cpu().numpy()

        # Extract all label dimensions
        snr = labels_dict['snr_db'] if isinstance(labels_dict['snr_db'], np.ndarray) else labels_dict['snr_db'].cpu().numpy()
        prop_mode = labels_dict['propagation_mode']
        qrn_type = labels_dict['qrn_type']
        symbol_rate = labels_dict['data_symbol_rate'] if isinstance(labels_dict['data_symbol_rate'], np.ndarray) else labels_dict['data_symbol_rate'].cpu().numpy()
        modulation = labels_dict['modulation']  # String labels
        num_messages = labels_dict['num_messages'] if isinstance(labels_dict['num_messages'], np.ndarray) else labels_dict['num_messages'].cpu().numpy()

        # Store each sample
        for i in range(batch_size):
            # Determine success (all predictions must be correct)
            pattern_correct = (pattern_preds[i] == pattern_targets[i])
            freq_correct = (freq_preds[i] == freq_targets[i])
            mod_correct = True if mod_preds is None else (mod_preds[i] == mod_targets[i])

            decode_success = pattern_correct and freq_correct and mod_correct

            self.results.append({
                'success': decode_success,
                'pattern_correct': pattern_correct,
                'freq_correct': freq_correct,
                'mod_correct': mod_correct,
                'snr_db': float(snr[i] if hasattr(snr, '__getitem__') else snr),
                'modulation': modulation[i] if isinstance(modulation, (list, np.ndarray)) else modulation,
                'symbol_rate': int(symbol_rate[i] if hasattr(symbol_rate, '__getitem__') else symbol_rate),
                'propagation_mode': prop_mode[i] if isinstance(prop_mode, (list, np.ndarray)) else prop_mode,
                'qrn_type': qrn_type[i] if isinstance(qrn_type, (list, np.ndarray)) else qrn_type,
                'num_messages': int(num_messages[i] if hasattr(num_messages, '__getitem__') else num_messages),
            })

    def compute_and_display_matrices(self, artifacts_dir, epoch):
        """Compute confusion matrices and save plots (minimal console output)."""
        if len(self.results) == 0:
            return

        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(self.results)
        overall_acc = df['success'].mean() * 100

        # Silently generate heatmaps (no console spam)
        snr_bins = [(-20, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 25)]
        self._generate_heatmaps(df, artifacts_dir, epoch, snr_bins)

    def _generate_heatmaps(self, df, artifacts_dir, epoch, snr_bins):
        """Generate heatmap visualizations (silent)."""
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'CASCADE Decode Success Analysis - Epoch {epoch}', fontsize=16, fontweight='bold')

        # 1. SNR vs Modulation heatmap
        ax = axes[0, 0]
        snr_bins_edges = [-20, -10, -5, 0, 5, 10, 15, 25]
        snr_labels = ['<-10', '-10 to -5', '-5 to 0', '0 to 5', '5 to 10', '10 to 15', '>15']
        modulations = ['BPSK', 'QPSK', '8-PSK', '16-APSK']

        heatmap_data = np.zeros((len(snr_labels), len(modulations)))
        counts = np.zeros((len(snr_labels), len(modulations)))

        for i, (low, high) in enumerate(zip(snr_bins_edges[:-1], snr_bins_edges[1:])):
            for j, mod in enumerate(modulations):
                mask = (df['snr_db'] >= low) & (df['snr_db'] < high) & (df['modulation'] == mod)
                if mask.sum() > 0:
                    heatmap_data[i, j] = df[mask]['success'].mean() * 100
                    counts[i, j] = mask.sum()
                else:
                    heatmap_data[i, j] = np.nan

        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(len(modulations)))
        ax.set_yticks(range(len(snr_labels)))
        ax.set_xticklabels(modulations)
        ax.set_yticklabels(snr_labels)
        ax.set_xlabel('Modulation')
        ax.set_ylabel('SNR (dB)')
        ax.set_title('Decode Success % by SNR and Modulation')

        # Add text annotations
        for i in range(len(snr_labels)):
            for j in range(len(modulations)):
                if not np.isnan(heatmap_data[i, j]):
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.0f}%\n({int(counts[i, j])})',
                                 ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=ax, label='Decode Success %')

        # 2. Symbol Rate vs SNR heatmap
        ax = axes[0, 1]
        symbol_rates = [75, 100, 125, 150, 175, 200, 250, 300]

        heatmap_data = np.zeros((len(snr_labels), len(symbol_rates)))
        counts = np.zeros((len(snr_labels), len(symbol_rates)))

        for i, (low, high) in enumerate(zip(snr_bins_edges[:-1], snr_bins_edges[1:])):
            for j, rate in enumerate(symbol_rates):
                mask = (df['snr_db'] >= low) & (df['snr_db'] < high) & (df['symbol_rate'] == rate)
                if mask.sum() > 0:
                    heatmap_data[i, j] = df[mask]['success'].mean() * 100
                    counts[i, j] = mask.sum()
                else:
                    heatmap_data[i, j] = np.nan

        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(len(symbol_rates)))
        ax.set_yticks(range(len(snr_labels)))
        ax.set_xticklabels(symbol_rates, rotation=45)
        ax.set_yticklabels(snr_labels)
        ax.set_xlabel('Symbol Rate (sym/s)')
        ax.set_ylabel('SNR (dB)')
        ax.set_title('Decode Success % by SNR and Symbol Rate')
        plt.colorbar(im, ax=ax, label='Decode Success %')

        # 3. Propagation Mode breakdown
        ax = axes[1, 0]
        prop_modes = df['propagation_mode'].unique()

        prop_accuracies = []
        prop_counts = []
        prop_labels = []

        for mode in sorted(prop_modes):
            mask = df['propagation_mode'] == mode
            acc = df[mask]['success'].mean() * 100
            count = mask.sum()
            prop_accuracies.append(acc)
            prop_counts.append(count)
            prop_labels.append(f"{mode}\n({count})")

        bars = ax.bar(range(len(prop_labels)), prop_accuracies, color=['green' if a > 80 else 'orange' if a > 60 else 'red' for a in prop_accuracies])
        ax.set_xticks(range(len(prop_labels)))
        ax.set_xticklabels(prop_labels, rotation=45, ha='right')
        ax.set_ylabel('Decode Success %')
        ax.set_title('Decode Success by Propagation Mode')
        ax.set_ylim([0, 105])
        ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% target')
        ax.legend()

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, prop_accuracies)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

        # 4. Collision Complexity
        ax = axes[1, 1]
        collision_bins = df['num_messages'].unique()

        coll_accuracies = []
        coll_counts = []
        coll_labels = []

        for num_msg in sorted(collision_bins):
            mask = df['num_messages'] == num_msg
            acc = df[mask]['success'].mean() * 100
            count = mask.sum()
            coll_accuracies.append(acc)
            coll_counts.append(count)
            label = f"{num_msg} msg" if num_msg == 1 else f"{num_msg} msgs"
            coll_labels.append(f"{label}\n({count})")

        bars = ax.bar(range(len(coll_labels)), coll_accuracies, color=['green' if a > 80 else 'orange' if a > 60 else 'red' for a in coll_accuracies])
        ax.set_xticks(range(len(coll_labels)))
        ax.set_xticklabels(coll_labels)
        ax.set_ylabel('Decode Success %')
        ax.set_title('Decode Success by Message Collisions')
        ax.set_ylim([0, 105])
        ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% target')
        ax.legend()

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, coll_accuracies)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save plot silently
        save_path = artifacts_dir / f'decode_analysis_epoch{epoch}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class IQEncoderTrainer:
    """Stage 1: Train IQ Encoder using autoencoder reconstruction."""

    @staticmethod
    def scale_invariant_mse(pred, target):
        """
        Scale-invariant MSE loss that normalizes signals before comparison.

        This is critical for signals with varying amplitudes (0.001 to 1000+).
        Normalizes both prediction and target to unit variance before computing MSE.

        NOTE: Since input signals are now pre-normalized (mean=0, std=1) from the dataset,
        this function is mainly for decoder output normalization.

        Args:
            pred: Predicted signal [batch, features]
            target: Target signal [batch, features]

        Returns:
            Scalar loss value
        """
        # FP16-safe epsilon (FP16 precision is ~1e-4, using 1e-3 for safety margin)
        eps = 1e-3

        # Normalize both to unit variance (per sample)
        pred_std = torch.std(pred, dim=-1, keepdim=True)
        target_std = torch.std(target, dim=-1, keepdim=True)

        # Clamp to prevent extreme values
        pred_std = torch.clamp(pred_std, min=eps)
        target_std = torch.clamp(target_std, min=eps)

        pred_norm = pred / pred_std
        target_norm = target / target_std

        # Also normalize mean to zero for better comparison
        pred_norm = pred_norm - torch.mean(pred_norm, dim=-1, keepdim=True)
        target_norm = target_norm - torch.mean(target_norm, dim=-1, keepdim=True)

        # Compute loss and clamp to prevent NaN propagation
        loss = F.mse_loss(pred_norm, target_norm)
        loss = torch.clamp(loss, max=1e6)  # Prevent exploding loss in FP16

        return loss

    @staticmethod
    def spectral_loss(pred, target, eps=1e-3):
        """
        Spectral loss: Compares FFT magnitude spectra of prediction and target.

        This helps the autoencoder preserve frequency-domain characteristics,
        which is critical for RF signal reconstruction.

        Args:
            pred: Predicted signal [batch, features]
            target: Target signal [batch, features]
            eps: Small epsilon for numerical stability

        Returns:
            Scalar loss value
        """
        # Compute FFT of both signals (returns complex tensor)
        pred_fft = torch.fft.fft(pred, dim=-1)
        target_fft = torch.fft.fft(target, dim=-1)

        # Get magnitude spectra (absolute value of complex FFT)
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Normalize magnitudes to unit energy (prevents scale issues)
        pred_mag = pred_mag / (torch.sum(pred_mag, dim=-1, keepdim=True) + eps)
        target_mag = target_mag / (torch.sum(target_mag, dim=-1, keepdim=True) + eps)

        # Compute MSE on normalized magnitude spectra
        loss = F.mse_loss(pred_mag, target_mag)

        return loss

    @staticmethod
    def combined_loss(pred, target, mse_weight=0.8, spectral_weight=0.2):
        """
        Combined loss: Weighted combination of scale-invariant MSE and spectral loss.

        Args:
            pred: Predicted signal [batch, features]
            target: Target signal [batch, features]
            mse_weight: Weight for MSE component (default: 0.8)
            spectral_weight: Weight for spectral component (default: 0.2)

        Returns:
            Scalar loss value
        """
        mse = IQEncoderTrainer.scale_invariant_mse(pred, target)
        spectral = IQEncoderTrainer.spectral_loss(pred, target)

        return mse_weight * mse + spectral_weight * spectral

    def __init__(self, device='cuda', use_amp=True, num_epochs=50, steps_per_epoch=100):
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()

        self.encoder = IQEmbeddingEncoder(output_size=512).to(device)

        # FIXED: ConvTranspose decoder for temporal reconstruction
        # Mirrors encoder structure: 512 embedding → [2, 2048] signal
        self.decoder = nn.Sequential(
            # Expand embedding to temporal features
            nn.Linear(512, 512 * 64),  # Match encoder's flattened output (32768)
            nn.Unflatten(1, (512, 64)),  # [batch, 32768] → [batch, 512, 64]

            # Transposed convolutions (mirror encoder, reverse order)
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64 → 128
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128 → 256
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),   # 256 → 512
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.ConvTranspose1d(64, 2, kernel_size=7, stride=2, padding=3, output_padding=1),     # 512 → 1024

            # Final upsampling to 2048 (2× more)
            nn.Upsample(size=2048, mode='linear', align_corners=False),

            # Flatten to [batch, 4096]
            nn.Flatten(1)
        ).to(device)

        # OPTIMIZATION: Initialize decoder weights
        self._initialize_decoder_weights()

        # OPTIMIZATION: Higher initial LR for faster convergence
        self.max_lr = 3e-3
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.max_lr,
            weight_decay=1e-5
        )

        # OPTIMIZATION: OneCycleLR for adaptive learning rate
        # - Warmup: 5 epochs (10% of training)
        # - Peak LR: 3e-3 at epoch 10
        # - Decay to: 3e-5 (100× reduction)
        self.scheduler_onecycle = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            total_steps=num_epochs * steps_per_epoch,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',  # Cosine annealing
            div_factor=10.0,  # Start at max_lr/10 = 3e-4
            final_div_factor=100.0  # End at max_lr/100 = 3e-5
        )

        # OPTIMIZATION: Backup scheduler - reduces LR if validation plateaus
        self.scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # OPTIMIZATION: Mixed precision training (2× faster on NVIDIA GPUs!)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # OPTIMIZATION: torch.compile for 15-20% speedup (PyTorch 2.0+)
        # DISABLED: Version incompatibility with current PyTorch install
        # if hasattr(torch, 'compile'):
        #     print("  🔥 Compiling encoder and decoder with torch.compile (reduce-overhead mode)")
        #     self.encoder = torch.compile(self.encoder, mode='reduce-overhead')
        #     self.decoder = torch.compile(self.decoder, mode='reduce-overhead')
        # else:
        #     print("  ⚠️  torch.compile not available (requires PyTorch 2.0+)")
        print("  ℹ️  torch.compile disabled (version incompatibility)")

        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0

    def _initialize_decoder_weights(self):
        """Initialize decoder weights using Xavier for Linear layers."""
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def train_epoch(self, train_loader):
        self.encoder.train()
        self.decoder.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_iq, batch_labels in tqdm(train_loader, desc="Training IQ Encoder"):
            batch_iq = batch_iq.to(self.device)

            # OPTIMIZATION: Mixed precision forward/backward pass
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    compressed = self.encoder(batch_iq)
                    reconstructed = self.decoder(compressed)

                # Reshape for loss computation (OUTSIDE autocast for numerical stability)
                batch_flat = batch_iq.view(batch_iq.size(0), -1)

                # Compute loss in FP32 for numerical stability
                with torch.amp.autocast('cuda', enabled=False):
                    # Cast to FP32 for loss computation
                    reconstructed_fp32 = reconstructed.float()
                    batch_flat_fp32 = batch_flat.float()
                    # OPTIMIZATION: Combined loss (MSE + spectral) for better reconstruction
                    loss = self.combined_loss(reconstructed_fp32, batch_flat_fp32,
                                            mse_weight=0.8, spectral_weight=0.2)

                # NaN detection and handling
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  ⚠️ NaN/Inf loss detected! Skipping batch.")
                    print(f"     Reconstructed range: [{reconstructed.min():.6f}, {reconstructed.max():.6f}]")
                    print(f"     Target range: [{batch_flat.min():.6f}, {batch_flat.max():.6f}]")
                    continue

                # OPTIMIZATION: set_to_none=True for faster gradient clearing
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    max_norm=1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                compressed = self.encoder(batch_iq)
                reconstructed = self.decoder(compressed)

                # Reshape for loss computation
                batch_flat = batch_iq.view(batch_iq.size(0), -1)
                # OPTIMIZATION: Combined loss (MSE + spectral) for better reconstruction
                loss = self.combined_loss(reconstructed, batch_flat,
                                        mse_weight=0.8, spectral_weight=0.2)

                # OPTIMIZATION: set_to_none=True for faster gradient clearing
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    max_norm=1.0
                )
                self.optimizer.step()

            # OPTIMIZATION: Step OneCycleLR after each batch
            self.scheduler_onecycle.step()

            epoch_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("No batches in IQ Encoder training! Check dataset.")

        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, val_loader):
        self.encoder.eval()
        self.decoder.eval()

        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_iq, batch_labels in tqdm(val_loader, desc="Validating IQ Encoder"):
                batch_iq = batch_iq.to(self.device)

                compressed = self.encoder(batch_iq)
                reconstructed = self.decoder(compressed)

                batch_flat = batch_iq.view(batch_iq.size(0), -1)
                # OPTIMIZATION: Combined loss (MSE + spectral) for better reconstruction
                loss = self.combined_loss(reconstructed, batch_flat,
                                        mse_weight=0.8, spectral_weight=0.2)

                val_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            raise RuntimeError("No batches in IQ Encoder validation! Check dataset.")

        avg_val_loss = val_loss / num_batches
        self.val_losses.append(avg_val_loss)

        # OPTIMIZATION: Step plateau scheduler on validation loss
        self.scheduler_plateau.step(avg_val_loss)

        return avg_val_loss

    def train(self, train_loader, val_loader, num_epochs=30, save_path=None,
              early_stop_patience=None, early_stop_delta=0.0001,
              early_stop_warmup=0, enable_early_stop=True,
              resume_from_checkpoint=True,
              train_dataset=None, val_dataset=None, regen_interval=0):
        """
        Train IQ Encoder with optional early stopping and auto-resume.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            save_path: Path to save model checkpoints
            early_stop_patience: Epochs to wait before stopping (default from env or 7)
            early_stop_delta: Minimum improvement threshold (default: 0.0001)
            early_stop_warmup: Minimum epochs before early stopping (default: 0)
            enable_early_stop: Enable early stopping (default: True)
            resume_from_checkpoint: Auto-resume from checkpoint if exists (default: True)
        """
        if save_path is None:
            save_path = 'checkpoints/stage1_iq_encoder.pth'

        # Check for existing checkpoint and resume if requested
        start_epoch = 0
        best_val_loss = float('inf')

        if resume_from_checkpoint and os.path.exists(save_path):
            print(f"\n📂 Found existing checkpoint: {save_path}")
            try:
                checkpoint = torch.load(save_path)
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint['val_loss']
                print(f"✅ Resuming from epoch {start_epoch} (best val_loss: {best_val_loss:.6f})")
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
                print("   Starting from scratch...")
                start_epoch = 0
                best_val_loss = float('inf')
        else:
            print("\n🆕 Starting training from scratch")

        # Initialize early stopping if enabled
        early_stopper = None
        if enable_early_stop:
            if early_stop_patience is None:
                early_stop_patience = 7  # Default patience
            early_stopper = EarlyStopping(
                patience=early_stop_patience,
                min_delta=early_stop_delta,
                mode='min',
                verbose=True,
                restore_best_weights=True,
                warmup_epochs=early_stop_warmup
            )
            print(f"\n Early stopping enabled:")
            print(f"    Patience: {early_stop_patience} epochs")
            print(f"    Min delta: {early_stop_delta}")
            print(f"    Warmup: {early_stop_warmup} epochs")

        print("\nStage 1: IQ Encoder Training")
        print("=" * 80)

        # Store datasets for regeneration
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._regen_interval = regen_interval

        for epoch in range(start_epoch, num_epochs):
            # Check for cache regeneration at configured intervals
            if regen_interval > 0 and train_dataset is not None:
                self._train_dataset, self._val_dataset, was_regenerated = regenerate_datasets_if_needed(
                    self._train_dataset, self._val_dataset, epoch, self._regen_interval, self.device
                )
                if was_regenerated:
                    # Rebuild dataloaders with fresh data
                    print("  Rebuilding DataLoaders with regenerated data...")
                    train_loader = torch.utils.data.DataLoader(
                        self._train_dataset, batch_size=train_loader.batch_size,
                        shuffle=True, num_workers=train_loader.num_workers,
                        pin_memory=True, persistent_workers=True
                    )
                    val_loader = torch.utils.data.DataLoader(
                        self._val_dataset, batch_size=val_loader.batch_size,
                        shuffle=False, num_workers=val_loader.num_workers,
                        pin_memory=True, persistent_workers=True
                    )
                    print("  ✓ DataLoaders rebuilt with fresh data")

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.scheduler_plateau.step(val_loss)

            saved_marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                }, save_path)
                saved_marker = " [saved]"

            print(f"Epoch {epoch+1}/{num_epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}{saved_marker}")

            # Early convergence check: Stop if val_loss < 0.5 (good enough for Stage 2)
            if val_loss < 0.5:
                print(f"Encoder converged! (val_loss={val_loss:.6f} < 0.5) - stopping early")
                break

            # Check early stopping
            if early_stopper is not None and early_stopper(val_loss, epoch):
                print(f"Training stopped early at epoch {epoch + 1}")
                break

        print(f"\nStage 1 complete! Best val_loss = {best_val_loss:.6f}")

        # Load best model
        checkpoint = torch.load(save_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

        return self.encoder


class JointRXTXTrainer:
    """
    Stage 4: Joint RX/TX training with embedding autoencoder.

    Trains RX model and TX embedding encoder together in closed loop:
    - RX decoder learns to predict embeddings
    - TX encoder learns to compress channel observations
    - Quantizer learns to compress embeddings to 113 bits
    - Consistency loss ensures RX and TX agree
    """

    def __init__(self, pretrained_rx_model, device='cuda'):
        """
        Initialize joint RX/TX trainer.

        Args:
            pretrained_rx_model: Pretrained CASCADE RX model (from Stage 2-3)
            device: 'cuda' or 'cpu'
        """
        self.device = device

        # RX model (can be fine-tuned or frozen)
        self.rx_model = pretrained_rx_model.to(device)

        # TX embedding components
        self.tx_embedding_encoder = EmbeddingEncoder().to(device)
        self.tx_quantizer = LearnedQuantizer().to(device)
        self.tx_embedding_decoder = EmbeddingDecoder().to(device)

        # Optimizer for all trainable components
        self.optimizer = torch.optim.Adam(
            list(self.rx_model.parameters()) +
            list(self.tx_embedding_encoder.parameters()) +
            list(self.tx_quantizer.parameters()) +
            list(self.tx_embedding_decoder.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader):
        """Train one epoch of joint RX/TX model."""
        self.rx_model.train()
        self.tx_embedding_encoder.train()
        self.tx_quantizer.train()
        self.tx_embedding_decoder.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_data in tqdm(train_loader, desc="Training Joint RX/TX"):
            # Unpack batch (reciprocal dataset has 3 elements)
            if len(batch_data) == 3:
                rx_iq, tx_observed_iq, batch_labels = batch_data
                rx_iq = rx_iq.to(self.device)
                tx_observed_iq = tx_observed_iq.to(self.device)
            else:
                # Fallback for non-reciprocal datasets (RX-only training)
                rx_iq, batch_labels = batch_data
                rx_iq = rx_iq.to(self.device)
                tx_observed_iq = None

            # ================================================================
            # RX FORWARD PASS
            # ================================================================
            context_kernels = None
            context_mask = None
            if 'context_kernels' in batch_labels and 'context_mask' in batch_labels:
                context_kernels = batch_labels['context_kernels'].to(self.device)
                context_mask = batch_labels['context_mask'].to(self.device)

            rx_outputs = self.rx_model(rx_iq, context_kernels, context_mask, return_expert_outputs=True)

            # ================================================================
            # TX FORWARD PASS (if reciprocal data available)
            # ================================================================
            tx_embedding_outputs = {}

            if tx_observed_iq is not None:
                # TX observes channel by processing RX beacon
                with torch.no_grad():
                    tx_compressed_iq = self.rx_model.encoder(tx_observed_iq)
                    tx_channel_features, _, _, _ = self.rx_model.experts['channel'](
                        tx_compressed_iq,
                        return_classification=True
                    )

                # TX embedding encoder generates continuous embedding
                tx_continuous_embedding = self.tx_embedding_encoder(tx_channel_features)

                # Quantize embedding (simulate 113-bit compression)
                tx_quantized_indices, tx_quantized_embedding = self.tx_quantizer(tx_continuous_embedding)

                # Refine dequantized embedding
                tx_refined_embedding = self.tx_embedding_decoder(tx_quantized_embedding)

                tx_embedding_outputs = {
                    'continuous': tx_continuous_embedding,
                    'quantized_indices': tx_quantized_indices,
                    'reconstructed': tx_refined_embedding
                }

            # ================================================================
            # COMPUTE LOSSES
            # ================================================================
            if tx_embedding_outputs:
                # Joint RX/TX loss
                loss_dict = compute_joint_rxtx_losses(
                    rx_outputs,
                    tx_embedding_outputs,
                    batch_labels,
                    self.device
                )
                loss = loss_dict['total_joint_loss']
            else:
                # RX-only loss (no TX data)
                loss_dict = compute_expert_losses(rx_outputs, batch_labels, self.device)
                loss = loss_dict['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.rx_model.parameters()) +
                list(self.tx_embedding_encoder.parameters()) +
                list(self.tx_quantizer.parameters()) +
                list(self.tx_embedding_decoder.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("No batches in Joint RX/TX training! Check dataset.")

        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, val_loader):
        """Validate joint RX/TX model."""
        self.rx_model.eval()
        self.tx_embedding_encoder.eval()
        self.tx_quantizer.eval()
        self.tx_embedding_decoder.eval()

        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validating Joint RX/TX"):
                # Unpack batch
                if len(batch_data) == 3:
                    rx_iq, tx_observed_iq, batch_labels = batch_data
                    rx_iq = rx_iq.to(self.device)
                    tx_observed_iq = tx_observed_iq.to(self.device)
                else:
                    rx_iq, batch_labels = batch_data
                    rx_iq = rx_iq.to(self.device)
                    tx_observed_iq = None

                # RX forward pass
                context_kernels = None
                context_mask = None
                if 'context_kernels' in batch_labels and 'context_mask' in batch_labels:
                    context_kernels = batch_labels['context_kernels'].to(self.device)
                    context_mask = batch_labels['context_mask'].to(self.device)

                rx_outputs = self.rx_model(rx_iq, context_kernels, context_mask, return_expert_outputs=True)

                # TX forward pass
                tx_embedding_outputs = {}
                if tx_observed_iq is not None:
                    tx_compressed_iq = self.rx_model.encoder(tx_observed_iq)
                    tx_channel_features, _, _, _ = self.rx_model.experts['channel'](
                        tx_compressed_iq,
                        return_classification=True
                    )
                    tx_continuous_embedding = self.tx_embedding_encoder(tx_channel_features)
                    tx_quantized_indices, tx_quantized_embedding = self.tx_quantizer(tx_continuous_embedding)
                    tx_refined_embedding = self.tx_embedding_decoder(tx_quantized_embedding)

                    tx_embedding_outputs = {
                        'continuous': tx_continuous_embedding,
                        'quantized_indices': tx_quantized_indices,
                        'reconstructed': tx_refined_embedding
                    }

                # Compute losses
                if tx_embedding_outputs:
                    loss_dict = compute_joint_rxtx_losses(
                        rx_outputs,
                        tx_embedding_outputs,
                        batch_labels,
                        self.device
                    )
                    loss = loss_dict['total_joint_loss']
                else:
                    loss_dict = compute_expert_losses(rx_outputs, batch_labels, self.device)
                    loss = loss_dict['total_loss']

                val_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            raise RuntimeError("No batches in Joint RX/TX validation! Check dataset.")

        avg_val_loss = val_loss / num_batches
        self.val_losses.append(avg_val_loss)

        return avg_val_loss

    def train(self, train_loader, val_loader, num_epochs=30, save_path=None,
              early_stop_patience=None, early_stop_delta=0.0001,
              early_stop_warmup=0, enable_early_stop=True,
              resume_from_checkpoint=True):
        """
        Train joint RX/TX model with optional early stopping and auto-resume.

        Args:
            train_loader: Training data loader (ReciprocalChannelDataset)
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            save_path: Path to save model checkpoints
            early_stop_patience: Epochs to wait before stopping
            early_stop_delta: Minimum improvement threshold
            early_stop_warmup: Minimum epochs before early stopping
            enable_early_stop: Enable early stopping
            resume_from_checkpoint: Auto-resume from checkpoint if exists (default: True)
        """
        if save_path is None:
            save_path = 'checkpoints/joint_rxtx_model.pth'

        # Check for existing checkpoint and resume if requested
        start_epoch = 0
        best_val_loss = float('inf')

        if resume_from_checkpoint and os.path.exists(save_path):
            print(f"\n📂 Found existing checkpoint: {save_path}")
            try:
                checkpoint = torch.load(save_path)
                self.rx_model.load_state_dict(checkpoint['rx_model_state_dict'])
                self.tx_embedding_encoder.load_state_dict(checkpoint['tx_embedding_encoder_state_dict'])
                self.tx_quantizer.load_state_dict(checkpoint['tx_quantizer_state_dict'])
                self.tx_embedding_decoder.load_state_dict(checkpoint['tx_embedding_decoder_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint['val_loss']
                print(f"✅ Resuming from epoch {start_epoch} (best val_loss: {best_val_loss:.6f})")
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
                print("   Starting from scratch...")
                start_epoch = 0
                best_val_loss = float('inf')
        else:
            print("\n🆕 Starting training from scratch")

        # Initialize early stopping
        early_stopper = None
        if enable_early_stop:
            if early_stop_patience is None:
                early_stop_patience = 7
            early_stopper = EarlyStopping(
                patience=early_stop_patience,
                min_delta=early_stop_delta,
                mode='min',
                verbose=True,
                restore_best_weights=True,
                warmup_epochs=early_stop_warmup
            )
            print(f"\n Early stopping enabled:")
            print(f"    Patience: {early_stop_patience} epochs")
            print(f"    Min delta: {early_stop_delta}")
            print(f"    Warmup: {early_stop_warmup} epochs")

        print("\nStage 4: Joint RX/TX Training (Embedding Autoencoder)")
        print("=" * 80)

        for epoch in range(start_epoch, num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'rx_model_state_dict': self.rx_model.state_dict(),
                    'tx_embedding_encoder_state_dict': self.tx_embedding_encoder.state_dict(),
                    'tx_quantizer_state_dict': self.tx_quantizer.state_dict(),
                    'tx_embedding_decoder_state_dict': self.tx_embedding_decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f"  → Saved best joint model (val_loss = {val_loss:.6f})")

            # Check early stopping
            if early_stopper is not None and early_stopper(val_loss, epoch):
                print(f"Training stopped early at epoch {epoch + 1}")
                break

        print(f"\nStage 4 complete! Best val_loss = {best_val_loss:.6f}")

        # Load best model
        checkpoint = torch.load(save_path)
        self.rx_model.load_state_dict(checkpoint['rx_model_state_dict'])
        self.tx_embedding_encoder.load_state_dict(checkpoint['tx_embedding_encoder_state_dict'])
        self.tx_quantizer.load_state_dict(checkpoint['tx_quantizer_state_dict'])
        self.tx_embedding_decoder.load_state_dict(checkpoint['tx_embedding_decoder_state_dict'])

        return {
            'rx_model': self.rx_model,
            'tx_embedding_encoder': self.tx_embedding_encoder,
            'tx_quantizer': self.tx_quantizer,
            'tx_embedding_decoder': self.tx_embedding_decoder
        }


def train_cascade_model(artifacts_dir: Path, train_dataset, val_dataset, device='cuda', start_stage=1):
    """
    Main training function for CASCADE model.

    Args:
        artifacts_dir: Directory for saving checkpoints and artifacts
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: Device to train on ('cuda' or 'cpu')
        start_stage: Stage to start training from (1, 2, or 3)
            - Stage 1: IQ Encoder training
            - Stage 2-3: Complete CASCADE model (experts + decoder)
            - Stage 4: Joint RX/TX training (optional, if TX observations available)
    """

    print("=" * 80)
    print("PHASE 3: CASCADE MODEL TRAINING")
    print("=" * 80)

    # Validate start_stage
    if start_stage not in [1, 2, 3, 4]:
        raise ValueError(f"Invalid start_stage: {start_stage}. Must be 1, 2, 3, or 4")

    # Validate checkpoints exist for stages we're skipping
    stage1_checkpoint_path = artifacts_dir / 'checkpoints' / 'stage1_iq_encoder.pth'
    stage23_checkpoint_path = artifacts_dir / 'checkpoints' / 'cascade_model.pth'
    stage4_checkpoint_path = artifacts_dir / 'checkpoints' / 'joint_rxtx_model.pth'

    if start_stage >= 2 and not stage1_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Cannot start from stage {start_stage}: Stage 1 checkpoint not found at {stage1_checkpoint_path}\n"
            f"Please complete Stage 1 first or start from --start-stage 1"
        )

    if start_stage >= 3 and not stage23_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Cannot start from stage {start_stage}: Stage 2-3 checkpoint not found at {stage23_checkpoint_path}\n"
            f"Please complete Stage 2-3 first or start from --start-stage 2"
        )

    if start_stage >= 4:
        # Stage 4 is optional, just warn if skipping without checkpoint
        if not hasattr(train_dataset, 'tx_observed_iq'):
            print("\n⚠️  WARNING: Stage 4 requested but dataset does not have TX observations")
            print("   Skipping Stage 4 (Joint RX/TX training)")
            start_stage = 3  # Effectively skip Stage 4

    print(f"\n📍 Starting from Stage {start_stage}")
    if start_stage > 1:
        print(f"   Skipping stages 1-{start_stage-1} (loading from checkpoints)")
    print()

    # Configuration (OPTIMIZED - see modules/training/core/streaming_cascade_dataset.py)
    BATCH_SIZE = int(os.getenv('CASCADE_BATCH_SIZE', '8192'))  # Increased from 512 (GPU has 72GB free!)
    NUM_EPOCHS = int(os.getenv('CASCADE_EPOCHS', '30'))

    # Cache regeneration configuration
    CACHE_REGEN_INTERVAL = int(os.getenv('CASCADE_CACHE_REGEN_INTERVAL', '5'))  # Regenerate every N epochs (0 = disabled)

    # Early stopping configuration from environment variables
    ENABLE_EARLY_STOP = os.getenv('CASCADE_ENABLE_EARLY_STOP', 'true').lower() in ('true', '1', 'yes')
    EARLY_STOP_PATIENCE = int(os.getenv('CASCADE_EARLY_STOP_PATIENCE', '10'))
    EARLY_STOP_DELTA = float(os.getenv('CASCADE_EARLY_STOP_DELTA', '0.0001'))
    EARLY_STOP_WARMUP = int(os.getenv('CASCADE_EARLY_STOP_WARMUP', '10'))

    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    print(f"\nCache Regeneration:")
    if CACHE_REGEN_INTERVAL > 0:
        print(f"  Enabled: Every {CACHE_REGEN_INTERVAL} epochs")
        print(f"  Fresh physics/noise each regeneration")
        print(f"  Prevents overfitting to static data")
    else:
        print(f"  Disabled (set CASCADE_CACHE_REGEN_INTERVAL > 0 to enable)")
    print(f"\nEarly Stopping:")
    print(f"  Enabled: {ENABLE_EARLY_STOP}")
    if ENABLE_EARLY_STOP:
        print(f"  Patience: {EARLY_STOP_PATIENCE} epochs")
        print(f"  Min delta: {EARLY_STOP_DELTA}")
        print(f"  Warmup: {EARLY_STOP_WARMUP} epochs")

    # Create dataloaders
    # DataLoader configuration depends on dataset type
    # StreamingCascadeDataset: Loads from HDF5 on-demand → needs workers for parallel I/O
    # EnhancedPhysicsDataset: Already in RAM → num_workers=0 to avoid duplication

    # Check dataset type and configure workers appropriately
    is_streaming_dataset = hasattr(train_dataset, 'hdf5_file') or hasattr(train_dataset, 'chunk_files')
    is_ram_cached = hasattr(train_dataset, 'ram_cache_streams') and train_dataset.ram_cache_streams is not None
    is_numpy_memmap = hasattr(train_dataset, 'using_numpy') and train_dataset.using_numpy

    print(f"\n{'='*60}")
    print(f"DATALOADER CONFIGURATION:")
    print(f"  Dataset type detection:")
    print(f"    - Streaming dataset: {is_streaming_dataset}")
    print(f"    - RAM cached: {is_ram_cached}")
    print(f"    - Numpy memmap: {is_numpy_memmap}")

    if is_numpy_memmap:
        # NUMPY MEMMAP: Thread-safe, use many workers!
        # OPTIMIZATION: Increased to 51 workers (matches CPU core count for optimal parallelism)
        dataloader_workers = int(os.getenv('CASCADE_DATALOADER_WORKERS', '51'))  # 51 workers for numpy (match CPU cores)
        print(f"  ✅ Numpy memmap detected - using {dataloader_workers} workers")
        print(f"     Numpy memmap is thread-safe (no serialization)")
        print(f"     {dataloader_workers} workers × 8 prefetch = {dataloader_workers*8} batches ready")
    elif is_ram_cached:
        # Dataset in RAM: No workers needed (data already in memory!)
        dataloader_workers = 0
        print(f"  ✅ RAM cached - using 0 workers (no I/O needed)")
    elif is_streaming_dataset:
        # HDF5-based dataset: Use moderate workers (file locking limits parallelism)
        dataloader_workers = int(os.getenv('CASCADE_DATALOADER_WORKERS', '4'))  # Only 4 for HDF5
        print(f"  ⚠️  HDF5 detected - using {dataloader_workers} workers (limited by file locking)")
        print(f"     Consider regenerating dataset (will auto-create numpy memmap)")
    else:
        # In-memory dataset: No workers to avoid duplication
        dataloader_workers = 0
        print(f"  Using 0 workers (in-memory dataset)")

    print(f"={'='*60}")

    # OPTIMIZATION: Increased prefetch_factor to 8 (masks data loading latency)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=dataloader_workers, pin_memory=(device == 'cuda'),
        prefetch_factor=8 if dataloader_workers > 0 else None,  # Increased from 4 (more aggressive prefetching)
        persistent_workers=True if dataloader_workers > 0 else False  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=dataloader_workers, pin_memory=(device == 'cuda'),
        prefetch_factor=8 if dataloader_workers > 0 else None,  # Increased from 4
        persistent_workers=True if dataloader_workers > 0 else False
    )

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Workers: {dataloader_workers}")
    print(f"  Prefetch factor: {4 if dataloader_workers > 0 else 'N/A'}")
    print(f"  Persistent workers: {dataloader_workers > 0}")

    # Helper function to create fresh dataloaders from datasets
    def create_dataloaders(train_ds, val_ds):
        """Create DataLoaders from datasets."""
        # Detect dataset type for worker configuration
        is_numpy_memmap = hasattr(train_ds, 'using_numpy') and train_ds.using_numpy
        is_ram_cached = hasattr(train_ds, 'ram_cache_streams') and train_ds.ram_cache_streams is not None

        if is_numpy_memmap:
            # OPTIMIZATION: Use 51 workers to match CPU cores
            workers = int(os.getenv('CASCADE_DATALOADER_WORKERS', '51'))
        elif is_ram_cached:
            workers = 0
        else:
            workers = int(os.getenv('CASCADE_DATALOADER_WORKERS', '4'))

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=workers, pin_memory=(device == 'cuda'),
            prefetch_factor=8 if workers > 0 else None,  # Increased from 4
            persistent_workers=True if workers > 0 else False
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=workers, pin_memory=(device == 'cuda'),
            prefetch_factor=8 if workers > 0 else None,  # Increased from 4
            persistent_workers=True if workers > 0 else False
        )
        return train_loader, val_loader

    # Stage 1: Train IQ Encoder
    print("\n" + "=" * 80)
    print("STAGE 1: IQ ENCODER TRAINING")
    print("=" * 80)

    # Calculate steps_per_epoch for OneCycleLR
    steps_per_epoch = len(train_loader)

    if start_stage > 1:
        # Skip Stage 1, load from checkpoint
        print(f"⏭️  Skipping Stage 1 (loading from checkpoint)")
        checkpoint = torch.load(str(stage1_checkpoint_path))
        print(f"   Loaded IQ Encoder from epoch {checkpoint['epoch'] + 1} (val_loss: {checkpoint['val_loss']:.6f})")
        use_amp_encoder = os.getenv('CASCADE_ENCODER_USE_AMP', 'true').lower() in ('true', '1', 'yes')
        iq_trainer = IQEncoderTrainer(device=device, use_amp=use_amp_encoder,
                                     num_epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch)
        iq_trainer.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        iq_encoder = iq_trainer.encoder
        # Load training history if available
        if 'train_losses' in checkpoint:
            iq_trainer.train_losses = checkpoint['train_losses']
            iq_trainer.val_losses = checkpoint['val_losses']
    elif stage1_checkpoint_path.exists():
        checkpoint = torch.load(str(stage1_checkpoint_path))
        if checkpoint['epoch'] >= NUM_EPOCHS - 1:
            print(f"✅ Stage 1 already complete (epoch {checkpoint['epoch'] + 1}/{NUM_EPOCHS})")
            print(f"   Loading trained IQ Encoder (val_loss: {checkpoint['val_loss']:.6f})")
            use_amp_encoder = os.getenv('CASCADE_ENCODER_USE_AMP', 'true').lower() in ('true', '1', 'yes')
            iq_trainer = IQEncoderTrainer(device=device, use_amp=use_amp_encoder,
                                         num_epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch)
            iq_trainer.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            iq_encoder = iq_trainer.encoder
            # Load training history if available
            if 'train_losses' in checkpoint:
                iq_trainer.train_losses = checkpoint['train_losses']
                iq_trainer.val_losses = checkpoint['val_losses']
        else:
            print(f"⏸️  Resuming Stage 1 from epoch {checkpoint['epoch'] + 1}")
            use_amp_encoder = os.getenv('CASCADE_ENCODER_USE_AMP', 'true').lower() in ('true', '1', 'yes')
            iq_trainer = IQEncoderTrainer(device=device, use_amp=use_amp_encoder,
                                         num_epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch)
            iq_encoder = iq_trainer.train(
                train_loader, val_loader,
                num_epochs=NUM_EPOCHS,
                save_path=str(stage1_checkpoint_path),
                early_stop_patience=EARLY_STOP_PATIENCE,
                early_stop_delta=EARLY_STOP_DELTA,
                early_stop_warmup=EARLY_STOP_WARMUP,
                enable_early_stop=ENABLE_EARLY_STOP,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                regen_interval=CACHE_REGEN_INTERVAL
            )
    else:
        print("🆕 Starting Stage 1 from scratch")
        use_amp_encoder = os.getenv('CASCADE_ENCODER_USE_AMP', 'true').lower() in ('true', '1', 'yes')
        print(f"  Mixed Precision (FP16): {'✅ ENABLED' if use_amp_encoder else '❌ DISABLED'}")
        print(f"  Adaptive LR: OneCycleLR (3e-4 → 3e-3 → 3e-5) + ReduceLROnPlateau backup")
        iq_trainer = IQEncoderTrainer(device=device, use_amp=use_amp_encoder,
                                     num_epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch)
        iq_encoder = iq_trainer.train(
            train_loader, val_loader,
            num_epochs=NUM_EPOCHS,
            save_path=str(stage1_checkpoint_path),
            early_stop_patience=EARLY_STOP_PATIENCE,
            early_stop_delta=EARLY_STOP_DELTA,
            early_stop_warmup=EARLY_STOP_WARMUP,
            enable_early_stop=ENABLE_EARLY_STOP,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            regen_interval=CACHE_REGEN_INTERVAL
        )

    # Plot IQ encoder training curves (only if trained in this run)
    if start_stage == 1 and iq_trainer.train_losses:  # Check if any training happened
        plot_training_curves(
            iq_trainer.train_losses,
            iq_trainer.val_losses,
            "Stage 1: IQ Encoder",
            artifacts_dir / 'stage1_iq_encoder.png'
        )
    elif start_stage > 1:
        print("  (Stage 1 was skipped - no new training curves to plot)")

    # Check for cache regeneration before Stage 2-3
    # Note: Regeneration happens at stage boundaries based on total epochs trained so far
    total_epochs_so_far = len(iq_trainer.train_losses) if iq_trainer.train_losses else 0
    train_dataset, val_dataset, was_regenerated = regenerate_datasets_if_needed(
        train_dataset, val_dataset, total_epochs_so_far, CACHE_REGEN_INTERVAL, device
    )

    if was_regenerated:
        # Rebuild dataloaders with fresh datasets
        print("  Rebuilding DataLoaders with fresh data...")
        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)
        print(f"  ✓ DataLoaders rebuilt")

    # Stage 2-3: Train complete CASCADE model with context
    print("\n" + "=" * 80)
    print("STAGE 2-3: COMPLETE CASCADE MODEL TRAINING (with context)")
    print("=" * 80)

    if start_stage > 2:
        # Skip Stage 2-3, load from checkpoint
        print(f"⏭️  Skipping Stage 2-3 (loading from checkpoint)")
        checkpoint = torch.load(str(stage23_checkpoint_path))
        print(f"   Loaded CASCADE Model from epoch {checkpoint['epoch'] + 1} (val_loss: {checkpoint['val_loss']:.4f})")
        cascade_trainer = CascadeModelTrainer(pretrained_encoder=iq_encoder, device=device, use_amp=True)
        cascade_trainer.model.load_state_dict(checkpoint['model_state_dict'])
        model = cascade_trainer.model
        # Load training history if available
        if 'train_losses' in checkpoint:
            cascade_trainer.train_losses = checkpoint['train_losses']
            cascade_trainer.val_losses = checkpoint['val_losses']
    elif stage23_checkpoint_path.exists():
        checkpoint = torch.load(str(stage23_checkpoint_path))
        if checkpoint['epoch'] >= NUM_EPOCHS - 1:
            print(f"✅ Stage 2-3 already complete (epoch {checkpoint['epoch'] + 1}/{NUM_EPOCHS})")
            print(f"   Loading trained CASCADE Model (val_loss: {checkpoint['val_loss']:.4f})")
            cascade_trainer = CascadeModelTrainer(pretrained_encoder=iq_encoder, device=device, use_amp=True)
            cascade_trainer.model.load_state_dict(checkpoint['model_state_dict'])
            model = cascade_trainer.model
            # Load training history if available
            if 'train_losses' in checkpoint:
                cascade_trainer.train_losses = checkpoint['train_losses']
                cascade_trainer.val_losses = checkpoint['val_losses']
        else:
            print(f"⏸️  Resuming Stage 2-3 from epoch {checkpoint['epoch'] + 1}")
            cascade_trainer = CascadeModelTrainer(pretrained_encoder=iq_encoder, device=device, use_amp=True)
            total_params = sum(p.numel() for p in cascade_trainer.model.parameters())
            print(f"\nCASCADE Model Summary:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Max context signals: 8")

            model = cascade_trainer.train(
                train_loader, val_loader,
                num_epochs=NUM_EPOCHS,
                save_path=str(stage23_checkpoint_path),
                early_stop_patience=EARLY_STOP_PATIENCE,
                early_stop_delta=EARLY_STOP_DELTA,
                early_stop_warmup=EARLY_STOP_WARMUP,
                enable_early_stop=ENABLE_EARLY_STOP
            )
    else:
        print("🆕 Starting Stage 2-3 with progressive encoder unfreezing")

        # Stage 2: Expert warmup with FROZEN encoder (5 epochs)
        print("\n--- Stage 2: Expert Warmup (encoder frozen) ---")
        cascade_trainer = CascadeModelTrainer(
            pretrained_encoder=iq_encoder,
            device=device,
            use_amp=True,
            freeze_encoder=True  # Freeze for warmup
        )

        total_params = sum(p.numel() for p in cascade_trainer.model.parameters())
        trainable_params = sum(p.numel() for p in cascade_trainer.model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,} ({trainable_params:,} trainable)")

        # Stage 2 warmup: No early stopping, no regeneration
        warmup_path = str(artifacts_dir / 'checkpoints' / 'cascade_warmup.pth')
        cascade_trainer.train(
            train_loader, val_loader,
            num_epochs=5,  # Short warmup
            save_path=warmup_path,
            early_stop_patience=None,
            enable_early_stop=False,  # Don't stop during warmup
            resume_from_checkpoint=False  # Always start fresh
        )

        # Stage 3: Joint E2E training with UNFROZEN encoder (25 epochs)
        print("\n--- Stage 3: Joint E2E Training (encoder unfrozen) ---")
        cascade_trainer.unfreeze_encoder(encoder_lr=1e-5)

        trainable_params = sum(p.numel() for p in cascade_trainer.model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable_params:,} (encoder now training)")

        # Stage 3: Full training with early stopping and regeneration
        model = cascade_trainer.train(
            train_loader, val_loader,
            num_epochs=25,  # Remaining epochs
            save_path=str(stage23_checkpoint_path),
            early_stop_patience=EARLY_STOP_PATIENCE,
            early_stop_delta=EARLY_STOP_DELTA,
            early_stop_warmup=EARLY_STOP_WARMUP,
            enable_early_stop=ENABLE_EARLY_STOP,
            resume_from_checkpoint=False  # Don't resume warmup checkpoint
        )

    # Plot CASCADE model training curves (only if trained in this run)
    if start_stage <= 2 and cascade_trainer.train_losses:
        plot_training_curves(
            cascade_trainer.train_losses,
            cascade_trainer.val_losses,
            "Stage 2-3: Complete CASCADE Model (with context)",
            artifacts_dir / 'stage2-3_cascade_model.png'
        )
    elif start_stage > 2:
        print("  (Stage 2-3 was skipped - no new training curves to plot)")

    # Check for cache regeneration before Stage 4
    total_epochs_so_far = (len(iq_trainer.train_losses) if iq_trainer.train_losses else 0) + \
                          (len(cascade_trainer.train_losses) if cascade_trainer.train_losses else 0)
    train_dataset, val_dataset, was_regenerated = regenerate_datasets_if_needed(
        train_dataset, val_dataset, total_epochs_so_far, CACHE_REGEN_INTERVAL, device
    )

    if was_regenerated:
        # Rebuild dataloaders with fresh datasets
        print("  Rebuilding DataLoaders with fresh data...")
        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)
        print(f"  ✓ DataLoaders rebuilt")

    # Stage 4: Joint RX/TX Training (OPTIONAL - only if using reciprocal dataset)
    joint_models = None
    joint_trainer = None
    tx_encoder_enabled = hasattr(train_dataset, 'tx_observed_iq')  # Check if reciprocal dataset

    if tx_encoder_enabled and start_stage <= 4:
        print("\n" + "=" * 80)
        print("STAGE 4: JOINT RX/TX TRAINING (Embedding Autoencoder)")
        print("=" * 80)

        if start_stage > 4:
            # Skip Stage 4, load from checkpoint
            print(f"⏭️  Skipping Stage 4 (loading from checkpoint)")
            checkpoint = torch.load(str(stage4_checkpoint_path))
            print(f"   Loaded Joint RX/TX Model from epoch {checkpoint['epoch'] + 1}")
            joint_trainer = JointRXTXTrainer(pretrained_rx_model=model, device=device)
            joint_trainer.tx_embedding_encoder.load_state_dict(checkpoint['tx_embedding_encoder_state_dict'])
            joint_trainer.tx_quantizer.load_state_dict(checkpoint['tx_quantizer_state_dict'])
            joint_trainer.tx_embedding_decoder.load_state_dict(checkpoint['tx_embedding_decoder_state_dict'])
            # Load training history if available
            if 'train_losses' in checkpoint:
                joint_trainer.train_losses = checkpoint['train_losses']
                joint_trainer.val_losses = checkpoint['val_losses']
            joint_models = {
                'rx_model': model,
                'tx_embedding_encoder': joint_trainer.tx_embedding_encoder,
                'tx_quantizer': joint_trainer.tx_quantizer,
                'tx_embedding_decoder': joint_trainer.tx_embedding_decoder
            }
            model = joint_models['rx_model']
        elif stage4_checkpoint_path.exists():
            checkpoint = torch.load(str(stage4_checkpoint_path))
            if checkpoint['epoch'] >= NUM_EPOCHS - 1:
                print(f"✅ Stage 4 already complete (epoch {checkpoint['epoch'] + 1}/{NUM_EPOCHS})")
                print(f"   Loading trained Joint RX/TX Model")
                joint_trainer = JointRXTXTrainer(pretrained_rx_model=model, device=device)
                joint_trainer.tx_embedding_encoder.load_state_dict(checkpoint['tx_embedding_encoder_state_dict'])
                joint_trainer.tx_quantizer.load_state_dict(checkpoint['tx_quantizer_state_dict'])
                joint_trainer.tx_embedding_decoder.load_state_dict(checkpoint['tx_embedding_decoder_state_dict'])
                # Load training history if available
                if 'train_losses' in checkpoint:
                    joint_trainer.train_losses = checkpoint['train_losses']
                    joint_trainer.val_losses = checkpoint['val_losses']
                joint_models = {
                    'rx_model': model,
                    'tx_embedding_encoder': joint_trainer.tx_embedding_encoder,
                    'tx_quantizer': joint_trainer.tx_quantizer,
                    'tx_embedding_decoder': joint_trainer.tx_embedding_decoder
                }
                model = joint_models['rx_model']
            else:
                print(f"⏸️  Resuming Stage 4 from epoch {checkpoint['epoch'] + 1}")
                joint_trainer = JointRXTXTrainer(pretrained_rx_model=model, device=device)
                # Print TX encoder summary
                tx_params = (
                    sum(p.numel() for p in joint_trainer.tx_embedding_encoder.parameters()) +
                    sum(p.numel() for p in joint_trainer.tx_quantizer.parameters()) +
                    sum(p.numel() for p in joint_trainer.tx_embedding_decoder.parameters())
                )
                print(f"\nTX Embedding Encoder Summary:")
                print(f"  Total TX parameters: {tx_params:,}")
                print(f"  Embedding dimension: 256")
                print(f"  Quantized bits: 113 (8-bit coarse + 105-bit fine)")

                # Train joint model
                joint_models = joint_trainer.train(
                    train_loader, val_loader,
                    num_epochs=NUM_EPOCHS,
                    save_path=str(stage4_checkpoint_path),
                    early_stop_patience=EARLY_STOP_PATIENCE,
                    early_stop_delta=EARLY_STOP_DELTA,
                    early_stop_warmup=EARLY_STOP_WARMUP,
                    enable_early_stop=ENABLE_EARLY_STOP
                )
                model = joint_models['rx_model']
        else:
            print("🆕 Starting Stage 4 from scratch")
            print("\nTraining TX embedding encoder with frozen RX model...")

            joint_trainer = JointRXTXTrainer(pretrained_rx_model=model, device=device)

            # Print TX encoder summary
            tx_params = (
                sum(p.numel() for p in joint_trainer.tx_embedding_encoder.parameters()) +
                sum(p.numel() for p in joint_trainer.tx_quantizer.parameters()) +
                sum(p.numel() for p in joint_trainer.tx_embedding_decoder.parameters())
            )
            print(f"\nTX Embedding Encoder Summary:")
            print(f"  Total TX parameters: {tx_params:,}")
            print(f"  Embedding dimension: 256")
            print(f"  Quantized bits: 113 (8-bit coarse + 105-bit fine)")

            # Train joint model
            joint_models = joint_trainer.train(
                train_loader, val_loader,
                num_epochs=NUM_EPOCHS,
                save_path=str(stage4_checkpoint_path),
                early_stop_patience=EARLY_STOP_PATIENCE,
                early_stop_delta=EARLY_STOP_DELTA,
                early_stop_warmup=EARLY_STOP_WARMUP,
                enable_early_stop=ENABLE_EARLY_STOP
            )

            # Update model to use joint-trained version
            model = joint_models['rx_model']

        # Plot joint training curves (only if trained in this run)
        if start_stage <= 4 and joint_trainer and joint_trainer.train_losses:
            plot_training_curves(
                joint_trainer.train_losses,
                joint_trainer.val_losses,
                "Stage 4: Joint RX/TX Training (Embedding Autoencoder)",
                artifacts_dir / 'stage4_joint_rxtx.png'
            )
        elif start_stage > 4:
            print("  (Stage 4 was skipped - no new training curves to plot)")

    # Save training history
    history = {
        'stage1_train_losses': iq_trainer.train_losses,
        'stage1_val_losses': iq_trainer.val_losses,
        'stage2_3_train_losses': cascade_trainer.train_losses,
        'stage2_3_val_losses': cascade_trainer.val_losses,
        'config': {
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'device': device,
            'max_context_signals': 8,
            'context_enabled': True,
            'tx_encoder_enabled': tx_encoder_enabled,
            'early_stopping': {
                'enabled': ENABLE_EARLY_STOP,
                'patience': EARLY_STOP_PATIENCE,
                'min_delta': EARLY_STOP_DELTA,
                'warmup_epochs': EARLY_STOP_WARMUP
            }
        }
    }

    # Add Stage 4 losses if trained
    if joint_trainer is not None:
        history['stage4_train_losses'] = joint_trainer.train_losses
        history['stage4_val_losses'] = joint_trainer.val_losses

    with open(artifacts_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training log saved to {artifacts_dir / 'training_log.json'}")

    # Return both RX model and TX components (if trained)
    if joint_models is not None:
        return joint_models, iq_encoder
    else:
        return model, iq_encoder


def regenerate_datasets_if_needed(train_dataset, val_dataset, current_epoch, regen_interval, device='cuda'):
    """
    Regenerate datasets if periodic regeneration is enabled and interval reached.

    Args:
        train_dataset: Current training dataset
        val_dataset: Current validation dataset
        current_epoch: Current epoch number (0-indexed)
        regen_interval: Regeneration interval in epochs (0 = disabled)
        device: Device for generation ('cuda' or 'cpu')

    Returns:
        Tuple of (new_train_dataset, new_val_dataset, was_regenerated)
    """
    # Check if regeneration needed
    if regen_interval <= 0:
        return train_dataset, val_dataset, False

    if current_epoch == 0 or current_epoch % regen_interval != 0:
        return train_dataset, val_dataset, False

    # Regeneration needed!
    print(f"\n{'='*80}")
    print(f"🔄 CACHE REGENERATION TRIGGERED (epoch {current_epoch}, interval={regen_interval})")
    print(f"{'='*80}")
    print(f"  Generating fresh dataset with new physics/noise conditions...")
    print(f"  This prevents overfitting to static data patterns")

    import time
    start_time = time.time()

    # Check if this is a StreamingCascadeDataset
    if hasattr(train_dataset, 'num_streams'):
        # StreamingCascadeDataset
        num_train_streams = train_dataset.num_streams
        num_val_streams = val_dataset.num_streams
        stream_duration = train_dataset.stream_duration_sec
        window_duration = train_dataset.window_duration_sec
        message_rate = train_dataset.message_arrival_rate
        batch_size = train_dataset.batch_size
        num_workers = train_dataset.num_workers
        enable_tx = train_dataset.enable_tx_observations
        load_to_ram = train_dataset.load_into_memory

        # Use epoch number as seed for reproducibility but variety
        train_seed = 42 + current_epoch
        val_seed = 1042 + current_epoch

        print(f"\n  Regenerating StreamingCascadeDataset:")
        print(f"    Train: {num_train_streams:,} streams (seed={train_seed})")
        print(f"    Val: {num_val_streams:,} streams (seed={val_seed})")

        # Import here to avoid circular imports
        from streaming_cascade_dataset import StreamingCascadeDataset

        new_train_dataset = StreamingCascadeDataset(
            num_streams=num_train_streams,
            stream_duration_sec=stream_duration,
            window_duration_sec=window_duration,
            message_arrival_rate=message_rate,
            seed=train_seed,
            regenerate_cache=True,  # Force regeneration
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            enable_tx_observations=enable_tx,
            load_into_memory=load_to_ram
        )

        new_val_dataset = StreamingCascadeDataset(
            num_streams=num_val_streams,
            stream_duration_sec=stream_duration,
            window_duration_sec=window_duration,
            message_arrival_rate=message_rate,
            seed=val_seed,
            regenerate_cache=True,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            enable_tx_observations=enable_tx,
            load_into_memory=load_to_ram
        )

    else:
        # Other dataset types - not supported yet
        print(f"  ⚠️  Cache regeneration not supported for this dataset type")
        print(f"     Dataset type: {type(train_dataset).__name__}")
        print(f"     Skipping regeneration")
        return train_dataset, val_dataset, False

    regen_time = time.time() - start_time
    print(f"\n  ✅ Cache regeneration complete in {regen_time:.1f}s")
    print(f"     New train samples: {len(new_train_dataset):,}")
    print(f"     New val samples: {len(new_val_dataset):,}")
    print(f"{'='*80}\n")

    return new_train_dataset, new_val_dataset, True


def plot_training_curves(train_losses, val_losses, title, output_path):
    """Plot training and validation curves."""
    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved training curves to {output_path}")
    plt.close()


def main():
    """Main entry point for Phase 3."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='CASCADE Model Training - Phase 3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Stages:
  1 - IQ Encoder training (bootstrap compression)
  2 - Complete CASCADE model (experts + decoder)
  3 - (Stage 2-3 combined in implementation)
  4 - Joint RX/TX training (optional, requires TX observations)

Examples:
  # Start from beginning (default)
  python phase3_model_training.py

  # Resume from Stage 2 (requires Stage 1 checkpoint)
  python phase3_model_training.py --start-stage 2

  # Start Stage 4 only (requires Stage 1 and Stage 2-3 checkpoints)
  python phase3_model_training.py --start-stage 4
        """
    )
    parser.add_argument(
        '--start-stage',
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help='Training stage to start from (1=IQ Encoder, 2=CASCADE Model, 4=Joint RX/TX). Default: 1'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help='Number of GPUs for parallel generation (default: auto-detect)'
    )
    parser.add_argument(
        '--parallel-generation',
        action='store_true',
        default=True,
        help='Use parallel multi-GPU generation (default: True if multiple GPUs detected)'
    )
    args = parser.parse_args()

    # Create artifacts directory
    artifacts_dir = create_artifacts_dir()
    print(f"Artifacts directory: {artifacts_dir}")

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nHardware Configuration:")
    print(f"=" * 60)

    # Detect number of GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.num_gpus is not None:
        num_gpus = args.num_gpus

    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPUs: {num_gpus}× {gpu_name}")
        print(f"  VRAM per GPU: {gpu_mem_gb:.1f} GB")
        print(f"  Total VRAM: {num_gpus * gpu_mem_gb:.1f} GB")

        # Check cluster type
        is_gh200 = 'H200' in gpu_name or 'GH200' in gpu_name or gpu_mem_gb > 80
        is_a100 = 'A100' in gpu_name
        is_cluster = num_gpus >= 4

        if is_gh200:
            print(f"  🚀 GH200 Grace Hopper Detected!")
        elif is_a100 and is_cluster:
            print(f"  🚀 A100 Cluster Detected!")
            print(f"  💡 Parallel generation recommended ({num_gpus} workers)")

        # Auto-enable parallel generation for clusters
        use_parallel = args.parallel_generation and num_gpus >= 4
        if use_parallel:
            print(f"\n  ✓ Parallel generation ENABLED ({num_gpus} GPUs)")
        else:
            print(f"\n  ℹ️  Single-GPU generation (use --parallel-generation for multi-GPU)")

    # CPU info
    import subprocess
    try:
        cpu_cores = int(subprocess.check_output(['nproc'], text=True).strip())
        print(f"  CPU Cores: {cpu_cores}")
    except:
        cpu_cores = 1

    # Memory info
    try:
        mem_info = subprocess.check_output(['free', '-h'], text=True).split('\n')[1].split()
        print(f"  System RAM: {mem_info[1]}")
    except:
        pass
    print(f"=" * 60)

    # Load or create datasets (with collision and QRM scenarios)
    NUM_TRAIN_SAMPLES = int(os.getenv('CASCADE_TRAIN_SAMPLES', '10000'))
    NUM_VAL_SAMPLES = int(os.getenv('CASCADE_VAL_SAMPLES', '2000'))

    # Dataset type configuration
    USE_STREAMING = os.getenv('CASCADE_USE_STREAMING', 'true').lower() in ('true', '1', 'yes')  # Default to streaming!

    # TX encoder training configuration
    ENABLE_TX_ENCODER = os.getenv('CASCADE_TX_ENCODER_TRAINING', 'false').lower() in ('true', '1', 'yes')

    # PRIORITY 1: Use Streaming Dataset (with or without TX observations)
    if STREAMING_DATASET_AVAILABLE and (USE_STREAMING or ENABLE_TX_ENCODER):
        enable_tx_obs = ENABLE_TX_ENCODER

        if enable_tx_obs:
            print("\n🚀 Using STREAMING DATASET with TX OBSERVATIONS (for joint RX/TX)...")
        else:
            print("\n🚀 Using STREAMING DATASET (RX-only)...")

        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Performance: ~{32 if enable_tx_obs else 33} windows/sec (vs 2.93 old method)")

        num_train_streams = NUM_TRAIN_SAMPLES // 5
        num_val_streams = NUM_VAL_SAMPLES // 5

        # Dataset generation configuration (from environment)
        DATASET_BATCH_SIZE = int(os.getenv('CASCADE_DATASET_BATCH_SIZE', '4096'))  # OPTIMIZED for 96GB VRAM
        DATASET_NUM_WORKERS = int(os.getenv('CASCADE_DATASET_NUM_WORKERS', '51'))
        REGENERATE_CACHE = os.getenv('CASCADE_REGENERATE_CACHE', 'false').lower() in ('true', '1', 'yes')

        print(f"  Dataset generation: batch_size={DATASET_BATCH_SIZE}, workers={DATASET_NUM_WORKERS}")
        if DATASET_BATCH_SIZE < 2048:
            print(f"  ⚠️  WARNING: Batch size {DATASET_BATCH_SIZE} is very small for GH200!")
            print(f"      Recommended: 4096-8192 for optimal GPU utilization")
            print(f"      Set: export CASCADE_DATASET_BATCH_SIZE=4096")

        # Window size must match IQ Encoder input (2048 samples = 42.7ms)
        WINDOW_DURATION_SEC = 2048 / 48000  # 0.0427 seconds

        # RAM caching: Only enable if explicitly requested (can use lots of RAM)
        # Set CASCADE_LOAD_TO_RAM=true to enable
        LOAD_TO_RAM = os.getenv('CASCADE_LOAD_TO_RAM', 'false').lower() in ('true', '1', 'yes')

        if LOAD_TO_RAM:
            print(f"\n  RAM Caching: ENABLED (will load dataset subset to RAM)")
        else:
            print(f"\n  RAM Caching: DISABLED (using HDF5 with parallel workers)")
            print(f"  Tip: Set CASCADE_LOAD_TO_RAM=true to enable RAM caching for faster training")

        # PARALLEL GENERATION: Launch if cluster detected and datasets don't exist
        if use_parallel and num_gpus >= 4:
            cache_dir = Path('/tmp/cascade_parallel')
            train_cache = cache_dir / 'train'
            val_cache = cache_dir / 'val'

            # Check if datasets already exist
            if not (train_cache.exists() and val_cache.exists()) or REGENERATE_CACHE:
                print(f"\n{'='*80}")
                print(f"LAUNCHING PARALLEL DATASET GENERATION ({num_gpus} GPUs)")
                print(f"{'='*80}")

                # Calculate optimal batch size based on GPU
                if gpu_mem_gb >= 150:  # B200
                    optimal_batch = 4096
                elif gpu_mem_gb >= 70:  # A100
                    optimal_batch = 3200
                else:  # Smaller GPUs
                    optimal_batch = 1024

                # Launch parallel generation script
                gen_script = Path(__file__).parent / 'generate_dataset_parallel.py'

                # Training set
                print(f"\nGenerating training set ({num_train_streams:,} streams)...")
                import subprocess
                train_cmd = [
                    sys.executable, str(gen_script),
                    '--num-streams', str(num_train_streams),
                    '--num-workers', str(num_gpus),
                    '--cache-dir', str(train_cache),
                    '--batch-size', str(optimal_batch)
                ]
                subprocess.run(train_cmd, check=True)

                # Validation set
                print(f"\nGenerating validation set ({num_val_streams:,} streams)...")
                val_cmd = [
                    sys.executable, str(gen_script),
                    '--num-streams', str(num_val_streams),
                    '--num-workers', str(num_gpus),
                    '--cache-dir', str(val_cache),
                    '--batch-size', str(optimal_batch),
                    '--validation'
                ]
                subprocess.run(val_cmd, check=True)

                print(f"\n✓ Parallel generation complete!")
                print(f"  Training: {train_cache}")
                print(f"  Validation: {val_cache}")

        train_dataset = StreamingCascadeDataset(
            num_streams=num_train_streams,
            stream_duration_sec=10.0,
            window_duration_sec=WINDOW_DURATION_SEC,  # Match IQ Encoder input size
            message_arrival_rate=0.8,
            seed=42,
            regenerate_cache=REGENERATE_CACHE,
            batch_size=DATASET_BATCH_SIZE,
            num_workers=DATASET_NUM_WORKERS,
            device='cuda',
            enable_tx_observations=enable_tx_obs,
            load_into_memory=LOAD_TO_RAM  # Load entire dataset to RAM (fast training, no HDF5 bottlenecks)
        )

        # Clear GPU memory before validation dataset (prevents resource conflicts)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("  GPU cache cleared between train/val datasets")

        val_dataset = StreamingCascadeDataset(
            num_streams=num_val_streams,
            stream_duration_sec=10.0,
            window_duration_sec=WINDOW_DURATION_SEC,
            message_arrival_rate=0.8,
            seed=1042,
            regenerate_cache=REGENERATE_CACHE,
            batch_size=DATASET_BATCH_SIZE,
            num_workers=DATASET_NUM_WORKERS,
            device='cuda',
            enable_tx_observations=enable_tx_obs,
            load_into_memory=LOAD_TO_RAM
        )

        print(f"  ✓ Train: {len(train_dataset):,} windows from {num_train_streams:,} streams")
        print(f"  ✓ Val: {len(val_dataset):,} windows from {num_val_streams:,} streams")

        train_collate_fn = cascade_collate_fn
        val_collate_fn = cascade_collate_fn

    # PRIORITY 2: Legacy reciprocal dataset (only if streaming not available)
    elif ENABLE_TX_ENCODER and RECIPROCAL_DATASET_AVAILABLE:
        print("\n🔗 Creating reciprocal channel datasets (for TX encoder training)...")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Mode: Joint RX/TX training with embedding autoencoder")
        print(f"  ⚠️  WARNING: Using legacy ReciprocalDataset (slow)")

        train_dataset = ReciprocalChannelDataset(
            num_samples=NUM_TRAIN_SAMPLES,
            sample_rate=48000,
            batch_size=4096,
            collision_probability=0.3,
            qrm_probability=0.2,
            seed=42,
            regenerate_cache=False,
            device='cuda'
        )

        val_dataset = ReciprocalChannelDataset(
            num_samples=NUM_VAL_SAMPLES,
            sample_rate=48000,
            batch_size=4096,
            collision_probability=0.3,
            qrm_probability=0.2,
            seed=1042,
            regenerate_cache=False,
            device='cuda'
        )

        print(f"  ✓ Train dataset: {len(train_dataset)} paired TX/RX samples")
        print(f"  ✓ Val dataset: {len(val_dataset)} paired TX/RX samples")

    # Use GPU-accelerated dataset if available (RX-only training)
    elif GPU_DATASET_AVAILABLE:
        print("\n🚀 Creating GPU-accelerated enhanced datasets...")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

        train_dataset = EnhancedPhysicsDataset(
            num_samples=NUM_TRAIN_SAMPLES,
            sample_rate=48000,
            for_test=False,
            batch_size=4096,  # GPU batch size
            collision_probability=0.3,  # 30% have collisions
            qrm_probability=0.2,  # 20% have QRM
            seed=42,
            regenerate_cache=False,  # Use cached if exists
            enable_visualization=False
        )

        val_dataset = EnhancedPhysicsDataset(
            num_samples=NUM_VAL_SAMPLES,
            sample_rate=48000,
            for_test=False,
            batch_size=4096,
            collision_probability=0.3,
            qrm_probability=0.2,
            seed=1042,
            regenerate_cache=False,
            enable_visualization=False
        )

        print(f"  ✓ Train dataset: {len(train_dataset)} samples (GPU-enhanced with continuous fading)")
        print(f"  ✓ Val dataset: {len(val_dataset)} samples (GPU-enhanced with continuous fading)")
    else:
        print("\n⚠️  GPU not available - using CPU collision-aware datasets...")

        signal_gen = SignalGenerator()

        train_dataset = CollisionAwareDataset(
            num_samples=NUM_TRAIN_SAMPLES,
            signal_generator=signal_gen,
            sample_rate=48000,
            for_test=False,
            collision_probability=0.3,  # 30% have collisions
            qrm_probability=0.2,  # 20% have QRM
            max_collisions=3,
            seed=42
        )

        val_dataset = CollisionAwareDataset(
            num_samples=NUM_VAL_SAMPLES,
            signal_generator=signal_gen,
            sample_rate=48000,
            for_test=False,
            collision_probability=0.3,
            qrm_probability=0.2,
            max_collisions=3,
            seed=1042
        )

        print(f"  ✓ Train dataset: {len(train_dataset)} samples (CPU with collisions/QRM/context)")
        print(f"  ✓ Val dataset: {len(val_dataset)} samples (CPU with collisions/QRM/context)")

    # Train model
    model, iq_encoder = train_cascade_model(artifacts_dir, train_dataset, val_dataset, device, start_stage=args.start_stage)

    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE")
    print("=" * 80)
    print("\nGenerated artifacts:")
    for artifact in sorted(artifacts_dir.rglob('*')):
        if artifact.is_file():
            print(f"  - {artifact.relative_to(artifacts_dir)}")
    print("\n✓ Phase 3: Model Training completed successfully!")

    return model


if __name__ == "__main__":
    main()
