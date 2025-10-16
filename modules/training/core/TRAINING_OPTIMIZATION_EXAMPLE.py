#!/usr/bin/env python3
"""
CASCADE Training with Performance Optimizations

This example shows how to apply all performance optimizations:
- DataLoader with multi-workers + pin_memory
- Mixed precision training (AMP)
- torch.compile() JIT compilation
- Gradient accumulation
- Optimal batch sizes

Expected speedup: 10-24× over baseline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
from pathlib import Path

# Import CASCADE components
from streaming_cascade_dataset import StreamingCascadeDataset

def create_optimized_dataloaders(
    num_train_samples: int = 100000,
    num_val_samples: int = 10000,
    batch_size: int = 512,
    num_workers: int = 32,
    device: str = 'cuda'
):
    """
    Create optimized DataLoaders for CASCADE training.

    Args:
        num_train_samples: Training samples
        num_val_samples: Validation samples
        batch_size: GPU batch size (512 for GH200)
        num_workers: Parallel workers (32 recommended)
        device: 'cuda' for GPU

    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    print("\n" + "="*70)
    print("CREATING OPTIMIZED DATALOADERS")
    print("="*70)

    # Training dataset (with GPU-accelerated generation)
    print(f"\nGenerating training dataset ({num_train_samples:,} samples)...")
    train_dataset = StreamingCascadeDataset(
        num_streams=num_train_samples // 5,  # 5 windows per stream
        stream_duration_sec=10.0,
        window_duration_sec=2.0,
        message_arrival_rate=0.8,
        sample_rate=48000,
        for_test=False,
        seed=42,
        cache_dir='./dataset_cache',
        use_cache=True,
        batch_size=512,  # ← OPTIMIZED: Large GPU batches
        device=device,
        num_workers=None  # Auto-detect CPU cores
    )

    # Validation dataset (smaller, harder scenarios)
    print(f"\nGenerating validation dataset ({num_val_samples:,} samples)...")
    val_dataset = StreamingCascadeDataset(
        num_streams=num_val_samples // 5,
        stream_duration_sec=10.0,
        window_duration_sec=2.0,
        message_arrival_rate=0.8,
        sample_rate=48000,
        for_test=True,  # Harder distribution
        seed=123,
        cache_dir='./dataset_cache',
        use_cache=True,
        batch_size=256,
        device=device
    )

    # Create DataLoaders with ALL optimizations
    print("\n" + "="*70)
    print("DATALOADER CONFIGURATION")
    print("="*70)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,        # ← CRITICAL: Parallel loading
        pin_memory=True,                 # ← Faster CPU→GPU transfer
        prefetch_factor=4,               # ← Pre-load 4 batches per worker
        persistent_workers=True,         # ← Reuse workers (avoid startup)
        drop_last=True                   # Consistent batch sizes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,    # Fewer workers for validation
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=False
    )

    print(f"✓ Training DataLoader:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Workers: {num_workers}")
    print(f"  - Batches per epoch: {len(train_loader)}")
    print(f"  - Prefetch factor: 4")
    print(f"  - Pin memory: True")

    print(f"\n✓ Validation DataLoader:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Workers: {num_workers // 2}")
    print(f"  - Batches: {len(val_loader)}")

    return train_loader, val_loader, train_dataset, val_dataset


def train_epoch_optimized(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int = 1,
    max_batches: int = None
):
    """
    Train one epoch with all optimizations:
    - Mixed precision (AMP)
    - Gradient accumulation
    - Efficient GPU usage

    Args:
        model: CASCADE model
        train_loader: Training DataLoader
        optimizer: Optimizer
        scaler: GradScaler for AMP
        device: CUDA device
        accumulation_steps: Gradient accumulation steps (effective batch = batch_size × steps)
        max_batches: Limit batches for quick testing

    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    start_time = time.time()

    for batch_idx, (rx_iq, labels) in enumerate(train_loader):
        if max_batches and batch_idx >= max_batches:
            break

        # Move to GPU (non-blocking with pin_memory=True)
        rx_iq = rx_iq.to(device, non_blocking=True)

        # Extract labels
        pattern_ids = labels['pattern_id'].to(device, non_blocking=True)
        frequency_triples = labels['frequency_triple'].to(device, non_blocking=True)

        # *** AUTOMATIC MIXED PRECISION (AMP) ***
        with autocast():
            # Forward pass in FP16 (2× faster matmul/conv)
            outputs = model(rx_iq)

            # Compute loss (automatically handled in FP32 for stability)
            pattern_loss = nn.functional.cross_entropy(
                outputs['pattern'],
                pattern_ids
            )
            frequency_loss = nn.functional.cross_entropy(
                outputs['frequency'],
                frequency_triples
            )
            loss = pattern_loss + frequency_loss

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Scaled backward pass (handles FP16→FP32 conversion)
        scaler.scale(loss).backward()

        # Update weights every N steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping (prevent exploding gradients)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with scaled gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        # Progress update every 100 batches
        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * train_loader.batch_size / elapsed
            print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                  f"Loss={loss.item()*accumulation_steps:.4f}, "
                  f"Speed={samples_per_sec:.0f} samples/sec")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
):
    """
    Validate model with mixed precision.

    Args:
        model: CASCADE model
        val_loader: Validation DataLoader
        device: CUDA device

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for rx_iq, labels in val_loader:
            rx_iq = rx_iq.to(device, non_blocking=True)
            pattern_ids = labels['pattern_id'].to(device, non_blocking=True)
            frequency_triples = labels['frequency_triple'].to(device, non_blocking=True)

            # Validation also benefits from AMP
            with autocast():
                outputs = model(rx_iq)
                pattern_loss = nn.functional.cross_entropy(outputs['pattern'], pattern_ids)
                frequency_loss = nn.functional.cross_entropy(outputs['frequency'], frequency_triples)
                loss = pattern_loss + frequency_loss

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main_training_loop_optimized(
    model: nn.Module,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 512,
    num_workers: int = 32,
    accumulation_steps: int = 1,
    use_compile: bool = True
):
    """
    Complete optimized training loop with all features.

    Args:
        model: CASCADE model
        num_epochs: Training epochs
        learning_rate: Initial learning rate
        batch_size: GPU batch size
        num_workers: DataLoader workers
        accumulation_steps: Gradient accumulation (effective_batch = batch_size × steps)
        use_compile: Use torch.compile() for JIT (PyTorch 2.0+)
    """
    print("\n" + "="*70)
    print("CASCADE OPTIMIZED TRAINING")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Move model to GPU
    model = model.to(device)

    # *** TORCH.COMPILE() - PyTorch 2.0+ JIT (1.3-1.8× speedup) ***
    if use_compile and hasattr(torch, 'compile'):
        print("\n🚀 Compiling model with torch.compile()...")
        print("  (First epoch will be slower due to compilation)")
        model = torch.compile(
            model,
            mode='reduce-overhead',  # Optimize for training
            # mode='max-autotune'    # Alternative: more aggressive
        )
        print("  ✓ Model compiled!")

    # Create optimized dataloaders
    train_loader, val_loader, _, _ = create_optimized_dataloaders(
        num_train_samples=10000,  # Small for demo
        num_val_samples=2000,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device
    )

    # Optimizer (AdamW with weight decay)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # *** MIXED PRECISION SCALER ***
    scaler = GradScaler()

    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {accumulation_steps}")
    print(f"Effective batch: {batch_size * accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Mixed precision: ✓ Enabled")
    print(f"torch.compile(): {'✓ Enabled' if use_compile else '✗ Disabled'}")
    print(f"DataLoader workers: {num_workers}")

    # Training loop
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch_optimized(
            model, train_loader, optimizer, scaler, device,
            accumulation_steps=accumulation_steps
        )

        # Validate
        val_loss = validate_epoch(model, val_loader, device)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = Path(f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    # Example: Create a dummy model
    class DummyCascadeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2)
            self.bn1 = nn.BatchNorm1d(64)
            self.pattern_head = nn.Linear(64, 4)
            self.frequency_head = nn.Linear(64, 43)

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.mean(x, dim=-1)
            return {
                'pattern': self.pattern_head(x),
                'frequency': self.frequency_head(x)
            }

    # Run optimized training
    model = DummyCascadeModel()

    main_training_loop_optimized(
        model,
        num_epochs=3,
        learning_rate=1e-3,
        batch_size=512,
        num_workers=32,
        accumulation_steps=1,
        use_compile=True
    )
