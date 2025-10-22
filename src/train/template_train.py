#!/usr/bin/env python3
"""
Template training script for audio source separation models

Usage:
    # Normal training
    python -m src.train.template_train --root-dir data/Libri2Mix --epochs 50

    # Debug mode (fast iteration)
    python -m src.train.template_train --root-dir data/Libri2Mix --debug

    # Resume from checkpoint
    python -m src.train.template_train --root-dir data/Libri2Mix --resume models/checkpoint.pth
"""

import argparse
from pathlib import Path
from typing import Dict, Optional
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.data.librimix_dataloader import create_train_val_test_loaders
from src.models.example_model import SimpleSourceSeparationModel, si_snr_loss
from src.utils.train_utils import (
    setup_training_device,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    count_parameters,
    save_training_config,
    AverageMeter,
    Timer,
    DebugDatasetWrapper
)
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Template training script for source separation")

    # Required arguments
    parser.add_argument("--root-dir", type=str, required=True,
                        help="Root directory of LibriMix dataset (e.g., data/Libri2Mix)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to dataset config file (YAML) containing dataset and dataloader settings")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="Gradient clipping threshold (0 = no clipping)")

    # Model arguments
    parser.add_argument("--encoder-channels", type=int, default=256,
                        help="Number of encoder channels")
    parser.add_argument("--hidden-channels", type=int, default=512,
                        help="Number of hidden channels")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of layers in bottleneck")

    # Optimization arguments
    parser.add_argument("--use-amp", action="store_true",
                        help="Use automatic mixed precision (AMP) for CUDA")
    parser.add_argument("--scheduler", type=str, default="reduce_on_plateau",
                        choices=["none", "step", "cosine", "reduce_on_plateau"],
                        help="Learning rate scheduler")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Early stopping patience (epochs)")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")

    # Debug arguments
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (implies --subset-size 100 if not set)")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Use only N samples from train/val sets (for debugging)")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Run 1-2 batches per epoch to test pipeline")
    parser.add_argument("--profile", action="store_true",
                        help="Enable profiling (time each step and data loading)")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--log-file", type=str, default="logs/training.log",
                        help="Log file path")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "mps", "cpu"],
                        help="Force specific device (auto-detect if not specified)")

    args = parser.parse_args()

    # Debug mode adjustments
    if args.debug and args.subset_size is None:
        args.subset_size = 100
        print("🐛 Debug mode: Setting subset_size to 100")

    return args


def create_dataloaders(args):
    """Create train/val/test dataloaders from config."""
    logger = setup_logger(__name__, log_file=args.log_file)

    logger.info(f"Loading LibriMix dataset from {args.root_dir}")
    logger.info(f"Loading dataset config from {args.config}")

    # Create dataloaders from config file
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        root_dir=args.root_dir,
        config_path=args.config,
    )

    # Apply debug wrapper if needed
    if args.subset_size is not None:
        logger.info(f"🐛 Applying debug subset: {args.subset_size} samples")
        from torch.utils.data import DataLoader

        # Wrap the datasets with debug wrapper
        train_dataset_wrapped = DebugDatasetWrapper(
            train_loader.dataset,
            subset_size=args.subset_size,
            seed=args.seed
        )
        val_dataset_wrapped = DebugDatasetWrapper(
            val_loader.dataset,
            subset_size=min(args.subset_size // 5, len(val_loader.dataset)),
            seed=args.seed
        )

        # Recreate dataloaders with wrapped datasets
        train_loader = DataLoader(
            train_dataset_wrapped,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
            collate_fn=train_loader.collate_fn
        )
        val_loader = DataLoader(
            val_dataset_wrapped,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory,
            collate_fn=val_loader.collate_fn
        )

    logger.info(f"✓ Dataset loaded: {len(train_loader.dataset)} train, "
                f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    return train_loader, val_loader, test_loader


def create_model(args, train_loader):
    """Create model instance."""
    # Get num_sources from dataset
    num_sources = train_loader.dataset.n_src

    model = SimpleSourceSeparationModel(
        num_sources=num_sources,
        encoder_channels=args.encoder_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers
    )

    num_params = count_parameters(model)
    print(f"✓ Model created: {num_params:,} trainable parameters")

    return model


def create_optimizer_and_scheduler(model, args):
    """Create optimizer and learning rate scheduler."""
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        scheduler = None

    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    args,
    scaler: Optional[object] = None  # GradScaler type, but imported conditionally
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    logger = setup_logger(__name__, log_file=args.log_file)

    loss_meter = AverageMeter("Loss")
    data_timer = Timer() if args.profile else None
    compute_timer = Timer() if args.profile else None

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

    batch_start_time = time.time()

    for batch_idx, batch in enumerate(pbar):
        if args.fast_dev_run and batch_idx >= 2:
            logger.info("🏃 Fast dev run: Stopping after 2 batches")
            break

        # track data loading time
        if data_timer:
            data_load_time = time.time() - batch_start_time
            data_timer.times.append(data_load_time)

        if compute_timer:
            compute_timer.start()

        # Move data to device
        mixture = batch['mixture'].to(device)  # [B, T]
        sources = batch['sources'].to(device)  # [B, num_sources, T]

        # Forward pass
        optimizer.zero_grad()

        if args.use_amp and device.type == "cuda":
            # Automatic mixed precision (CUDA only)
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(mixture)  # [B, num_sources, T]
                loss = criterion(outputs, sources)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(mixture)
            loss = criterion(outputs, sources)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), mixture.size(0))

        # track compute time
        if compute_timer:
            compute_timer.stop()

        # Update progress bar
        postfix = {
            'loss': f'{loss_meter.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        }

        if args.profile and data_timer and compute_timer:
            postfix['data_ms'] = f'{data_timer.times[-1]*1000:.0f}'
            postfix['comp_ms'] = f'{compute_timer.times[-1]*1000:.0f}'

        pbar.set_postfix(postfix)

        # reset timer for next batch
        if data_timer:
            batch_start_time = time.time()

    # log profiling summary
    if data_timer and compute_timer:
        avg_data_time = data_timer.average() * 1000
        avg_compute_time = compute_timer.average() * 1000
        total_time = avg_data_time + avg_compute_time

        logger.info(f"\nprofiling summary:")
        logger.info(f"  avg data loading time: {avg_data_time:.1f}ms ({avg_data_time/total_time*100:.1f}%)")
        logger.info(f"  avg compute time: {avg_compute_time:.1f}ms ({avg_compute_time/total_time*100:.1f}%)")
        logger.info(f"  total batch time: {total_time:.1f}ms")

        # warn if data loading is bottleneck
        if avg_data_time > avg_compute_time:
            logger.warning(f"⚠️  data loading is slower than compute! consider:")
            logger.warning(f"    - increasing --num-workers")
            logger.warning(f"    - using --cache-size or --preload-to-ram")

    return {'loss': loss_meter.avg}


def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    epoch: int,
    args
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    logger = setup_logger(__name__, log_file=args.log_file)

    loss_meter = AverageMeter("Val Loss")

    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx >= 2:
                logger.info("🏃 Fast dev run: Stopping after 2 batches")
                break

            # Move data to device
            mixture = batch['mixture'].to(device)
            sources = batch['sources'].to(device)

            # Forward pass
            outputs = model(mixture)
            loss = criterion(outputs, sources)

            # Update metrics
            loss_meter.update(loss.item(), mixture.size(0))

            pbar.set_postfix({'val_loss': f'{loss_meter.avg:.4f}'})

    return {'loss': loss_meter.avg}


def main():
    """Main training function."""
    args = parse_args()

    # Setup
    set_seed(args.seed)
    logger = setup_logger(__name__, log_file=args.log_file, level='INFO')

    logger.info("=" * 80)
    logger.info("Starting training script")
    logger.info("=" * 80)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_training_config(vars(args), save_dir)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(args)

    # Create model
    model = create_model(args, train_loader)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = None  # Auto-detect

    model, device = setup_training_device(model, device)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args)

    # Loss function
    criterion = si_snr_loss

    # AMP scaler (CUDA only)
    scaler = None
    if args.use_amp and device.type == "cuda":
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        logger.info("✓ Using Automatic Mixed Precision (AMP)")

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"✓ Resumed from epoch {start_epoch - 1}")

    # Early stopping tracker
    patience_counter = 0

    # Training loop
    logger.info("Starting training loop...")

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info("-" * 40)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args, scaler
        )
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, args)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")

        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # Check for improvement
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            logger.info(f"✓ New best model! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epoch(s)")

        # Save checkpoint
        if epoch % args.save_every == 0 or is_best:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_val_loss,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'args': vars(args)
            }

            save_checkpoint(
                checkpoint_state,
                save_dir,
                filename=f"checkpoint_epoch_{epoch}.pth",
                is_best=is_best
            )

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {save_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
