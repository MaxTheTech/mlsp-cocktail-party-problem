import argparse
from pathlib import Path
from typing import Dict, Optional
import time
import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from src.data.librimix_dataloader import create_train_val_test_loaders
from src.models.conv_tasnet import ConvTasNet, ConvTasNetMultiScale, si_snr_loss
from src.utils.train_utils import (
    set_training_device,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    count_parameters,
    save_training_config,
    generate_unique_model_id,
    AverageMeter
)
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified Conv-TasNet training script for audio source separation (supports standard and multi-scale variants)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model selection
    parser.add_argument("--model-type", type=str, default="multi_scale", 
                        choices=["standard", "multi_scale"],
                        help="Model architecture: 'standard' for original Conv-TasNet, 'multi_scale' for Multi-Scale variant")

    # Required arguments
    parser.add_argument("--root-dir-data", type=str, required=True,
                        help="Root directory of LibriMix dataset (e.g., data/Libri2Mix)")
    parser.add_argument("--config-data", type=str, required=True,
                        help="Path to dataset config file (YAML)")
    parser.add_argument("--config-model", type=str, required=True,
                        help="Path to model config file (YAML) containing model and training hyperparameters")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="models/conv_tasnet",
                        help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Enable periodic checkpoint saving (in addition to best model)")

    # Debug arguments
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (implies --subset-size 100 if not set)")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Use only N samples from train/val sets (for debugging)")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Run 1-2 batches per epoch to test pipeline")
    parser.add_argument("--profile", action="store_true",
                        help="Enable profiling (time each step and data loading)")

    # Multi-GPU arguments
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Enable multi-GPU training with DataParallel")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training with DistributedDataParallel (DDP)")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3')")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="Local rank for distributed training (set by torch.distributed.launch)")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Number of processes for distributed training")
    
    # Other
    parser.add_argument("--log-file", type=str, default=None,
                        help="Optional log file path")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging verbosity level (default: INFO)")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging")

    args = parser.parse_args()

    # Debug mode adjustments
    if args.debug and args.subset_size is None:
        args.subset_size = 100
        print("🐛 Debug mode: Setting subset_size to 100")
    
    # Multi-GPU validation
    if args.multi_gpu and args.distributed:
        raise ValueError("Cannot use both --multi-gpu and --distributed. Choose one.")
    
    # Parse GPU IDs
    if args.gpu_ids:
        args.gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    
    return args


def setup_distributed(args):
    """Setup for distributed training"""
    if args.distributed:
        # Initialize distributed backend
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ['RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.local_rank = int(os.environ['LOCAL_RANK'])
        elif args.local_rank != -1:
            args.rank = args.local_rank
            args.world_size = torch.cuda.device_count()
        else:
            print("Not using distributed mode")
            args.distributed = False
            args.rank = 0  # Default rank for non-distributed
            return args
        
        # Setup device
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device('cuda', args.local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank
        )

        # Note: seed will be set later after loading config

        print(f'Distributed training: rank {args.rank}/{args.world_size}, local_rank {args.local_rank}')
    else:
        # Non-distributed training
        args.rank = 0
        args.world_size = 1
    
    return args


def is_main_process(args):
    """Check if this is the main process"""
    if args.distributed:
        return args.rank == 0
    return True


def cleanup_distributed(args):
    """Cleanup distributed training"""
    if args.distributed:
        dist.destroy_process_group()


def create_dataloaders(args):
    """Create train/val/test dataloaders from config."""
    logger = setup_logger(__name__, log_file=args.log_file)

    logger.info(f"Loading LibriMix dataset from {args.root_dir_data}")
    logger.info(f"Loading dataset config from {args.config_data}")

    # Create dataloaders from config file
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        root_dir_data=args.root_dir_data,
        config_path_data=args.config_data,
        include_test=True
    )

    # Note: Debug subset functionality (--subset-size) removed due to missing DebugDatasetWrapper
    # To debug with smaller dataset, modify the config file to use fewer samples
    if args.subset_size is not None:
        logger.warning(f"⚠️  --subset-size is not supported (DebugDatasetWrapper unavailable)")
        logger.warning(f"    To train on subset, modify your dataset config file instead")

    # Add distributed sampler for DDP
    train_sampler = None
    val_sampler = None
    
    if args.distributed:
        from torch.utils.data import DataLoader
        
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=args.seed
        )
        val_sampler = DistributedSampler(
            val_loader.dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False
        )
        
        # Recreate loaders with distributed samplers
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=train_sampler,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
            collate_fn=train_loader.collate_fn
        )
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=val_loader.batch_size,
            sampler=val_sampler,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory,
            collate_fn=val_loader.collate_fn
        )
        
        logger.info(f"✓ Using DistributedSampler for rank {args.rank}/{args.world_size}")

    logger.info(f"✓ Dataset loaded: {len(train_loader.dataset)} train, "
                f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    return train_loader, val_loader, test_loader, train_sampler, val_sampler


def create_model(model_config, dataset_config, logger, args):
    """Create Conv-TasNet model instance (standard or multi-scale based on args.model_type)."""
    # Get num_sources from dataset config
    num_sources = dataset_config['n_src']

    # Create model based on model_type
    if args.model_type == "standard":
        # Standard Conv-TasNet
        model = ConvTasNet(
            num_sources=num_sources,
            encoder_channels=model_config['encoder_channels'],
            bottleneck_channels=model_config['bottleneck_channels'],
            hidden_channels=model_config['hidden_channels'],
            kernel_size=model_config.get('kernel_size', 16),
            tcn_kernel_size=model_config['tcn_kernel_size'],
            num_blocks=model_config['num_blocks'],
            num_repeats=model_config['num_repeats'],
            stride=model_config.get('stride', 8)
        )
        
        num_params = count_parameters(model)
        
        if is_main_process(args):
            logger.info("=" * 60)
            logger.info("Conv-TasNet Model Configuration:")
            logger.info("-" * 60)
            logger.info(f"  Number of sources:       {num_sources}")
            logger.info(f"  Encoder channels (N):    {model_config['encoder_channels']}")
            logger.info(f"  Bottleneck channels (B): {model_config['bottleneck_channels']}")
            logger.info(f"  Hidden channels (H):     {model_config['hidden_channels']}")
            logger.info(f"  Encoder kernel (L):      {model_config.get('kernel_size', 16)}")
            logger.info(f"  TCN kernel (P):          {model_config['tcn_kernel_size']}")
            logger.info(f"  Num blocks (X):          {model_config['num_blocks']}")
            logger.info(f"  Num repeats (R):         {model_config['num_repeats']}")
            logger.info(f"  Encoder stride:          {model_config.get('stride', 8)}")
            logger.info(f"  Total TCN layers:        {model_config['num_blocks'] * model_config['num_repeats']}")
            logger.info("-" * 60)
            logger.info(f"  Trainable parameters:    {num_params:,}")
            logger.info("=" * 60)
            print(f"✓ Conv-TasNet created: {num_params:,} trainable parameters")
            
    elif args.model_type == "multi_scale":
        # Multi-Scale Conv-TasNet
        model = ConvTasNetMultiScale(
            num_sources=num_sources,
            encoder_channels=model_config['encoder_channels'],
            bottleneck_channels=model_config['bottleneck_channels'],
            hidden_channels=model_config['hidden_channels'],
            encoder_kernel_sizes=model_config['encoder_kernel_sizes'],
            tcn_kernel_size=model_config['tcn_kernel_size'],
            num_blocks=model_config['num_blocks'],
            num_repeats=model_config['num_repeats'],
            num_tcn_scales=model_config['num_tcn_scales'],
            use_frequency_aware=model_config['use_frequency_aware']
        )
        
        num_params = count_parameters(model)
        
        if is_main_process(args):
            logger.info("=" * 60)
            logger.info("Conv-TasNet Multi-Scale Model Configuration:")
            logger.info("-" * 60)
            logger.info(f"  Number of sources:         {num_sources}")
            logger.info(f"  Encoder channels (N):      {model_config['encoder_channels']}")
            logger.info(f"  Bottleneck channels (B):   {model_config['bottleneck_channels']}")
            logger.info(f"  Hidden channels (H):       {model_config['hidden_channels']}")
            logger.info(f"  Encoder kernel sizes:      {model_config['encoder_kernel_sizes']}")
            logger.info(f"  TCN kernel (P):            {model_config['tcn_kernel_size']}")
            logger.info(f"  Num blocks (X):            {model_config['num_blocks']}")
            logger.info(f"  Num repeats (R):           {model_config['num_repeats']}")
            logger.info(f"  Num TCN scales:            {model_config['num_tcn_scales']}")
            logger.info(f"  Use frequency-aware:       {model_config['use_frequency_aware']}")
            logger.info(f"  Total TCN layers/branch:   {model_config['num_blocks'] * model_config['num_repeats']}")
            logger.info(f"  Total TCN branches:        {model_config['num_tcn_scales']}")
            logger.info("-" * 60)
            logger.info(f"  Trainable parameters:      {num_params:,}")
            logger.info("=" * 60)
            logger.info("")
            logger.info("Multi-Scale Features:")
            logger.info(f"  ✓ Multi-scale encoders/decoders: {len(model_config['encoder_kernel_sizes'])} scales")
            logger.info(f"  ✓ Hierarchical TCN: {model_config['num_tcn_scales']} branches")
            logger.info(f"  ✓ Frequency-aware convolutions: {'Yes' if model_config['use_frequency_aware'] else 'No'}")
            logger.info("=" * 60)
            print(f"✓ Conv-TasNet Multi-Scale created: {num_params:,} trainable parameters")
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    return model


def wrap_model_for_multi_gpu(model, args, device):
    """Wrap model for multi-GPU training"""
    if args.distributed:
        # DistributedDataParallel
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True  # Required for Conv-TasNet architecture
        )
        if is_main_process(args):
            print(f"✓ Using DistributedDataParallel on {args.world_size} GPUs")
    
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        # DataParallel (simpler but less efficient)
        if args.gpu_ids:
            model = nn.DataParallel(model, device_ids=args.gpu_ids)
            print(f"✓ Using DataParallel on GPUs: {args.gpu_ids}")
        else:
            model = nn.DataParallel(model)
            print(f"✓ Using DataParallel on {torch.cuda.device_count()} GPUs")
    
    return model


def create_optimizer_and_scheduler(model, training_config, logger):
    """Create optimizer and learning rate scheduler."""
    lr = float(training_config['learning_rate'])
    weight_decay = float(training_config['weight_decay'])

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    logger.debug(f"Optimizer: Adam (lr={lr}, weight_decay={weight_decay})")

    # Learning rate scheduler
    scheduler_type = training_config.get('scheduler', 'none')
    scheduler_params = training_config.get('scheduler_params', {})

    if scheduler_type == "step":
        step_size = scheduler_params.get('step', {}).get('step_size', 10)
        gamma = scheduler_params.get('step', {}).get('gamma', 0.5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.debug(f"Scheduler: StepLR (step_size={step_size}, gamma={gamma})")
    elif scheduler_type == "cosine":
        epochs = int(training_config['epochs'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        logger.debug(f"Scheduler: CosineAnnealingLR (T_max={epochs})")
    elif scheduler_type == "reduce_on_plateau":
        factor = scheduler_params.get('reduce_on_plateau', {}).get('factor', 0.5)
        patience = scheduler_params.get('reduce_on_plateau', {}).get('patience', 5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience
        )
        logger.debug(f"Scheduler: ReduceLROnPlateau (patience={patience}, factor={factor})")
    else:
        scheduler = None
        logger.debug("Scheduler: None")

    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    training_config: dict,
    logger,
    args,
    scaler: Optional[object] = None  # GradScaler type, but imported conditionally
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter("Loss")

    # Simple timer implementation for profiling
    data_times = [] if args.profile else None
    compute_times = [] if args.profile else None
    compute_start = None

    # Get training parameters from config
    epochs = int(training_config['epochs'])
    gradient_clip_norm = float(training_config['gradient_clip_norm'])
    use_amp = training_config.get('use_amp', False)

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")

    batch_start_time = time.time()
    
    # Track for memory cleanup
    batch_count = 0

    for batch_idx, batch in enumerate(pbar):
        if args.fast_dev_run and batch_idx >= 2:
            logger.info("🏃 Fast dev run: Stopping after 2 batches")
            break

        # track data loading time
        if data_times is not None:
            data_load_time = time.time() - batch_start_time
            data_times.append(data_load_time)

        if compute_times is not None:
            compute_start = time.time()

        # Move data to device
        mixture = batch['mixture'].to(device)  # [B, T]
        sources = batch['sources'].to(device)  # [B, num_sources, T]

        # Forward pass
        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            # Automatic mixed precision (CUDA only)
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(mixture)  # [B, num_sources, T]
                loss = criterion(outputs, sources)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(mixture)
            loss = criterion(outputs, sources)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), mixture.size(0))

        # track compute time
        if compute_times is not None and compute_start is not None:
            compute_times.append(time.time() - compute_start)

        # Update progress bar
        postfix = {
            'loss': f'{loss_meter.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        }

        if args.profile and data_times and compute_times:
            postfix['data_ms'] = f'{data_times[-1]*1000:.0f}'
            postfix['comp_ms'] = f'{compute_times[-1]*1000:.0f}'

        pbar.set_postfix(postfix)

        # reset timer for next batch
        if data_times is not None:
            batch_start_time = time.time()
        
        # Periodic memory cleanup (every 50 batches)
        batch_count += 1
        if batch_count % 50 == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # log profiling summary
    if data_times and compute_times:
        avg_data_time = sum(data_times) / len(data_times) * 1000
        avg_compute_time = sum(compute_times) / len(compute_times) * 1000
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
    
    # Final memory cleanup at end of epoch
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {'loss': loss_meter.avg}


def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    epoch: int,
    training_config: dict,
    logger,
    args
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()

    loss_meter = AverageMeter("Val Loss")

    epochs = int(training_config['epochs'])
    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")

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
    
    # Memory cleanup after validation
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {'loss': loss_meter.avg}


def main():
    """Main training function for Conv-TasNet (standard or multi-scale)."""
    args = parse_args()
    
    # Fix for NFS temporary file issues with multiprocessing
    # Use local /tmp instead of NFS-mounted directories
    import tempfile
    if os.path.exists('/tmp') and os.access('/tmp', os.W_OK):
        tempfile.tempdir = '/tmp'
        os.environ['TMPDIR'] = '/tmp'
    
    # Setup distributed training if needed
    args = setup_distributed(args)

    # Setup logger for all processes
    if is_main_process(args):
        # Main process logs to file and console
        logger = setup_logger(__name__, log_file=args.log_file, level=args.log_level)

        logger.info("=" * 80)
        model_name = "Conv-TasNet Multi-Scale" if args.model_type == "multi_scale" else "Conv-TasNet"
        logger.info(f"{model_name} Training Script")
        logger.info("=" * 80)
        logger.info(f"Model Type: {args.model_type}")

        if args.multi_gpu:
            logger.info("Multi-GPU mode: DataParallel")
        elif args.distributed:
            logger.info(f"Multi-GPU mode: DistributedDataParallel ({args.world_size} processes)")
        logger.info("=" * 80)
    else:
        # Non-main processes: create a simple console logger (minimal output)
        import logging
        logger = logging.getLogger(f"{__name__}_rank{args.rank}")
        logger.setLevel(logging.WARNING)  # Only show warnings and errors
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(f'[Rank {args.rank}] %(levelname)s: %(message)s'))
            logger.addHandler(handler)

    # ============================================
    # DETERMINE IF RESUMING AND LOAD CONFIGS
    # ============================================
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")

        # Extract model directory from checkpoint path
        checkpoint_path = Path(args.resume)
        model_dir = str(checkpoint_path.parent)

        # Load config from checkpoint directory
        config_path = checkpoint_path.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                f"Cannot resume without original config."
            )

        with open(config_path, 'r') as f:
            full_training_config = json.load(f)

        # Extract sub-configs
        dataset_config = full_training_config['dataset']
        run_config = full_training_config['run']
        model_config = full_training_config['model']
        training_config = full_training_config['training']

        # Restore model_type from saved config (if available)
        saved_model_type = run_config.get('model_type', None)
        if saved_model_type is not None:
            args.model_type = saved_model_type
            logger.info(f"Restored model_type from checkpoint: {saved_model_type}")
        else:
            # Infer from model config (backward compatibility)
            if 'encoder_kernel_sizes' in model_config:
                args.model_type = 'multi_scale'
            else:
                args.model_type = 'standard'
            logger.info(f"Inferred model_type from config: {args.model_type}")

        run_id = full_training_config.get('run', {}).get('run_id', 'resumed_run')

        logger.info(f"Loaded config from checkpoint")
        logger.info(f"Resuming run: {run_id}")
        logger.info(f"Using existing directory: {model_dir}")

    else:
        logger.info("Starting fresh training")

        # Load configs from YAML files
        with open(args.config_data) as f:
            full_data_config = yaml.safe_load(f)
        dataset_config = full_data_config['dataset']

        with open(args.config_model) as f:
            full_model_config = yaml.safe_load(f)

        run_config = full_model_config['run']
        model_config = full_model_config['model']
        training_config = full_model_config['training']

        # Generate unique model ID
        model_type = Path(args.save_dir).name
        model_dir, run_id = generate_unique_model_id(logger, model_type, args.save_dir)

        # Merge configs and save
        full_training_config = full_data_config | full_model_config
        full_training_config['run']['run_id'] = run_id
        full_training_config['run']['model_type'] = args.model_type  # Save model_type to config
        save_training_config(logger, full_training_config, model_dir)

    logger.debug("Model config:")
    for key, value in model_config.items():
        logger.debug(f"  {key}: {value}")
    logger.debug("Training config:")
    for key, value in training_config.items():
        logger.debug(f"  {key}: {value}")

    # Set seed
    seed = run_config['seed']
    if args.distributed:
        set_seed(logger, seed + args.rank)
    else:
        set_seed(logger, seed)

    # Create save directory
    save_dir = Path(model_dir)

    # Create dataloaders
    train_loader, val_loader, test_loader, train_sampler, val_sampler = create_dataloaders(args)

    # Create model using config
    model = create_model(model_config, dataset_config, logger, args)

    # Setup device
    device_config = run_config.get('device', None)
    if args.distributed:
        device = args.device  # Already set in setup_distributed
        model = model.to(device)
    else:
        model, device = set_training_device(logger, model, device_config)

    # Wrap model for multi-GPU
    model = wrap_model_for_multi_gpu(model, args, device)

    # Create optimizer and scheduler using config
    optimizer, scheduler = create_optimizer_and_scheduler(model, training_config, logger)

    # Loss function
    criterion = si_snr_loss

    # AMP scaler (CUDA only)
    scaler = None
    use_amp = training_config.get('use_amp', False)
    if use_amp and device.type == "cuda":
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        logger.info("✓ Using Automatic Mixed Precision (AMP)")

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume:
        checkpoint, start_epoch, best_val_loss = load_checkpoint(logger, args.resume, model, optimizer, device)
        logger.info(f"✓ Resumed from epoch {start_epoch - 1}")

    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    epochs = int(training_config['epochs'])
    save_every = training_config.get('save_every', 5)
    early_stopping_patience = training_config.get('early_stopping_patience', None)

    if args.save_checkpoints:
        logger.info(f"Checkpointing: saving every {save_every} epochs + best model")
    else:
        logger.info("Checkpointing: only saving best model")

    patience_counter = 0

    for epoch in range(start_epoch, epochs + 1):
        # Synchronize all processes at the start of each epoch (CRITICAL for DDP!)
        if args.distributed:
            print(f"[Rank {args.rank}] Starting epoch {epoch} - entering barrier", flush=True)
            dist.barrier()
            print(f"[Rank {args.rank}] Passed start barrier for epoch {epoch}", flush=True)

        # Set epoch for distributed samplers (required for proper shuffling in DDP)
        if args.distributed:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
                print(f"[Rank {args.rank}] Set train_sampler epoch to {epoch}", flush=True)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
                print(f"[Rank {args.rank}] Set val_sampler epoch to {epoch}", flush=True)

        logger.info("")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info("-" * 40)

        # Train
        if args.distributed:
            print(f"[Rank {args.rank}] Starting training for epoch {epoch}", flush=True)
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            training_config, logger, args, scaler
        )
        if args.distributed:
            print(f"[Rank {args.rank}] Finished training for epoch {epoch}", flush=True)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")

        # Validate
        if args.distributed:
            print(f"[Rank {args.rank}] Starting validation for epoch {epoch}", flush=True)
        val_metrics = validate(
            model, val_loader, criterion, device, epoch,
            training_config, logger, args
        )
        if args.distributed:
            print(f"[Rank {args.rank}] Finished validation for epoch {epoch}", flush=True)
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

        # Save checkpoint (only main process)
        should_save_periodic = args.save_checkpoints and (epoch % save_every == 0)
        should_save_best = is_best

        if args.distributed:
            print(f"[Rank {args.rank}] Before checkpoint section for epoch {epoch}", flush=True)

        if is_main_process(args) and (should_save_periodic or should_save_best):
            print(f"[Rank {args.rank}] Saving checkpoint for epoch {epoch}", flush=True)
            # Get model state dict (unwrap if using DataParallel or DDP)
            if isinstance(model, (nn.DataParallel, DDP)):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_val_loss,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'config': full_training_config
            }

            # Save periodic checkpoint
            if should_save_periodic:
                checkpoint_filename = f"checkpoint_epoch_{epoch}.pth"
                save_checkpoint(logger, checkpoint_state, save_dir, filename=checkpoint_filename, is_best=False)
                logger.info(f"Saved checkpoint: {checkpoint_filename}")

            # Save best model (separate file)
            if should_save_best:
                save_checkpoint(logger, checkpoint_state, save_dir, filename="best_model.pth", is_best=True)
                logger.info("Saved best model")
            
            # Clear checkpoint_state to free memory
            del checkpoint_state
            
            # Memory cleanup after checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[Rank {args.rank}] Checkpoint saved for epoch {epoch}", flush=True)
        
        if args.distributed:
            print(f"[Rank {args.rank}] After checkpoint section for epoch {epoch}", flush=True)

        # Early stopping
        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered (patience={early_stopping_patience})")
            break

        # Final synchronization at end of epoch
        if args.distributed:
            print(f"[Rank {args.rank}] Entering end-of-epoch barrier", flush=True)
            dist.barrier()
            print(f"[Rank {args.rank}] Passed end-of-epoch barrier for epoch {epoch}", flush=True)

    if is_main_process(args):
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Model saved to: {save_dir}")
        logger.info("=" * 80)
    
    # Cleanup distributed
    cleanup_distributed(args)


if __name__ == "__main__":
    main()
