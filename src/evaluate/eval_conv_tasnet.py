import argparse
import os
import json
from pathlib import Path
import torch
from tqdm import tqdm
from src.models.conv_tasnet import ConvTasNet, ConvTasNetMultiScale
from src.data.librimix_dataloader import create_librimix_dataloader
from src.utils.logger import setup_logger
from src.utils.eval_utils import si_snr_loss
from itertools import permutations


def compute_si_snr_with_pit(estimated, target, eps=1e-8):
    """
    Compute SI-SNR with Permutation Invariant Training (PIT)

    Args:
        estimated: [batch, n_src, time] - estimated sources
        target: [batch, n_src, time] - target sources
        eps: small constant for numerical stability

    Returns:
        si_snr: [batch, n_src] - SI-SNR for each source using best permutation
    """
    batch_size, n_src, _ = estimated.shape

    # Get all possible permutations
    perms = list(permutations(range(n_src)))

    # Compute SI-SNR for each permutation
    best_si_snr = None
    best_perm = None

    for perm in perms:
        # Reorder estimated sources according to this permutation
        estimated_perm = estimated[:, perm, :]

        # Compute SI-SNR for this permutation: [batch, n_src]
        si_snr = si_snr_loss(estimated_perm, target, eps)

        # Average across sources for comparison: [batch]
        si_snr_mean = si_snr.mean(dim=1)

        # Keep track of best permutation
        if best_si_snr is None:
            best_si_snr = si_snr
            best_si_snr_mean = si_snr_mean
            best_perm = perm
        else:
            # Update where this permutation is better
            mask = si_snr_mean > best_si_snr_mean
            for i in range(batch_size):
                if mask[i]:
                    best_si_snr[i] = si_snr[i]
            best_si_snr_mean = torch.where(mask, si_snr_mean, best_si_snr_mean)

    return best_si_snr


def load_model(checkpoint_path, device, logger):
    """
    Load trained Conv-TasNet model from checkpoint (supports both standard and multi-scale)

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        logger: logger instance

    Returns:
        Loaded Conv-TasNet model in eval mode
    """
    logger.info(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configuration from checkpoint
    if 'config' in checkpoint:
        # New format with full config
        config = checkpoint['config']
        model_config = config['model']
        dataset_config = config['dataset']
        run_config = config.get('run', {})
        
        # Determine model type
        model_type = run_config.get('model_type', None)
        
        # Infer model type if not specified (backward compatibility)
        if model_type is None:
            if 'encoder_kernel_sizes' in model_config:
                model_type = 'multi_scale'
            else:
                model_type = 'standard'
        
        logger.info(f"Detected model type: {model_type}")
        
        if model_type == 'multi_scale':
            model = ConvTasNetMultiScale(
                num_sources=dataset_config['n_src'],
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
        else:  # standard
            model = ConvTasNet(
                num_sources=dataset_config['n_src'],
                encoder_channels=model_config['encoder_channels'],
                bottleneck_channels=model_config['bottleneck_channels'],
                hidden_channels=model_config['hidden_channels'],
                kernel_size=model_config['kernel_size'],
                tcn_kernel_size=model_config['tcn_kernel_size'],
                num_blocks=model_config['num_blocks'],
                num_repeats=model_config['num_repeats'],
                stride=model_config['stride']
            )
    else:
        # Old format - try to infer from state dict
        logger.warning("Checkpoint doesn't contain config. Attempting to infer model architecture...")

        # Default 2-source model
        num_sources = 2

        # Try to infer from state dict keys if available
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Create model with default standard config (assume standard for old checkpoints)
        model = ConvTasNet(
            num_sources=num_sources,
            encoder_channels=256,
            bottleneck_channels=256,
            hidden_channels=512,
            kernel_size=16,
            tcn_kernel_size=3,
            num_blocks=8,
            num_repeats=3,
            stride=8
        )

    # Load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    logger.info("✓ Model loaded successfully")
    return model


def evaluate_conv_tasnet(model_path, root_dir_data, config_path_data, split, device, logger):
    """
    Evaluate Conv-TasNet on specified split (dev or test)

    Args:
        model_path: Path to trained model checkpoint
        root_dir_data: Path to LibriMix dataset
        config_path_data: Path to data config YAML file
        split: 'dev' or 'test'
        device: torch device
        logger: logger instance

    Returns:
        results: Dictionary with metrics
    """
    logger.info(f"Evaluating Conv-TasNet on {split} split...")

    # Load model
    model = load_model(model_path, device, logger)

    # Create dataloader
    dataloader = create_librimix_dataloader(
        root_dir_data=root_dir_data,
        config_path_data=config_path_data,
        split=split
    )

    logger.info(f"Loaded {len(dataloader.dataset)} samples from {split} split")

    all_si_snr = []
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
            mixture = batch['mixture'].to(device)
            sources = batch['sources'].to(device)

            # Forward pass
            separated = model(mixture)

            # Compute SI-SNR with PIT (returns positive SI-SNR in dB)
            # This finds the best permutation and returns [batch, n_src] SI-SNR values
            si_snr = compute_si_snr_with_pit(separated, sources)  # [batch, n_src]

            # Store SI-SNR values
            all_si_snr.append(si_snr.cpu())

            batch_count += 1

    # Concatenate all batches
    all_si_snr = torch.cat(all_si_snr, dim=0)  # [num_samples, num_sources]
    num_samples, num_sources = all_si_snr.shape

    results = {
        'model_path': str(Path(model_path).name),
        'config_data': Path(config_path_data).name,
        'split': split,
        'num_samples': num_samples,
        'num_sources': num_sources,
        'num_batches': batch_count,
        'mean_si_snr': all_si_snr.mean(dim=0).tolist(),
        'std_si_snr': all_si_snr.std(dim=0).tolist(),
        'min_si_snr': all_si_snr.min(dim=0).values.tolist(),
        'max_si_snr': all_si_snr.max(dim=0).values.tolist(),
        'overall_mean_si_snr': all_si_snr.mean().item(),
        'overall_std_si_snr': all_si_snr.std().item(),
    }

    logger.info(f"Conv-TasNet results on {split}:")
    logger.info(f"  Num samples: {num_samples}")
    logger.info(f"  Overall mean SI-SNR: {results['overall_mean_si_snr']:.2f} dB")
    logger.info(f"  Overall std SI-SNR: {results['overall_std_si_snr']:.2f} dB")
    for src_idx in range(num_sources):
        logger.info(f"  Source {src_idx}: {results['mean_si_snr'][src_idx]:.2f} +- {results['std_si_snr'][src_idx]:.2f} dB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Conv-TasNet on LibriMix")

    parser.add_argument('--model-path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--root-dir-data', required=True, help='Path to LibriMix root directory')
    parser.add_argument('--config-data', required=True, help='Path to config YAML file')
    parser.add_argument('--split', default='dev', choices=['dev', 'test'], help='Dataset split to evaluate on (default: dev)')
    parser.add_argument('--output-dir', default='output/results/conv_tasnet', help='Directory to save results')
    parser.add_argument('--log-level', default='INFO', help='Log level (default: INFO)')
    parser.add_argument('--log-file', default=None, help='Optional log file path')
    parser.add_argument('--device', default=None, choices=['cuda', 'mps', 'cpu'], help='Device to use (auto-detect if not specified)')

    args = parser.parse_args()

    logger = setup_logger(__name__, log_file=args.log_file, level=args.log_level)

    logger.info("="*50)
    logger.info("Conv-TasNet Evaluation")
    logger.info("="*50)
    logger.info(f"model_path: {args.model_path}")
    logger.info(f"config_data: {args.config_data}")
    logger.info(f"root_dir_data: {args.root_dir_data}")
    logger.info(f"split: {args.split}")

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    logger.info(f"Device: {device}")

    # Run evaluation
    results = evaluate_conv_tasnet(
        model_path=args.model_path,
        root_dir_data=args.root_dir_data,
        config_path_data=args.config_data,
        split=args.split,
        device=device,
        logger=logger
    )

    # Save results to JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract model name from path for filename
    model_name = Path(args.model_path).parent.name
    results_file = output_dir / f'{model_name}_results_{args.split}.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    logger.info("="*50)


if __name__ == '__main__':
    main()
