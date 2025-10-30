import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm
from src.models.oracle_irm import OracleIRM
from src.data.librimix_dataloader import create_librimix_dataloader
from src.utils.logger import setup_logger
from src.utils.eval_utils import si_snr_loss


def evaluate_oracle_irm(root_dir_data, config_path_data, split, device, logger):
    """
    Evaluate Oracle IRM on specified split (dev or test)

    Args:
        root_dir_data: Path to LibriMix dataset
        config_path_data: Path to data config YAML file
        split: 'dev' or 'test'
        device: torch device
        logger: logger instance

    Returns:
        results: Dictionary with metrics
    """
    logger.info(f"Evaluating Oracle IRM on {split} split...")

    # create model and dataloader
    model = OracleIRM(n_fft=1024, hop_length=256)
    model.eval()

    dataloader = create_librimix_dataloader(root_dir_data=root_dir_data, config_path_data=config_path_data, split=split)

    logger.info(f"Loaded {len(dataloader.dataset)} samples from {split} split")

    all_si_snr = []
    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"evaluating {split}"):
            mixture = batch['mixture']
            sources = batch['sources']

            # forward pass with oracle IRM
            separated = model(mixture, sources)

            # compute SI-SNR
            si_snr = si_snr_loss(separated, sources)

            all_si_snr.append(si_snr)
            batch_count += 1
    
    all_si_snr = torch.cat(all_si_snr, dim=0)
    num_samples, num_sources = all_si_snr.shape

    results = {
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

    logger.info(f"Oracle IRM results on {split}:")
    logger.info(f"  Num samples: {num_samples}")
    logger.info(f"  Overall mean SI-SNR: {results['overall_mean_si_snr']:.2f} dB")
    logger.info(f"  Overall std SI-SNR: {results['overall_std_si_snr']:.2f} dB")
    for src_idx in range(num_sources):
        logger.info(f"  Source {src_idx}: {results['mean_si_snr'][src_idx]:.2f} +- {results['std_si_snr'][src_idx]:.2f} dB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Oracle IRM baseline on LibriMix")

    parser.add_argument('--root-dir-data', required=True, help='Path to LibriMix root directory')
    parser.add_argument('--config-data', required=True, help='Path to config YAML file')
    parser.add_argument('--split', default='dev', choices=['dev', 'test'], help='Dataset split to evaluate on (default: dev)')
    parser.add_argument('--output-dir', default='output/results/oracle_irm', help='Directory to save results')
    parser.add_argument('--log-level', default='INFO', help='Log level (default: INFO)')
    parser.add_argument('--log-file', default=None, help='Optional log file path')

    args = parser.parse_args()

    logger = setup_logger(__name__, log_file=args.log_file, level=args.log_level)

    logger.info("="*50)
    logger.info("Oracle IRM Evaluation - Upper Bound Baseline")
    logger.info("="*50)
    logger.info(f"config_data: {args.config_data}")
    logger.info(f"root_dir_data: {args.root_dir_data}")
    logger.info(f"split: {args.split}")

    # setup device (STFT not supported by MPS, so use CPU)
    device = torch.device('cpu')
    logger.info("Device: using CPU")

    # run evaluation
    results = evaluate_oracle_irm(
        root_dir_data=args.root_dir_data,
        config_path_data=args.config_data,
        split=args.split,
        device=device,
        logger=logger
    )

    # save results to JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f'oracle_irm_results_{args.split}.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    logger.info("="*50)


if __name__ == '__main__':
    main()


