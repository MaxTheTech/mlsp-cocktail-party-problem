#!/usr/bin/env python3
"""
benchmark script for LibriMix dataloader performance

usage:
    python -m src.utils.benchmark_dataloader \
        --root-dir data/Libri2Mix \
        --config config/libri2mix_16k_2src.yaml \
        --num-batches 50
"""

import argparse
import time
import tempfile
import torch
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import json
from src.data.librimix_dataloader import create_librimix_dataloader
from src.utils.logger import setup_logger


def benchmark_dataloader(dataloader, num_batches=50, warmup_batches=5, device='cpu'):
    """
    benchmark dataloader performance

    args:
        dataloader: pytorch dataloader to benchmark
        num_batches: number of batches to time
        warmup_batches: number of warmup batches (not timed)
        device: device to move data to (simulates training)

    returns:
        dict with timing statistics
    """
    times = []
    total_samples = 0

    # warmup
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches:
            break
        # move to device to simulate training
        _ = batch['mixture'].to(device)

    # actual benchmark
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        batch_start = time.time()

        # move to device to simulate training
        mixture = batch['mixture'].to(device)
        sources = batch['sources'].to(device)

        batch_time = time.time() - batch_start
        times.append(batch_time)
        total_samples += mixture.size(0)

    total_time = time.time() - start_time

    return {
        'total_time': total_time,
        'num_batches': len(times),
        'total_samples': total_samples,
        'avg_batch_time': sum(times) / len(times),
        'min_batch_time': min(times),
        'max_batch_time': max(times),
        'samples_per_sec': total_samples / total_time,
        'batch_times': times,
    }


def run_benchmark_suite(root_dir, config_path, num_batches=50, split='train'):
    """
    run comprehensive benchmark comparing different worker configurations

    all other settings (batch size, segment length, etc.) are loaded from config file
    """
    logger = setup_logger(__name__)

    logger.info("="*80)
    logger.info("dataloader performance benchmark")
    logger.info("="*80)
    logger.info(f"config file: {config_path}")

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"using device: {device}")

    results = {}

    # configuration matrix - test different worker counts
    # all other params (batch size, segment length) come from config file
    configs = [
        {
            'name': 'baseline (4 workers)',
            'num_workers': 4,
        },
        {
            'name': 'optimized (8 workers)',
            'num_workers': 8,
        },
        {
            'name': 'auto-detect workers',
            'num_workers': None,  # will auto-configure in dataloader
        },
    ]

    for config in configs:
        logger.info(f"\n{'='*80}")
        logger.info(f"testing: {config['name']}")
        logger.info(f"{'='*80}")

        # temporarily override num_workers in config if specified
        # by reading config, modifying, and passing back
        config_file = Path(config_path)
        with open(config_file) as f:
            full_config = yaml.safe_load(f)

        if config['num_workers'] is not None:
            full_config['dataloader']['num_workers'] = config['num_workers']
            logger.info(f"num_workers: {config['num_workers']}")
        else:
            logger.info(f"num_workers: auto-detect")

        # create temporary config file with modified settings
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(full_config, tmp)
            tmp_config_path = tmp.name

        try:
            # create dataloader from modified config
            dataloader = create_librimix_dataloader(
                root_dir=root_dir,
                config_path=tmp_config_path,
                split=split,
            )

            # run benchmark
            logger.info(f"benchmarking {num_batches} batches...")
            stats = benchmark_dataloader(dataloader, num_batches=num_batches, device=device)

            # log results
            logger.info(f"total time: {stats['total_time']:.2f}s")
            logger.info(f"avg batch time: {stats['avg_batch_time']*1000:.1f}ms")
            logger.info(f"throughput: {stats['samples_per_sec']:.1f} samples/sec")
            logger.info(f"min/max batch time: {stats['min_batch_time']*1000:.1f}ms / {stats['max_batch_time']*1000:.1f}ms")

            results[config['name']] = stats
        finally:
            # cleanup temp file
            Path(tmp_config_path).unlink()

    return results


def plot_results(results, save_path=None):
    """create visualization of benchmark results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    config_names = list(results.keys())

    # throughput comparison
    ax = axes[0, 0]
    throughputs = [results[name]['samples_per_sec'] for name in config_names]
    bars = ax.bar(range(len(config_names)), throughputs, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=15, ha='right')
    ax.set_ylabel('samples/sec', fontsize=11)
    ax.set_title('throughput comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, throughputs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)

    # avg batch time comparison
    ax = axes[0, 1]
    batch_times = [results[name]['avg_batch_time']*1000 for name in config_names]
    bars = ax.bar(range(len(config_names)), batch_times, color='coral', alpha=0.7)
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=15, ha='right')
    ax.set_ylabel('milliseconds', fontsize=11)
    ax.set_title('average batch time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # add value labels
    for bar, val in zip(bars, batch_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}ms',
                ha='center', va='bottom', fontsize=9)

    # batch time distribution (first config)
    ax = axes[1, 0]
    first_config = config_names[0]
    times = [t*1000 for t in results[first_config]['batch_times']]
    ax.hist(times, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('batch time (ms)', fontsize=11)
    ax.set_ylabel('frequency', fontsize=11)
    ax.set_title(f'batch time distribution: {first_config}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # batch time distribution (best config)
    ax = axes[1, 1]
    best_config = min(config_names, key=lambda n: results[n]['avg_batch_time'])
    times = [t*1000 for t in results[best_config]['batch_times']]
    ax.hist(times, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('batch time (ms)', fontsize=11)
    ax.set_ylabel('frequency', fontsize=11)
    ax.set_title(f'batch time distribution: {best_config}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="benchmark LibriMix dataloader performance")

    parser.add_argument('--root-dir', type=str, required=True,
                        help="root directory of LibriMix dataset (e.g., data/Libri2Mix)")
    parser.add_argument('--config', type=str, required=True,
                        help="path to dataset config file (YAML)")
    parser.add_argument('--num-batches', type=int, default=50,
                        help="number of batches to benchmark")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'dev', 'test'],
                        help="dataset split to benchmark")
    parser.add_argument('--save-plot', type=str, default='output/figures/dataloader_benchmark.png',
                        help="path to save benchmark plot")
    parser.add_argument('--save-json', type=str, default='output/logs/dataloader_benchmark.json',
                        help="path to save benchmark results as JSON")

    args = parser.parse_args()

    # run benchmark suite
    results = run_benchmark_suite(
        root_dir=args.root_dir,
        config_path=args.config,
        num_batches=args.num_batches,
        split=args.split
    )

    # save results to JSON
    if args.save_json:
        # remove batch_times from json (too verbose)
        json_results = {}
        for name, stats in results.items():
            json_results[name] = {k: v for k, v in stats.items() if k != 'batch_times'}

        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nsaved results to {args.save_json}")

    # plot results
    plot_results(results, save_path=args.save_plot)

    # print summary
    print("\n" + "="*80)
    print("benchmark summary")
    print("="*80)

    baseline_name = list(results.keys())[0]
    baseline_throughput = results[baseline_name]['samples_per_sec']

    for name, stats in results.items():
        speedup = stats['samples_per_sec'] / baseline_throughput
        print(f"\n{name}:")
        print(f"  throughput: {stats['samples_per_sec']:.1f} samples/sec")
        print(f"  speedup: {speedup:.2f}x vs baseline")
        print(f"  avg batch time: {stats['avg_batch_time']*1000:.1f}ms")


if __name__ == '__main__':
    main()
