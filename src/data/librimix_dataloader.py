"""DataLoader utilities for LibriMix dataset"""

import os
import torch
from torch.utils.data import DataLoader
from src.data.librimix_dataset import LibriMixDataset, collate_fn_librimix


def create_librimix_dataloader(root_dir, split='train', batch_size=8,
                                sample_rate='16k', n_src=2, mode='min',
                                mixture_type='mix_clean', segment_length=None,
                                num_workers=None, shuffle=None,
                                return_speaker_info=False, return_metrics=False,
                                preload_to_ram=False, cache_size=0):
    """
    Create DataLoader for LibriMix dataset with auto-configured settings

    Args:
        root_dir: Path to LibriMix root (e.g., 'data/Libri2Mix')
        split: 'train', 'dev', or 'test'
        batch_size: Batch size
        sample_rate: '8k' or '16k'
        n_src: Number of sources (2 or 3)
        mode: 'min' or 'max'
        mixture_type: 'mix_clean', 'mix_both', or 'mix_single'
        segment_length: Fixed segment length in samples (None for variable)
        num_workers: Number of workers (auto-configured if None)
        shuffle: Shuffle data (auto: True for train, False otherwise)
        return_speaker_info: Include speaker IDs and genders
        return_metrics: Include SNR metrics
    """

    # create dataset
    dataset = LibriMixDataset(
        root_dir=root_dir,
        split=split,
        sample_rate=sample_rate,
        n_src=n_src,
        mode=mode,
        mixture_type=mixture_type,
        segment_length=segment_length,
        return_speaker_info=return_speaker_info,
        return_metrics=return_metrics,
        preload_to_ram=preload_to_ram,
        cache_size=cache_size,
    )

    # auto-configure settings
    if shuffle is None:
        shuffle = (split == 'train')

    if num_workers is None:
        # increase worker count for I/O-bound audio loading
        # use more workers since audio loading is disk-bound, not CPU-bound
        cpu_count = os.cpu_count() or 1
        num_workers = min(cpu_count, 8)  # increased from 4 to 8

    # check if GPU/MPS available for pin_memory
    has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    pin_memory = has_gpu and num_workers > 0

    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_librimix,
        pin_memory=pin_memory,
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataloader


def create_train_val_test_loaders(root_dir, batch_size_train=16,
                                   batch_size_val=8, batch_size_test=8,
                                   sample_rate='16k', n_src=2, mode='min',
                                   mixture_type='mix_clean',
                                   segment_length_train=64000,  # 4s at 16kHz
                                   segment_length_val=None,
                                   segment_length_test=None,
                                   num_workers=None,
                                   return_speaker_info=False,
                                   return_metrics=False,
                                   preload_to_ram=False,
                                   cache_size=0):
    """Create train, val, and test dataloaders"""

    train_loader = create_librimix_dataloader(
        root_dir=root_dir,
        split='train',
        batch_size=batch_size_train,
        sample_rate=sample_rate,
        n_src=n_src,
        mode=mode,
        mixture_type=mixture_type,
        segment_length=segment_length_train,
        num_workers=num_workers,
        return_speaker_info=return_speaker_info,
        return_metrics=return_metrics,
        preload_to_ram=preload_to_ram,
        cache_size=cache_size,
    )

    val_loader = create_librimix_dataloader(
        root_dir=root_dir,
        split='dev',
        batch_size=batch_size_val,
        sample_rate=sample_rate,
        n_src=n_src,
        mode=mode,
        mixture_type=mixture_type,
        segment_length=segment_length_val,
        num_workers=num_workers,
        return_speaker_info=return_speaker_info,
        return_metrics=return_metrics,
        preload_to_ram=preload_to_ram,
        cache_size=cache_size,
    )

    test_loader = create_librimix_dataloader(
        root_dir=root_dir,
        split='test',
        batch_size=batch_size_test,
        sample_rate=sample_rate,
        n_src=n_src,
        mode=mode,
        mixture_type=mixture_type,
        segment_length=segment_length_test,
        num_workers=num_workers,
        return_speaker_info=return_speaker_info,
        return_metrics=return_metrics,
        preload_to_ram=preload_to_ram,
        cache_size=cache_size,
    )

    return train_loader, val_loader, test_loader
