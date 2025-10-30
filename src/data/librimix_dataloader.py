import os
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from src.data.librimix_dataset import LibriMixDataset, collate_fn_librimix


def create_librimix_dataloader(root_dir_data, config_path_data, split='train'):
    """
    Create DataLoader for LibriMix dataset from config file

    Args:
        root_dir_data: Path to LibriMix root (e.g., 'data/Libri2Mix')
        config_path_data: Path to YAML config file (contains dataset and dataloader settings)
        split: 'train', 'dev', or 'test'

    Returns:
        DataLoader configured with settings from config file
    """

    # load config from YAML file
    config_path_data = Path(config_path_data)
    if not config_path_data.exists():
        raise FileNotFoundError(f"Config file not found: {config_path_data}")

    with open(config_path_data) as f:
        full_config = yaml.safe_load(f)

    # extract dataset and dataloader sections
    if 'dataset' not in full_config:
        raise ValueError("Config file must have 'dataset' section")
    if 'dataloader' not in full_config:
        raise ValueError("Config file must have 'dataloader' section")

    dataloader_config = full_config['dataloader']

    # extract dataloader parameters
    batch_sizes = {'train': dataloader_config['batch_size_train'],
                   'dev': dataloader_config['batch_size_val'],
                   'test': dataloader_config['batch_size_test']}
    batch_size = batch_sizes[split]

    num_workers = dataloader_config.get('num_workers', None)
    shuffle = (split == 'train')

    # create dataset (dataset loads config internally)
    dataset = LibriMixDataset(
        root_dir_data=root_dir_data,
        config_path_data=config_path_data,
        split=split,
    )

    # auto-configure num_workers if not specified
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(cpu_count, 8)

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


def create_train_val_test_loaders(root_dir_data, config_path_data):
    """
    Create train, validation, and test dataloaders from config file

    Args:
        root_dir_data: Path to LibriMix root (e.g., 'data/Libri2Mix')
        config_path_data: Path to YAML config file

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader = create_librimix_dataloader(
        root_dir_data=root_dir_data,
        config_path_data=config_path_data,
        split='train'
    )
    val_loader = create_librimix_dataloader(
        root_dir_data=root_dir_data,
        config_path_data=config_path_data,
        split='dev'
    )
    test_loader = create_librimix_dataloader(
        root_dir_data=root_dir_data,
        config_path_data=config_path_data,
        split='test'
    )
    return train_loader, val_loader, test_loader
