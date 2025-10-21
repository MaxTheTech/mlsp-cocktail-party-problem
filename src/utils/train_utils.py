"""Training utilities for PyTorch models"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from pathlib import Path
import json
import time
import numpy as np


def get_device():
    """Get best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def setup_training_device(model, device=None):
    """Move model to device and enable optimizations"""
    if device is None:
        device = get_device()

    model = model.to(device)

    # Enable cuDNN benchmarking for faster training
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    return model, device


class DebugDatasetWrapper(Dataset):
    """Wrapper to use a small subset for debugging"""

    def __init__(self, dataset, subset_size=None, seed=42):
        self.dataset = dataset

        if subset_size is not None and subset_size < len(dataset):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(dataset), size=subset_size, replace=False)
            indices = sorted(indices.tolist())
            self.subset = Subset(dataset, indices)
            print(f"Debug mode: using {subset_size}/{len(dataset)} samples")
        else:
            self.subset = dataset
            print(f"Using full dataset: {len(dataset)} samples")

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]


class AverageMeter:
    """Track average values during training"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.values = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.values.append(val)

    def std(self):
        """Compute standard deviation of tracked values"""
        if len(self.values) < 2:
            return 0.0
        return float(np.std(self.values))

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class Timer:
    """Simple timer for profiling"""

    def __init__(self):
        self.times = []
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)
        return elapsed

    def average(self):
        return np.mean(self.times) if self.times else 0.0

    def reset(self):
        self.times = []


def save_checkpoint(state, save_dir, filename="checkpoint.pth", is_best=False):
    """Save model checkpoint"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / filename
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = save_dir / "best_model.pth"
        torch.save(state, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device=None):
    """Load model checkpoint"""
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best Loss: {checkpoint.get('best_loss', 'unknown')}")

    return checkpoint


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"Set random seed to {seed}")


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_training_config(config, save_dir):
    """Save training config to JSON"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / "config.json"

    # Convert non-serializable objects to strings
    serializable = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool, list, dict, type(None))):
            serializable[key] = value
        else:
            serializable[key] = str(value)

    with open(config_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved config to {config_path}")


def gradient_norm(model):
    """Compute total gradient norm for debugging"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
