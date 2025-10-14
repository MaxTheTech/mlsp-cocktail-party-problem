"""
PyTorch Dataset for LibriMix (Libri2Mix/Libri3Mix)
"""
import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, List


class LibriMixDataset(Dataset):
    """
    PyTorch Dataset for LibriMix (Speech Separation Dataset)
    
    Args:
        root_dir: Path to LibriMix root directory (e.g., 'data/Libri2Mix')
        split: 'train', 'dev', or 'test'
        sample_rate: '8k' or '16k'
        n_src: Number of sources (2 or 3)
        mode: 'min' or 'max' (mixture duration mode)
        mixture_type: 'mix_clean', 'mix_both', or 'mix_single'
        segment: Optional segment length in samples for chunking (None = full utterance)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        sample_rate: str = '16k',
        n_src: int = 2,
        mode: str = 'min',
        mixture_type: str = 'mix_clean',
        segment: Optional[int] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.mode = mode
        self.mixture_type = mixture_type
        self.segment = segment
        
        # Construct path to dataset directory
        self.dataset_path = os.path.join(
            root_dir, 
            f'wav{sample_rate}',
            mode,
            self._get_split_dir()
        )
        
        # Load metadata
        self.metadata_path = os.path.join(
            self.dataset_path.replace(f'/{self._get_split_dir()}', ''),
            'metadata',
            f'mixture_{self._get_split_dir()}_{mixture_type}.csv'
        )
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")
        
        self.metadata = pd.read_csv(self.metadata_path)
        print(f"Loaded {len(self.metadata)} mixtures from {self.metadata_path}")
        
        # Get actual sample rate as integer
        self.fs = int(sample_rate.replace('k', '')) * 1000
        
    def _get_split_dir(self) -> str:
        """Get directory name for split"""
        split_map = {
            'train': 'train-100',
            'dev': 'dev',
            'test': 'test'
        }
        return split_map.get(self.split, self.split)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing:
            - mixture: Mixed audio signal [samples]
            - sources: List of clean source signals [n_src, samples]
            - noise: Noise signal (if mix_both or mix_single) [samples]
            - mixture_id: String identifier
        """
        row = self.metadata.iloc[idx]
        
        # Load mixture
        mixture_path = row['mixture_path']
        mixture, fs = torchaudio.load(mixture_path)
        mixture = mixture.squeeze(0)  # Remove channel dimension if mono
        
        # Load sources
        sources = []
        for i in range(self.n_src):
            source_path = row[f'source_{i+1}_path']
            source, _ = torchaudio.load(source_path)
            source = source.squeeze(0)
            sources.append(source)
        
        sources = torch.stack(sources)  # [n_src, samples]
        
        # Load noise if applicable
        noise = None
        if self.mixture_type in ['mix_both', 'mix_single']:
            noise_path = row['noise_path']
            noise, _ = torchaudio.load(noise_path)
            noise = noise.squeeze(0)
        
        # Apply segmentation if specified
        if self.segment is not None:
            mixture, sources, noise = self._segment_audio(mixture, sources, noise)
        
        result = {
            'mixture': mixture,
            'sources': sources,
            'mixture_id': row['mixture_ID']
        }
        
        if noise is not None:
            result['noise'] = noise
            
        return result
    
    def _segment_audio(
        self, 
        mixture: torch.Tensor, 
        sources: torch.Tensor,
        noise: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extract random segment from audio"""
        length = mixture.shape[-1]
        
        if length <= self.segment:
            # Pad if too short
            pad_len = self.segment - length
            mixture = torch.nn.functional.pad(mixture, (0, pad_len))
            sources = torch.nn.functional.pad(sources, (0, pad_len))
            if noise is not None:
                noise = torch.nn.functional.pad(noise, (0, pad_len))
        else:
            # Random crop
            start = torch.randint(0, length - self.segment, (1,)).item()
            mixture = mixture[start:start + self.segment]
            sources = sources[:, start:start + self.segment]
            if noise is not None:
                noise = noise[start:start + self.segment]
        
        return mixture, sources, noise
    
    def get_infos(self) -> Dict:
        """Get dataset information"""
        return {
            'dataset': f'Libri{self.n_src}Mix',
            'split': self.split,
            'sample_rate': self.fs,
            'mode': self.mode,
            'mixture_type': self.mixture_type,
            'num_samples': len(self),
            'segment_length': self.segment
        }


def create_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 4,
    num_workers: int = 4,
    sample_rate: str = '16k',
    n_src: int = 2,
    mode: str = 'min',
    mixture_type: str = 'mix_clean',
    segment: Optional[int] = None,
    shuffle: bool = True
):
    """
    Create a DataLoader for LibriMix
    
    Example:
        train_loader = create_dataloader(
            root_dir='data/Libri2Mix',
            split='train',
            batch_size=8,
            segment=32000  # 2 seconds at 16kHz
        )
    """
    dataset = LibriMixDataset(
        root_dir=root_dir,
        split=split,
        sample_rate=sample_rate,
        n_src=n_src,
        mode=mode,
        mixture_type=mixture_type,
        segment=segment
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == '__main__':
    # Example usage
    dataset = LibriMixDataset(
        root_dir='data/Libri2Mix',
        split='train',
        sample_rate='16k',
        n_src=2,
        mode='min',
        mixture_type='mix_clean'
    )
    
    print(f"Dataset info: {dataset.get_infos()}")
    print(f"Number of samples: {len(dataset)}")
    
    # Load one sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Mixture shape: {sample['mixture'].shape}")
    print(f"  Sources shape: {sample['sources'].shape}")
    print(f"  Mixture ID: {sample['mixture_id']}")