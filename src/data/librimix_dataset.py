import os
import yaml
from pathlib import Path
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


class LibriMixDataset(Dataset):
    """Dataset for LibriMix - loads mixed audio and clean sources from config"""

    def __init__(self, root_dir_data, config_path_data, split='train'):
        """
        Args:
            root_dir_data: Path to LibriMix root (e.g., 'data/Libri2Mix')
            config_path_data: Path to YAML config file (contains dataset and dataloader settings)
            split: 'train', 'dev', or 'test'
        """
        self.root_dir_data = Path(root_dir_data)
        self.split = split

        # load config from YAML file
        config_path_data = Path(config_path_data)
        if not config_path_data.exists():
            raise FileNotFoundError(f"Config file not found: {config_path_data}")

        with open(config_path_data) as f:
            full_config = yaml.safe_load(f)

        dataset_config = full_config['dataset']

        # extract dataset parameters from config
        self.sample_rate = dataset_config['sample_rate']
        self.n_src = dataset_config['n_src']
        self.mode = dataset_config['mode']
        self.mixture_type = dataset_config['mixture_type']
        self.return_speaker_info = dataset_config.get('return_speaker_info', False)
        self.return_metrics = dataset_config.get('return_metrics', False)
        self.preload_to_ram = dataset_config.get('preload_to_ram', False)

        # get segment length based on split
        if split == 'train':
            self.segment_length = dataset_config['segment_length_train']
        elif split == 'dev':
            self.segment_length = dataset_config['segment_length_val']
        elif split == 'test':
            self.segment_length = dataset_config['segment_length_test']
        else:
            raise ValueError(f"Invalid split: {split}")

        # validate config values
        assert split in ['train', 'dev', 'test']
        assert self.sample_rate in ['8k', '16k']
        assert self.n_src in [2, 3]

        # convert sample rate to Hz
        self.fs = int(self.sample_rate.replace('k', '')) * 1000

        # figure out split directory name
        split_names = {'train': 'train-100', 'dev': 'dev', 'test': 'test'}
        self.split_dir = split_names[split]

        # build path to wav files
        self.wav_dir = self.root_dir_data / f'wav{self.sample_rate}' / self.mode

        # load metadata CSV
        metadata_file = f'mixture_{self.split_dir}_{self.mixture_type}.csv'
        self.metadata_path = self.wav_dir / 'metadata' / metadata_file

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Can't find metadata at {self.metadata_path}")

        self.metadata = pd.read_csv(self.metadata_path)

        # precompute and cache normalized CSV paths for efficiency
        self._cache_normalized_paths()

        # validate sample rate on first load
        self._validate_sample_rate()

        # optionally load SNR metrics
        self.metrics_df = None
        if self.return_metrics:
            metrics_file = f'metrics_{self.split_dir}_{self.mixture_type}.csv'
            metrics_path = self.wav_dir / 'metadata' / metrics_file
            if metrics_path.exists():
                self.metrics_df = pd.read_csv(metrics_path)

        # optionally load speaker info
        self.speaker_lookup = {}
        if self.return_speaker_info:
            self._load_speaker_metadata()

        # setup RAM cache
        self.ram_cache = {}

        # preload all data to RAM if requested
        if self.preload_to_ram:
            self._preload_dataset_to_ram()

    def _cache_normalized_paths(self):
        """Precompute normalized CSV paths for all samples to avoid repeated computation"""
        self.cached_paths = []

        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]

            paths = {
                'mixture': self._normalize_path(row['mixture_path'])
            }

            # cache source paths
            for i in range(self.n_src):
                paths[f'source_{i}'] = self._normalize_path(row[f'source_{i+1}_path'])

            # cache noise path if applicable
            if self.mixture_type in ['mix_both', 'mix_single']:
                paths['noise'] = self._normalize_path(row['noise_path'])

            self.cached_paths.append(paths)

    def _load_speaker_metadata(self):
        """Load speaker IDs and genders from LibriSpeech metadata"""
        metadata_dir = self.wav_dir / 'metadata' / 'LibriSpeech'

        for csv_file in ['dev-clean.csv', 'test-clean.csv', 'train-clean-100.csv']:
            csv_path = metadata_dir / csv_file
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    self.speaker_lookup[int(row['speaker_ID'])] = row['sex']

    def _normalize_path(self, stored_path):
        """Convert CSV paths to actual file paths"""
        stored_path = str(stored_path)

        if os.path.isabs(stored_path) and os.path.exists(stored_path):
            return Path(stored_path)

        # remove Libri2Mix/Libri3Mix prefix if present
        for prefix in ['Libri2Mix/', 'Libri3Mix/']:
            if prefix in stored_path:
                stored_path = stored_path.split(prefix, 1)[1]
                break

        return self.root_dir_data / stored_path

    def _validate_sample_rate(self):
        """Validate that first file has correct sample rate"""
        if len(self.cached_paths) == 0:
            return

        first_file = self.cached_paths[0]['mixture']
        with sf.SoundFile(str(first_file)) as f:
            actual_sr = f.samplerate
            if actual_sr != self.fs:
                raise ValueError(
                    f"Sample rate mismatch: expected {self.fs} Hz but got {actual_sr} Hz. "
                    f"Check your dataset configuration (sample_rate='{self.sample_rate}')"
                )

    def _load_audio_segment_impl(self, path, start_frame=None, num_frames=None):
        """
        Load audio file with optional seeking for efficient segment loading

        args:
            path: path to audio file
            start_frame: starting frame (None = start from beginning)
            num_frames: number of frames to read (None = read all)

        returns:
            audio data as torch tensor
        """
        path_str = str(path)

        with sf.SoundFile(path_str) as f:
            if start_frame is not None:
                f.seek(start_frame)

            if num_frames is not None:
                audio = f.read(frames=num_frames, dtype='float32')
            else:
                audio = f.read(dtype='float32')

        return torch.from_numpy(audio)

    def _load_audio_segment(self, path, start_frame=None, num_frames=None):
        """
        Load audio segment, checking RAM cache if preload is enabled
        """
        # check if entire file is preloaded to RAM
        if self.preload_to_ram:
            path_key = str(path)
            if path_key in self.ram_cache:
                audio = self.ram_cache[path_key]

                # apply segmentation if needed
                if start_frame is not None and num_frames is not None:
                    audio = audio[start_frame:start_frame + num_frames]
                elif start_frame is not None:
                    audio = audio[start_frame:]

                return audio

        # otherwise use direct loader
        return self._load_audio_segment_impl(str(path), start_frame, num_frames)

    def _preload_dataset_to_ram(self):
        """preload all audio files to RAM for maximum speed"""
        print(f"preloading {len(self)} samples to RAM...")

        # load all unique audio files
        unique_files = set()
        for paths in self.cached_paths:
            unique_files.add(str(paths['mixture']))
            for i in range(self.n_src):
                unique_files.add(str(paths[f'source_{i}']))
            if 'noise' in paths:
                unique_files.add(str(paths['noise']))

        print(f"loading {len(unique_files)} unique audio files...")

        for file_path in unique_files:
            with sf.SoundFile(file_path) as f:
                audio = f.read(dtype='float32')
            self.ram_cache[file_path] = torch.from_numpy(audio)

        print(f"preloaded {len(unique_files)} files to RAM")

    def _segment_audio(self, mixture, sources, noise=None):
        """Randomly crop or pad audio to fixed length"""
        current_length = mixture.shape[-1]

        if current_length < self.segment_length:
            # pad if too short
            pad_amount = self.segment_length - current_length
            mixture = torch.nn.functional.pad(mixture, (0, pad_amount))
            sources = torch.nn.functional.pad(sources, (0, pad_amount))
            if noise is not None:
                noise = torch.nn.functional.pad(noise, (0, pad_amount))

        elif current_length > self.segment_length:
            # random crop if too long
            start_idx = torch.randint(0, current_length - self.segment_length + 1, (1,)).item()
            end_idx = start_idx + self.segment_length
            mixture = mixture[start_idx:end_idx]
            sources = sources[:, start_idx:end_idx]
            if noise is not None:
                noise = noise[start_idx:end_idx]

        return mixture, sources, noise

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        paths = self.cached_paths[idx]

        # get audio length and compute segment parameters
        original_length = int(row['length'])

        # determine if we should load full file or just a segment
        if self.segment_length is not None and original_length > self.segment_length:
            # optimization: load only the segment we need via seek
            start_frame = torch.randint(0, original_length - self.segment_length + 1, (1,)).item()
            num_frames = self.segment_length
        else:
            # load full file (either no segmentation or file is shorter than segment)
            start_frame = None
            num_frames = None

        # load mixed audio (using cached path)
        mixture = self._load_audio_segment(paths['mixture'], start_frame, num_frames)

        # load clean sources (using cached paths)
        sources = []
        for i in range(self.n_src):
            source = self._load_audio_segment(paths[f'source_{i}'], start_frame, num_frames)
            sources.append(source)

        sources = torch.stack(sources)

        # load noise if needed (using cached path)
        noise = None
        if self.mixture_type in ['mix_both', 'mix_single']:
            noise = self._load_audio_segment(paths['noise'], start_frame, num_frames)

        # pad if necessary (only needed if original file was shorter than segment_length)
        if self.segment_length is not None and mixture.shape[0] < self.segment_length:
            pad_amount = self.segment_length - mixture.shape[0]
            mixture = torch.nn.functional.pad(mixture, (0, pad_amount))
            sources = torch.nn.functional.pad(sources, (0, pad_amount))
            if noise is not None:
                noise = torch.nn.functional.pad(noise, (0, pad_amount))

        # pack everything into dict
        sample = {
            'mixture': mixture,
            'sources': sources,
            'mixture_id': row['mixture_ID'],
            'length': original_length,
        }

        if noise is not None:
            sample['noise'] = noise

        # add speaker info if requested
        if self.return_speaker_info:
            mixture_id = row['mixture_ID']
            parts = mixture_id.split('_')

            speaker_ids = []
            speaker_genders = []
            for part in parts:
                speaker_id = int(part.split('-')[0])
                speaker_ids.append(speaker_id)
                speaker_genders.append(self.speaker_lookup.get(speaker_id, 'U'))

            sample['speaker_ids'] = speaker_ids
            sample['speaker_genders'] = speaker_genders

        # add SNR metrics if requested
        if self.return_metrics and self.metrics_df is not None:
            metrics_row = self.metrics_df[
                self.metrics_df['mixture_ID'] == row['mixture_ID']
            ]

            if len(metrics_row) > 0:
                metrics_row = metrics_row.iloc[0]
                sample['metrics'] = {
                    'source_1_snr': float(metrics_row['source_1_SNR']),
                    'source_2_snr': float(metrics_row['source_2_SNR']),
                }
                if 'noise_SNR' in metrics_row and pd.notna(metrics_row['noise_SNR']):
                    sample['metrics']['noise_snr'] = float(metrics_row['noise_SNR'])

        return sample

    def get_infos(self):
        """Return dataset info"""
        return {
            'dataset': f'Libri{self.n_src}Mix',
            'split': self.split,
            'sample_rate': self.fs,
            'mode': self.mode,
            'mixture_type': self.mixture_type,
            'num_samples': len(self),
            'segment_length': self.segment_length,
        }


def collate_fn_librimix(batch, pad_value=0.0):
    """
    optimized collate function for batching variable-length samples
    reduces memory allocations and uses more efficient tensor operations
    """
    batch_size = len(batch)

    # early return for single sample batch
    if batch_size == 1:
        sample = batch[0]
        return {
            'mixture': sample['mixture'].unsqueeze(0),
            'sources': sample['sources'].unsqueeze(0),
            'lengths': torch.tensor([sample['mixture'].shape[0]], dtype=torch.long),
            'mixture_ids': [sample['mixture_id']],
            **(
                {'noise': sample['noise'].unsqueeze(0)}
                if 'noise' in sample
                else {}
            ),
            **(
                {'speaker_ids': [sample['speaker_ids']]}
                if 'speaker_ids' in sample
                else {}
            ),
            **(
                {'speaker_genders': [sample['speaker_genders']]}
                if 'speaker_genders' in sample
                else {}
            ),
            **(
                {'metrics': [sample.get('metrics')]}
                if 'metrics' in sample
                else {}
            ),
        }

    # extract metadata
    n_src = batch[0]['sources'].shape[0]
    has_noise = 'noise' in batch[0]
    has_speaker_ids = 'speaker_ids' in batch[0]
    has_speaker_genders = 'speaker_genders' in batch[0]
    has_metrics = 'metrics' in batch[0]

    # find max length in batch (single pass)
    lengths_list = [sample['mixture'].shape[0] for sample in batch]
    max_length = max(lengths_list)

    # pre-allocate tensors with final size
    mixtures = torch.zeros(batch_size, max_length, dtype=batch[0]['mixture'].dtype)
    sources = torch.zeros(batch_size, n_src, max_length, dtype=batch[0]['sources'].dtype)
    lengths = torch.tensor(lengths_list, dtype=torch.long)

    if has_noise:
        noises = torch.zeros(batch_size, max_length, dtype=batch[0]['noise'].dtype)

    # preallocate lists
    mixture_ids = [None] * batch_size
    speaker_ids_batch = [None] * batch_size if has_speaker_ids else None
    speaker_genders_batch = [None] * batch_size if has_speaker_genders else None
    metrics_batch = [None] * batch_size if has_metrics else None

    # fill tensors in single loop with optimized indexing
    for i, sample in enumerate(batch):
        length = lengths_list[i]

        # use narrow() to avoid creating intermediate tensors
        mixtures[i, :length] = sample['mixture']
        sources[i, :, :length] = sample['sources']
        mixture_ids[i] = sample['mixture_id']

        if has_noise:
            noises[i, :length] = sample['noise']

        if has_speaker_ids:
            speaker_ids_batch[i] = sample['speaker_ids']
        if has_speaker_genders:
            speaker_genders_batch[i] = sample['speaker_genders']
        if has_metrics:
            metrics_batch[i] = sample.get('metrics')

    # build output dict with conditional fields
    result = {
        'mixture': mixtures,
        'sources': sources,
        'lengths': lengths,
        'mixture_ids': mixture_ids,
    }

    if has_noise:
        result['noise'] = noises
    if has_speaker_ids:
        result['speaker_ids'] = speaker_ids_batch
    if has_speaker_genders:
        result['speaker_genders'] = speaker_genders_batch
    if has_metrics:
        result['metrics'] = metrics_batch

    return result
