import torchaudio

# List all available dataset classes
# print("Available torchaudio datasets:")
# for ds in dir(torchaudio.datasets) :
#     print(f"  - {ds}")

from torchaudio.datasets import LibriMix

# Load a manageable subset - use the dev set
dataset = LibriMix(
    root='./data',
    subset='dev',  # smaller than 'train'
    num_speakers=2,
    sample_rate=8000,  # or 16000
    task='sep_clean',  # clean separation task
    download=True
)

# Check the size
print(f"Dataset size: {len(dataset)}")

# Load one sample
mixture, sources, metadata = dataset[0]
print(f"Mixture shape: {mixture.shape}")
print(f"Sources shape: {sources.shape}")  # [num_speakers, channels, time]