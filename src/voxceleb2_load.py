from datasets import load_dataset, Dataset
import os
from pathlib import Path

# Get repo root (parent of src directory)
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw" / "voxceleb2_subset"

print("Streaming and downloading first 10,000 samples...")

# Stream the dataset
dataset_stream = load_dataset(
    "acul3/voxceleb2",
    split="train",
    streaming=True,
)

# More memory-efficient: use from_generator instead of list
subset_stream = dataset_stream.take(10000)
dataset = Dataset.from_generator(lambda: iter(subset_stream))

print(f"Downloaded {len(dataset)} samples")

# Save to disk
DATA_DIR.mkdir(parents=True, exist_ok=True)
dataset.save_to_disk(str(DATA_DIR))
print(f"✓ Dataset saved to: {DATA_DIR}")