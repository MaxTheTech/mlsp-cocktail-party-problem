import os
# os.environ['DATASETS_AUDIO_BACKEND'] = 'soundfile' # needed to make audio decoding work

# import datasets
# datasets.config.AUDIO_DECODE_BACKEND = 'soundfile'

from datasets import load_dataset, Dataset
from pathlib import Path
import argparse
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Download any Hugging Face dataset (full or subset).")
    
    ### REQUIRED ARGUMENTS
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the Hugging Face dataset to download (e.g. 'acul3/voxceleb2', etc).",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        required=True,
        help="Name of the output folder to save the dataset under (e.g. 'voxceleb2_subset').",
    )

    ### OPTIONAL ARGUMENTS
    parser.add_argument(
        "--num_samples",
        type=int,
        default=int(os.getenv("NUM_SAMPLES", 10_000)),
        help="Number of samples to download (default: 10,000). Use 0 to download the full dataset.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download, check on dataset Hugging Face page (default: 'train').",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # build relative paths
    REPO_ROOT = Path(__file__).parent.parent.parent  # get repo root (src/data/ --> src/ --> repo root)
    DATA_DIR = REPO_ROOT / "data" / "raw" / args.out_name # destination for downloaded data

    # setup logger
    logger = setup_logger(__name__, level=os.getenv("LOG_LEVEL", "INFO"))

    logger.info(f"Preparing to download dataset: {args.dataset} (split: {args.split})")
    logger.info(f"Output directory: {DATA_DIR}")

    # full dataset download with no streaming
    if args.num_samples == 0:
        logger.warning("Full dataset download selected.")
        logger.info("Starting full dataset download...")
        dataset = load_dataset(args.dataset, split=args.split) # full load, no streaming
        logger.info(f"Full dataset downloaded: {len(dataset)} samples")
    # stream subset and download num_samples sized subset
    else:
        logger.info(f"Streaming and downloading first {args.num_samples} samples...")
        dataset_stream = load_dataset(args.dataset, split=args.split, streaming=True)
        subset_stream = dataset_stream.take(args.num_samples)
        dataset = Dataset.from_generator(lambda: iter(subset_stream))
        logger.info(f"Downloaded {len(dataset)} samples")

    # save to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(DATA_DIR))
    logger.info(f"Dataset saved to: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()