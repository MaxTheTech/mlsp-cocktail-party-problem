"""
Simple script to make paths relative in LibriMix metadata
"""
import pandas as pd
from pathlib import Path

LIBRIMIX_ROOT = 'data/Libri2Mix'
metadata_dir = Path(LIBRIMIX_ROOT) / 'wav16k/min/metadata'

# Get all mixture CSV files
csv_files = list(metadata_dir.glob('mixture_*.csv'))

for csv_file in csv_files:
    print(f"Processing {csv_file.name}...")
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Remove /mnt/data/ from all path columns
    path_columns = ['mixture_path', 'source_1_path', 'source_2_path', 'noise_path']
    
    for col in path_columns:
        if col in df.columns:
            df[col] = df[col].str.replace('/mnt/data/', '')
    
    # Save
    df.to_csv(csv_file, index=False)
    print(f"  ✅ Fixed!")

print("\nDone! All paths updated.")
