import os
import pandas as pd
from glob import glob

# Directory containing the parquet parts
PARQUET_DIR = '/mnt/bn/liuwenzhuo-hl-data/datasets/parquet/llava665k'
PATTERN     = os.path.join(PARQUET_DIR, 'llava665k_part*.parquet')

def read_and_print_first_row(path):
    try:
        df = pd.read_parquet(path, engine='pyarrow')
        print(f"=== File: {path} ===")
        print("Columns:", df.columns.tolist())
        print("First record:\n", df.iloc[0])
    except Exception as e:
        print(f"Failed to read {path}: {e}")

def main():
    files = sorted(glob(PATTERN))
    if not files:
        print(f"No parquet files found matching {PATTERN}")
        return

    for path in files:
        read_and_print_first_row(path)

if __name__ == '__main__':
    main()
