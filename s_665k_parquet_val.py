import os
import pandas as pd

file_debug = '/mnt/bn/liuwenzhuo-hl-data/datasets/parquet/llava665k/debug.parquet'

def read_and_print_one(path):
    df = pd.read_parquet(path, engine='pyarrow')
    print(f"=== File: {path} ===")
    print("Columns:", df.columns.tolist())
    print("First record:\n", df.iloc[100])

if __name__ == '__main__':
    read_and_print_one(file_debug)
