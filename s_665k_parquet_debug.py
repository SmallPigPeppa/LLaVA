import os
import json
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Config
JSON_PATH   = '/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava665k/llava_v1_5_mix665k_filter.json'
IMAGE_DIR   = '/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava665k'
OUTPUT_FILE = '/mnt/bn/liuwenzhuo-hl-data/datasets/parquet/llava665k/debug.parquet'
COMPRESSION = 'snappy'
MAX_WORKERS = 128
DEBUG_LIMIT = 10000
RANDOM_SEED = 42

def process_record(item):
    """
    Convert JSON entry to dict, embedding image bytes and preserving id.
    """
    img_bytes = None
    img_path = item.get('image')
    if img_path:
        try:
            with open(os.path.join(IMAGE_DIR, img_path), 'rb') as f:
                img_bytes = f.read()
        except FileNotFoundError:
            pass

    return {
        'id': str(item.get('id', '')),
        'images': [img_bytes] if img_bytes else None,
        'conversations': item.get('conversations', [])
    }

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load full JSON
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    # Shuffle with fixed seed, then take first DEBUG_LIMIT
    random.seed(RANDOM_SEED)
    random.shuffle(entries)
    entries = entries[:DEBUG_LIMIT]

    # Parallel processing
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        records = list(tqdm(pool.map(process_record, entries),
                            total=len(entries),
                            desc='Processing'))

    # Save to Parquet
    pd.DataFrame(records).to_parquet(
        OUTPUT_FILE,
        engine='pyarrow',
        compression=COMPRESSION,
        index=False
    )
    print(f"Wrote {len(records)} rows to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
