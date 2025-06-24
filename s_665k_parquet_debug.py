import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configuration
JSON_PATH     = '/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava665k/llava_v1_5_mix665k_filter.json'
IMAGE_DIR     = '/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava665k'
OUTPUT_FILE   = '/mnt/bn/liuwenzhuo-hl-data/datasets/parquet/llava665k/debug.parquet'
COMPRESSION   = 'snappy'
MAX_WORKERS   = 128
DEBUG_LIMIT   = 200  # only process first 100 entries

def process_record(item):
    """Convert a JSON entry to a dict, embedding any image bytes."""
    image_bytes = None
    img_path = item.get('image')
    if img_path:
        full_path = os.path.join(IMAGE_DIR, img_path)
        try:
            with open(full_path, 'rb') as f:
                image_bytes = f.read()
        except Exception:
            pass
    return {
        'images': [image_bytes] if image_bytes else None,
        'conversations': item.get('conversations', [])
    }

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load only the first DEBUG_LIMIT entries
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        entries = json.load(f)[:DEBUG_LIMIT]

    # Parallel conversion
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        results = list(tqdm(pool.map(process_record, entries),
                            total=len(entries),
                            desc='Processing'))

    # Write a single Parquet file
    df = pd.DataFrame(results)
    df.to_parquet(OUTPUT_FILE,
                  engine='pyarrow',
                  compression=COMPRESSION,
                  index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
