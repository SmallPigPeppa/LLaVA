import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configuration
JSON_PATH = '/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava665k/llava_v1_5_mix665k_filter.json'
IMAGE_DIR = '/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava665k'
OUTPUT_DIR = '/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava665k/parquet_output'
COMPRESSION = 'snappy'
CHUNK_SIZE = 200000  # approx rows per file
MAX_WORKERS = 128


def process_record(item):
    """Convert a JSON entry to a dict with optional image bytes."""
    image_bytes = None
    img_path = item.get('image')
    if img_path:
        full_path = os.path.join(IMAGE_DIR, img_path)
        try:
            with open(full_path, 'rb') as f:
                image_bytes = f.read()
        except Exception:
            image_bytes = None
    return {
        'images': [image_bytes] if image_bytes else None,
        'conversations': item.get('conversations', [])
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    batch, part = [], 1

    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        for record in tqdm(pool.map(process_record, entries), total=len(entries), desc='Processing'):
            batch.append(record)
            if len(batch) >= CHUNK_SIZE:
                out_file = os.path.join(OUTPUT_DIR, f'part_{part:04d}.parquet')
                pd.DataFrame(batch).to_parquet(out_file, engine='pyarrow', compression=COMPRESSION, index=False)
                tqdm.write(f"Wrote {len(batch)} rows to {out_file}")
                part += 1
                batch.clear()

    if batch:
        out_file = os.path.join(OUTPUT_DIR, f'part_{part:04d}.parquet')
        pd.DataFrame(batch).to_parquet(out_file, engine='pyarrow', compression=COMPRESSION, index=False)
        tqdm.write(f"Wrote final {len(batch)} rows to {out_file}")


if __name__ == '__main__':
    main()
