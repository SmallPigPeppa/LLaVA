import os
import json
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# —— 配置区 —— #
JSON_PATH   = '/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava665k/llava_v1_5_mix665k_filter.json'
IMAGE_DIR   = '/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava665k'
OUTPUT_DIR  = '/mnt/bn/liuwenzhuo-hl-data/datasets/parquet/llava665k'
COMPRESSION = 'snappy'
MAX_WORKERS = 512
CHUNK_SIZE  = 10000   # 每个 parquet 文件写入 10000 条
SHUFFLE     = True    # 是否在分块前打乱顺序
SEED        = 42
# ———————— #

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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 读取并（可选）打乱全量数据
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    if SHUFFLE:
        random.seed(SEED)
        random.shuffle(entries)

    total = len(entries)
    print(f"Total records: {total}")

    # 2. 并行处理 & 分块写入
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        for part_idx, start in enumerate(range(0, total, CHUNK_SIZE)):
            end = min(start + CHUNK_SIZE, total)
            subset = entries[start:end]

            # 并行转换
            records = list(tqdm(
                pool.map(process_record, subset),
                total=len(subset),
                desc=f'Chunk {part_idx + 1}/{(total-1)//CHUNK_SIZE + 1}'
            ))

            # 写入 parquet
            df = pd.DataFrame(records)
            out_path = os.path.join(
                OUTPUT_DIR,
                f'part{part_idx:03d}.parquet'
            )
            df.to_parquet(
                out_path,
                engine='pyarrow',
                compression=COMPRESSION,
                index=False
            )
            print(f"Wrote records {start}–{end-1} to {out_path}")

if __name__ == '__main__':
    main()
