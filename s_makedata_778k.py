import os
import json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

# 数据集路径和输出目录
dataset_path = "/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava779k"
image_folder = "/mnt/bn/liuwenzhuo-hl-data/datasets/llava779k/images"
output_json = "/mnt/bn/liuwenzhuo-hl-data/datasets/llava779k/llava_v1_6_mix779k.json"

os.makedirs(image_folder, exist_ok=True)

# 加载数据集
data = load_dataset(dataset_path, split="train")

converted_data = []

for da in tqdm(data, desc="Converting images"):
    entry = {
        "id": da["id"],
        "conversations": da["conversations"],
    }

    img = da.get("image", None)
    if img is not None:
        try:
            # 如果不是 RGB，就统一 convert 到 RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            filename = f"{da['id']}.jpg"
            img.save(os.path.join(image_folder, filename), format="JPEG", quality=95)
            entry["image"] = filename

        except (OSError, UnidentifiedImageError) as e:
            # 如果依然保存失败，就打印警告并跳过
            print(f"[Warning] could not save image {da['id']} (mode={img.mode}): {e}")

    converted_data.append(entry)

# 写出最终的 JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
