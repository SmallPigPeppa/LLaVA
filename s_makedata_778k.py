import os
import json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# 数据集路径和输出目录
dataset_path = "/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava779k"
image_folder = "/mnt/bn/liuwenzhuo-hl-data/datasets/llava779k/images"
output_json = "/mnt/bn/liuwenzhuo-hl-data/datasets/llava779k/llava_v1_6_mix779k.json"

# 确保输出目录存在
os.makedirs(image_folder, exist_ok=True)

# 加载数据集
data = load_dataset(dataset_path, split="train")

converted_data = []

for da in tqdm(data, desc="Converting images"):
    json_data = {"id": da["id"], "conversations": da["conversations"]}

    img = da.get("image", None)
    if img is not None:
        # 若是 RGBA，则转换为 RGB
        if img.mode == "RGBA":
            img = img.convert("RGB")
        # 构造文件名并保存为 JPEG
        filename = f"{da['id']}.jpg"
        img.save(os.path.join(image_folder, filename))
        json_data["image"] = filename

    converted_data.append(json_data)

# 将转换后的元数据写入 JSON 文件
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
