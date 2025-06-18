import os
from datasets import load_dataset
from tqdm import tqdm
import json

# # 之前下载在/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava779k/
# data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train")
data = load_dataset(
    "/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava779k",
    split="train"
)

image_folder = "/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava779k/images"

converted_data = []

for da in tqdm(data):
    json_data = {}
    json_data["id"] = da["id"]
    if da["image"] is not None:
        json_data["image"] = f"{da['id']}.jpg"
        da["image"].save(os.path.join(image_folder, json_data["image"]))
    json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)


with open("llava_v1_6_mix779k.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
