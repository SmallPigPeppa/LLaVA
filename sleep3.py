import json
import random

# 加载 coco 和 others 数据集
with open('playground/data/domain-incremental/llava_v1_5_mix665k-coco.json', 'r') as f:
    coco_data = json.load(f)

with open('playground/data/domain-incremental/llava_v1_5_mix665k-others.json', 'r') as f:
    others_data = json.load(f)

# 统计样本数量
coco_count = len(coco_data)
others_count = len(others_data)

print(f"COCO 数据集的样本数量: {coco_count}")
print(f"Others 数据集的样本数量: {others_count}")

# 打乱 others 数据集
random.shuffle(others_data)

# 计算需要复制多少次 others 数据集，以使其样本数量和 coco 一致
if others_count < coco_count:
    repeat_times = coco_count // others_count
    remainder = coco_count % others_count
    others_data_extended = others_data * repeat_times + others_data[:remainder]
else:
    others_data_extended = others_data[:coco_count]

# 合并 coco 数据集和扩展后的 others 数据集
merged_data = coco_data + others_data_extended

# 打乱合并后的数据集顺序
random.shuffle(merged_data)

# 保存为新的文件
with open('playground/data/domain-incremental-mse/coco-with-othersv2.json', 'w') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"合并成功，共生成 {len(merged_data)} 个样本。")
