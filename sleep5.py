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

# 计算目标 others 样本数量（按 coco 的 0.5 比例）
target_others_count = int(coco_count * 0.2)

# 打乱 others 数据集
random.shuffle(others_data)

# 根据目标数量调整 others 数据集
if others_count < target_others_count:
    # 需要重复 others 数据集
    repeat_times = target_others_count // others_count
    remainder = target_others_count % others_count
    others_data_adjusted = others_data * repeat_times + others_data[:remainder]
else:
    # 截取部分 others 数据
    others_data_adjusted = others_data[:target_others_count]

# 打印调整后的 others 数据集数量
print(f"调整后的 Others 数据集样本数量: {len(others_data_adjusted)}")

# 合并 coco 数据集和调整后的 others 数据集
merged_data = coco_data + others_data_adjusted

# 打乱合并后的数据集顺序
random.shuffle(merged_data)

# 保存为新的文件
output_file = 'playground/data/domain-incremental-mse/coco-with-othersv4.json'
with open(output_file, 'w') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"合并成功，共生成 {len(merged_data)} 个样本，保存为 {output_file}。")
