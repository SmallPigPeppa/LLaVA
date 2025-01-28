import json
import random

# 加载exp1/part2.json和domain-incremental/others.json
with open('playground/data/fine-tune/CLEVR-Math/train_4w-filter.json', 'r') as f:
    ocr_data = json.load(f)

with open('playground/data/domain-incremental/llava_v1_5_mix665k-others.json', 'r') as f:
    others_data = json.load(f)



# 合并两个数据集
merged_data = ocr_data + others_data

# 打乱合并后的数据集顺序
random.shuffle(merged_data)

with open('playground/data/fine-tune/CLEVR-Math/train_4w-filter-with-others.json', 'w') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"合并成功，共生成 {len(merged_data)} 个样本。")
