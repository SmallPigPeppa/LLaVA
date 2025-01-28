import json
import random

# 读取原始数据文件
input_file = '/ppio_net0/code/LLaVA/playground/data/fine-tune/CLEVR-Math/train_4w.json'
output_file = '/ppio_net0/code/LLaVA/playground/data/fine-tune/CLEVR-Math/train_2w.json'

# 加载原始数据
with open(input_file, 'r') as f:
    data = json.load(f)

# 随机选择1000个样本
k=20000
sampled_data = random.sample(data, k)

# 保存为新的 JSON 文件
with open(output_file, 'w') as f:
    json.dump(sampled_data, f, indent=4)

print(f"Successfully selected {k} samples and saved to {output_file}.")
