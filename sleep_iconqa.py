import json
import random

# 读取原始数据文件
input_file = '/ppio_net0/code/LLaVA/playground/data/fine-tune/iconqa/val.json'
output_file = '/ppio_net0/code/LLaVA/playground/data/fine-tune/iconqa/val-2000.json'

# 加载原始数据
with open(input_file, 'r') as f:
    data = json.load(f)

# 随机选择1000个样本
sampled_data = random.sample(data, 2000)

# 保存为新的 JSON 文件
with open(output_file, 'w') as f:
    json.dump(sampled_data, f, indent=4)

print(f"Successfully selected 1000 samples and saved to {output_file}.")
