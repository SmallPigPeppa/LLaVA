import json

# 文件路径
input_file = './playground/data/eval/MME/llava_mme.jsonl'
output_file = './playground/data/eval/MME/llava_mme_refined.jsonl'

# 打开输入文件，逐行处理
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        if not line.strip():
            continue
        # 加载每行 JSON 数据
        data = json.loads(line.strip())

        # 修改 text 字段
        if 'text' in data:
            data['text'] += '\n Please answer Yes or No.'

        # 将修改后的数据写入输出文件
        json.dump(data, outfile)
        outfile.write('\n')  # 每条记录后换行

print(f"Refined JSONL file has been saved to: {output_file}")
