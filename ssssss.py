import os
import json

# 输入和输出文件路径
input_file = '/ppio_net0/code/LLaVA/playground/data/eval/seed_bench/llava-seed-bench.jsonl'
output_file = '/ppio_net0/code/LLaVA/playground/data/eval/seed_bench/llava-seed-bench-modified.jsonl'

# 路径修改规则
def modify_image_path(image_path, question_id):
    # 提取任务号
    task_id = image_path.split('/')[1].split('_')[0]
    # 根据任务号确定新路径
    if task_id == "10":
        new_path = f"SEED-Bench-video-image/task10/ssv2_8_frame/{question_id}/4.png"
    elif task_id == "11":
        new_path = f"SEED-Bench-video-image/task11/kitchen_8_frame/{question_id}/4.png"
    elif task_id == "12":
        new_path = f"SEED-Bench-video-image/task12/breakfast_8_frame/{question_id}/4.png"
    else:
        new_path = image_path  # 如果任务号不匹配，保留原路径
    return new_path

# 修改JSONL文件
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        entry = json.loads(line.strip())  # 读取每行数据为字典
        if 'v' in entry['question_id']:  # 检查 question_id 是否包含 'v'
            entry['image'] = modify_image_path(entry['image'], entry['question_id'])
        json.dump(entry, outfile)  # 写入修改后的数据
        outfile.write('\n')  # 添加换行符
