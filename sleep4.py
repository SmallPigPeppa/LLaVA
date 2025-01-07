import json
import random


def merge_datasets(coco_file, others_file, output_file, interval=3):
    # 加载coco数据和others数据
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    with open(others_file, 'r') as f:
        others_data = json.load(f)

    # 打乱coco数据和others数据
    random.shuffle(coco_data)
    random.shuffle(others_data)

    # 合并数据集
    merged_data = []
    others_index = 0  # 用来追踪`others_data`中的当前样本位置

    for i in range(0, len(coco_data), interval):  # 每`interval`个coco样本一组
        # 先添加coco样本
        merged_data.extend(coco_data[i:i + interval])

        # 如果others数据有剩余，插入一个others样本
        if others_index < len(others_data):
            merged_data.append(others_data[others_index])
            others_index += 1
        else:
            # 如果others样本用完了，重新从头开始使用
            others_index = 0
            merged_data.append(others_data[others_index])
            others_index += 1

    # 如果coco_data的长度不是interval的倍数，剩下的样本会直接添加到merged_data中
    if len(coco_data) % interval != 0:
        merged_data.extend(coco_data[-(len(coco_data) % interval):])

    # 保存合并后的数据集
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"合并成功，共生成 {len(merged_data)} 个样本。")


# 调用函数
merge_datasets(
    coco_file='playground/data/domain-incremental/llava_v1_5_mix665k-coco.json',
    others_file='playground/data/domain-incremental/llava_v1_5_mix665k-others.json',
    output_file='playground/data/domain-incremental-mse/coco-with-othersv3.json',
    interval=7  # 控制每n个coco样本插入1个others样本
)
