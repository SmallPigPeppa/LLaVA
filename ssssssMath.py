import json

# 输入和输出文件路径
input_file = "playground/data/fine-tune/CLEVR-Math/train_4w-Aold.json"
output_file = "playground/data/fine-tune/CLEVR-Math/train_4w-filter.json"

# 统计被删除的项目数
deleted_count = 0

# 加载 JSON 数据
with open(input_file, "r") as infile:
    data = json.load(infile)

# 处理后的数据存储
filtered_data = []

# 遍历每个项目
for item in data:
    # 获取对话内容
    conversations = item.get("conversations", [])
    # 提取 gpt 和 old-model 的 value
    gpt_value = None
    old_model_value = None

    for convo in conversations:
        if convo["from"] == "gpt":
            gpt_value = convo["value"]
        elif convo["from"] == "old-model":
            old_model_value = convo["value"]

    # 如果 gpt 和 old-model 的 value 相等，删除整个项目
    if gpt_value == old_model_value:
        deleted_count += 1
    else:
        # 如果不相等，删除 old-model 相关内容，保留项目
        item["conversations"] = [convo for convo in conversations if convo["from"] != "old-model"]
        filtered_data.append(item)

# 将处理后的数据保存到输出文件
with open(output_file, "w") as outfile:
    json.dump(filtered_data, outfile, indent=4)

# 输出被删除的项目数
print(f"Deleted {deleted_count} items.")
