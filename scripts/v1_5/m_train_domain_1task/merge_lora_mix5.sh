#!/bin/bash

# Set Hugging Face cache directory
export HF_HOME=/ppio_net0/huggingface

# 设置模型基路径
MODEL_BASE="lmsys/vicuna-7b-v1.5"

# 设置模型路径（以数组形式存储路径）
MODEL_PATHS=(
  "continual-ckpt/domain-1task/llava-v1.5-7b-lora-task-others"
  "continual-ckpt/domain-1task/llava-v1.5-7b-lora-task-coco"
  "continual-ckpt/domain-1task/llava-v1.5-7b-lora-task-ocr_vqa"
  "continual-ckpt/domain-1task/llava-v1.5-7b-lora-task-textvqa"
  "continual-ckpt/domain-1task/llava-v1.5-7b-lora-task-gqa"
  "continual-ckpt/domain-1task/llava-v1.5-7b-lora-task-vg"
)


# 设置模型权重比例
MODEL_WEIGHTS="8.32,56.39,11.17,3.17,12.42,13.74"
#MODEL_WEIGHTS="8.32,56.39,3.17"

# 设置输出路径
OUTPUT_DIR="continual-ckpt/domain-1task-mix/all"

# 将模型路径列表转换为以逗号分隔的字符串，并传递给 Python 脚本
MODEL_PATHS_STR=$(IFS=,; echo "${MODEL_PATHS[*]}")

# 运行 Python 脚本
python -m llava.eval.model_vqa_save_weight_hf_mix_multi \
  --model-base $MODEL_BASE \
  --model-path "$MODEL_PATHS_STR" \
  --model-weights $MODEL_WEIGHTS \
  --save-path $OUTPUT_DIR

