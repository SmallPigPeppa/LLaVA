#!/bin/bash

# 配置
SPLIT="mmbench_dev_cn_20231003"
export HF_HOME="/ppio_net0/huggingface"

# 定义模型路径列表
MODELS=(
#    "checkpoints/llava-v1.5-7b-lora-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-others-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-ocr_vqa-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-textvqa-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-gqa-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-vg-merged"
    "continual-ckpt/llava-c/llava-v1.5c-7b-lora-task-vg-merged"
)

# 创建目录
UPLOAD_DIR="playground/data/eval/mmbench_cn/answers_upload/$SPLIT"
mkdir -p "$UPLOAD_DIR"

# 循环执行每个模型
for MODEL in "${MODELS[@]}"; do
    echo "Processing model: $MODEL"

    # 执行模型评估
    python -m llava.eval.model_vqa_mmbench \
        --model-path "$MODEL" \
        --question-file "./playground/data/eval/mmbench_cn/$SPLIT.tsv" \
        --answers-file "./playground/data/eval/mmbench_cn/answers/$SPLIT/$(basename $MODEL).jsonl" \
        --lang cn \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1

    # 转换结果以供上传
    python scripts/convert_mmbench_for_submission.py \
        --annotation-file "./playground/data/eval/mmbench_cn/$SPLIT.tsv" \
        --result-dir "./playground/data/eval/mmbench_cn/answers/$SPLIT" \
        --upload-dir "$UPLOAD_DIR" \
        --experiment "$(basename $MODEL)"

    echo "Finished processing model: $MODEL"
    echo "-----------------------------------"
done

echo "All models processed."
