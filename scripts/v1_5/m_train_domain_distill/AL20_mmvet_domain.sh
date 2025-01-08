#!/bin/bash

#export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
#export CUDA_VISIBLE_DEVICES=4

# 模型路径列表
MODELS=(
#    "checkpoints/llava-v1.5-7b-lora-merged"
    "continual-ckpt/domain/llava-v1.5-7b-lora-task-others-merged"
    "continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
    "continual-ckpt/domain/llava-v1.5-7b-lora-task-ocr_vqa-merged"
    "continual-ckpt/domain/llava-v1.5-7b-lora-task-textvqa-merged"
    "continual-ckpt/domain/llava-v1.5-7b-lora-task-gqa-merged"
    "continual-ckpt/domain/llava-v1.5-7b-lora-task-vg-merged"
)

# 循环执行每个模型
for MODEL in "${MODELS[@]}"; do
    echo "Running evaluation for model: $MODEL"

    # 动态生成 answers-file 名称
    MODEL_NAME=$(basename $MODEL)
    ANSWERS_FILE="./playground/data/eval/mm-vet/answers/${MODEL_NAME}.jsonl"
    RESULTS_FILE="./playground/data/eval/mm-vet/results/${MODEL_NAME}.json"

    # 运行第一个脚本
    python -m llava.eval.model_vqa \
        --model-path $MODEL \
        --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
        --image-folder ./playground/data/eval/mm-vet/images \
        --answers-file $ANSWERS_FILE \
        --temperature 0 \
        --conv-mode vicuna_v1

    # 创建结果目录
    mkdir -p ./playground/data/eval/mm-vet/results

    # 运行第二个脚本
    python scripts/convert_mmvet_for_eval.py \
        --src $ANSWERS_FILE \
        --dst $RESULTS_FILE
done
