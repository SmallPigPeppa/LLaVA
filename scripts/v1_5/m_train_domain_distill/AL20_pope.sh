#!/bin/bash
#export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
#export CUDA_VISIBLE_DEVICES=7
#tmux set-option history-limit 1000000
# 模型路径列表
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

# 循环执行每个模型
for MODEL in "${MODELS[@]}"; do
    echo "Running evaluation for model: $MODEL"

    # 运行第一个脚本
    python -m llava.eval.model_vqa_loader \
        --model-path $MODEL \
        --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
        --image-folder ./playground/data/eval/pope/val2014 \
        --answers-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    # 运行第二个脚本
    python llava/eval/eval_pope.py \
        --annotation-dir ./playground/data/eval/pope/coco \
        --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
        --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl
done
