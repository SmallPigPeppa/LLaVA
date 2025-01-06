#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0
SPLIT="mmbench_dev_20230712"

# List of model paths
MODELS=(
    "continual-ckpt/domain/llava-v1.5-7b-lora-task-others-merged"
    "continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-textvqa-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-gqa-merged"
    "continual-ckpt/domain/llava-v1.5-7b-lora-task-vg-merged"
)

# Iterate over models and run evaluation
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"

    # Run evaluation for the current model
    python -m llava.eval.model_vqa_mmbench \
        --model-path $MODEL \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$(basename $MODEL).jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1

    # Create directory for results
    mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

    # Convert results for submission
    python scripts/convert_mmbench_for_submission.py \
        --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
        --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
        --experiment $(basename $MODEL)
done
