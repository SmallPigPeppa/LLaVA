#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
#export CUDA_VISIBLE_DEVICES=5
tmux set-option history-limit 1000000

# List of model paths
MODELS=(
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-others-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-ocr_vqa-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-textvqa-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-gqa-merged"
#    "continual-ckpt/domain/llava-v1.5-7b-lora-task-vg-merged"
     "continual-ckpt/llava-c/llava-v1.5c-7b-lora-task-ocr-merged"
     "continual-ckpt/llava-c/llava-v1.5c-7b-lora-task-textvqa-merged"
     "continual-ckpt/llava-c/llava-v1.5c-7b-lora-task-gqa-merged"
#     "continual-ckpt/llava-c/llava-v1.5c-7b-lora-task-vg-merged"
)

# Iterate over models and run evaluation
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"
    cd /ppio_net0/code/LLaVA
    # Run evaluation for the current model
    python -m llava.eval.model_vqa_loader \
        --model-path $MODEL \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/llava-v1.5-13b-debug.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    cd ./playground/data/eval/MME

    python convert_answer_to_mme.py --experiment llava-v1.5-13b-debug

    cd eval_tool

    python calculation.py --results_dir answers/llava-v1.5-13b-debug
done

