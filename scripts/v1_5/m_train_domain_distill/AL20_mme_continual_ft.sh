#!/bin/bash

export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

# Define the list of model paths
MODEL_PATHS=(
    "finetune-ckpt/continual-ft/llava-v1.5-7b-lora-task1-arxiv-merged"
    "finetune-ckpt/continual-ft/llava-v1.5-7b-lora-task2-math-merged"
    "finetune-ckpt/continual-ft/llava-v1.5-7b-lora-task3-super-merged"
    "finetune-ckpt/continual-ft/llava-v1.5-7b-lora-task4-iconqa-merged"
    "finetune-ckpt/continual-ft/llava-v1.5-7b-lora-task5-figure-merged"
    "finetune-ckpt/continual-ft/llava-v1.5-7b-lora-task6-scienceqa-merged"
)

# Prepare the MME evaluation tool
rm -rf ./playground/data/eval/MME/eval_tool
unzip ./playground/data/eval/MME/eval_tool.zip -d ./playground/data/eval/MME

# Iterate over each model path
for MODEL in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename "${MODEL}")

    echo "Starting evaluation for model: ${MODEL_NAME}"

    # Step 1: Generate answers
    python -m llava.eval.model_vqa_loader \
        --model-path "${MODEL}" \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/${MODEL_NAME}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    # Step 2: Convert answers to MME format
    cd ./playground/data/eval/MME
    python convert_answer_to_mme.py --experiment "${MODEL_NAME}"

    # Step 3: Calculate metrics
    cd eval_tool
    python calculation.py --results_dir "answers/${MODEL_NAME}"

    # Return to the original directory
    cd /ppio_net0/code/LLaVA

    echo "Finished evaluation for model: ${MODEL_NAME}"
    echo "-------------------------------------"
done

echo "All models have been evaluated."
