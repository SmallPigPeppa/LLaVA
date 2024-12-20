#!/bin/bash
export HF_HOME=/ppio_net0/huggingface

# Define task suffix
TASK="task-task1-merged"

# Create the reviews directory if it doesn't exist
mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews


# Set the model for the current task
MODEL="llava-v1.5-7b-lora-$TASK"

# Evaluate the model
echo "Generate A-old from $MODEL"
python -m llava.eval.model_vqa_generate_Aold \
    --model-path "continual-ckpt/data-incremental/$MODEL" \
    --image-folder ./playground/data \
    --dataset-file ./playground/data/debug/llava_v1_5_mix665k-random-8.json \
    --output-file ./playground/data/debug/llava_v1_5_mix665k-random-8-Aold.json \
    --conv-mode llava_v1

