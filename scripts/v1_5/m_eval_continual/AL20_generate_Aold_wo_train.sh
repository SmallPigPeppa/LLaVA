#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface

# Set the model for the current task
MODEL="llava-v1.5-7b-lora-wo-train-merged"

# Evaluate the model
echo "Generate A-old from $MODEL"

python -m llava.eval.model_vqa_generate_Aold \
    --model-path "continual-ckpt/distill/$MODEL" \
    --image-folder ./playground/data \
    --dataset-file ./playground/data/debug/llava_v1_5_mix665k-random-8.json \
    --output-file ./playground/data/debug/llava_v1_5_mix665k-random-8-Aold.json \
    --conv-mode llava_v1

