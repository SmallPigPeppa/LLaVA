#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface
export CUDA_VISIBLE_DEVICES=1
# Set the model for the current task
MODEL="llava-v1.5-7b-wo-train"

# Evaluate the model
echo "Generate A-old from $MODEL"

python -m llava.eval.model_vqa_generate_Aold_text_debug_random \
    --model-path "continual-ckpt/distill/$MODEL" \
    --image-folder ./playground/data \
    --dataset-file playground/data/exp1/part2-others-debug.json \
    --output-file playground/data/exp1/part2-others-debug-Aold.json \
    --conv-mode llava_v1

