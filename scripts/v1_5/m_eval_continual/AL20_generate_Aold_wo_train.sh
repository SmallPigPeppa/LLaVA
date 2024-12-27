#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Set the model for the current task
MODEL="llava-v1.5-7b-wo-train"

# Evaluate the model
echo "Generate A-old from $MODEL"

python -m llava.eval.model_vqa_generate_Aold_text \
    --model-path "continual-ckpt/distill/$MODEL" \
    --image-folder ./playground/data \
    --dataset-file ./playground/data/domain-incremental/llava_v1_5_mix665k-others.json \
    --output-file ./playground/data/domain-incremental/llava_v1_5_mix665k-others-Aold.json \
    --conv-mode llava_v1

