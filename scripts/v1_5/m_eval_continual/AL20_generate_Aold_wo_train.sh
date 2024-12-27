#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface

# Set the model for the current task
MODEL="llava-v1.5-7b-wo-train"

# Evaluate the model
echo "Generate A-old from $MODEL"

python -m llava.eval.model_vqa_generate_Aold \
    --model-path "continual-ckpt/distill/$MODEL" \
    --image-folder ./playground/data \
    --dataset-file ./playground/data/domain-incremental/others.json \
    --output-file /playground/data/domain-incremental/others-Aold.json \
    --conv-mode llava_v1

