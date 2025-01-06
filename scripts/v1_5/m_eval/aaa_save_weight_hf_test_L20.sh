#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export HF_HOME=/mnt/disk3/wzliu/huggingface

# Set model as a variable
MODEL_PATH="checkpoints/llava-v1.5-7b-lora"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
# Evaluate the model
python -m llava.eval.model_vqa_save_weight_hf \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --save-path $MODEL_PATH-merged
