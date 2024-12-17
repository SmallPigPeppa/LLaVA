#!/bin/bash
export HF_HOME=/ppio_net0/huggingface

# Set model as a variable
MODEL_PATH="checkpoints/llava-v1.5-7b-lora"
MODEL_BASE="checkpoints/llava-v1.5-7b-lora-x"
# Evaluate the model
python -m llava.eval.model_vqa_save_weight_hf \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --save-path $MODEL_PATH-merged
