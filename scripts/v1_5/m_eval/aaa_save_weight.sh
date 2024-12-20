#!/bin/bash
export HF_HOME=/ppio_net0/huggingface

# Set model as a variable
MODEL="llava-v1.5-7b-lora"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
# Evaluate the model
python -m llava.eval.model_vqa_save_weight \
    --model-path "checkpoints/$MODEL" \
    --model-base $MODEL_BASE \
    --save-path "checkpoints/$MODEL/llava_weights.pth"
