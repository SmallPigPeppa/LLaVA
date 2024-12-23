#!/bin/bash

# Set Hugging Face cache directory
export HF_HOME=/ppio_net0/huggingface
# Limit script to only use GPU 0
#export CUDA_VISIBLE_DEVICES=0
# Manually specify model and vision configuration
MODEL_BASE="continual-ckpt/domain/llava-v1.5-7b-lora-task-ocr_vqa-merged"
MODEL_PATH="continual-ckpt/domain/llava-v1.5-7b-lora-task-textvqa"
Ratio=0.2
#OUT_PATH=${MODEL_PATH}-merged-ratio${Ratio}
OUT_PATH="continual-ckpt/domain-mix/llava-v1.5-7b-lora-task-textvqa-merged-ratio${Ratio}"

echo "OCR task training completed successfully!"

python -m llava.eval.model_vqa_save_weight_hf_mix \
  --model-path ${MODEL_PATH} \
  --model-base ${MODEL_BASE} \
  --save-path ${OUT_PATH} \
  --lora-scale-factor ${Ratio}
