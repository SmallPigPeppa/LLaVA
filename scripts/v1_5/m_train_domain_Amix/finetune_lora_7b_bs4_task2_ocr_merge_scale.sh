#!/bin/bash

# Set Hugging Face cache directory
export HF_HOME=/ppio_net0/huggingface
# Limit script to only use GPU 0
#export CUDA_VISIBLE_DEVICES=0
# Manually specify model and vision configuration
MODEL_BASE="continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
MODEL_PATH="continual-ckpt/domain/llava-v1.5-7b-lora-task-ocr_vqa"
Ratio=1.0
OUT_PATH=${MODEL_PATH}-merged-ratio{Ratio}

echo "OCR task training completed successfully!"

python -m llava.eval.model_vqa_save_weight_hf \
  --model-path ${MODEL_PATH} \
  --model-base ${MODEL_BASE} \
  --save-path ${OUT_PATH} \
  --lora-scale-factor ${Ratio}
