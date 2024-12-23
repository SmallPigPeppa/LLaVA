#!/bin/bash

# Set Hugging Face cache directory
export HF_HOME=/ppio_net0/huggingface
MODEL_BASE="continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
MODEL_PATH="continual-ckpt/exp1/llava-v1.5-7b-lora-task-ocr-mix-v5"
OUT_PATH=${MODEL_PATH}-merged

python -m llava.eval.model_vqa_save_weight_hf \
  --model-path ${MODEL_PATH} \
  --model-base ${MODEL_BASE} \
  --save-path ${OUT_PATH}
