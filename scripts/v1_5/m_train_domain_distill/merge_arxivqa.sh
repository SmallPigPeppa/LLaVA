#!/bin/bash

export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

MODEL_BASE="liuhaotian/llava-v1.5-7b"

# List of model paths to evaluate
MODEL_PATHS=(
  "finetune-ckpt/fine-tune/llava-v1.5-7b-lora-arxivqa"
  "finetune-ckpt/fine-tune/llava-v1.5-7b-lora-arxivqa-lambda1.0"
)

# Loop through each model path
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  OUT_PATH="${MODEL_PATH}-merged"

  # Evaluate the model (save weights)
  echo "Evaluating model: ${MODEL_PATH}"
  python -m llava.eval.model_vqa_save_weight_hf \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --save-path ${OUT_PATH}
done
