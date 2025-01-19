#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
#export HF_HOME=/mnt/disk3/wzliu/huggingface
export CUDA_VISIBLE_DEVICES=0


MODEL_BASE="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-gqa-svdv3-lambda1.0-merged-mix0.1-svdv3"

# First, evaluate the model without lambda
MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-vg-svdv3-lambda1.0"

OUT_PATH="${MODEL_PATH}-merged"

# Evaluate the model (1st command - saving weights)
python -m llava.eval.model_vqa_save_weight_hf \
  --model-path ${MODEL_PATH} \
  --model-base ${MODEL_BASE} \
  --save-path ${OUT_PATH}
