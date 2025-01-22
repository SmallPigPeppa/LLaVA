#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
#export HF_HOME=/mnt/disk3/wzliu/huggingface
export CUDA_VISIBLE_DEVICES=0


MODEL_BASE="liuhaotian/llava-v1.5-7b"

# First, evaluate the model without lambda
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-iconqa"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-scienceqa"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-scienceqa-lambda1.0-1w"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-scienceqa-lambda1.0"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-scienceqa-mm"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-scienceqa-lambda1.0-1w-2"

OUT_PATH="${MODEL_PATH}-merged"

# Evaluate the model (1st command - saving weights)
python -m llava.eval.model_vqa_save_weight_hf \
  --model-path ${MODEL_PATH} \
  --model-base ${MODEL_BASE} \
  --save-path ${OUT_PATH}

