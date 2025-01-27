#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH_A="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-iconqa-lambda1.0-merged-mix0.6-svdv3"
MODEL_PATH_B="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-super-lambda1.0-merged-mix0.6-svdv3"


MIX_RATIO=0.5
SAVE_PATH="finetune-ckpt/llava-c/llava-v1.5-7b-lora-task1-iconqa-task2-super-merged"


# Evaluate the model (1st command - saving weights)
python -m llava.eval.model_vqa_save_weight_hf_mixv3 \
  --model-path-a ${MODEL_PATH_A} \
  --model-path-b ${MODEL_PATH_B} \
  --mix-ratio ${MIX_RATIO} \
  --save-path ${SAVE_PATH}
