#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH_A="finetune-ckpt/fine-tune/"
MODEL_PATH_B="finetune-ckpt/fine-tune/"


MIX_RATIO=0.5
SAVE_PATH="${MODEL_PATH_B}-mix${MIX_RATIO}"


# Evaluate the model (1st command - saving weights)
python -m llava.eval.model_vqa_save_weight_hf_mixv3 \
  --model-path-a ${MODEL_PATH_A} \
  --model-path-b ${MODEL_PATH_B} \
  --mix-ratio ${MIX_RATIO} \
  --save-path ${SAVE_PATH}
