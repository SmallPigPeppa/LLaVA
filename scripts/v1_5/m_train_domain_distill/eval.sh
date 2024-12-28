#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

MODEL_BASE="continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"

# First, evaluate the model without lambda
MODEL_PATH="continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-distill-lambda1.0-nodatamix"
OUT_PATH="${MODEL_PATH}-mix-merged"
MODEL="${OUT_PATH##*/}"

# Evaluate the model (1st command - saving weights)
python -m llava.eval.model_vqa_save_weight_hf \
  --model-path ${MODEL_PATH} \
  --model-base ${MODEL_BASE} \
  --save-path ${OUT_PATH}

# Evaluate the model (2nd command - actual evaluation)
python -m llava.eval.model_vqa \
    --model-path "continual-ckpt/distill/$MODEL" \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
