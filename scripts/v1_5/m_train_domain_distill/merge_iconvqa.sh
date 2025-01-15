#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
#export HF_HOME=/mnt/disk3/wzliu/huggingface
export CUDA_VISIBLE_DEVICES=0

#MODEL_BASE="continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
#MODEL_BASE="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda1.0-merged"
#MODEL_BASE="lmsys/vicuna-7b-v1.5"
#MODEL_BASE="continual-ckpt/domain/llava-v1.5-7b-lora-task-others-merged"
MODEL_BASE="liuhaotian/llava-v1.5-7b"

# First, evaluate the model without lambda
#MODEL_PATH="continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-distill-lambda1.0-nodatamix"
#MODEL_PATH="continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-distill-lambda100.0-nodatamix-mse-all"
MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-lambda10"
MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda5"
MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda1.0"
MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-lambda1.0"
MODEL_PATH="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco-textvqa"
MODEL_PATH="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco2text-lambda1-v2"
MODEL_PATH="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco2text-lambda1"
MODEL_PATH="ablation-ckpt/fine-tune/llava-v1.5-7b-lora-iconqa"
#MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-ocr-oinit-lambda1.0"
#MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-lambda1.0-llama"

OUT_PATH="${MODEL_PATH}-merged"
#MODEL="${OUT_PATH##*/}"
MODEL_NAME=$(basename $OUT_PATH)

# Evaluate the model (1st command - saving weights)
python -m llava.eval.model_vqa_save_weight_hf \
  --model-path ${MODEL_PATH} \
  --model-base ${MODEL_BASE} \
  --save-path ${OUT_PATH}

# Evaluate the model (2nd command - actual evaluation)
python -m llava.eval.model_vqa \
    --model-path $OUT_PATH \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
