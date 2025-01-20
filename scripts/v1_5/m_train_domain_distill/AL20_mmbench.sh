#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0
SPLIT="mmbench_dev_20230712"
MODEL=continual-ckpt/domain-Amix/llava-v1.5-7b-lora-task-ocr-merged
MODEL=continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-distill-lambda100.0-nodatamix-mse-all-mix-merged
MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-ocr_vqa-merged
MODEL=continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda1.0-merged
MODEL="continual-ckpt/llava-c/llava-v1.5c-7b-lora-task-vg-merged"
#MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-lambda1.0"
python -m llava.eval.model_vqa_mmbench \
    --model-path $MODEL \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-13b
