#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"
#export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
#export CUDA_VISIBLE_DEVICES=4
MODEL=checkpoints/llava-v1.5-7b-lora-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-others-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-ocr_vqa-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-textvqa-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-gqa-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-vg-merged
python -m llava.eval.model_vqa_mmbench \
    --model-path $MODEL \
    --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench_cn/answers/$SPLIT/llava-v1.5-13b.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment llava-v1.5-13b
