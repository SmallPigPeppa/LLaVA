#!/bin/bash

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

python -m llava.eval.model_vqa \
    --model-path $MODEL \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-v1.5-13b.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-v1.5-13b.json

