#!/bin/bash
#export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
#export CUDA_VISIBLE_DEVICES=7

MODEL=checkpoints/llava-v1.5-7b-lora-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-others-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-ocr_vqa-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-textvqa-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-gqa-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-vg-merged
python -m llava.eval.model_vqa_loader \
    --model-path $MODEL \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl
