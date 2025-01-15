#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0
MODEL=checkpoints/llava-v1.5-7b-lora-merged
MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-others-merged
#MODEL="continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-ocr_vqa-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-textvqa-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-gqa-merged
#MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-vg-merged
#MODEL=continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-lambda0-merged
#MODEL="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-ocr-oinit-lambda1.0-merged"
#MODEL="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-ocr-oinit-lambda1.0-merged-mix0.2-svdv3"

MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda1.0"
#MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-lambda1.0"

MODEL="${MODEL_PATH}-merged"

rm -rf ./playground/data/eval/MME/eval_tool
unzip ./playground/data/eval/MME/eval_tool.zip -d ./playground/data/eval/MME

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-13b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-13b




