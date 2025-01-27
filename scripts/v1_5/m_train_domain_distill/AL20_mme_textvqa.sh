#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0
MODEL="checkpoints/llava-v1.5-7b-lora-merged"
MODEL="continual-ckpt/domain/llava-v1.5-7b-lora-task-others-merged"
MODEL="continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
MODEL="continual-ckpt/domain/llava-v1.5-7b-lora-task-ocr_vqa-merged"
MODEL="continual-ckpt/domain/llava-v1.5-7b-lora-task-textvqa-merged"
#MODEL="continual-ckpt/domain/llava-v1.5-7b-lora-task-gqa-merged"
#MODEL="continual-ckpt/domain/llava-v1.5-7b-lora-task-vg-merged"

MODEL_NAME=$(basename $MODEL)

rm -rf ./playground/data/eval/MME/eval_tool
unzip ./playground/data/eval/MME/eval_tool.zip -d ./playground/data/eval/MME

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL \
    --question-file ./playground/data/eval/MME/llava_mme_new.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme_new.py --experiment ${MODEL_NAME}

cd eval_tool

python calculation.py --results_dir answers/${MODEL_NAME}





