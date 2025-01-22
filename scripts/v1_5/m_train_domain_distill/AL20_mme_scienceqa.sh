#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0
MODEL="checkpoints/llava-v1.5-7b-lora-merged"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-iconqa"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-iconqa-lambda1.0"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-scienceqa-lambda1.0-1w"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-scienceqa-lambda1.0"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-scienceqa-mm"

MODEL="${MODEL_PATH}-merged"

#MODEL="liuhaotian/llava-v1.5-7b"

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






