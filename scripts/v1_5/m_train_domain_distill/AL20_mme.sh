#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface
#export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=6
MODEL=checkpoints/llava-v1.5-7b-lora-merged
MODEL=continual-ckpt/domain/llava-v1.5-7b-lora-task-others-merged
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

