#!/bin/bash
#export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
#export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="continual-ckpt/llava-c/llava-v1.5c-7b-lora-task-gqa-merged"
MODEL_PATH="liuhaotian/llava-v1.5-7b"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-scienceqa-merged"
MODEL=$(basename $OUT_PATH)

python -m llava.eval.model_vqa_science \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${MODEL}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${MODEL}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${MODEL}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${MODEL}_result.json
