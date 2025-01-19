#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
#export HF_HOME=/mnt/disk3/wzliu/huggingface
export CUDA_VISIBLE_DEVICES=0


MODEL_PATH="ablation-ckpt/fine-tune/llava-v1.5-7b-lora-iconqa"
OUT_PATH="${MODEL_PATH}-merged"
MODEL=$(basename $OUT_PATH)



python -m llava.eval.model_vqa \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/data/fine-tune/iconqa/val.json \
    --image-folder ./playground/data \
    --answers-file ./playground/data/eval/iconvqa/answers/${MODEL}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


python -m llava.eval.eval_iconqa \
    --annotation-file ./playground/data/fine-tune/iconqa/val.json \
    --result-file /playground/data/eval/iconvqa/result/${MODEL}.jsonl \
    --output-dir /playground/data/eval/iconvqa/output/${MODEL} \
