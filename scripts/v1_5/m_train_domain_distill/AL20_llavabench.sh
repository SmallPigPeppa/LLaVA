#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0
# Set model as a variable

#llava-v1.5-7b-lora-task-ocr-distill-mix-merged.jsonl
MODEL_PATH="continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-distill-mix"
MODEL_PATH="continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-distill-lambda1.0-datamix-mix"
MODEL_PATH="continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-no-distill-datamix-mix"
MODEL_PATH="continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-distill-lambda1.0-all-datamix-mix"
MODEL_PATH="continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-distill-lambda1.0-nodatamix-mix"
MODEL_PATH="continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-distill-lambda100.0-nodatamix-mse-mix"
MODEL_PATH="continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-distill-lambda100.0-nodatamix-mse-all-mix"
MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-lambda10"
MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda5"
OUT_PATH="${MODEL_PATH}-merged"
#MODEL="${OUT_PATH##*/}"
MODEL=$(basename $OUT_PATH)



# Create the reviews directory
mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

# Generate reviews
python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$MODEL.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$MODEL.jsonl

# Summarize reviews
python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$MODEL.jsonl



