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
MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda1.0"
#MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-lambda1.0"
MODEL_PATH="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco-textvqa"
MODEL_PATH="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco2text-lambda1-v2"
MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-ocr-oinit-lambda1.0"
MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-lambda1.0-llama"
MODEL_PATH="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco2text-lambda1"
MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-ocr-cocoinit-lambda1.0"
#MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v0-oinit-lambda1.0"
#MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-textvqa-cocoinit-lambda1.0"
OUT_PATH="${MODEL_PATH}-merged"
#OUT_PATH="${MODEL_PATH}-merged-mix0.1-svdv3"
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



