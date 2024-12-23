#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0
# Set model as a variable
Ratio=0.2
MODEL="llava-v1.5-7b-lora-task-ocr_vqa-merged-ratio${Ratio}"
# Evaluate the model
python -m llava.eval.model_vqa \
    --model-path "continual-ckpt/domain-mix/$MODEL" \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

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


