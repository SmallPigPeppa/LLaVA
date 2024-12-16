#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export OPENAI_API_KEY="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"
# Set model as a variable
MODEL="llava-v1.5-7b-lora"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
# Evaluate the model
python -m llava.eval.model_vqa \
    --model-path "checkpoints/$MODEL" \
    --model-base $MODEL_BASE \
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
