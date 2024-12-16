#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
#export OPENAI_API_KEY="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"
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
