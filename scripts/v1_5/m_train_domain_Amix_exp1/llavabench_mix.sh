#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0
# Set model as a variable
MODEL="llava-v1.5-7b-lora-task-gqa-merged-exp1-ratio1.0"

export HF_HOME=/ppio_net0/huggingface
MODEL_BASE="continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
MODEL_PATH="continual-ckpt/exp1/llava-v1.5-7b-lora-task-gqa"
OUT_PATH="continual-ckpt/exp1/llava-v1.5-7b-lora-task-gqa-merged-exp1-ratio1.0"

python -m llava.eval.model_vqa_save_weight_hf_mix \
  --model-path ${MODEL_PATH} \
  --model-base ${MODEL_BASE} \
  --save-path ${OUT_PATH} \
  --lora-scale-factor 0


# Evaluate the model
python -m llava.eval.model_vqa \
    --model-path "continual-ckpt/exp1/$MODEL" \
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


