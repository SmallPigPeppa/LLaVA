#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface
export CUDA_VISIBLE_DEVICES=0

MODEL_BASE="continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"

# First, evaluate the model without lambda
MODEL_PATH="continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-distill"
OUT_PATH="${MODEL_PATH}-mix-merged"
MODEL="${OUT_PATH##*/}"

# Evaluate the model (1st command - saving weights)
python -m llava.eval.model_vqa_save_weight_hf \
  --model-path ${MODEL_PATH} \
  --model-base ${MODEL_BASE} \
  --save-path ${OUT_PATH}

# Evaluate the model (2nd command - actual evaluation)
python -m llava.eval.model_vqa \
    --model-path "continual-ckpt/distill/$MODEL" \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "Finished evaluation for model without lambda"

# Now, loop through lambda values: 0.0, 1.0, 10.0
for lambda_value in 0.0 1.0 10.0; do
    # Update the MODEL_PATH and OUT_PATH based on the current lambda value
    MODEL_PATH="continual-ckpt/distill/llava-v1.5-7b-lora-task-ocr-distill-lambda${lambda_value}"
    OUT_PATH="${MODEL_PATH}-merged"
    MODEL="${OUT_PATH##*/}"

    # Evaluate the model (1st command - saving weights)
    python -m llava.eval.model_vqa_save_weight_hf \
      --model-path ${MODEL_PATH} \
      --model-base ${MODEL_BASE} \
      --save-path ${OUT_PATH}

    # Evaluate the model (2nd command - actual evaluation)
    python -m llava.eval.model_vqa \
        --model-path "continual-ckpt/distill/$MODEL" \
        --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
        --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$MODEL.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    echo "Finished evaluation for lambda=${lambda_value}"
done
