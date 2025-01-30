#!/bin/bash

export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

# List of model paths
MODEL_PATHS=(
#    "liuhaotian/llava-v1.5-7b"
#    "finetune-ckpt/fine-tune/llava-v1.5-7b-lora-super-merged"
#    "finetune-ckpt/fine-tune/llava-v1.5-7b-lora-super-lambda1.0-merged"
    "finetune-ckpt/llava-c/llava-v1.5-7b-lora-task4-figureqa-lambda1.0-merged-mix0.13-svdv3"
)

# Iterate through each model path
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    OUT_PATH="${MODEL_PATH}"
    MODEL=$(basename "$OUT_PATH")  # Extract the model name

    echo "Evaluating model: ${MODEL_PATH}"

    # Step 1: Run model evaluation for super-CLEVR
#    python -m llava.eval.model_super_clevr \
#        --model-path "${MODEL_PATH}" \
#        --question-file ./playground/data/fine-tune/super-CLEVR/test_2k.json \
#        --image-folder ./playground/data \
#        --answers-file ./playground/data/eval/super-CLEVR/answers/${MODEL}.jsonl \
#        --temperature 0 \
#        --conv-mode vicuna_v1

    # Step 2: Evaluate results
    python -m llava.eval.eval_super_clevr \
        --annotation-file ./playground/data/fine-tune/super-CLEVR/test_2k.json \
        --result-file ./playground/data/eval/super-CLEVR/answers/${MODEL}.jsonl \
        --output-dir ./playground/data/eval/super-CLEVR/answers/${MODEL}

    echo "Finished evaluating model: ${MODEL_PATH}"
done
