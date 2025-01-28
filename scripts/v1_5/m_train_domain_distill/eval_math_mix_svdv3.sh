#!/bin/bash

export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=1

# List of model paths
MODEL_PATHS=(
#    "liuhaotian/llava-v1.5-7b"
#    "finetune-ckpt/fine-tune/llava-v1.5-7b-lora-math-merged"
#    "finetune-ckpt/fine-tune/llava-v1.5-7b-lora-math-lambda1.0-merged"
    "finetune-ckpt/llava-c/llava-v1.5-7b-lora-task3-math-lambda1.0-merged"
)

MIX_RATIOS=(
#  0.0
  0.1
#  0.125
#  0.11
#  0.12
#  0.13
  0.2
  0.3
  0.4
#  0.5
#  0.6
#  0.7
#  0.8
#  0.9
#  1.0
)


# Iterate through each model path
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  for MIX_RATIO in "${MIX_RATIOS[@]}"; do
    CURRENT_MODEL_PATH="${MODEL_PATH}-mix${MIX_RATIO}-svdv3"
    MODEL=$(basename "${MODEL_PATH}")  # Extract the model name

#    echo "Evaluating model: ${MODEL_PATH}"
    echo "Processing model: ${CURRENT_MODEL_PATH}"

    # Step 1: Run model evaluation for CLEVR-Math
    python -m llava.eval.model_clevr_math \
        --model-path "${CURRENT_MODEL_PATH}" \
        --question-file ./playground/data/fine-tune/CLEVR-Math/test_2k.json \
        --image-folder ./playground/data \
        --answers-file ./playground/data/eval/clevr-math/answers/${MODEL}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    # Step 2: Evaluate results
    python -m llava.eval.eval_clevr_math \
        --annotation-file ./playground/data/fine-tune/CLEVR-Math/test_2k.json \
        --result-file ./playground/data/eval/clevr-math/answers/${MODEL}.jsonl \
        --output-dir ./playground/data/eval/clevr-math/answers/${MODEL}

    echo "Finished evaluating model: ${CURRENT_MODEL_PATH}"
  done
done
