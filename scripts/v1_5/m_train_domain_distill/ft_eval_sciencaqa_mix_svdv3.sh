#!/bin/bash

# Set the Hugging Face home directory
export HF_HOME=/ppio_net0/huggingface

# Set the model base path
MODEL_PATH_B1="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-scienceqa-merged"

# List of mix ratios to iterate over
MIX_RATIOS=(
#  0.0
  0.1
  0.2
  0.3
  0.4
  0.5
  0.6
  0.7
  0.8
  0.9
#  1.0
)


# Loop through each mix ratio
for MIX_RATIO in "${MIX_RATIOS[@]}"
do
    # Modify the model path based on the mix ratio
    MODEL_PATH="${MODEL_PATH_B1}-mix${MIX_RATIO}-svdv3"
    MODEL=$(basename "${MODEL_PATH}")
    echo "Processing with MIX_RATIO: $MIX_RATIO"

    # Execute model evaluation
    python -m llava.eval.model_vqa_science \
        --model-path "${MODEL_PATH}" \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/images/test \
        --answers-file ./playground/data/eval/scienceqa/answers/${MODEL}.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1

    # Evaluate the results
    python llava/eval/eval_science_qa.py \
        --base-dir ./playground/data/eval/scienceqa \
        --result-file ./playground/data/eval/scienceqa/answers/${MODEL}.jsonl \
        --output-file ./playground/data/eval/scienceqa/answers/${MODEL}_output.jsonl \
        --output-result ./playground/data/eval/scienceqa/answers/${MODEL}_result.json
done
