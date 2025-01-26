#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

# List of model paths to evaluate
MODEL_PATHS=(
#    "liuhaotian/llava-v1.5-7b"
    "finetune-ckpt/fine-tune/llava-v1.5-7b-lora-figureqa-merged"
    "finetune-ckpt/fine-tune/llava-v1.5-7b-lora-figureqa-lambda1.0-merged"
)

# Iterate through each model path
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    OUT_PATH="${MODEL_PATH}"
    MODEL=$(basename $OUT_PATH)  # Extract the model name

    echo "Evaluating model: ${MODEL_PATH}"

    # Step 1: Run model evaluation for FigureQA
    python -m llava.eval.model_figureqa \
        --model-path ${MODEL_PATH} \
        --question-file ./playground/data/fine-tune/FigureQA/test_2k.json \
        --image-folder ./playground/data \
        --answers-file ./playground/data/eval/figureqa/answers/${MODEL}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    # Step 2: Evaluate results
    python -m llava.eval.eval_figureqa \
        --annotation-file ./playground/data/fine-tune/FigureQA/test_2k.json \
        --result-file ./playground/data/eval/figureqa/answers/${MODEL}.jsonl \
        --output-dir ./playground/data/eval/figureqa/answers/${MODEL}
done
