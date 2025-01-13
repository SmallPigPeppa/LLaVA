#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH_A="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda1.0-merged"
MODEL_PATH_B="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco2text-lambda1-merged"

# Directory for results
RESULTS_DIR="playground/data/eval/llava-bench-in-the-wild/answers"
REVIEWS_DIR="playground/data/eval/llava-bench-in-the-wild/reviews"

mkdir -p $RESULTS_DIR
mkdir -p $REVIEWS_DIR

# List of mix_ratios to evaluate
MIX_RATIOS=(
  0.0
  0.1
  0.2
#  0.3
#  0.4
#  0.5
#  0.6
#  0.7
#  0.8
#  0.9
#  1.0
)

# Loop through the list of mix_ratios
for MIX_RATIO in "${MIX_RATIOS[@]}"; do
    SAVE_PATH="${MODEL_PATH_B}-mix${MIX_RATIO}"
    MODEL_NAME=$(basename $SAVE_PATH)

    echo "Evaluating mix_ratio=${MIX_RATIO}"

    # Step 1: Save mixed model weights
    python -m llava.eval.model_vqa_save_weight_hf_mixv3 \
        --model-path-a ${MODEL_PATH_A} \
        --model-path-b ${MODEL_PATH_B} \
        --mix-ratio ${MIX_RATIO} \
        --save-path ${SAVE_PATH}

    # Step 2: Evaluate the model
    python -m llava.eval.model_vqa \
        --model-path ${SAVE_PATH} \
        --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
        --answers-file ${RESULTS_DIR}/${MODEL_NAME}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    # Step 3: Generate reviews
    python llava/eval/eval_gpt_review_bench.py \
        --question ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --context ./playground/data/eval/llava-bench-in-the-wild/context.jsonl \
        --rule llava/eval/table/rule.json \
        --answer-list \
            ./playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
            ${RESULTS_DIR}/${MODEL_NAME}.jsonl \
        --output ${REVIEWS_DIR}/${MODEL_NAME}.jsonl

    # Step 4: Summarize reviews
    python llava/eval/summarize_gpt_review.py \
        -f ${REVIEWS_DIR}/${MODEL_NAME}.jsonl
done

echo "Evaluation complete for all mix ratios."
