#!/bin/bash

export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="finetune-ckpt/continual-ft/llava-v1.5-7b-lora-task6-scienceqa-merged"
MODEL_PATH="finetune-ckpt/continual-ft-order2/llava-v1.5-7b-lora-task5-figure-merged"
#MODEL_PATH="finetune-ckpt/llava-c/llava-v1.5-7b-lora-task1-iconqa-task2-super-merged"
#MODEL_PATH="finetune-ckpt/llava-c/llava-v1.5-7b-lora-task2-super-lambda1.0-merged"
#MODEL_PATH="finetune-ckpt/lwf-ft-order2-lambda0.2/llava-v1.5-7b-lora-task5-figure-merged"
#MODEL_PATH="finetune-ckpt/lwf-ft-order3-lambda0.2/llava-v1.5-7b-lora-task5-arxiv-merged"
#MODEL_PATH="finetune-ckpt/continual-ft-order3/llava-v1.5-7b-lora-task5-arxiv-merged"
#MODEL_PATH="finetune-ckpt/lwf-ft-order3-lambda0.2/llava-v1.5-7b-lora-task5-arxiv-merged"
MODEL_PATH='finetune-ckpt/llava-c/llava-v1.5-7b-lora-task4-figureqa-lambda1.0-merged-mix0.1-svdv3'

MODEL=$(basename "${MODEL_PATH}")


# Function to evaluate a task
evaluate_task() {
    local task_name=$1
    local model_path=$2
    local question_file=$3
    local annotation_file=$4
    local answers_dir=$5
    local eval_dir=$6
    local eval_script=$7
    local model_script=$8

    echo "Starting evaluation for ${task_name}"

    # Step 1: Generate answers
    python -m ${model_script} \
        --model-path "${model_path}" \
        --question-file "${question_file}" \
        --image-folder ./playground/data \
        --answers-file "${answers_dir}/${MODEL}.jsonl" \
        --temperature 0 \
        --conv-mode vicuna_v1

    # Step 2: Evaluate results
    python -m ${eval_script} \
        --annotation-file "${annotation_file}" \
        --result-file "${answers_dir}/${MODEL}.jsonl" \
        --output-dir "${eval_dir}/${MODEL}"

    echo "Finished evaluation for ${task_name}"
    echo "-------------------------------------"
}

# Task configurations
tasks=(
    "IconQA playground/data/fine-tune/iconqa/val-1000.json playground/data/fine-tune/iconqa/val-1000.json playground/data/eval/iconvqa/answers playground/data/eval/iconvqa/output llava.eval.eval_iconqa llava.eval.model_iconqa"
    "Super-CLEVR playground/data/fine-tune/super-CLEVR/test_2k.json playground/data/fine-tune/super-CLEVR/test_2k.json playground/data/eval/super-CLEVR/answers playground/data/eval/super-CLEVR/answers llava.eval.eval_clevr_math llava.eval.model_super_clevr"
    "CLEVR-Math playground/data/fine-tune/CLEVR-Math/test_2k.json playground/data/fine-tune/CLEVR-Math/test_2k.json playground/data/eval/clevr-math/answers playground/data/eval/clevr-math/answers llava.eval.eval_clevr_math llava.eval.model_clevr_math"
    "FigureQA playground/data/fine-tune/FigureQA/test_2k.json playground/data/fine-tune/FigureQA/test_2k.json playground/data/eval/figureqa/answers playground/data/eval/figureqa/answers llava.eval.eval_figureqa llava.eval.model_figureqa"
#    "ArxivQA playground/data/fine-tune/ArxivQA/test_2k.json playground/data/fine-tune/ArxivQA/test_2k.json playground/data/eval/arxivqa/answers playground/data/eval/arxivqa/answers llava.eval.eval_arxivqa llava.eval.model_arxivqa"
)

# Iterate over tasks and evaluate sequentially
for task in "${tasks[@]}"; do
    IFS=" " read -r task_name question_file annotation_file answers_dir eval_dir eval_script model_script <<< "${task}"
    evaluate_task "${task_name}" "${MODEL_PATH}" "${question_file}" "${annotation_file}" "${answers_dir}" "${eval_dir}" "${eval_script}" "${model_script}"
done

echo "All tasks have been evaluated."

