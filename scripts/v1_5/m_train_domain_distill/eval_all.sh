#!/bin/bash

export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="finetune-ckpt/continual-ft/llava-v1.5-7b-lora-task6-scienceqa-merged"
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
        --conv-mode vicuna_v1 &

    wait # Ensure Step 1 is completed before moving to Step 2

    # Step 2: Evaluate results
    python -m ${eval_script} \
        --annotation-file "${annotation_file}" \
        --result-file "${answers_dir}/${MODEL}.jsonl" \
        --output-dir "${eval_dir}/${MODEL}" &

    echo "Finished evaluation for ${task_name}"
}

# Task configurations
tasks=(
#    "ArxivQA playground/data/fine-tune/ArxivQA/test_2k.json playground/data/fine-tune/ArxivQA/test_2k.json playground/data/eval/arxivqa/answers playground/data/eval/arxivqa/answers llava.eval.eval_arxivqa llava.eval.model_arxivqa"
    "CLEVR-Math playground/data/fine-tune/CLEVR-Math/test_2k.json playground/data/fine-tune/CLEVR-Math/test_2k.json playground/data/eval/clevr-math/answers playground/data/eval/clevr-math/answers llava.eval.eval_clevr_math llava.eval.model_clevr_math"
    "Super-CLEVR playground/data/fine-tune/super-CLEVR/test_2k.json playground/data/fine-tune/super-CLEVR/test_2k.json playground/data/eval/super-CLEVR/answers playground/data/eval/super-CLEVR/answers llava.eval.eval_clevr_math llava.eval.model_super_clevr"
    "IconQA playground/data/fine-tune/iconqa/val-1000.json playground/data/fine-tune/iconqa/val-1000.json playground/data/eval/iconvqa/answers playground/data/eval/iconvqa/output llava.eval.eval_iconqa llava.eval.model_iconqa"
    "FigureQA playground/data/fine-tune/FigureQA/test_2k.json playground/data/fine-tune/FigureQA/test_2k.json playground/data/eval/figureqa/answers playground/data/eval/figureqa/answers llava.eval.eval_figureqa llava.eval.model_figureqa"
)

# Iterate over tasks and evaluate in parallel
for task in "${tasks[@]}"; do
    IFS=" " read -r task_name question_file annotation_file answers_dir eval_dir eval_script model_script <<< "${task}"
    evaluate_task "${task_name}" "${MODEL_PATH}" "${question_file}" "${annotation_file}" "${answers_dir}" "${eval_dir}" "${eval_script}" "${model_script}" &
done

# Wait for all tasks to complete
wait

echo "All tasks have been evaluated."
