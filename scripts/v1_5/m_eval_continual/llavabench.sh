#!/bin/bash
export HF_HOME=/ppio_net0/huggingface

# Set model base variable for the first task
MODEL_BASE="lmsys/vicuna-7b-v1.5"

# Define task suffixes in a list
TASKS=("task-task1" "task-task2" "task-task3" "task-task4" "task-task5")

# Define tasks to skip
SKIP_TASKS=("task-task1" )  # Add tasks you want to skip

# Create the reviews directory if it doesn't exist
mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

# Loop through each task in the list
for ((i = 0; i < ${#TASKS[@]}; i++)); do
    TASK="${TASKS[$i]}"

    # Check if the task should be skipped
    if [[ " ${SKIP_TASKS[@]} " =~ " $TASK " ]]; then
        echo "Skipping $TASK..."
        continue  # Skip this task and go to the next iteration
    fi

    # Set the model for the current task
    MODEL="llava-v1.5-7b-lora-$TASK"

    # If it's not the first task, set the base model to the previous task's merged model
    if [ $i -gt 0 ]; then
        PREVIOUS_TASK="${TASKS[$i-1]}"
        MODEL_BASE="continual-ckpt/data-incremental/llava-v1.5-7b-lora-${PREVIOUS_TASK}-merged"
    fi

    # Evaluate the model
    echo "Evaluating $MODEL with base model $MODEL_BASE..."
    python -m llava.eval.model_vqa \
        --model-path "continual-ckpt/data-incremental/$MODEL" \
        --model-base $MODEL_BASE \
        --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
        --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$MODEL.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    # Generate reviews for the current task
    echo "Generating reviews for $MODEL..."
    python llava/eval/eval_gpt_review_bench.py \
        --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
        --rule llava/eval/table/rule.json \
        --answer-list \
            playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
            playground/data/eval/llava-bench-in-the-wild/answers/$MODEL.jsonl \
        --output \
            playground/data/eval/llava-bench-in-the-wild/reviews/$MODEL.jsonl

    # Summarize reviews for the current task
    echo "Summarizing reviews for $MODEL..."
    python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$MODEL.jsonl

    echo "============================================"
done

echo "All tasks have been evaluated and reviews summarized."
