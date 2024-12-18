#!/bin/bash
export HF_HOME=/ppio_net0/huggingface


# Define task suffixes in a list
TASKS=("task-task1-merged" "task-task2-merged" "task-task3-merged" "task-task4-merged" "task-task5-merged")
TASKS=("task-task5-merged")


# Create the reviews directory if it doesn't exist
mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

# Loop through each task in the list
for ((i = 0; i < ${#TASKS[@]}; i++)); do
    TASK="${TASKS[$i]}"

    # Set the model for the current task
    MODEL="llava-v1.5-7b-lora-$TASK"

    # Evaluate the model
    echo "Evaluating $MODEL"
    python -m llava.eval.model_vqa \
        --model-path "continual-ckpt/data-incremental/$MODEL" \
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
