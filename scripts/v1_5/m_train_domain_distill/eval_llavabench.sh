#!/bin/bash

export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0


# Define the list of model paths
MODEL_PATHS=(
#    "finetune-ckpt/continual-ft-order2/llava-v1.5-7b-lora-task1-iconqa-merged"
#    "finetune-ckpt/continual-ft-order2/llava-v1.5-7b-lora-task2-super-merged"
#    "finetune-ckpt/continual-ft-order2/llava-v1.5-7b-lora-task3-math-merged"
#    "finetune-ckpt/continual-ft-order2/llava-v1.5-7b-lora-task4-arxiv-merged"
#    "finetune-ckpt/continual-ft-order2/llava-v1.5-7b-lora-task5-figure-merged"
#    "finetune-ckpt/continual-ft-order2/llava-v1.5-7b-lora-task6-scienceqa-merged"
    "finetune-ckpt/fine-tune/llava-v1.5-7b-lora-iconqa-lambda1.0-merged-mix0.3-svdv3"
#    "finetune-ckpt/llava-c/llava-v1.5-7b-lora-task1-iconqa-task2-super-merged"
)



# Iterate through each model path
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    source /etc/proxy/net_proxy
    MODEL=$(basename ${MODEL_PATH})  # Extract model name

    echo "Starting evaluation for model: ${MODEL}"

    # Step 1: Evaluate the model
    python -m llava.eval.model_vqa \
        --model-path "${MODEL_PATH}" \
        --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
        --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/${MODEL}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    # Step 2: Create the reviews directory
    mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

    # Step 3: Generate reviews
    python llava/eval/eval_gpt_review_bench.py \
        --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
        --rule llava/eval/table/rule.json \
        --answer-list \
            playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
            playground/data/eval/llava-bench-in-the-wild/answers/${MODEL}.jsonl \
        --output \
            playground/data/eval/llava-bench-in-the-wild/reviews/${MODEL}.jsonl

    # Step 4: Summarize reviews
    python llava/eval/summarize_gpt_review.py \
        -f playground/data/eval/llava-bench-in-the-wild/reviews/${MODEL}.jsonl

    echo "Finished evaluation for model: ${MODEL}"
    echo "---------------------------------------------"
done

echo "All models have been evaluated."
