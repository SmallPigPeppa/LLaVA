#!/bin/bash
export HF_HOME=/ppio_net0/huggingface


# Define task suffixes in a list
TASK="task-coco-merged"




# Create the reviews directory if it doesn't exist
mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews



# Set the model for the current task
MODEL="llava-v1.5-7b-lora-$TASK"

# Evaluate the model
echo "Generate A-old from $MODEL"
#python -m llava.eval.model_vqa \
#    --model-path "continual-ckpt/domain/$MODEL" \
#    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
#    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$MODEL.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1

python -m llava.eval.model_vqa_generate_Aold \
    --model-path "continual-ckpt/domain/$MODEL" \
    --image-folder ./playground/data \
    --dataset-file ./playground/data/debug/llava_v1_5_mix665k-random-8.json \
    --output-file ./playground/data/debug/llava_v1_5_mix665k-random-8-Aold.json \
    --conv-mode llava_v1

