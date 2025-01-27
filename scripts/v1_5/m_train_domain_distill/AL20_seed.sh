#!/bin/bash
export HF_HOME=/ppio_net0/huggingface

#gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
#IFS=',' read -ra GPULIST <<< "$gpu_list"

gpu_list="0,1,2,3,4,5,6,7"  # Define your GPU sequence here
IFS=',' read -ra GPULIST <<< "$gpu_list"


# Number of GPUs
CHUNKS=${#GPULIST[@]}

# Display GPU usage information
echo "Using ${CHUNKS} GPU(s): ${gpu_list}"
echo "GPU IDs: ${GPULIST[*]}"

MODEL_PATH="continual-ckpt/llava-c/llava-v1.5c-7b-lora-task-vg-merged"
MODEL_PATH="continual-ckpt/domain/llava-v1.5-7b-lora-task-vg-merged"
MODEL_PATH="checkpoints/llava-v1.5-7b-lora-merged"
MODEL_PATH="continual-ckpt/domain/llava-v1.5-7b-lora-task-others-merged"
MODEL_NAME=$(basename ${MODEL_PATH})
OUT_FILE="./playground/data/eval/seed_bench/answers/${MODEL_NAME}/merge.jsonl"

#for IDX in $(seq 0 $((CHUNKS-1))); do
#    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#        --model-path ${MODEL_PATH} \
#        --question-file ./playground/data/eval/seed_bench/llava-seed-bench-modified.jsonl \
#        --image-folder ./playground/data/eval/seed_bench \
#        --answers-file ./playground/data/eval/seed_bench/answers/${MODEL_NAME}/${CHUNKS}_${IDX}.jsonl \
#        --num-chunks $CHUNKS \
#        --chunk-idx $IDX \
#        --temperature 0 \
#        --conv-mode vicuna_v1 &
#done
#
#wait



## Clear out the output file if it exists.
#> ${OUT_FILE}

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/answers/${MODEL_NAME}/${CHUNKS}_${IDX}.jsonl >> ${OUT_FILE}
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file ${OUT_FILE} \
    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/${MODEL_NAME}.jsonl

