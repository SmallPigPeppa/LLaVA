#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=1
IDX=0

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='/mnt/haiyangguo/mywork/FCIT/CoIN/checkpoints/LLaVA/FCIT/multi_task/multask_llava_lora_ours/llava_lora_finetune_epoch_9'
else
    MODELPATH=$2
fi

if [ ! -n "$3" ] ;then
    GPU=0
else
    GPU=$3
fi

RESULT_DIR="./results/FCIT/each_dataset/IconQA"

# for IDX in $(seq 0 $((CHUNKS-1))); do
CUDA_VISIBLE_DEVICES=$GPU python -m ETrain.Eval.LLaVA.CoIN.model_iconqa \
    --model-path $MODELPATH \
    --model-base /mnt/haiyangguo/mywork/FCIT/pre_trained/llava-v1.5-7b \
    --question-file /mnt/ShareDB_6TB/datasets/FCIT_data/instructions/IconQA/val.json \
    --image-folder /mnt/ShareDB_6TB/datasets/FCIT_data/datasets \
    --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
    --cur-task 0 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &
# done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m ETrain.Eval.LLaVA.CoIN.eval_iconqa \
    --annotation-file /mnt/ShareDB_6TB/datasets/FCIT_data/instructions/IconQA/val.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \

# python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
#     --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
#     --questions ./playground/Instructions_Original/ScienceQA/test.json \
#     --results $output_file \
