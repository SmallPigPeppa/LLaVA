#!/bin/bash

# Set Hugging Face cache directory
export HF_HOME=/ppio_net0/huggingface

# Base model and vision configuration
MODEL_BASE="lmsys/vicuna-7b-v1.5"
VISION_TOWER="openai/clip-vit-large-patch14-336"
PRETRAIN_ADAPTER="./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin"

# Define task names
TASK_NAMES=("others" "coco" "ocr_vqa" "textvqa" "gqa" "vg")

DATA_ROOT="./playground/data/domain-incremental"
OUTPUT_DIR_ROOT="./continual-ckpt/domain-1task"
LOG_DIR="./scripts/v1_5/m_train_domain_1task"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Loop through each task
for TASK_NAME in "${TASK_NAMES[@]}"; do
    # Define the data path and output directory for the current task
    DATA_PATH="${DATA_ROOT}/llava_v1_5_mix665k-${TASK_NAME}.json"  # Path to JSON file
    OUTPUT_DIR="${OUTPUT_DIR_ROOT}/llava-v1.5-7b-lora-task-${TASK_NAME}"  # Task output directory
    LOG_FILE="${LOG_DIR}/training_time_${TASK_NAME}.txt"  # Log file for the current task

    # Get start time
    START_TIME=$(date +%s)

    # Build the training command
    deepspeed llava/train/train_mem.py \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --deepspeed ./scripts/zero3.json \
        --model_name_or_path ${MODEL_BASE} \
        --pretrain_mm_mlp_adapter ${PRETRAIN_ADAPTER} \
        --version v1 \
        --data_path ${DATA_PATH} \
        --image_folder ./playground/data \
        --vision_tower ${VISION_TOWER} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ${OUTPUT_DIR} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb

    # Get end time
    END_TIME=$(date +%s)

    # Calculate the duration (in seconds)
    TRAINING_TIME=$((END_TIME - START_TIME))

    # Log the training time into the file
    echo "Task: ${TASK_NAME}, Training Time: ${TRAINING_TIME} seconds" >> ${LOG_FILE}

    # Optionally print the training time to the console
    echo "Task: ${TASK_NAME}, Training Time: ${TRAINING_TIME} seconds"
done
