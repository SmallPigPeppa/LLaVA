#!/bin/bash

export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

# Define task names
TASKS=("Arxiv" "Math" "Super" "IconQA" "Figure" "ScienceQA")

# Define data paths for each task
DATA_PATHS=(
    "playground/data/fine-tune/ArxivQA/train_4w.json"
    "playground/data/fine-tune/CLEVR-Math/train_4w.json"
    "playground/data/fine-tune/super-CLEVR/train.json"
    "playground/data/fine-tune/iconqa/train.json"
    "playground/data/fine-tune/FigureQA/train.json"
    "playground/data/fine-tune/ScienceQA/train.json"
)

# Define initial model base
MODEL_BASE="liuhaotian/llava-v1.5-7b"

# Define output directory prefix
OUTPUT_DIR_PREFIX="finetune-ckpt/continual-ft"

# Loop through tasks
for i in "${!TASKS[@]}"; do
    TASK=${TASKS[i]}
    DATA_PATH=${DATA_PATHS[i]}
    OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/llava-v1.5-7b-lora-task$((i+1))-${TASK,,}"  # Lowercase task name
    OUTPUT_DIR_MERGED="${OUTPUT_DIR}-merged"

    echo "Training task: ${TASK}"
    echo "Data path: ${DATA_PATH}"
    echo "Model base: ${MODEL_BASE}"
    echo "Output directory: ${OUTPUT_DIR}"

    # Step 1: Train the model
    deepspeed llava/train/train_mem.py \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --distill False \
        --deepspeed ./scripts/zero3.json \
        --model_name_or_path "${MODEL_BASE}" \
        --version v1 \
        --data_path "${DATA_PATH}" \
        --image_folder ./playground/data \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir "${OUTPUT_DIR}" \
        --num_train_epochs 1 \
        --max_steps 1 \
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

    # Step 2: Merge the model weights
    echo "Merging model weights for task: ${TASK}"
    python -m llava.eval.model_vqa_save_weight_hf \
        --model-base "${MODEL_BASE}" \
        --model-path "${OUTPUT_DIR}" \
        --save-path "${OUTPUT_DIR_MERGED}"

    # Update MODEL_BASE for the next task
    MODEL_BASE="${OUTPUT_DIR_MERGED}"

    echo "Task ${TASK} completed."
    echo "-------------------------------------"
done

echo "All tasks completed."
