#!/bin/bash

# Set Hugging Face cache directory
export HF_HOME=/ppio_net0/huggingface
export HF_HOME=/mnt/disk3/wzliu/huggingface
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=0
# Manually specify model and vision configuration
MODEL_PATH="lmsys/vicuna-7b-v1.5"
VISION_TOWER="openai/clip-vit-large-patch14-336"
DATA_PATH="playground/data/domain-incremental-mse/coco-with-othersv4.json"
OUTPUT_DIR="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-lambda0"
PRETRAIN_ADAPTER="./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin"



# Training command for OCR task
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --distill True \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ./playground/data \
    --vision_tower ${VISION_TOWER} \
    --pretrain_mm_mlp_adapter ${PRETRAIN_ADAPTER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --max_steps -1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
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

/ppio_net0/code/openapi.sh stop 1ad06ee0e8bb7b15

