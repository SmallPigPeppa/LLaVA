#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
#export HF_HOME=/mnt/disk3/wzliu/huggingface

MODEL_PATH_A="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda1.0-merged"
MODEL_PATH_B="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco2text-lambda1-merged"


MIX_RATIO=0.9
SVD_RATIO=0.5
SAVE_PATH="${MODEL_PATH_B}-mix${MIX_RATIO}-svd${SVD_RATIO}"
SAVE_PATH="${MODEL_PATH_B}-mix${MIX_RATIO}"
MODEL_NAME=$(basename $SAVE_PATH)

#python -m llava.eval.model_vqa_loader \
#    --model-path $SAVE_PATH \
#    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#    --image-folder ./playground/data/eval/textvqa/train_images \
#    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl
