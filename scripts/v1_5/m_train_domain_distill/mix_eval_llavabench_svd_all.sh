#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

# 模型路径
MODEL_PATH_A="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda1.0-merged"
MODEL_PATH_B="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco2text-lambda1-merged"

# 固定的 SVD 保留比例
SVD_RATIO=0.5

# 遍历 MIX_RATIO 从 0 到 1
for i in $(seq 0 0.1 1.0); do
    MIX_RATIO=$(printf "%.1f" $i)  # 保留一位小数
    SAVE_PATH="${MODEL_PATH_B}-mix${MIX_RATIO}-svd${SVD_RATIO}"
    MODEL_NAME=$(basename $SAVE_PATH)

    echo "Evaluating mix ratio: ${MIX_RATIO} with SVD ratio: ${SVD_RATIO}"

    # 调用模型保存和权重融合脚本
    python -m llava.eval.model_vqa_save_weight_hf_mixv3_svd \
        --model-path-a ${MODEL_PATH_A} \
        --model-path-b ${MODEL_PATH_B} \
        --mix-ratio ${MIX_RATIO} \
        --save-path ${SAVE_PATH} \
        --retain-ratio ${SVD_RATIO}

    echo "Evaluation for mix ratio ${MIX_RATIO} completed."
done

echo "All evaluations completed."
