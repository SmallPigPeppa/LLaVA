#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH_A1="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-gqa-svdv3-lambda1.0-merged-mix0.1-svdv3"
MODEL_PATH_B1="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-vg-svdv3-lambda1.0-merged"

# 定义模型组合
MODELS=(
    "$MODEL_PATH_A1 $MODEL_PATH_B1"
)

# 定义比例列表
MIX_RATIOS=(
0.01
0.02
0.03
0.04
0.05
)
SVD_RATIO=0.2
SCALE_RATIO=0.5

# 遍历每组模型
for MODEL_PAIR in "${MODELS[@]}"; do
    # 解析 MODEL_PATH_A 和 MODEL_PATH_B
    MODEL_PATH_A=$(echo $MODEL_PAIR | awk '{print $1}')
    MODEL_PATH_B=$(echo $MODEL_PAIR | awk '{print $2}')

    echo "Processing Model Pair: A=${MODEL_PATH_A}, B=${MODEL_PATH_B}"

    # 遍历 MIX_RATIO 列表
    for MIX_RATIO in "${MIX_RATIOS[@]}"; do
        SAVE_PATH="${MODEL_PATH_B}-mix${MIX_RATIO}-svdv3"
        MODEL_NAME=$(basename $SAVE_PATH)

        echo "Evaluating mix ratio: ${MIX_RATIO} with SVD V3"

        # 调用模型保存和权重融合脚本
        python -m llava.eval.model_vqa_save_weight_hf_mixv3_svdv3 \
            --model-path-a ${MODEL_PATH_A} \
            --model-path-b ${MODEL_PATH_B} \
            --mix-ratio ${MIX_RATIO} \
            --save-path ${SAVE_PATH} \
            --retain-ratio ${SVD_RATIO} \
            --scale-ratio ${SCALE_RATIO}

        echo "Evaluation for mix ratio ${MIX_RATIO} completed."
    done

    echo "Completed processing Model Pair: A=${MODEL_PATH_A}, B=${MODEL_PATH_B}"
done

echo "All evaluations completed."
