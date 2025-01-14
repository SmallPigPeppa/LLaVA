#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

# 定义模型路径
MODEL_PATH_A1="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda1.0-merged"
MODEL_PATH_B1="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco2text-lambda1-merged"

MODEL_PATH_A2="continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
MODEL_PATH_B2="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco-textvqa-merged"

# 固定的 SVD 保留比例
SVD_RATIO=0.5
SCALE_RATIO=0.5

# 定义模型组合
MODELS=(
    "$MODEL_PATH_A1 $MODEL_PATH_B1"
#    "$MODEL_PATH_A2 $MODEL_PATH_B2"
)

# 遍历每组模型
for MODEL_PAIR in "${MODELS[@]}"; do
    # 解析 MODEL_PATH_A 和 MODEL_PATH_B
    MODEL_PATH_A=$(echo $MODEL_PAIR | awk '{print $1}')
    MODEL_PATH_B=$(echo $MODEL_PAIR | awk '{print $2}')

    echo "Processing Model Pair: A=${MODEL_PATH_A}, B=${MODEL_PATH_B}"

    # 遍历 MIX_RATIO 从 0 到 1
    for i in $(seq 0.1 0.1 1.0); do
        MIX_RATIO=$(printf "%.1f" $i)  # 保留一位小数
        SAVE_PATH="${MODEL_PATH_B}-mix${MIX_RATIO}-svdv2${SVD_RATIO}-scale${SCALE_RATIO}"
        MODEL_NAME=$(basename $SAVE_PATH)

        echo "Evaluating mix ratio: ${MIX_RATIO} with SVD ratio: ${SVD_RATIO} SVD scale: ${SCALE_RATIO}"

        # 调用模型保存和权重融合脚本
        python -m llava.eval.model_vqa_save_weight_hf_mixv3_svdv2 \
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


/ppio_net0/code/openapi.sh stop ce8af4451be5be30
