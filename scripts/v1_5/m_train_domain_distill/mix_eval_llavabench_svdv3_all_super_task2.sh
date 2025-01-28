#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

# 定义模型路径

MODEL_PATH_A1="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-iconqa-lambda1.0-merged-mix0.3-svdv3"
MODEL_PATH_B1="finetune-ckpt/llava-c/llava-v1.5-7b-lora-task2-super-lambda1.0-merged"

# 固定的 SVD 保留比例
SVD_RATIO=0.2
SCALE_RATIO=0.5
MIX_RATIOS=(
#  0.0
#  0.1
  0.125
#  0.2
#  0.3
#  0.4
#  0.5
#  0.6
#  0.7
#  0.8
#  0.9
#  1.0
)


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



#/ppio_net0/code/openapi.sh stop ce8af4451be5be30
