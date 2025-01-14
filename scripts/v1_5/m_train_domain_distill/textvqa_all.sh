#!/bin/bash
export HF_HOME=/ppio_net0/huggingface

# 模型路径
MODEL_PATH_A="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda1.0-merged"
MODEL_PATH_B1="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco2text-lambda1-merged"
#MODEL_PATH_B2="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco-textvqa-merged"

# 问题与答案路径
QUESTION_FILE="./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMAGE_FOLDER="./playground/data/eval/textvqa/train_images"
ANNOTATION_FILE="./playground/data/eval/textvqa/TextVQA_0.5.1_val.json"

# 输出目录
ANSWERS_DIR="./playground/data/eval/textvqa/answers"
mkdir -p $ANSWERS_DIR

# 定义模型路径列表
MODEL_PATH_B_LIST=("$MODEL_PATH_B1" "$MODEL_PATH_B2")

# 对每个模型路径进行评估
for MODEL_PATH_B in "${MODEL_PATH_B_LIST[@]}"; do
    echo "Evaluating model path: ${MODEL_PATH_B}"

    # 对 MIX_RATIO 从 0 到 1 进行评估
    for i in $(seq 0 0.1 1.0); do
        MIX_RATIO=$(printf "%.1f" $i)  # 保留一位小数
        SAVE_PATH="${MODEL_PATH_B}-mix${MIX_RATIO}"
        MODEL_NAME=$(basename $SAVE_PATH)
        ANSWERS_FILE="${ANSWERS_DIR}/${MODEL_NAME}.jsonl"

        echo "Evaluating mix ratio: ${MIX_RATIO}"

#        # 调用模型加载和推理脚本
#        python -m llava.eval.model_vqa_loader \
#            --model-path $SAVE_PATH \
#            --question-file $QUESTION_FILE \
#            --image-folder $IMAGE_FOLDER \
#            --answers-file $ANSWERS_FILE \
#            --temperature 0 \
#            --conv-mode vicuna_v1

        # 调用评估脚本
        python -m llava.eval.eval_textvqa \
            --annotation-file $ANNOTATION_FILE \
            --result-file $ANSWERS_FILE

        echo "Evaluation for mix ratio ${MIX_RATIO} completed."
    done
done

echo "All evaluations completed."
#/ppio_net0/code/openapi.sh stop c9b6332a877ee875
