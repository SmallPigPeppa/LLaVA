#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

# 定义模型路径列表
MODEL_PATHS=(
#    "finetune-ckpt/fine-tune/llava-v1.5-7b-lora-iconqa-lambda1.0-merged-mix0.3-svdv3"
#    "finetune-ckpt/llava-c/llava-v1.5-7b-lora-task2-super-lambda1.0-merged-mix0.12-svdv3"
#    "finetune-ckpt/llava-c/llava-v1.5-7b-lora-task3-math-lambda1.0-merged-mix0.25-svdv3"
#    "finetune-ckpt/llava-c/llava-v1.5-7b-lora-task4-figureqa-lambda1.0-merged-mix0.13-svdv3"
#    "finetune-ckpt/llava-c/llava-v1.5-7b-lora-task5-arxiv-lambda1.0-merged-mix0.15-svdv3"
#    "finetune-ckpt/lwf-ft-order3-lambda0.2/llava-v1.5-7b-lora-task5-arxiv-merged"
#    "finetune-ckpt/lwf-ft-order3-lambda1.0/llava-v1.5-7b-lora-task5-arxiv-merged"
#    "finetune-ckpt/continual-ft-order3/llava-v1.5-7b-lora-task5-arxiv-merged"
#     "finetune-ckpt/lwf-pretrain-lambda0.2/llava-v1.5-7b-lora-task6-vg-merged"
     "finetune-ckpt/llava-c/llava-v1.5-7b-lora-task3-arxiv-lambda1.0-ablation-merged-mix0.15-svdv3"
)



# 遍历所有模型路径并执行评估
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  # 解压工具
    rm -rf ./playground/data/eval/MME/eval_tool
    unzip ./playground/data/eval/MME/eval_tool.zip -d ./playground/data/eval/MME

    # 切换到 LLaVA 目录
    cd /ppio_net0/code/LLaVA
    MODEL_NAME=$(basename ${MODEL_PATH})
    echo "Evaluating : ${MODEL_PATH}"

    # 加载模型并生成答案文件
    python -m llava.eval.model_vqa_loader \
        --model-path ${MODEL_PATH} \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/${MODEL_NAME}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    # 转换答案格式
    cd ./playground/data/eval/MME
    python convert_answer_to_mme.py --experiment ${MODEL_NAME}

    # 计算结果
    cd eval_tool
    cp /ppio_net0/code/LLaVA/sssssssMME-cal.py ./calculation_new.py
    python calculation_new.py --results_dir answers/${MODEL_NAME}

    echo "Evaluation for ${MODEL_PATH} completed."

done

echo "All evaluations completed."
