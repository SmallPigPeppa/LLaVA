#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

# 定义模型路径

#MODEL_PATH_B="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-arxivqa-merged"
MODEL_PATH_B="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-super-lambda1.0-merged"
MODEL_PATH_B="finetune-ckpt/llava-c/llava-v1.5-7b-lora-task2-super-lambda1.0-merged"

# 定义所有的 mix_ratios
MIX_RATIOS=(
#  0.0
#  0.1
  0.125
  0.12
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



# 解压工具
rm -rf ./playground/data/eval/MME/eval_tool
unzip ./playground/data/eval/MME/eval_tool.zip -d ./playground/data/eval/MME

# 遍历每个 MIX_RATIO
for MIX_RATIO in "${MIX_RATIOS[@]}"; do
    MODEL="${MODEL_PATH_B}-mix${MIX_RATIO}-svdv3"
    echo "Evaluating mix ratio: ${MIX_RATIO}"

    cd /ppio_net0/code/LLaVA

    # 加载模型并生成答案文件
    python -m llava.eval.model_vqa_loader \
        --model-path ${MODEL} \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/${MODEL}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    # 转换答案格式
    cd ./playground/data/eval/MME
    python convert_answer_to_mme.py --experiment ${MODEL}

#     计算结果
    cd eval_tool
    cp /ppio_net0/code/LLaVA/sssssssMME-cal.py ./calculation_new.py
#    python calculation.py --results_dir answers/${MODEL}
    python calculation_new.py --results_dir answers/${MODEL}

    echo "Evaluation for mix ratio ${MIX_RATIO} completed."


done

echo "All evaluations completed."
