#!/bin/bash

#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="liuhaotian/llava-v1.5-7b"
MODEL_PATH="finetune-ckpt/fine-tune-1task/llava-v1.5-7b-lora-arxivqa-datamix-merged"
#MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-arxivqa-lambda1.0-merged"


# 定义Mix率的不同值
MIX_RATIOS=(
  0.0
  0.1
  0.2
  0.3
  0.4
  0.5
  0.6
  0.7
  0.8
  0.9
  1.0
)

# 遍历每个Mix率
for MIX_RATIO in "${MIX_RATIOS[@]}"; do
  # 设置带有Mix率的模型路径
  CURRENT_MODEL_PATH="${MODEL_PATH}-mix${MIX_RATIO}"
  MODEL=$(basename ${CURRENT_MODEL_PATH})

  # 输出当前正在处理的模型路径
  echo "Processing model: ${CURRENT_MODEL_PATH}"

  python -m llava.eval.model_arxivqa \
    --model-path ${CURRENT_MODEL_PATH} \
    --question-file ./playground/data/fine-tune/ArxivQA/test_2k.json \
    --image-folder ./playground/data \
    --answers-file ./playground/data/eval/arxivqa/answers/${MODEL}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


    python -m llava.eval.eval_arxivqa \
    --annotation-file ./playground/data/fine-tune/ArxivQA/test_2k.json \
    --result-file ./playground/data/eval/arxivqa/answers/${MODEL}.jsonl \
    --output-dir ./playground/data/eval/arxivqa/answers/${MODEL} \

  # 输出已完成的提示
  echo "Finished processing model: ${CURRENT_MODEL_PATH}"
done
