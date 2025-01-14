#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
#export HF_HOME=/mnt/disk3/wzliu/huggingface
export CUDA_VISIBLE_DEVICES=0

#MODEL_BASE="continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
MODEL_PATH_A="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-oinit-lambda1.0-merged"
MODEL_PATH_B="ablation-ckpt/exp1-model-mix/llava-v1.5-7b-lora-coco2text-lambda1-merged"


MIX_RATIO=0.9
SVD_RATIO=0.2
SAVE_PATH="${MODEL_PATH_B}-mix${MIX_RATIO}-svd${SVD_RATIO}"
MODEL_NAME=$(basename $SAVE_PATH)

# Evaluate the model (1st command - saving weights)
python -m llava.eval.model_vqa_save_weight_hf_mixv3_svd \
  --model-path-a ${MODEL_PATH_A} \
  --model-path-b ${MODEL_PATH_B} \
  --mix-ratio ${MIX_RATIO} \
  --save-path ${SAVE_PATH} \
  --retain-ratio ${SVD_RATIO}
#
## Evaluate the model (2nd command - actual evaluation)
#python -m llava.eval.model_vqa \
#    --model-path ${SAVE_PATH} \
#    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
#    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/${MODEL_NAME}.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
#
## Create the reviews directory
#mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews
#
## Generate reviews
#python llava/eval/eval_gpt_review_bench.py \
#    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
#    --rule llava/eval/table/rule.json \
#    --answer-list \
#        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
#        playground/data/eval/llava-bench-in-the-wild/answers/$MODEL_NAME.jsonl \
#    --output \
#        playground/data/eval/llava-bench-in-the-wild/reviews/$MODEL_NAME.jsonl
#
## Summarize reviews
#python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$MODEL_NAME.jsonl
#
