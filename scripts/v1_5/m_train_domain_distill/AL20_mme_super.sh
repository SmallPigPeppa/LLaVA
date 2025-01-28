#!/bin/bash
export HF_HOME=/mnt/disk3/wzliu/huggingface
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="checkpoints/llava-v1.5-7b-lora-merged"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-arxivqa-merged"
MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-super-lambda1.0-merged"
MODEL_PATH="finetune-ckpt/llava-c/llava-v1.5-7b-lora-task1-iconqa-task2-super-merged"
MODEL_PATH="finetune-ckpt/llava-c/llava-v1.5-7b-lora-task2-super-lambda1.0-merged"

MODEL_NAME=$(basename ${MODEL_PATH})

rm -rf ./playground/data/eval/MME/eval_tool
unzip ./playground/data/eval/MME/eval_tool.zip -d ./playground/data/eval/MME

python -m llava.eval.model_vqa_loader \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment ${MODEL_NAME}

cd eval_tool

#python calculation.py --results_dir answers/${MODEL_NAME}
cp /ppio_net0/code/LLaVA/sssssssMME-cal.py ./calculation_new.py
python calculation_new.py --results_dir answers/${MODEL_NAME}




