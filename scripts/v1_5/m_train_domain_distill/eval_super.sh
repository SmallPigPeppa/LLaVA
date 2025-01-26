#!/bin/bash

export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=1

MODEL_PATH="liuhaotian/llava-v1.5-7b"
#MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-figureqa-merged"
OUT_PATH="${MODEL_PATH}"
MODEL=$(basename $OUT_PATH)


python -m llava.eval.model_super_clevr \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/data/fine-tune/super-CLEVR/test_2k.json \
    --image-folder ./playground/data \
    --answers-file ./playground/data/eval/super-CLEVR/answers/${MODEL}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_clevr_math \
    --annotation-file ./playground/data/fine-tune/super-CLEVR/test_2k.json \
    --result-file ./playground/data/eval/super-CLEVR/answers/${MODEL}.jsonl \
    --output-dir ./playground/data/eval/super-CLEVR/answers/${MODEL} \
