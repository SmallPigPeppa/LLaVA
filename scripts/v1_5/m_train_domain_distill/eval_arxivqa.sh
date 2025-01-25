#!/bin/bash

#!/bin/bash
export HF_HOME=/ppio_net0/huggingface
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="finetune-ckpt/fine-tune/llava-v1.5-7b-lora-arxivqa-merged"
OUT_PATH="${MODEL_PATH}"
MODEL=$(basename $OUT_PATH)


#python -m llava.eval.model_arxivqa \
#    --model-path ${MODEL_PATH} \
#    --question-file ./playground/data/fine-tune/ArxivQA/test_2k.json \
#    --image-folder ./playground/data \
#    --answers-file ./playground/data/eval/arxivqa/answers/${MODEL}.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1


python -m llava.eval.eval_arxivqa \
    --annotation-file ./playground/data/fine-tune/ArxivQA/test_2k.json \
    --result-file ./playground/data/eval/arxivqa/answers/${MODEL}.jsonl \
    --output-dir ./playground/data/eval/arxivqa/answers/${MODEL} \

