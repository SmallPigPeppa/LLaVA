#!/usr/bin/env bash

python main_new_fixbugv4.py \
  --api_key 7132bb0a-223c-4fd0-9d9e-893d1bcb17b0 \
  --input_file /ppio_net0/code/LLaVA/playground/data/exp1/part3-Aold.json \
  --output_file /ppio_net0/code/LLaVA/playground/data/exp1/part3-mix.json \
  --base_url https://api.ppinfra.com/v3/openai \
  --model qwen/qwen-2.5-72b-instruct \
  --max_tokens 2048 \
  --max_workers 100

