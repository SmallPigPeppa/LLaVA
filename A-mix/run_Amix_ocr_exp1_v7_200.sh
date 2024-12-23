#!/usr/bin/env bash

python main_new_fixbugv7.py \
  --api_key dcf59e46-a998-4a69-adfd-31aeabac1760 \
  --input_file /ppio_net0/code/LLaVA/playground/data/exp1/part2.json \
  --output_file /ppio_net0/code/LLaVA/playground/data/exp1/part2-mix-v7-200.json \
  --base_url https://api.ppinfra.com/v3/openai \
  --model qwen/qwen-2.5-72b-instruct \
  --max_tokens 2048 \
  --max_workers 200

