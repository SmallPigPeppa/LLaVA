#!/usr/bin/env bash

    # ----------------------------
    # Adjust the Python command as needed
python main_new_fixbugv2.py \
  --api_key 7132bb0a-223c-4fd0-9d9e-893d1bcb17b0 \
  --input_file input/part2.json \
  --output_file output-qwen-v2/part2.json \
  --base_url https://api.ppinfra.com/v3/openai \
  --model qwen/qwen2.5-32b-instruct \
  --max_tokens 4096 \
  --max_workers 100

