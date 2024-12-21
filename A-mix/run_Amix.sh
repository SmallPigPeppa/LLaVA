python main.py \
  --api_key dcf59e46-a998-4a69-adfd-31aeabac1760 \
  --input_file input/llava_v1_5_mix665k-ocr_vqa-Aold-step1-stack-dedupedv2.json \
  --output_file output/debug.json \
  --base_url https://api.ppinfra.com/v3/openai \
  --model qwen/qwen2.5-32b-instruct \
  --max_tokens 2048 \
  --max_workers 3
