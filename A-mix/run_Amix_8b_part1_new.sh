#!/usr/bin/env bash

# 脚本名：run_loop.sh
# 功能：重复执行 python main_new.py 命令，最多 100 次。如果任意一次执行时间小于30秒，则提前停止。

MAX_RUNS=100
MIN_TIME=30  # 小于 30 秒就停止

for ((i=1; i<=$MAX_RUNS; i++))
do
    echo "===> 开始执行第 $i 次... "

    # 记录开始时间
    START_TIME=$(date +%s)

    # 这里是你的 python 命令，请根据实际需求调整参数
    python main_new.py \
      --api_key 09604121-6e12-4e83-8e2e-277b6450b32c \
      --input_file input/part1.json \
      --output_file output-8b/part1.json \
      --base_url https://api.novita.ai/v3/openai \
      --model meta-llama/llama-3.1-8b-instruct \
      --max_tokens 2048 \
      --max_workers 32

    # 记录结束时间
    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))

    echo "===> 第 $i 次执行耗时：$ELAPSED 秒"

    # 如果本次执行时间小于 30 秒，则停止脚本
    if [ "$ELAPSED" -lt "$MIN_TIME" ]; then
        echo "本次执行时间 < ${MIN_TIME} 秒，提前停止后续执行。"
        break
    fi

    echo
done

echo "脚本执行完毕。"
