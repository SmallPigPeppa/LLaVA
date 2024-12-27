#!/bin/bash

# 设置Hugging Face缓存目录
export HF_HOME=/mnt/disk3/wzliu/huggingface

# 定义模型和任务相关参数
MODEL="llava-v1.5-7b-wo-train"

# 定义输入和输出文件
INPUT_FILE="./playground/data/domain-incremental/llava_v1_5_mix665k-others.json"
OUTPUT_FILE="./playground/data/domain-incremental/llava_v1_5_mix665k-others-Aold.json"

# 生成基于输出文件的临时目录名
TMP_DIR="./c-llava-cache-${OUTPUT_FILE##*/}"
mkdir -p $TMP_DIR

# 检测可用的GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $NUM_GPUS 个GPU。"

# 使用jq拆分数据集为NUM_GPUS部分
echo "使用jq将数据集拆分为 $NUM_GPUS 部分..."
TOTAL_LINES=$(jq '. | length' $INPUT_FILE)
LINES_PER_GPU=$((TOTAL_LINES / NUM_GPUS))
REMAINDER=$((TOTAL_LINES % NUM_GPUS))

echo "总数据条数: $TOTAL_LINES"
echo "每个GPU处理条数: $LINES_PER_GPU"
echo "剩余条数（将分配给最后一个GPU）: $REMAINDER"

for i in $(seq 0 $((NUM_GPUS - 1)))
do
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        # 最后一个GPU获取剩余的部分
        START_IDX=$((i * LINES_PER_GPU))
        END_IDX=$TOTAL_LINES
    else
        START_IDX=$((i * LINES_PER_GPU))
        END_IDX=$((START_IDX + LINES_PER_GPU))
    fi

    # 使用jq切片数据集
    TMP_OUTPUT_FILE="$TMP_DIR/part_$((i + 1)).json"
    jq ".[$START_IDX:$END_IDX]" $INPUT_FILE > $TMP_OUTPUT_FILE
    echo "已创建分片文件: $TMP_OUTPUT_FILE (索引范围: $START_IDX 到 $END_IDX)"
done

# 并行在每个GPU上运行评估任务
echo "在每个GPU上并行运行评估任务..."
for i in $(seq 0 $((NUM_GPUS - 1)))
do
    CUDA_VISIBLE_DEVICES=$i python -m llava.eval.model_vqa_generate_Aold_text \
        --model-path "continual-ckpt/distill/$MODEL" \
        --image-folder ./playground/data \
        --dataset-file "$TMP_DIR/part_$((i + 1)).json" \
        --output-file "$TMP_DIR/output_part_$((i + 1)).json" \
        --conv-mode llava_v1 &
    echo "已启动GPU $i 上的评估任务。"
done

# 等待所有后台任务完成
wait
echo "所有GPU上的评估任务已完成。"

# 合并所有部分结果到最终输出文件
echo "合并所有部分结果到最终输出文件..."
jq -s 'add' $TMP_DIR/output_part_*.json > $OUTPUT_FILE

# 清理临时目录
rm -rf $TMP_DIR
echo "已清理临时文件。"

echo "处理完成。最终结果已保存到 $OUTPUT_FILE。"
