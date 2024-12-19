#!/bin/bash
export HF_HOME=/ppio_net0/huggingface

# Define task suffix
TASK="task-coco-merged"

# Define the input and output files
input_file="./playground/data/debug/llava_v1_5_mix665k-random-8.json"
output_file="./playground/data/debug/llava_v1_5_mix665k-random-8-Aold.json"

# Generate temporary directory name based on the output file
tmp_dir="./c-llava-cache-${output_file##*/}"
mkdir -p $tmp_dir

# Number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Split the dataset into n parts using jq
echo "Splitting the dataset into $NUM_GPUS parts using jq..."
TOTAL_LINES=$(jq '. | length' $input_file)
LINES_PER_GPU=$((TOTAL_LINES / NUM_GPUS))
REMAINDER=$((TOTAL_LINES % NUM_GPUS))

# Create split files in tmp_dir
for i in $(seq 0 $((NUM_GPUS - 1)))
do
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        # Last GPU gets the remainder
        START_IDX=$((i * LINES_PER_GPU))
        END_IDX=$TOTAL_LINES
    else
        START_IDX=$((i * LINES_PER_GPU))
        END_IDX=$((START_IDX + LINES_PER_GPU))
    fi

    # Use jq to slice the dataset for each GPU part
    tmp_output_file="$tmp_dir/part_$((i + 1)).json"
    jq ".[$START_IDX:$END_IDX]" $input_file > $tmp_output_file
done

# Run the model evaluation in parallel on each GPU
echo "Running evaluations on each GPU..."
for i in $(seq 0 $((NUM_GPUS - 1)))
do
    CUDA_VISIBLE_DEVICES=$i python -m llava.eval.model_vqa_generate_Aold \
        --model-path "continual-ckpt/domain/llava-v1.5-7b-lora-$TASK" \
        --image-folder ./playground/data \
        --dataset-file "$tmp_dir/part_$((i + 1)).json" \
        --output-file "$tmp_dir/output_part_$((i + 1)).json" \
        --conv-mode llava_v1 &
done

# Wait for all processes to finish
wait

# Combine all results into the final output file without sorting
echo "Combining results without sorting using jq..."
jq -s 'add' $tmp_dir/output_part_*.json > $output_file

# Clean up part files in tmp_dir
rm -rf $tmp_dir

echo "Processing complete. Final results saved to $output_file."
