#!/bin/bash
export HF_HOME=/ppio_net0/huggingface

# Define task suffix
TASK="task-coco-merged"

# Define the input and output files
input_file="./playground/data/debug/llava_v1_5_mix665k-random-8.json"
output_file="./playground/data/debug/llava_v1_5_mix665k-random-8_Aold.json"

# Define temporary directory for intermediate results
tmp_dir="./c-llava-cache"
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
    START_IDX=$((i * LINES_PER_GPU))
    END_IDX=$((START_IDX + LINES_PER_GPU + (i < REMAINDER ? 1 : 0)))

    # Use jq to slice the dataset for each GPU part
    tmp_output_file="$tmp_dir/llava_v1_5_mix665k-random-8_part_$((i + 1)).json"
    jq ".[$START_IDX:$END_IDX]" $input_file > $tmp_output_file
done

# Run the model evaluation in parallel on each GPU
echo "Running evaluations on each GPU..."
for i in $(seq 0 $((NUM_GPUS - 1)))
do
    CUDA_VISIBLE_DEVICES=$i python -m llava.eval.model_vqa_generate_Aoldv3 \
        --model-path "continual-ckpt/domain/llava-v1.5-7b-lora-$TASK" \
        --image-folder ./playground/data \
        --dataset-file "$tmp_dir/llava_v1_5_mix665k-random-8_part_$((i + 1)).json" \
        --output-file "$tmp_dir/llava_v1_5_mix665k-random-8_Aold_part_$((i + 1)).json" \
        --conv-mode llava_v1 &
done

# Wait for all processes to finish
wait

# Combine all results into the final output file, ensuring they are in the correct order by id
echo "Combining results..."
cat $tmp_dir/llava_v1_5_mix665k-random-8_Aold_part_*.json | jq -s 'add | sort_by(.id)' > $output_file

# Clean up part files in tmp_dir
rm -rf $tmp_dir

echo "Processing complete. Final results saved to $output_file."
