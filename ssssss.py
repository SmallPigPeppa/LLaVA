import os
import json

# Input and output file paths
input_file = '/ppio_net0/code/LLaVA/playground/data/eval/seed_bench/llava-seed-bench.jsonl'
output_file = '/ppio_net0/code/LLaVA/playground/data/eval/seed_bench/llava-seed-bench-modified.jsonl'

# Function to modify the image path based on the task ID
def modify_image_path(image_path, question_id):
    # Extract the task ID from the image path
    task_id = image_path.split('/')[1].split('_')[0]
    # Construct the new path based on the task ID
    if task_id == "10":
        new_path = f"SEED-Bench-video-image/task10/ssv2_8_frame/{question_id}/4.png"
    elif task_id == "11":
        new_path = f"SEED-Bench-video-image/task11/kitchen_8_frame/{question_id}/4.png"
    elif task_id == "12":
        new_path = f"SEED-Bench-video-image/task12/breakfast_8_frame/{question_id}/4.png"
    else:
        new_path = image_path  # Keep the original path if the task ID is not recognized
    return new_path

# Process the JSONL file
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Parse each line as a JSON object
        entry = json.loads(line.strip())
        # Check if the question ID contains 'v'
        if 'v' in entry['question_id']:
            # Modify the image path
            entry['image'] = modify_image_path(entry['image'], entry['question_id'])
        # Write the modified entry to the output file
        json.dump(entry, outfile)
        outfile.write('\n')  # Add a newline after each entry
