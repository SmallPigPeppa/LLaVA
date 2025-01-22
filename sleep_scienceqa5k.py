import json
import random

# Load OCR data
with open('playground/data/fine-tune/ScienceQA/train-mm.json', 'r') as f:
    ocr_data = json.load(f)

# Load others data
with open('playground/data/domain-incremental/llava_v1_5_mix665k-others.json', 'r') as f:
    others_data = json.load(f)

# Randomly select 20,000 samples from others_data
selected_others_data = random.sample(others_data, 5000)

# Merge the selected others data with OCR data
merged_data = ocr_data + selected_others_data

# Shuffle the merged dataset
random.shuffle(merged_data)

# Save the merged and shuffled data to a new JSON file
with open('playground/data/fine-tune/ScienceQA/train-mm-with-others-5k.json', 'w') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"合并成功，共生成 {len(merged_data)} 个样本。")

