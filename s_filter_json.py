import os
import json
from PIL import Image

# Define base directory and file paths
base_dir = '/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/llava665k'
input_path = os.path.join(base_dir, 'llava_v1_5_mix665k.json')
output_path = os.path.expanduser('~/llava665k_validated.json')
possible_extensions = ['.jpg', '.png', '.gif']

# Load original JSON
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

validated = []
for item in data:
    original_image = item['image']
    image_path = os.path.join(base_dir, original_image)

    paths_to_try = [image_path]
    if not os.path.exists(image_path):
        name, _ = os.path.splitext(original_image)
        paths_to_try = [os.path.join(base_dir, name + ext) for ext in possible_extensions]

    found = False
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                # Try opening the image
                with Image.open(path) as img:
                    img.convert('RGB')
                # If successful, update image field if extension changed
                rel_path = os.path.relpath(path, base_dir)
                item['image'] = rel_path.replace('\\', '/')
                validated.append(item)
                found = True
                break
            except Exception as e:
                # Failed to open; try next extension
                continue

    if not found:
        print(f"Entry {item['id']} removed: file not found or unreadable.")

# Save validated JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(validated, f, ensure_ascii=False, indent=2)

print(f"Validated JSON saved to {output_path} with {len(validated)} entries.")
