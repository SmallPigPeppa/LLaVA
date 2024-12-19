import multiprocessing
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
import multiprocessing
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

# Ensure the 'spawn' method is used for multiprocessing
def eval_model(args):
    multiprocessing.set_start_method('spawn', force=True)  # Set spawn method

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Load the dataset
    with open(os.path.expanduser(args.dataset_file), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")

    # Calculate the base chunk size and distribute the remainder evenly
    base_chunk_size = len(dataset) // num_gpus
    remainder = len(dataset) % num_gpus

    # Split the dataset into chunks
    chunks = []
    start = 0
    for i in range(num_gpus):
        end = start + base_chunk_size + (1 if i < remainder else 0)  # Distribute the remainder
        chunks.append(dataset[start:end])
        start = end

    # Prepare a multiprocessing queue to collect results from all processes
    result_queue = multiprocessing.Queue()

    # Create a list to hold the processes
    processes = []

    # Start one process per GPU
    for i in range(num_gpus):
        p = multiprocessing.Process(target=generate_answers_on_gpu, args=(
            i, model, tokenizer, image_processor, chunks[i], args.image_folder, model_name, args, result_queue))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Collect the results from the queue
    updated_dataset = []
    while not result_queue.empty():
        updated_dataset.extend(result_queue.get())

    # Save the updated dataset to a new file
    save_updated_dataset(updated_dataset, args.output_file)


def save_updated_dataset(conversations, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--model-base", type=str, default=None, help="Base model (optional)")
    parser.add_argument("--image-folder", type=str, required=True, help="Folder containing images")
    parser.add_argument("--dataset-file", type=str, required=True, help="Path to the dataset file (JSONL)")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the updated dataset")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation template mode")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling for diversity")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")

    args = parser.parse_args()

    eval_model(args)

