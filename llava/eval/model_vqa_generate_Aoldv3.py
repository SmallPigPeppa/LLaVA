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


# Function to process a chunk of the dataset on a specific GPU
def generate_answers_on_gpu(device_id, model, tokenizer, image_processor, conversations, image_folder, model_name, args,
                            result_queue):
    # Set the current device to the one assigned by the GPU ID
    torch.cuda.set_device(device_id)
    updated_conversations = []

    # Process each conversation chunk on the assigned GPU
    for conv in tqdm(conversations):
        gpt_index = -1  # Initialize to an invalid index
        conversation_history = []  # This will hold the entire conversation history

        for i, dialogue in enumerate(conv['conversations']):
            if dialogue['from'] == 'human':
                human_question = dialogue['value']
                conversation_history.append(f"Human: {human_question}")
                cur_prompt = "\n".join(conversation_history)

                conv_ = conv_templates[args.conv_mode].copy()
                conv_.append_message(conv_.roles[0], cur_prompt)
                conv_.append_message(conv_.roles[1], None)
                prompt = conv_.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                    0).to(device_id)

                # Load and process the image
                image_file = conv["image"]
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                image_tensor = process_images([image], image_processor, model.config)[0].to(device_id)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().to(device_id),
                        image_sizes=[image.size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=1024,
                        use_cache=True
                    )

                generated_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                gpt_index = i + 1
                if gpt_index < len(conv['conversations']) and conv['conversations'][gpt_index]['from'] == 'gpt':
                    conv['conversations'].insert(gpt_index + 1, {
                        "from": "old-model",
                        "value": generated_answer
                    })

                conversation_history.append(f"GPT: {generated_answer}")

        updated_conversations.append(conv)

    # Put the result in the queue to merge later
    result_queue.put(updated_conversations)


# Function to split the dataset and process it in parallel
def eval_model(args):
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
