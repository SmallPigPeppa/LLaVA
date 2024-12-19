import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class ImageDataset(Dataset):
    def __init__(self, conversations, image_folder, tokenizer, image_processor, model_config):
        self.conversations = conversations
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]
        gpt_index = -1
        conversation_history = []

        for i, dialogue in enumerate(conv['conversations']):
            if dialogue['from'] == 'human':
                human_question = dialogue['value']
                conversation_history.append(f"Human: {human_question}")
                cur_prompt = "\n".join(conversation_history)

                # Adding the corresponding GPT response
                conv_ = conv_templates[args.conv_mode].copy()
                conv_.append_message(conv_.roles[0], cur_prompt)  # Append the concatenated history
                conv_.append_message(conv_.roles[1], None)
                prompt = conv_.get_prompt()

                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)

                # Load and process the image
                image_file = conv["image"]
                image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
                image_tensor = process_images([image], self.image_processor, self.model_config)[0]

                return input_ids, image_tensor, conv, i  # Return the input for each chunk


def generate_answers_from_model(model, tokenizer, image_processor, conversations, image_folder, model_name, args):
    dataset = ImageDataset(conversations, image_folder, tokenizer, image_processor, model.config)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    updated_conversations = []

    for batch in tqdm(data_loader):
        input_ids, image_tensors, batch_conversations, indices = batch

        # Move input data to GPUs
        input_ids = input_ids.cuda()
        image_tensors = image_tensors.cuda()

        # Generate answers using the model in batch mode
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=[image.size for image in batch_conversations],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True
            )

        # Decode and clean up the generated answer
        for i, output_id in zip(indices, output_ids):
            generated_answer = tokenizer.batch_decode(output_id, skip_special_tokens=True)[0].strip()
            # Add the generated answer to the conversation history and update the dataset
            conv = batch_conversations[i]
            gpt_index = i + 1
            if gpt_index < len(conv['conversations']) and conv['conversations'][gpt_index]['from'] == 'gpt':
                conv['conversations'].insert(gpt_index + 1, {
                    "from": "old-model",
                    "value": generated_answer
                })
            updated_conversations.append(conv)

    return updated_conversations


def save_updated_dataset(conversations, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def eval_model(args):
    # Model setup
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Use DataParallel for multi-GPU usage
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.cuda()  # Ensure the model is moved to the GPU

    # Load the dataset
    with open(os.path.expanduser(args.dataset_file), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Split dataset into chunks (for parallelization)
    dataset = get_chunk(dataset, args.num_chunks, args.chunk_idx)

    # Generate answers using the model
    updated_dataset = generate_answers_from_model(model, tokenizer, image_processor, dataset, args.image_folder,
                                                  model_name, args)

    # Save the updated dataset to a new file
    save_updated_dataset(updated_dataset, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--model-base", type=str, default=None, help="Base model (optional)")
    parser.add_argument("--image-folder", type=str, required=True, help="Folder containing images")
    parser.add_argument("--dataset-file", type=str, required=True, help="Path to the dataset file (JSONL)")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the updated dataset")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation template mode")
    parser.add_argument("--num-chunks", type=int, default=1, help="Number of chunks to split the dataset into")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Index of the chunk to process")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling for diversity")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")

    args = parser.parse_args()

    eval_model(args)
