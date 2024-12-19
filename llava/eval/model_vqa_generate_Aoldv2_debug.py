import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from torch.utils.data import DataLoader, Dataset
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class ConversationDataset(Dataset):
    """A custom Dataset to handle chunked conversations"""

    def __init__(self, conversations, image_folder, tokenizer, image_processor, model, args):
        self.conversations = conversations
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model = model
        self.args = args

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]
        updated_conversations = []

        for dialogue in conv['conversations']:
            if dialogue['from'] == 'human':
                human_question = dialogue['value']
                conversation_history = [f"Human: {human_question}"]

                cur_prompt = "\n".join(conversation_history)
                conv_ = conv_templates[self.args.conv_mode].copy()
                conv_.append_message(conv_.roles[0], cur_prompt)
                conv_.append_message(conv_.roles[1], None)
                prompt = conv_.get_prompt()

                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                                  return_tensors='pt').unsqueeze(0).cuda()

                image_file = conv["image"]
                image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
                image_tensor = process_images([image], self.image_processor, self.model.config)[0]

                # Model inference
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        do_sample=True if self.args.temperature > 0 else False,
                        temperature=self.args.temperature,
                        top_p=self.args.top_p,
                        num_beams=self.args.num_beams,
                        max_new_tokens=1024,
                        use_cache=True
                    )

                generated_answer = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                gpt_index = len(conv['conversations']) + 1
                if gpt_index < len(conv['conversations']) and conv['conversations'][gpt_index]['from'] == 'gpt':
                    conv['conversations'].insert(gpt_index + 1, {
                        "from": "old-model",
                        "value": generated_answer
                    })
                conversation_history.append(f"GPT: {generated_answer}")

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

    # Prepare DataLoader with dataset
    with open(os.path.expanduser(args.dataset_file), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Split dataset into chunks (for parallelization)
    dataset = get_chunk(dataset, args.num_chunks, args.chunk_idx)

    # Use custom dataset
    conversation_dataset = ConversationDataset(dataset, args.image_folder, tokenizer, image_processor, model, args)

    # Initialize DataLoader for batch processing
    dataloader = DataLoader(conversation_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Prepare the model for distributed inference
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[args.local_rank])

    # Generate answers using the model in batches
    updated_dataset = []
    for batch in tqdm(dataloader):
        updated_dataset.extend(batch)

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
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

    args = parser.parse_args()

    # Initialize distributed training if necessary
    if args.local_rank == 0:
        torch.distributed.init_process_group(backend="nccl")

    eval_model(args)
