import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

def setup_distributed(local_rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

def cleanup_distributed():
    dist.destroy_process_group()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def generate_answers_from_model(model, tokenizer, image_processor, conversations, image_folder, model_name, args, device):
    updated_conversations = []

    model.eval()
    for conv in tqdm(conversations, desc=f"Processing on GPU {device}"):
        # We will collect pairs of "gpt" and "human" indices
        gpt_index = -1  # Initialize to an invalid index
        conversation_history = []  # This will hold the entire conversation history

        for i, dialogue in enumerate(conv['conversations']):
            if dialogue['from'] == 'human':
                # Extract the question (from human)
                human_question = dialogue['value']

                # Create the prompt for the model with conversation history
                conversation_history.append(f"Human: {human_question}")

                # Concatenate previous conversation history with the current human question
                cur_prompt = "\n".join(conversation_history)

                # Adding the corresponding GPT response
                conv_ = conv_templates[args.conv_mode].copy()
                conv_.append_message(conv_.roles[0], cur_prompt)  # Append the concatenated history
                conv_.append_message(conv_.roles[1], None)
                prompt = conv_.get_prompt()

                # Tokenize and move to the correct device
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(device)

                # Load and process the image
                image_file = conv["image"]
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                image_tensor = process_images([image], image_processor, model.module.config)[0].to(device)

                # Generate answer from model
                with torch.inference_mode():
                    output_ids = model.module.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half(),
                        image_sizes=[image.size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=1024,
                        use_cache=True
                    )

                # Decode and clean up the generated answer
                generated_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                # Now find the "gpt" index that corresponds to this human question
                gpt_index = i + 1  # The next index is the gpt response
                if gpt_index < len(conv['conversations']) and conv['conversations'][gpt_index]['from'] == 'gpt':
                    # Insert the "old-model" response after the "gpt" message
                    conv['conversations'].insert(gpt_index + 1, {
                        "from": "old-model",
                        "value": generated_answer
                    })

                # Add the GPT's answer to the conversation history
                conversation_history.append(f"GPT: {generated_answer}")

        updated_conversations.append(conv)

    return updated_conversations

def save_updated_dataset(conversations, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write the entire list as a JSON array in one go
        json.dump(conversations, f, ensure_ascii=False, indent=2)

class ConversationDataset(Dataset):
    def __init__(self, conversations):
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return self.conversations[idx]

def eval_model(args):
    # Setup distributed environment
    local_rank = args.local_rank
    world_size = args.world_size
    setup_distributed(local_rank, world_size)
    device = torch.device(f'cuda:{local_rank}')

    # Model setup
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Load the dataset
    with open(os.path.expanduser(args.dataset_file), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Split dataset into chunks based on world_size
    datasets = split_list(dataset, world_size)
    if local_rank >= len(datasets):
        # No data for this rank
        local_dataset = []
    else:
        local_dataset = datasets[local_rank]

    # Generate answers using the model
    updated_dataset = generate_answers_from_model(
        model, tokenizer, image_processor, local_dataset, args.image_folder,
        model_name, args, device
    )

    # Save the updated dataset to a new file with rank suffix
    output_file = f"{args.output_file}_rank{local_rank}"
    save_updated_dataset(updated_dataset, output_file)

    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--model-base", type=str, default=None, help="Base model (optional)")
    parser.add_argument("--image-folder", type=str, required=True, help="Folder containing images")
    parser.add_argument("--dataset-file", type=str, required=True, help="Path to the dataset file (JSON)")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the updated dataset")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation template mode")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling for diversity")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")

    # Distributed training parameters
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs available")

    args = parser.parse_args()

    eval_model(args)

    # After all processes finish, aggregate the results
    if args.local_rank == 0:
        aggregated_data = []
        for rank in range(args.world_size):
            rank_output_file = f"{args.output_file}_rank{rank}"
            if os.path.exists(rank_output_file):
                with open(rank_output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    aggregated_data.extend(data)
                os.remove(rank_output_file)  # Clean up individual rank files

        save_updated_dataset(aggregated_data, args.output_file)
        print(f"Aggregated results saved to {args.output_file}")

if __name__ == "__main__":
    main()
