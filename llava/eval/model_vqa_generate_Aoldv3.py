import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from torch.utils.data import Dataset, DataLoader

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

class ConversationDataset(Dataset):
    def __init__(self, dataset_file):
        with open(dataset_file, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def collate_fn(batch):
    return batch

def generate_answers_from_model(model, tokenizer, image_processor, conversations_batch, image_folder, model_name, args):
    updated_batch = []

    # Prepare lists for batch processing
    prompts = []
    images = []
    original_conversations = []

    for conv in conversations_batch:
        conversation_history = []
        for dialogue in conv['conversations']:
            if dialogue['from'] == 'human':
                human_question = dialogue['value']
                conversation_history.append(f"Human: {human_question}")
        cur_prompt = "\n".join(conversation_history)
        conv_ = conv_templates[args.conv_mode].copy()
        conv_.append_message(conv_.roles[0], cur_prompt)  # Append the concatenated history
        conv_.append_message(conv_.roles[1], None)
        prompt = conv_.get_prompt()
        prompts.append(prompt)

        # Load and process the image
        image_file = conv["image"]
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        images.append(image)

        original_conversations.append(conv)

    # Tokenize prompts
    input_ids = tokenizer_image_token(prompts, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

    # Process images in batch
    image_tensors = process_images(images, image_processor, model.config)

    # Generate answers in batch
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors.half().cuda(),
            image_sizes=[img.size for img in images],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=1024,
            use_cache=True
        )

    # Decode generated answers
    generated_answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Insert 'old-model' responses into conversations
    for conv, answer in zip(original_conversations, generated_answers):
        # Find the index where to insert 'old-model' response
        # Assuming 'gpt' response is immediately after the 'human' question
        gpt_index = -1
        for i, dialogue in enumerate(conv['conversations']):
            if dialogue['from'] == 'human':
                gpt_index = i + 1
                break
        if gpt_index != -1 and gpt_index < len(conv['conversations']) and conv['conversations'][gpt_index]['from'] == 'gpt':
            conv['conversations'].insert(gpt_index + 1, {
                "from": "old-model",
                "value": answer.strip()
            })
        else:
            # 如果没有找到对应的 'gpt' 回复，可以选择其他处理方式
            conv['conversations'].append({
                "from": "old-model",
                "value": answer.strip()
            })
        updated_batch.append(conv)

    return updated_batch

def save_updated_dataset(conversations, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

def eval_model(args):
    # Model setup
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Prepare Dataset and DataLoader
    dataset = ConversationDataset(args.dataset_file)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    updated_dataset = []

    for batch in tqdm(dataloader, desc="Processing Batches"):
        # Generate answers using the model for the current batch
        updated_batch = generate_answers_from_model(model, tokenizer, image_processor, batch, args.image_folder,
                                                    model_name, args)
        updated_dataset.extend(updated_batch)

    # Save the updated dataset to a new file
    save_updated_dataset(updated_dataset, args.output_file)

if __name__ == "__main__":
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

    args = parser.parse_args()

    eval_model(args)
