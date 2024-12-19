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


def generate_answers_from_model(model, tokenizer, image_processor, conversations, image_folder, model_name, args):
    device = model.device_ids[0] if isinstance(model, torch.nn.DataParallel) else model.device
    updated_conversations = []

    for conv in tqdm(conversations):
        gpt_index = -1
        conversation_history = []

        for i, dialogue in enumerate(conv['conversations']):
            if dialogue['from'] == 'human':
                human_question = dialogue['value']
                conversation_history.append(f"Human: {human_question}")
                cur_prompt = "\n".join(conversation_history)

                conv_ = conv_templates[args.conv_mode].copy()
                conv_.append_message(conv_.roles[0], cur_prompt)
                conv_.append_message(conv_.roles[1], None)
                prompt = conv_.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(device)

                image_file = conv["image"]
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                image_tensor = process_images([image], image_processor, model.module.config if isinstance(model,
                                                                                                          torch.nn.DataParallel) else model.config)[
                    0]
                image_tensor = image_tensor.to(device).half()

                with torch.inference_mode():
                    output_ids = model.module.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0),
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

    # Move model to GPUs and wrap with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Load the dataset
    with open(os.path.expanduser(args.dataset_file), "r", encoding="utf-8") as f:
        dataset = json.load(f)

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
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling for diversity")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")

    args = parser.parse_args()

    eval_model(args)
