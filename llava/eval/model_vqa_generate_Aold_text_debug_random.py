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


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



"""
只支持单轮QA
"""
# def generate_answers_from_model(model, tokenizer, image_processor, conversations, image_folder, model_name, args):
#     updated_conversations = []
#
#     for conv in tqdm(conversations):
#         # We will collect pairs of "gpt" and "human" indices
#         gpt_index = -1  # Initialize to an invalid index
#         for i, dialogue in enumerate(conv['conversations']):
#             if dialogue['from'] == 'human':
#                 # Extract the question (from human)
#                 human_question = dialogue['value']
#
#                 # Create the prompt for the model
#                 image_file = conv["image"]
#                 cur_prompt = human_question
#
#                 conv_ = conv_templates[args.conv_mode].copy()
#                 conv_.append_message(conv_.roles[0], cur_prompt)
#                 conv_.append_message(conv_.roles[1], None)
#                 prompt = conv_.get_prompt()
#
#                 input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
#                     0).cuda()
#
#                 image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
#                 image_tensor = process_images([image], image_processor, model.config)[0]
#                 with torch.inference_mode():
#                     output_ids = model.generate(
#                         input_ids,
#                         images=image_tensor.unsqueeze(0).half().cuda(),
#                         image_sizes=[image.size],
#                         do_sample=True if args.temperature > 0 else False,
#                         temperature=args.temperature,
#                         top_p=args.top_p,
#                         num_beams=args.num_beams,
#                         max_new_tokens=1024,
#                         use_cache=True
#                     )
#
#                 generated_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#
#                 # Now find the "gpt" index that corresponds to this human question
#                 gpt_index = i + 1  # The next index is the gpt response
#                 if gpt_index < len(conv['conversations']) and conv['conversations'][gpt_index]['from'] == 'gpt':
#                     # Insert the "old-model" response after the "gpt" message
#                     conv['conversations'].insert(gpt_index + 1, {
#                         "from": "old-model",
#                         "value": generated_answer
#                     })
#
#         updated_conversations.append(conv)
#
#     return updated_conversations


def eval_model(model_name, questions_file, answers_file):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16).cuda()


    ques_file = open(os.path.expanduser(questions_file), "r")
    ans_file = open(os.path.expanduser(answers_file), "w")
    for i, line in enumerate(tqdm(ques_file)):
        idx = json.loads(line)["question_id"]
        qs = json.loads(line)["text"]
        cat = json.loads(line)["category"]
        conv = default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            use_cache=True,
            temperature=0.7,
            max_new_tokens=1024,)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        try:
            index = outputs.index(conv.sep, len(prompt))
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep, len(prompt))

        outputs = outputs[len(prompt) + len(conv.roles[1]) + 2:index].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
"""
支持多轮QA
"""

def generate_answers_from_model(model, tokenizer, image_processor, items, image_folder, model_name, args):
    updated_conversations = []

    for item in tqdm(items):
        # We will collect pairs of "gpt" and "human" indices
        gpt_index = -1  # Initialize to an invalid index
        conversation_history = []  # This will hold the entire conversation history
        conv = conv_templates[args.conv_mode].copy()

        for i, qa in enumerate(item['conversations']):
            if qa['from'] == 'human':
                # Extract the question (from human)
                qs = qa['value']
                conv.append_message(conv.roles[0], qs)  # Append the concatenated history
                prompt = conv.get_prompt()
                inputs = tokenizer([prompt])
                input_ids = torch.as_tensor(inputs.input_ids).cuda()

                import pdb; pdb.set_trace()

                output_ids = model.generate(
                    input_ids,
                    do_sample=True,
                    use_cache=True,
                    temperature=0.7,
                    max_new_tokens=1024, )

                import pdb; pdb.set_trace()
                output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                # Now find the "gpt" index that corresponds to this human question
                gpt_index = i + 1  # The next index is the gpt response
                if gpt_index < len(item['conversations']) and item['conversations'][gpt_index]['from'] == 'gpt':
                    # Insert the "old-model" response after the "gpt" message
                    item['conversations'].insert(gpt_index + 1, {
                        "from": "old-model",
                        "value": output,
                    })

                conv.append_message(conv.roles[1], output)

        updated_conversations.append(item)

    return updated_conversations


def save_updated_dataset(conversations, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # import pdb;pdb.set_trace()
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write the entire list as a JSON array in one go
        # import pdb;pdb.set_trace()
        json.dump(conversations, f, ensure_ascii=False, indent=2)



def eval_model(args):
    # Model setup
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Load the dataset
    # dataset = [json.loads(line) for line in open(os.path.expanduser(args.dataset_file), "r")]
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
