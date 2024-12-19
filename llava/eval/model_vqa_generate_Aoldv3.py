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
from torch.nn import DataParallel


def generate_answers_from_model(model, tokenizer, image_processor, conversations, image_folder, model_name, args,
                                device):
    updated_conversations = []

    # 准备批量处理
    batch_size = args.batch_size
    num_batches = math.ceil(len(conversations) / batch_size)

    for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
        batch_convs = conversations[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_prompts = []
        batch_images = []
        batch_conv_indices = []

        # 收集批量中的所有提示和图像
        for conv_idx, conv in enumerate(batch_convs):
            conversation_history = []
            for dialogue in conv['conversations']:
                if dialogue['from'] == 'human':
                    human_question = dialogue['value']
                    conversation_history.append(f"Human: {human_question}")
                    cur_prompt = "\n".join(conversation_history)

                    conv_ = conv_templates[args.conv_mode].copy()
                    conv_.append_message(conv_.roles[0], cur_prompt)
                    conv_.append_message(conv_.roles[1], None)
                    prompt = conv_.get_prompt()

                    batch_prompts.append(prompt)
                    image_file = conv["image"]
                    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                    image_tensor = process_images([image], image_processor, model.config)[0]
                    batch_images.append(image_tensor)
                    batch_conv_indices.append(conv_idx)

        if not batch_prompts:
            # 当前批次没有需要处理的对话
            for conv in batch_convs:
                updated_conversations.append(conv)
            continue

        # 准备输入张量
        input_ids = tokenizer_image_token(batch_prompts, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(device)
        images = torch.stack(batch_images).to(device).half()

        # 生成回答
        with torch.inference_mode():
            output_ids = model.module.generate(
                input_ids=input_ids,
                images=images,
                image_sizes=[(img.size[1], img.size[0]) for img in
                             [Image.open(os.path.join(image_folder, conv["image"])).convert('RGB') for conv in
                              batch_convs]],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True
            )

        # 解码生成的答案
        generated_answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # 将生成的答案插入到相应的对话中
        for i, answer in enumerate(generated_answers):
            conv = batch_convs[i]
            gpt_index = 0  # 根据具体逻辑调整索引
            if gpt_index < len(conv['conversations']) and conv['conversations'][gpt_index]['from'] == 'gpt':
                conv['conversations'].insert(gpt_index + 1, {
                    "from": "old-model",
                    "value": answer.strip()
                })
            updated_conversations.append(conv)

    return updated_conversations


def save_updated_dataset(conversations, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def eval_model(args):
    # 模型设置
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # 检查是否有多个 GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    else:
        print("Using a single GPU or CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 加载数据集
    with open(os.path.expanduser(args.dataset_file), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 生成答案
    updated_dataset = generate_answers_from_model(model, tokenizer, image_processor, dataset, args.image_folder,
                                                  model_name, args, device)

    # 保存更新后的数据集
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
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing")

    args = parser.parse_args()

    eval_model(args)
