import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_dataset(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_answer(model, tokenizer, image_processor, entry, conv_mode, device):
    image_path = entry['image']
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0].to(device)

    conversations = entry['conversations']
    prompt = ""
    conv = conv_templates[conv_mode].copy()

    for message in conversations:
        if message['from'] == 'human':
            qs = message['value']
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv.append_message(conv.roles[0], qs)
        elif message['from'] == 'gpt':
            conv.append_message(conv.roles[1], message['value'])

    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def update_dataset(data, model, tokenizer, image_processor, conv_mode, device):
    for entry in tqdm(data, desc="Processing entries"):
        # 生成新的回答
        new_answer = generate_answer(model, tokenizer, image_processor, entry, conv_mode, device)

        # 创建新的对话条目
        new_conversation = {
            "from": "old-model",
            "value": new_answer
        }

        # 将新的回答添加到 conversations 中
        entry['conversations'].append(new_conversation)

    return data


def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    print("Loading dataset...")
    data = load_dataset(args.input_file)

    # 加载模型
    print("Loading model...")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.to(device)
    model.eval()

    # 更新数据集
    print("Updating dataset with model-generated answers...")
    updated_data = update_dataset(data, model, tokenizer, image_processor, args.conv_mode, device)

    # 保存新的数据集
    print(f"Saving updated dataset to {args.output_file}...")
    save_dataset(updated_data, args.output_file)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update dataset with model-generated answers.")
    parser.add_argument("--model-path", type=str, required=True, help="路径到训练好的模型")
    parser.add_argument("--model-base", type=str, default=None, help="模型基础名称（如果有）")
    parser.add_argument("--input-file", type=str, default="llava_v1_5_mix665k-random-100.json", help="输入数据集文件")
    parser.add_argument("--output-file", type=str, default="llava_v1_5_mix665k-random-100-oldmodel.json",
                        help="输出更新后的数据集文件")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="对话模板模式")
    args = parser.parse_args()

    main(args)
