import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import math
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def split_list(lst, n):
    """将列表拆分为n个（大致）相等的块"""
    chunk_size = math.ceil(len(lst) / n)  # 向上取整
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """获取第k个块"""
    chunks = split_list(lst, n)
    if k >= len(chunks):
        raise ValueError(f"chunk_idx {k} 超出范围，最大索引为 {len(chunks) - 1}")
    return chunks[k]


def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_dataset(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_answer(model, tokenizer, image_processor, entry, conv_mode, device, temperature, top_p, num_beams):
    image_path = entry['image']
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0].to(device)

    conversations = entry['conversations']
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
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=1024,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def update_dataset(data, model, tokenizer, image_processor, conv_mode, device, temperature, top_p, num_beams):
    for entry in tqdm(data, desc="Processing entries"):
        try:
            # 生成新的回答
            new_answer = generate_answer(model, tokenizer, image_processor, entry, conv_mode, device, temperature,
                                         top_p, num_beams)

            # 创建新的对话条目
            new_conversation = {
                "from": "old-model",
                "value": new_answer
            }

            # 将新的回答添加到 conversations 中
            entry['conversations'].append(new_conversation)
        except Exception as e:
            print(f"Error processing entry ID {entry.get('id', 'unknown')}: {e}")

    return data


def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    print("Loading dataset...")
    data = load_dataset(args.input_file)

    # 分割数据集
    if args.num_chunks > 1:
        print(f"Splitting dataset into {args.num_chunks} chunks...")
        data_chunks = split_list(data, args.num_chunks)
        if args.chunk_idx >= len(data_chunks):
            raise ValueError(f"chunk_idx {args.chunk_idx} 超出范围，最大索引为 {len(data_chunks) - 1}")
        data = data_chunks[args.chunk_idx]
        print(f"Processing chunk {args.chunk_idx} with {len(data)} entries.")
    else:
        print("Processing entire dataset as a single chunk.")

    # 加载模型
    print("Loading model...")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.to(device)
    model.half()  # 使用半精度提高效率
    model.eval()

    # 更新数据集
    print("Updating dataset with model-generated answers...")
    updated_data = update_dataset(
        data, model, tokenizer, image_processor, args.conv_mode, device,
        args.temperature, args.top_p, args.num_beams
    )

    # 如果有分块，只保存当前块
    if args.num_chunks > 1:
        # 加载完整的数据集
        full_data = load_dataset(args.input_file)
        # 替换当前块的数据
        full_data_chunks = split_list(full_data, args.num_chunks)
        full_data_chunks[args.chunk_idx] = updated_data
        # 合并所有块
        updated_full_data = []
        for chunk in full_data_chunks:
            updated_full_data.extend(chunk)
    else:
        updated_full_data = updated_data

    # 保存新的数据集
    print(f"Saving updated dataset to {args.output_file}...")
    save_dataset(updated_full_data, args.output_file)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update dataset with model-generated answers.")
    parser.add_argument("--model-path", type=str, required=True, help="路径到训练好的模型")
    parser.add_argument("--model-base", type=str, default=None, help="模型基础名称（如果有）")
    parser.add_argument("--input-file", type=str, default="llava_v1_5_mix665k-random-100.json", help="输入数据集文件")
    parser.add_argument("--output-file", type=str, default="llava_v1_5_mix665k-random-100-oldmodel.json",
                        help="输出更新后的数据集文件")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="对话模板模式")
    parser.add_argument("--num-chunks", type=int, default=1, help="将数据集分割成的块数，用于并行处理")
    parser.add_argument("--chunk-idx", type=int, default=0, help="处理的块的索引，从0开始")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成时的温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="生成时的 top_p 参数")
    parser.add_argument("--num_beams", type=int, default=1, help="生成时的 beam 数量")
    args = parser.parse_args()

    main(args)
