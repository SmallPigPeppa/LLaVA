import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def split_list(lst, n):
    """将列表分割成n个（大致）等大小的块"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def update_dataset(args):
    # 禁用torch初始化以节省内存
    disable_torch_init()

    # 加载模型
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.eval().cuda()

    # 读取原始数据集
    with open(os.path.expanduser(args.input_file), "r", encoding="utf-8") as f:
        data = json.load(f)

    # 如果需要处理数据集的一部分，可以使用分块
    if args.num_chunks > 1:
        total_chunks = split_list(data, args.num_chunks)
        data = total_chunks[args.chunk_idx]

    # 遍历数据集并更新回答
    for entry in tqdm(data, desc="Updating dataset"):
        image_path = os.path.join(args.image_folder, entry["image"])

        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}. 跳过此条目.")
            continue

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"无法打开图片 {image_path}: {e}. 跳过此条目.")
            continue

        image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().cuda()

        # 假设每个条目中的conversations按顺序成对出现
        conversations = entry.get("conversations", [])
        if len(conversations) % 2 != 0:
            print(f"条目 {entry['id']} 的对话数量不匹配. 跳过此条目.")
            continue

        for i in range(0, len(conversations), 2):
            gpt_msg = conversations[i]
            human_msg = conversations[i + 1]

            if gpt_msg["from"] != "gpt" or human_msg["from"] != "human":
                print(f"条目 {entry['id']} 中的对话格式不符合预期. 跳过此对话.")
                continue

            gpt_question = gpt_msg["value"]

            # 准备提示语
            qs = gpt_question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()

            # 生成回答
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True
                )

            generated_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # 更新human的回答
            human_msg["value"] = generated_answer

    # 保存更新后的数据集
    with open(os.path.expanduser(args.output_file), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"新的数据集已保存到 {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用模型更新数据集中human的回答")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--model-base", type=str, default=None, help="模型基准")
    parser.add_argument("--image-folder", type=str, required=True, help="图片文件夹路径")
    parser.add_argument("--input-file", type=str, required=True, help="输入的JSON数据集文件")
    parser.add_argument("--output-file", type=str, default="llava_v1_5_mix665k-random-100-oldmodel.json",
                        help="输出的JSON数据集文件")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="对话模式")
    parser.add_argument("--num-chunks", type=int, default=1, help="数据集分块数量")
    parser.add_argument("--chunk-idx", type=int, default=0, help="处理的数据块索引")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p采样")
    parser.add_argument("--num_beams", type=int, default=1, help="束搜索的数量")
    args = parser.parse_args()

    update_dataset(args)
