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

        # 预处理图像
        image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().cuda()

        # 获取对话列表
        conversations = entry.get("conversations", [])

        # 维护对话历史
        conversation_history = []

        # 使用while循环以便在遍历过程中插入新消息
        i = 0
        while i < len(conversations):
            msg = conversations[i]
            conversation_history.append(msg)

            if msg["from"] == "gpt":
                # 构建对话历史作为提示语
                prompt = ""
                for conv_msg in conversation_history:
                    prompt += f"{conv_msg['from']}: {conv_msg['value']}\n"

                # 添加 <image> token 和相关标记
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                final_prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(final_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                                  return_tensors='pt').unsqueeze(0).cuda()

                # 生成回答
                with torch.inference_mode():
                    # import pdb; pdb.set_trace()
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

                # 创建新的"old-model"回答
                new_old_model_msg = {
                    "from": "old-model",
                    "value": generated_answer
                }

                # 将新的回答插入到当前"gpt"消息后面
                conversations.insert(i + 1, new_old_model_msg)

                # 将新的回答添加到对话历史
                conversation_history.append(new_old_model_msg)

                # 跳过新插入的消息
                i += 2
            else:
                i += 1

        # 更新样本的对话列表
        entry["conversations"] = conversations

    # 保存更新后的数据集
    with open(os.path.expanduser(args.output_file), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"新的数据集已保存到 {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用模型更新数据集中gpt的回答，并添加为old-model")
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
