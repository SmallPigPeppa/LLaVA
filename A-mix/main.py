import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from prompt import rule_description
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(description="Process JSON files with OpenAI API.")
    parser.add_argument("--base_url", type=str, default="https://api.ppinfra.com/v3/openai",
                        help="OpenAI API base URL")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key for OpenAI")
    parser.add_argument("--model", type=str, default="qwen/qwen2.5-32b-instruct",
                        help="Model identifier")
    parser.add_argument("--stream", action='store_true',
                        help="Enable streaming mode")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSON file path")
    parser.add_argument("--output_file", type=str, default="output_improved.json",
                        help="Output JSON file path")
    parser.add_argument("--max_workers", type=int, default=3,
                        help="Number of workers for parallel processing")
    return parser.parse_args()


def load_data(input_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist. Please check the path.")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("The root element of the JSON is not a list. Please adjust accordingly.")
    return data


def extract_json_from_text(response_text: str):
    start_index = response_text.find('{')
    end_index = response_text.rfind('}')
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None
    possible_json = response_text[start_index:end_index + 1]
    try:
        return json.loads(possible_json)
    except json.JSONDecodeError:
        return None


def process_item(client, item, args, retry_count=1):
    messages = [
        {"role": "system",
         "content": "You are a professional AI assistant. Please improve the conversation based on the following rules: " + rule_description},
        {"role": "user",
         "content": "Here is the original conversation data:\n" + json.dumps(item["conversations"], ensure_ascii=False,
                                                                             indent=2) + "\n\nPlease return the improved JSON result."}
    ]
    import pdb;pdb.set_trace()
    attempt = 0
    last_error = None
    while attempt < retry_count:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=args.stream,
                max_tokens=args.max_tokens
            )
            response_text = response.choices[0].message.content if not args.stream else "".join(
                [chunk.choices[0].delta["content"] for chunk in response])
            parsed_json = extract_json_from_text(response_text)
            if parsed_json is not None:
                return parsed_json
            else:
                attempt += 1
        except Exception as e:
            attempt += 1
            last_error = e
    # If all retries fail, log the error but do not append to improved_data
    print(f"Failed to process item with ID {item.get('id')}: {last_error}, after {retry_count} attempts.")
    return None  # Return None to signify failure


import threading

def process_data(data, args):
    """
    改进后的 process_data 函数示例：
      1. 如果 output_file 已经存在，则读取其中的内容并存储到 improved_data。
      2. 生成 existing_ids 集合，用于跳过重复处理。
      3. 每获得一次结果，就写回 output_file。
    """

    # 1. 如果输出文件已存在，则读取已处理的数据
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            try:
                improved_data = json.load(f)
            except json.JSONDecodeError:
                improved_data = []
    else:
        improved_data = []

    # 2. 根据已经存在的 improved_data，生成已处理过的ID集合
    existing_ids = set()
    for d in improved_data:
        if "id" in d:
            existing_ids.add(d["id"])

    # 初始化客户端和锁
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )
    lock = threading.Lock()

    # 使用线程池进行处理
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_item = {}

        # 3. 只有当 item["id"] 不在 existing_ids 中时，才提交到线程池进行处理
        for item in data:
            if item["id"] not in existing_ids:
                future = executor.submit(process_item, client, item, args)
                future_to_item[future] = item

        # 4. 处理完每个 future 后，及时写回到 output_file
        for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Processing"):
            result = future.result()
            if result is not None:
                with lock:
                    # 找到对应的原始项
                    item = future_to_item[future]

                    # 直接用 result 替换原始的 conversations
                    item["conversations"] = result.get("conversations", item["conversations"])  # 更新 conversations

                    # 添加到 improved_data，并更新 existing_ids
                    improved_data.append(item)
                    existing_ids.add(item["id"])

                    # 立即写回到文件，保证多线程环境下数据及时落盘
                    with open(args.output_file, "w", encoding="utf-8") as f:
                        json.dump(improved_data, f, ensure_ascii=False, indent=2)

    return improved_data



def save_data(output_file, data):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the data to the file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nProcessing completed. Results have been written to: {output_file}")


def main():
    args = parse_args()
    data = load_data(args.input_file)
    improved_data = process_data(data, args)
    # save_data(args.output_file, improved_data)


if __name__ == "__main__":
    main()
