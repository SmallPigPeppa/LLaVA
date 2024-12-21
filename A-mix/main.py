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


def process_item(client, item, args, retry_count=2):
    messages = [
        {"role": "system",
         "content": "You are a professional AI assistant. Please improve the conversation based on the following rules: " + rule_description},
        {"role": "user",
         "content": "Here is the original conversation data:\n" + json.dumps(item["conversations"], ensure_ascii=False,
                                                                             indent=2) + "\n\nPlease return the improved JSON result."}
    ]
    attempt = 0
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


def process_data(data, args):
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )
    improved_data = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_item = {executor.submit(process_item, client, item, args): item for item in data}
        for future in tqdm(as_completed(future_to_item), total=len(data), desc="Processing"):
            result = future.result()
            if result is not None:  # Only append if result is not None
                improved_data.append(result)
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
    save_data(args.output_file, improved_data)


if __name__ == "__main__":
    main()
