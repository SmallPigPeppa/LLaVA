from openai import OpenAI
from prompt import rule_description
# Initialize the OpenAI client
client = OpenAI(
    # base_url="https://api.ppinfra.com/v3/openai",
    # api_key='dcf59e46-a998-4a69-adfd-31aeabac1760',
    base_url="https://api.novita.ai/v3/openai",
    api_key='09604121-6e12-4e83-8e2e-277b6450b32c'
)

# Model configuration
# model = "qwen/qwen-2.5-72b-instruct"
model = "qwen/qwen2.5-32b-instruct"
model = "meta-llama/llama-3.1-8b-instruct"
# model = 'meta-llama/llama-3.2-7b-instruct'
# model = 'thudm/glm-4-9b-chat'
stream = True  # Streamed output or not
max_tokens = 2048

# Define the input JSON
input_json =    {
    "id": "1451670761",
    "image": "ocr_vqa/images/1451670761.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWho is the author of this book?\nAnswer the question using a single word or phrase."
      },
      {
        "from": "gpt",
        "value": "Dick Crouser"
      },
      {
        "from": "old-model",
        "value": "Dick crawser"
      },
      {
        "from": "human",
        "value": "What is the title of this book?"
      },
      {
        "from": "gpt",
        "value": "Funny (but true) Golf Anecdotes: about Tiger, Phil, Bubba, Rory, Rickie, Jack, Arnie, and all the rest."
      },
      {
        "from": "old-model",
        "value": "Funny but true golf anecdotes"
      },
      {
        "from": "human",
        "value": "What type of book is this?"
      },
      {
        "from": "gpt",
        "value": "Humor & Entertainment"
      },
      {
        "from": "old-model",
        "value": "Golf"
      }
    ]
  },

# Define the rule description

# Prepare the messages for the API call
messages = [
    {
        "role": "system",
        "content": f"You are a professional AI assistant that processes JSON input and improves Q&A quality based on defined rules {rule_description}."
    },
    {
        "role": "user",
        "content": f"Here is the input data: {input_json}. Please improve the json file:"
    }
]

# Call the API
chat_completion_res = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=stream,
    max_tokens=max_tokens,
)

# Process and print the response
if stream:
    for chunk in chat_completion_res:
        print(chunk.choices[0].delta.content or "", end="")
else:
    print(chat_completion_res.choices[0].message.content)
