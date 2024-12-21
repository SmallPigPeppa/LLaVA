from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://api.ppinfra.com/v3/openai",
    api_key='dcf59e46-a998-4a69-adfd-31aeabac1760',
)

# Model configuration
# model = "qwen/qwen-2.5-72b-instruct"
model = "qwen/qwen2.5-32b-instruct"
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
rule_description = """
Rule Description:

    Objective: Improve the quality of questions (Q) from humans and answers (A) from GPT.
        Optimize original Q and A using old-model responses as context.
    Q Improvement:
        Add context from old-model behind the every original question, e.g., "the old answer is 'xxxxx'.
    A Improvement:
        Combine old-model and GPT answers to generate a more accurate and contextually relevant answer.
    New Q&A Pairs:
        Q1: Summarize the differences between the updated and old answers, identifying more than three differences.
            A1: Provide a concise, complete response.
        Q2: Generate description for the image base on updated and previous answers, respectively. evaluate relevance between them with a score (0–10).
            A2: Provide a concise, complete response.
    Output Format:
        Return improved JSON data, Remove old-model answers.

demo input

  {
    "id": "140031996X",
    "image": "ocr_vqa/images/140031996X.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWho is the author of this book?\nAnswer the question using a single word or phrase."
      },
      {
        "from": "gpt",
        "value": "H. Jackson Brown"
      },
      {
        "from": "old-model",
        "value": "Jackson brown"
      },
      {
        "from": "human",
        "value": "What is the title of this book?"
      },
      {
        "from": "gpt",
        "value": "Life's Little Instruction Book: Simple Wisdom and a Little Humor for Living a Happy and Rewarding Life"
      },
      {
        "from": "old-model",
        "value": "Life's little instruction book"
      },
    ]
  },
}


demo output：

{
  "id": "140031996X",
  "image": "ocr_vqa/images/140031996X.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWho is the author of this book?, Old answer is 'Jackson brown'. \nAnswer the question using a single word or phrase."
    },
    {
      "from": "gpt",
      "value": "H. Jackson Brown"
    },
    {
      "from": "human",
      "value": "What is the title of this book? Old answer is 'Life's little instruction book'."
    },
    {
      "from": "gpt",
      "value": "Life's Little Instruction Book: Simple Wisdom and a Little Humor for Living a Happy and Rewarding Life"
    },
    {
      "from": "human",
      "value": "Summarize the differences between the updated and old answers, identifying more than three differences."
    },
    {
      "from": "gpt",
      "value": “xxxxx.”
    },
    {
      "from": "human",
      "value": "Generate description for the image based on the updated and old answers, and evaluate relevance between them with a score (0–10)."
    },
    {
      "from": "gpt",
      "value": "xxxxx"
    }
  ]
}
"""

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
