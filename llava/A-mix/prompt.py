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
