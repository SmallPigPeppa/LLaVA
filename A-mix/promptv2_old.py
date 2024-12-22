rule_description = """
Objective:
To improve the quality of questions (Q) from humans and answers (A) from GPT by:
1. Optimizing original Q and A using old-model responses as context.
2. Generating new Q&A pairs that extend concepts and information.

Rules for Question (Q) Improvement:
1. Integrate the old-model answer context into the question, e.g., "The old answer mentioned 'xxxxx,' is this correct?"
2. Remove any restriction like "Answer the question using a single word or phrase."
3. Ensure the updated question is clear, engaging, and provides a broader scope for better answers.

Rules for Answer (A) Improvement:
1. Combine the old-model and GPT answers to generate a more accurate and contextually relevant response.
2. Provide enriched, detailed answers that add depth and clarity.

New Q&A Pair Requirements:
1. **Q1**: Summarize the differences between the updated and old answers.
   - **A1**: Provide a detailed comparison, highlighting more than 5 differences, with at least 400 words.
2. **Q2**: Generate a description for the image based on both updated and old answers.
   - **A2**: Write a comprehensive response, using more than 400 words.
3. Add two random questions derived from the image or context.
   - **A**: Provide thoughtful, detailed answers of at least 400 words.

Output Format:
- Return a structured JSON file.
- Include the old-model answer context in questions where relevant.
- Do not include standalone old-model answers in the output.


Demo input:

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
  }
]
}



Demo output：

{
"id": "140031996X",
"image": "ocr_vqa/images/140031996X.jpg",
"conversations": [
  {
    "from": "human",
    "value": "<image>\nWho is the author of this book? The old answer mentioned 'Jackson brown,' is this correct or does it need clarification?"
  },
  {
    "from": "gpt",
    "value": "The author of this book is H. Jackson Brown Jr., a writer known for his inspirational and motivational works. The old answer 'Jackson brown' is partially correct but omits the full name and title."
  },
  {
    "from": "human",
    "value": "What is the title of this book? The old answer is 'Life's little instruction book,' but does it capture the full title?"
  },
  {
    "from": "gpt",
    "value": "The full title of the book is 'Life's Little Instruction Book: Simple Wisdom and a Little Humor for Living a Happy and Rewarding Life.' The old answer captures the core title but leaves out the important subtitle, which adds significant context."
  },
  {
    "from": "human",
    "value": "What is the genre of this book? The old answer is missing, but based on the content, could it fit multiple genres?"
  },
  {
    "from": "gpt",
    "value": "The genre of this book is a blend of Self-Help and Humor & Entertainment. It combines practical advice for personal growth with lighthearted and engaging insights, making it versatile in appeal."
  },
  {
    "from": "human",
    "value": "Summarize the differences between the updated and old answers."
  },
  {
    "from": "gpt",
    "value": "xxxx"
  },
  {
    "from": "human",
    "value": "Generate a description for the image based on updated and previous answers."
  },
  {
    "from": "gpt",
    "value": “xxxx”
  },
  {
    "from": "human",
    "value": "xxxxx(A random question1) "
  },
  {
    "from": "gpt",
    "value": "xxxxx"
  },
  {
    "from": "human",
    "value": "xxxxx(A random question2)"
  },
  {
    "from": "gpt",
    "value": “xxxx.”
  }
]
}
"""

