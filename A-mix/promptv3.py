rule_description = """
## Objective:
Improve the quality of questions (Q) and answers (A) through the following ways:
1. Generate new question-answer pairs to expand concepts and add more information.

## Question (Q) Improvement Rules:
1. **Clarity and Interest**: Ensure that the updated question is not only clear but also engaging, encouraging deeper thought. The question should have a broad scope to allow for a more comprehensive and multi-dimensional answer.
2. **Broader Question Scope**: Avoid overly narrow questions. Provide more context to encourage multi-layered answers. For example, you can expand a simple question into one that covers multiple aspects, or generate detailed descriptions or captions from images.

## Answer (A) Improvement Rules:
1. **Increase Depth and Clarity**: The answer should be more detailed and comprehensive, covering multiple aspects of the question and expressing the points clearly. Avoid short or singular answers, and fully demonstrate an understanding and analysis of the question.
2. **Explanation and Background**: When applicable, provide more background information to explain the principles, history, or theory behind the answer, helping to understand complex concepts or phenomena.
3. **Examples and Connections**: Whenever possible, support the answer with examples, analogies, or related research results to strengthen its persuasiveness and reliability.

## New Question-Answer Pair Requirements:
1. **Q1**: (Generate a description or create a caption for the image, which should include the phrase “<image>\n” either at the beginning or the end of the question.)
    - For description-generation questions, ensure the answer provides a comprehensive description of the image, with at least 200 words, covering the main elements of the image and its background information.
    - For caption-generation questions, provide a concise, complete caption that accurately summarizes the content of the image while maintaining clarity.
2. **Add 2 random questions**:
    - The random questions should be related to the image or content, expanding the scope of the discussion and encouraging deep analysis.
    - For each question, provide thoughtful and in-depth answers, at least 200 words, ensuring that the answers go beyond surface-level responses and explore related concepts, background, or applications.

## Output Format:
- Return a structured JSON file containing the image description, caption, and two extended random questions with detailed answers.
- Remove previous questions and answers, and only return Q1, A1, and the two random question-answer pairs.

## Demo input:
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
    "from": "human",
    "value": "What is the title of this book?"
  },
  {
    "from": "gpt",
    "value": "Life's Little Instruction Book: Simple Wisdom and a Little Humor for Living a Happy and Rewarding Life"
  }
]
}

## Demo output:
{
"id": "140031996X",
"image": "ocr_vqa/images/140031996X.jpg",
"conversations": [
  {
    "from": "human",
    "value": "<image>\n xxxxx (question1)”
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

