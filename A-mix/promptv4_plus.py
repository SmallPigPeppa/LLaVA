rule_description = """
## Objective:
Improve the quality of answers (A) through the following ways:

1. The content with "from": "human" in the JSON file corresponds to Q (question), and the content with "from": "gpt" corresponds to A (answer).
2. Iterate through each QA pair in the JSON file and rewrite A (the answer) in a more flexible and varied way.
3. The modified answer (A) should be significantly different from the original in terms of sentence structure, content, and expression. The goal is to provide a response that retains the same meaning but presents it in a more varied and creative way, using different phrasing and structure.

"""

