rule_description = """
## Objective:

Improve the quality of questions ("from": "human") and answers ("from": "gpt") through the following steps:

Step 1: Rewrite every A into a smooth and natural paragraph. Remove the constrain "\nAnswer the question using a single word or phrase. in Q1"
Step 2: randomly generate 2~4 new questions ("from": "human") and answers ("from": "gpt") pairs related to the image (answers should be between 50–100 words) and add them to the "conversations."
Step 3:

"""

