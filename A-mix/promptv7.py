rule_description = """
## Objective:
Improve the quality of questions (Q) and answers (A) through the following ways:

Step 1: Rewrite every A into a smooth and natural paragraph. Remove the constrain "\nAnswer the question using a single word or phrase. in Q1"
Step 2: Generate new QA JSON data based on the input QA JSON data, randomly generate 2–4 new QA pairs related to the image (A should be between 50–100 words) and add them to the "conversations."
Step 3: Assume that the expanded conversation has 1–N QA pairs. Randomly shuffle the order of QA pairs from 2 to N in the "conversations."


"""

