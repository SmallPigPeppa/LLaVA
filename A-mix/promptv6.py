rule_description = """
## Objective:
Improve the quality of questions (Q) and answers (A) through the following ways:

Generate new QA json data based on the input QA json data. The background of the new QA data's questions is based on the old model's description, and the relevant description is improved through the input image. Improve input QA JSON file as follows:
step1. Summarize the old model's QA to generate an old caption/description for the image in a smooth, and natural paragraph.
step2. At the beginning of the conversation, provide the old model's description (generated in step 1) and task background.
step3. Remove the old model's data and return the improved QA JSON file.


## Demo Input:
{
    "id": "967097339",
    "image": "ocr_vqa/images/967097339.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nWho is the author of this book?\nAnswer the question using a single word or phrase."
        },
        {
            "from": "gpt",
            "value": "Allan H. Goodman"
        },
        {
            "from": "old-model",
            "value": "Alan h goodman"
        },
        {
            "from": "human",
            "value": "What is the title of this book?"
        },
        {
            "from": "gpt",
            "value": "Basic Skills for the New Mediator, Second Edition"
        },
        {
            "from": "old-model",
            "value": "Basic skills for new mediator"
        },
        {
            "from": "human",
            "value": "What type of book is this?"
        },
        {
            "from": "gpt",
            "value": "Law"
        },
        {
            "from": "old-model",
            "value": "Self help"
        },
        {
            "from": "human",
            "value": "Is this book related to Law?"
        },
        {
            "from": "gpt",
            "value": "Yes"
        },
        {
            "from": "old-model",
            "value": "No"
        }
    ]
},

## Demo Output:
{
    "id": "967097339",
    "image": "ocr_vqa/images/967097339.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "The old model's description of this image is: "The image describes xxxx." Based on the understanding of the input image, correct/improve the relevant description."
        },
        {
            "from": "gpt",
            "value": "I am ready"
        },
        {
            "from": "human",
            "value": "<image>\nWho is the author of this book?\nAnswer the question using a single word or phrase."
        },
        {
            "from": "gpt",
            "value": "Allan H. Goodman."
        },
        {
            "from": "human",
            "value": "What is the title of this book?"
        },
        {
            "from": "gpt",
            "value": "Basic Skills for the New Mediator, Second Edition."
        },
        xxxx (remaining questions and answers)
    ]
}

"""

