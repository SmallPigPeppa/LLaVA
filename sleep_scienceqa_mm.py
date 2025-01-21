import json

def filter_and_save_json(input_path, output_path):
    # Load the original JSON data from the file
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Filter out entries that do not contain an "image" key
    filtered_data = [entry for entry in data if "image" in entry]

    # Save the filtered data to a new JSON file
    with open(output_path, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=4)

# Define the input and output file paths
input_path = "playground/data/fine-tune/ScienceQA/train.json"
output_path = "playground/data/fine-tune/ScienceQA/train-mm.json"

# Call the function to perform the filtering and saving
filter_and_save_json(input_path, output_path)
