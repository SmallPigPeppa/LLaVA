import argparse
import torch
import os
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from transformers import Trainer, TrainingArguments


def average_models(weights_list, model_weights):
    """Averages the models' weights based on the given weights."""
    # Ensure the weights sum to 1 for proper averaging
    weight_sum = sum(model_weights)
    if weight_sum != 1.0:
        model_weights = [w / weight_sum for w in model_weights]

    # Create a dictionary to store the averaged weights
    averaged_weights = None

    # Iterate over the list of model weights and average them
    for weights, weight in zip(weights_list, model_weights):
        if averaged_weights is None:
            averaged_weights = {key: value * weight for key, value in weights.items()}
        else:
            for key in weights:
                averaged_weights[key] += weights[key] * weight

    return averaged_weights


def save_model_with_trainer(args):
    # Disable torch initialization to avoid unnecessary operations
    disable_torch_init()

    # Load the pre-trained models from the provided paths
    model_paths = args.model_path.split(",")  # Support multiple paths by splitting at commas
    model_weights = list(map(float, args.model_weights.split(",")))  # Parse the weight ratios

    if len(model_paths) != len(model_weights):
        raise ValueError("Number of models and weights must match.")

    # Extract the weights from each model and free up memory
    model = None
    weights_list = []
    for model_path in model_paths:
        model_path = os.path.expanduser(model_path)
        model_name = os.path.basename(model_path)  # Extract the model name from the path
        tokenizer, model, _, _ = load_pretrained_model(model_path, args.model_base, model_name)

        # Extract weights from the model and move them to the CPU
        weights = model.state_dict()
        weights = {key: value.cpu() for key, value in weights.items()}
        import pdb;pdb.set_trace()# Ensure weights are on CPU
        weights_list.append(weights)

    # Average the weights based on the provided weights
    averaged_weights = average_models(weights_list, model_weights)

    # Load the averaged weights into the new model
    model.load_state_dict(averaged_weights)

    # Set up training arguments (used for saving only, no training or evaluation)
    training_args = TrainingArguments(
        output_dir=os.path.expanduser(args.save_path),  # Directory to save the model and tokenizer
        do_train=False,  # No training
        do_eval=False,  # No evaluation
        logging_dir=None,  # No logging required
        save_strategy="no",  # No automatic saving during training
    )

    # Initialize the Hugging Face Trainer with the averaged model and tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer  # Include the tokenizer to save it with the model
    )

    trainer.model.generation_config.do_sample = True
    # Save the averaged model and tokenizer
    trainer.save_model()
    print(f"Averaged model and tokenizer saved to {os.path.expanduser(args.save_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Comma-separated paths to the pre-trained models")
    parser.add_argument("--model-base", type=str, default=None, help="Base model type (optional)")
    parser.add_argument("--model-weights", type=str, required=True, help="Comma-separated weights for each model")
    parser.add_argument("--save-path", type=str, required=True,
                        help="Directory to save the averaged model and tokenizer")
    args = parser.parse_args()

    save_model_with_trainer(args)
