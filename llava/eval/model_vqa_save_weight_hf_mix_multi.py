import argparse
import torch
import os
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from transformers import Trainer, TrainingArguments


def average_models(models, weights):
    """Averages the models based on the given weights."""
    # Ensure the weights sum to 1 for proper averaging
    weight_sum = sum(weights)
    if weight_sum != 1.0:
        weights = [w / weight_sum for w in weights]

    # Get the state_dict of the first model to initialize the averaged model
    model_state_dict = models[0].state_dict()

    # Iterate over the models and average their weights
    for model, weight in zip(models, weights):
        state_dict = model.state_dict()
        for key in model_state_dict:
            model_state_dict[key] += state_dict[key] * weight

    # Create a new model with the averaged weights
    averaged_model = models[0]
    averaged_model.load_state_dict(model_state_dict)

    return averaged_model


def save_model_with_trainer(args):
    # Disable torch initialization to avoid unnecessary operations
    disable_torch_init()

    # Load the pre-trained models from the provided paths
    model_paths = args.model_path.split(",")  # Support multiple paths by splitting at commas
    model_weights = list(map(float, args.model_weights.split(",")))  # Parse the weight ratios

    if len(model_paths) != len(model_weights):
        raise ValueError("Number of models and weights must match.")

    # Load each model
    models = []
    for model_path in model_paths:
        model_path = os.path.expanduser(model_path)
        model_name = os.path.basename(model_path)  # Extract the model name from the path
        tokenizer, model, _, _ = load_pretrained_model(model_path, args.model_base, model_name)
        models.append(model)

    # Average the models based on the provided weights
    averaged_model = average_models(models, model_weights)

    # Set up training arguments (used for saving only, no training or evaluation)
    training_args = TrainingArguments(
        output_dir=os.path.expanduser(args.save_path),  # Directory to save the model and tokenizer
        do_train=False,  # No training
        do_eval=False,  # No evaluation
        logging_dir=None,  # No logging required
        save_strategy="no",  # No automatic saving during training
    )

    # Initialize the Hugging Face Trainer with the averaged model and tokenizer
    tokenizer, _, _, _ = load_pretrained_model(model_paths[0], args.model_base, os.path.basename(model_paths[0]))  # Use the tokenizer of the first model
    trainer = Trainer(
        model=averaged_model,
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
    parser.add_argument("--save-path", type=str, required=True, help="Directory to save the averaged model and tokenizer")
    args = parser.parse_args()

    save_model_with_trainer(args)
