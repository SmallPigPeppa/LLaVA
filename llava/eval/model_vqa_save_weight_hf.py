import argparse
import torch
import os
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from transformers import Trainer, TrainingArguments


def save_model_with_trainer(args):
    # Disable torch initialization to avoid unnecessary operations
    disable_torch_init()

    # Load the pre-trained model
    model_path = os.path.expanduser(args.model_path)
    model_name = os.path.basename(model_path)  # Extract the model name from the path
    _, model, _, _ = load_pretrained_model(model_path, args.model_base, model_name)

    # Set up training arguments (used for model saving only)
    training_args = TrainingArguments(
        output_dir=os.path.expanduser(args.save_path),  # Directory to save the model
        do_train=False,  # No training
        do_eval=False,  # No evaluation
        logging_dir=None,  # No logging required
        save_strategy="no",  # No automatic saving during training
    )

    # Initialize the Hugging Face Trainer without tokenizer
    trainer = Trainer(
        model=model,
        args=training_args
    )

    # Save the model weights using Trainer's built-in method
    trainer.model.generation_config.do_sample = True
    trainer.save_model()
    print(f"Model saved to {os.path.expanduser(args.save_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--model-base", type=str, default=None, help="Base model type (optional)")
    parser.add_argument("--save-path", type=str, required=True, help="Directory to save the model weights")
    args = parser.parse_args()

    save_model_with_trainer(args)
