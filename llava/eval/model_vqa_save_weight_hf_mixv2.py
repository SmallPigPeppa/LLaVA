import argparse
import torch
import os
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from transformers import Trainer, TrainingArguments

def load_and_mix_models(model_path_a, model_path_b, mix_ratio=0.5):
    # Disable torch initialization to avoid unnecessary operations
    disable_torch_init()

    # Load both models (A and B)
    model_name_a = os.path.basename(model_path_a)  # Extract the model name from the path
    model_name_b = os.path.basename(model_path_b)  # Extract the model name from the path

    tokenizer_a, model_a, _, _ = load_pretrained_model(model_path=model_path_a, model_base=None, model_name=model_name_a)
    tokenizer_b, model_b, _, _ = load_pretrained_model(model_path=model_path_b, model_base=None, model_name=model_name_b)

    # Ensure both models have the same architecture
    assert model_a.config.to_dict() == model_b.config.to_dict(), "Models must have the same architecture for mixing."

    # Mix the weights of the two models according to the specified ratio
    mixed_model = model_a
    for name, param_a in model_a.named_parameters():
        if name in model_b.state_dict():
            param_b = model_b.state_dict()[name]
            # Mix the parameters
            param_a.data = (1 - mix_ratio) * param_a.data + mix_ratio * param_b.data

    # Return the mixed model and tokenizer from model A (tokenizers are typically shared)
    return tokenizer_a, mixed_model

def save_mixed_model(args, mix_ratio=0.5):
    # Load and mix the models
    tokenizer, mixed_model = load_and_mix_models(args.model_path_a, args.model_path_b, mix_ratio)

    # Set up training arguments (used for saving only, no training or evaluation)
    training_args = TrainingArguments(
        output_dir=os.path.expanduser(args.save_path),  # Directory to save the model and tokenizer
        do_train=False,  # No training
        do_eval=False,  # No evaluation
        logging_dir=None,  # No logging required
        save_strategy="no",  # No automatic saving during training
    )

    # Initialize the Hugging Face Trainer with model and tokenizer
    trainer = Trainer(
        model=mixed_model,
        args=training_args,
        tokenizer=tokenizer
    )

    trainer.model.generation_config.do_sample = True
    # Save the mixed model and tokenizer
    trainer.save_model()
    print(f"Mixed model and tokenizer saved to {os.path.expanduser(args.save_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path-a", type=str, required=True, help="Path to the pre-trained model A")
    parser.add_argument("--model-path-b", type=str, required=True, help="Path to the pre-trained model B")
    parser.add_argument("--save-path", type=str, required=True, help="Directory to save the mixed model and tokenizer")
    parser.add_argument("--mix-ratio", type=float, default=0.5, help="Mixing ratio between model A and model B (0.0 to 1.0)")
    args = parser.parse_args()

    save_mixed_model(args, args.mix_ratio)
