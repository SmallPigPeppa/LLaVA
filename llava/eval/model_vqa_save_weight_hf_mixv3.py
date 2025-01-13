import argparse
import torch
import os
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from transformers import Trainer, TrainingArguments

def load_and_mix_models(model_path_a, model_path_b, mix_ratio=0.5):
    # Disable torch initialization to avoid unnecessary operations
    disable_torch_init()

    # Load model A and save its state_dict to CPU
    model_name_a = os.path.basename(model_path_a)
    tokenizer_a, model_a, _, _ = load_pretrained_model(model_path=model_path_a, model_base=None, model_name=model_name_a)
    state_dict_a = {name: param.cpu() for name, param in model_a.state_dict().items()}
    del model_a  # Free GPU memory
    torch.cuda.empty_cache()

    # Load model B and save its state_dict to CPU
    model_name_b = os.path.basename(model_path_b)
    tokenizer_b, model_b, _, _ = load_pretrained_model(model_path=model_path_b, model_base=None, model_name=model_name_b)
    state_dict_b = {name: param.cpu() for name, param in model_b.state_dict().items()}
    del model_b  # Free GPU memory
    torch.cuda.empty_cache()

    # Rebuild model A to mix parameters
    tokenizer, model_a, _, _ = load_pretrained_model(model_path=model_path_a, model_base=None, model_name=model_name_a)
    for name, param in model_a.named_parameters():
        if name in state_dict_b:
            # Mix the parameters from state_dict_a and state_dict_b
            param.data = ((1 - mix_ratio) * state_dict_a[name] + mix_ratio * state_dict_b[name]).to(param.device)

    # Return the mixed model and tokenizer from model A
    return tokenizer_a, model_a

def save_mixed_model(args, mix_ratio=0.5):
    # Load and mix the models
    tokenizer, mixed_model = load_and_mix_models(args.model_path_a, args.model_path_b, mix_ratio)

    # Set up training arguments (used for saving only, no training or evaluation)
    training_args = TrainingArguments(
        output_dir=os.path.expanduser(args.save_path),
        do_train=False,
        do_eval=False,
        logging_dir=None,
        save_strategy="no",
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
