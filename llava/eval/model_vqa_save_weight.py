import argparse
import torch
import os
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def save_model_weights(args):
    # Disable torch initialization to prevent unnecessary initialization
    disable_torch_init()

    # Load the model
    model_path = os.path.expanduser(args.model_path)
    model_name = os.path.basename(model_path)  # Get the model name from the path
    _, model, _, _ = load_pretrained_model(model_path, args.model_base, model_name)

    # Save the model weights
    save_path = os.path.expanduser(args.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--model-base", type=str, default=None, help="Base model type (optional)")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the model weights")
    args = parser.parse_args()

    save_model_weights(args)
