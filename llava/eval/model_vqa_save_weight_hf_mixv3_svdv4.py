import argparse
import torch
import os
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from transformers import Trainer, TrainingArguments
from tqdm.auto import tqdm


import torch

import torch

import torch

def filter_delta(delta, scale_ratio=0.3, pre_scaling_ratio=0.5):
    """
    对 delta 参数进行特征值分解，动态调整特征值。
    如果 total_variance 为 0 或无法找到满足条件的 K，则直接返回原始 delta。

    Args:
        delta (torch.Tensor): 输入的张量，形状为 (..., n)。
        scale_ratio (float): 缩放比率，默认为 0.3。

    Returns:
        torch.Tensor: 经过特征值调整后的 delta 张量，形状与输入相同。
    """
    # Flatten delta to 2D matrix
    original_shape = delta.shape
    flat_delta = delta.view(-1, delta.size(-1))  # Reshape to (m, n)

    # Move delta to GPU if available
    flat_delta = flat_delta.to(torch.float32)
    if torch.cuda.is_available():
        flat_delta = flat_delta.to('cuda')  # 将数据移动到 GPU

    # Perform SVD
    U, S, Vh = torch.linalg.svd(flat_delta, full_matrices=False)

    # Compute total variance
    total_variance = S.sum().item()

    # If total variance is 0, return original delta
    if total_variance == 0:
        print("Total variance is 0. Returning original delta.")
        return delta

    # Calculate the pre-scaling coefficient
    pre_scaling_factor = 1 - (1 - scale_ratio) * pre_scaling_ratio

    # Apply pre-scaling to singular values
    scaled_S = S * pre_scaling_factor

    # Compute the target value based on scaled singular values
    target_value = S[0] * scale_ratio
    cumulative_sum = torch.cumsum(scaled_S, dim=0)

    try:
        # Find the first K where condition is met
        K = next((i for i in range(1, len(scaled_S) + 1) if cumulative_sum[i - 1] / i <= target_value), None)
    except StopIteration:
        K = None

    # If no valid K is found, return original delta
    if K is None:
        print("No valid K found. Returning original delta.")
        return delta

    # Dynamically adjust singular values
    modified_S = torch.zeros_like(S)
    for i in range(len(S)):
        if i < len(S) - K + 1:
            # Compute dynamic average for valid range
            modified_S[i] = torch.sum(scaled_S[i:i + K]) / K
        else:
            # For the last few singular values, reduce K dynamically
            modified_S[i] = torch.sum(scaled_S[i:]) / (len(S) - i)

    # Reconstruct filtered delta
    filtered_delta = (U @ torch.diag(modified_S) @ Vh)

    # Move result back to CPU and clear GPU memory
    filtered_delta = filtered_delta.to('cpu')
    del U, S, Vh, flat_delta, scaled_S  # 清理中间变量
    torch.cuda.empty_cache()  # 释放 GPU 显存

    # Reshape back to the original shape
    return filtered_delta.view(*original_shape)





def load_and_mix_models(model_path_a, model_path_b, mix_ratio=0.5, retain_ratio=0.9):
    disable_torch_init()

    # Load model A and its state_dict
    model_name_a = os.path.basename(model_path_a)
    tokenizer_a, model_a, _, _ = load_pretrained_model(model_path=model_path_a, model_base=None,
                                                       model_name=model_name_a)
    state_dict_a = {name: param.cpu() for name, param in model_a.state_dict().items()}
    del model_a
    torch.cuda.empty_cache()

    # Load model B and its state_dict
    model_name_b = os.path.basename(model_path_b)
    tokenizer_b, model_b, _, _ = load_pretrained_model(model_path=model_path_b, model_base=None,
                                                       model_name=model_name_b)
    state_dict_b = {name: param.cpu() for name, param in model_b.state_dict().items()}
    del model_b
    torch.cuda.empty_cache()

    # Rebuild model A and perform parameter fusion with progress bar
    tokenizer, model_a, _, _ = load_pretrained_model(model_path=model_path_a, model_base=None, model_name=model_name_a)

    param_names = list(model_a.named_parameters())  # Get parameter names for progress tracking
    with tqdm(total=len(param_names), desc="Parameter Fusion Progress") as pbar:
        for name, param in param_names:
            if name in state_dict_b:
                delta = state_dict_b[name] - state_dict_a[name]
                filtered_delta = filter_delta(delta, scale_ratio=mix_ratio)
                param.data = (state_dict_a[name] + filtered_delta).to(param.device)
            pbar.update(1)  # Update progress bar after processing each parameter

    return tokenizer_a, model_a


def save_mixed_model(args, mix_ratio=0.5, retain_ratio=0.9):
    tokenizer, mixed_model = load_and_mix_models(args.model_path_a, args.model_path_b, mix_ratio, retain_ratio)

    # Set up training arguments for saving
    training_args = TrainingArguments(
        output_dir=os.path.expanduser(args.save_path),
        do_train=False,
        do_eval=False,
        logging_dir=None,
        save_strategy="no",
    )

    # Initialize Trainer and save the model
    trainer = Trainer(
        model=mixed_model,
        args=training_args,
        tokenizer=tokenizer
    )

    trainer.model.generation_config.do_sample = True
    trainer.save_model()
    print(f"Mixed model saved to {os.path.expanduser(args.save_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path-a", type=str, required=True, help="Path to pre-trained model A")
    parser.add_argument("--model-path-b", type=str, required=True, help="Path to pre-trained model B")
    parser.add_argument("--save-path", type=str, required=True, help="Directory to save the mixed model and tokenizer")
    parser.add_argument("--mix-ratio", type=float, default=0.5, help="Mixing ratio between model A and B (0.0 to 1.0)")
    parser.add_argument("--retain-ratio", type=float, default=0.9,
                        help="Ratio of variance to retain in filtered delta (0.0 to 1.0)")
    parser.add_argument("--scale-ratio", type=float, default=0.5,
                        help="Ratio of variance to retain in filtered delta (0.0 to 1.0)")
    args = parser.parse_args()

    save_mixed_model(args, args.mix_ratio, args.retain_ratio)
