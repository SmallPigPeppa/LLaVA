import argparse
import torch
import os
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from transformers import Trainer, TrainingArguments
from tqdm.auto import tqdm


def filter_delta_old(delta, retain_ratio=0.9):
    """
    对 delta 参数进行特征值分解，并过滤特征值，仅保留累积贡献达到 retain_ratio 的部分。
    """
    # Flatten delta to a 2D matrix for SVD
    original_shape = delta.shape
    flat_delta = delta.view(-1, delta.size(-1))

    # Perform SVD (Singular Value Decomposition)
    U, S, V = torch.svd(flat_delta)
    total_variance = S.sum().item()

    # Retain the largest singular values until the retain_ratio is met
    cumulative_variance = 0
    selected_indices = []
    for i, singular_value in enumerate(S):
        cumulative_variance += singular_value.item()
        selected_indices.append(i)
        if cumulative_variance / total_variance >= retain_ratio:
            break

    # Zero out remaining singular values
    filtered_S = torch.zeros_like(S)
    for i in selected_indices:
        filtered_S[i] = S[i]

    # Reconstruct filtered delta using updated singular values
    filtered_delta = torch.mm(torch.mm(U, torch.diag(filtered_S)), V.T)

    # Reshape back to the original shape of delta
    return filtered_delta.view(*original_shape)

# def filter_delta(delta, retain_ratio=0.9):
#     """
#     对 delta 参数进行特征值分解，并过滤特征值，仅保留累积贡献达到 retain_ratio 的部分。
#     如果 total_variance 为 0，则直接返回原始 delta。
#     """
#     # Flatten delta to 2D matrix
#     original_shape = delta.shape
#     flat_delta = delta.view(-1, delta.size(-1))  # Reshape to (m, n)
#
#     # Perform SVD
#     flat_delta = flat_delta.to(torch.float32)
#     # if torch.cuda.is_available():
#     #     flat_delta = flat_delta.to('cuda')  # 将数据移动到 GPU
#
#     U, S, Vh = torch.linalg.svd(flat_delta, full_matrices=False)
#
#     # Compute total variance
#     total_variance = S.sum().item()
#
#     # If total variance is 0, return original delta
#     if total_variance == 0:
#         print("Total variance is 0. Returning original delta.")
#         return delta
#
#     # Filter singular values to retain 90% variance
#     cumulative_variance = 0
#     selected_indices = []
#     for i, singular_value in enumerate(S):
#         cumulative_variance += singular_value.item()
#         selected_indices.append(i)
#         if cumulative_variance / total_variance >= retain_ratio:
#             break
#
#     # Zero out unselected singular values
#     filtered_S = torch.zeros_like(S)
#     for i in selected_indices:
#         filtered_S[i] = S[i]
#
#     # Reconstruct filtered delta
#     filtered_delta = (U @ torch.diag(filtered_S) @ Vh)
#
#     # Reshape back to the original shape
#     return filtered_delta.view(*original_shape)


import torch

def filter_delta(delta, retain_ratio=0.9):
    """
    对 delta 参数进行特征值分解，并过滤特征值，仅保留累积贡献达到 retain_ratio 的部分。
    如果 total_variance 为 0，则直接返回原始 delta。
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

    # Filter singular values to retain specified variance
    cumulative_variance = 0
    selected_indices = []
    for i, singular_value in enumerate(S):
        cumulative_variance += singular_value.item()
        selected_indices.append(i)
        if cumulative_variance / total_variance >= retain_ratio:
            break

    # Zero out unselected singular values
    filtered_S = torch.zeros_like(S)
    for i in selected_indices:
        filtered_S[i] = S[i]

    # Reconstruct filtered delta
    filtered_delta = (U @ torch.diag(filtered_S) @ Vh)

    # Move result back to CPU and clear GPU memory
    filtered_delta = filtered_delta.to('cpu')
    del U, S, Vh, flat_delta  # 清理中间变量
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
                filtered_delta = filter_delta(delta, retain_ratio=retain_ratio)
                param.data = (state_dict_a[name] + mix_ratio * filtered_delta).to(param.device)
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
    args = parser.parse_args()

    save_mixed_model(args, args.mix_ratio, args.retain_ratio)
