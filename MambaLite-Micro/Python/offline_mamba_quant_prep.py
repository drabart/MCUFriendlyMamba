import argparse
import math
import os
import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.tiny_mamba_har import TinyMambaHAR
from utils.data import load_har_data


def hadamard_matrix(size: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if size < 1:
        raise ValueError(f"Hadamard size must be positive, got {size}")
    if size & (size - 1):
        raise ValueError(f"Hadamard matrix requires a power-of-two size, got {size}")
    if size == 1:
        return torch.ones((1, 1), device=device, dtype=dtype)

    # Recursive step
    half = hadamard_matrix(size // 2, device=device, dtype=dtype)
    
    # Construction: 1/sqrt(2) * [[H, H], [H, -H]]
    # This maintains orthogonality at every recursive step
    scale = 1.0 / math.sqrt(2)
    top = torch.cat([half, half], dim=1)
    bottom = torch.cat([half, -half], dim=1)
    
    return torch.cat([top, bottom], dim=0) * scale


def klt_rotation(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim != 2:
        samples = samples.reshape(-1, samples.shape[-1])

    if samples.shape[0] < 2:
        return torch.eye(samples.shape[-1], dtype=samples.dtype)

    centered = samples - samples.mean(dim=0, keepdim=True)
    cov = centered.T @ centered
    cov = cov / float(centered.shape[0] - 1)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    rotation = eigvecs[:, order]

    dominant = rotation.abs().argmax(dim=0)
    signs = torch.sign(rotation[dominant, torch.arange(rotation.shape[1])])
    signs[signs == 0] = 1
    rotation = rotation * signs
    return rotation

HOOKED_LAYERS = [
    "mamba.state_proj",
    "classifier",
]


def collect_layer_inputs(
    model: nn.Module,
    loader: DataLoader,
    max_batches: int,
) -> Dict[str, List[torch.Tensor]]: 
    collected: Dict[str, List[torch.Tensor]] = {name: [] for name in HOOKED_LAYERS}
    hooks = []

    for name, module in model.named_modules():
        if name not in collected:
            continue

        def pre_hook(_module, inputs, layer_name=name):
            if not inputs:
                return
            x = inputs[0].detach().cpu()
            if x.ndim < 2:
                x = x.reshape(1, -1)
            else:
                x = x.reshape(-1, x.shape[-1])
            collected[layer_name].append(x)

        hooks.append(module.register_forward_pre_hook(pre_hook))

    model.eval()
    with torch.no_grad():
        for batch_index, (xb, _) in enumerate(loader):
            if batch_index >= max_batches:
                break
            model(xb)

    for hook in hooks:
        hook.remove()

    return collected

def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    current = model
    for part in module_name.split("."):
        current = getattr(current, part)
    return current

def offline_prepare_rotations(
    input_model: nn.Module,
    data: Tuple[torch.Tensor, torch.Tensor],
    max_batches: int,
) -> nn.Module:
    loader = DataLoader(TensorDataset(data[0], data[1]), batch_size=64, shuffle=False)

    activations = collect_layer_inputs(input_model, loader, max_batches=max_batches)
    new_model = copy.deepcopy(input_model)

    for layer_name in HOOKED_LAYERS:
        samples = activations[layer_name]
        if not samples:
            print(f"⚠️ No samples collected for layer {layer_name}, skipping rotation")
            continue

        sample_tensor = torch.cat(samples, dim=0)
        klt = klt_rotation(sample_tensor)
        hadamard = hadamard_matrix(klt.shape[0], device=klt.device, dtype=klt.dtype)

        hk = klt @ hadamard
        hkt = hk.T

        print(klt.shape, len(samples), samples[0].shape, sample_tensor.shape)

        if layer_name == "mamba.state_proj":
            module = get_module_by_name(new_model, "linear_in")
            with torch.no_grad():
                module.weight.copy_((module.weight.T @ hkt).T)
                module.bias.copy_(module.bias @ hkt)

            module = get_module_by_name(new_model, "mamba.state_proj")
            with torch.no_grad():
                module.weight.copy_((hk @ module.weight.T).T)
            module = get_module_by_name(new_model, "mamba.gate_proj")
            with torch.no_grad():
                module.weight.copy_((hk @ module.weight.T).T)

        elif layer_name == "classifier":
            module = get_module_by_name(new_model, "mamba.out_proj")
            with torch.no_grad():
                module.weight.copy_((module.weight.T @ hkt).T)
                module.bias.copy_(module.bias @ hkt)

            module = get_module_by_name(new_model, "classifier")
            with torch.no_grad():
                module.weight.copy_((hk @ module.weight.T).T)

    return new_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Hadamard/KLT prep for MambaLite-Micro")
    parser.add_argument("--checkpoint", type=str, default="./Python/models/mamba_har_model.sd.pt")
    parser.add_argument("--data-dir", type=str, default="../UCI HAR Dataset")
    parser.add_argument("--rotated-checkpoint-out", type=str, default="./Python/models/mamba_har_model_rotated.sd.pt")
    parser.add_argument("--channel-width", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--max-batches", type=int, default=8)

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    input_model = TinyMambaHAR(input_dim=57, mamba_channel_width=args.channel_width, num_classes=args.num_classes).cpu().eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    input_model.load_state_dict(state)

    x_train, y_train, _, _ = load_har_data(args.data_dir)

    model = offline_prepare_rotations(
        input_model=input_model,
        data=(x_train, y_train),
        max_batches=args.max_batches,
    )

    torch.save(model.state_dict(), args.rotated_checkpoint_out)
    print("Saved rotated checkpoint to:", args.rotated_checkpoint_out)

if __name__ == "__main__":
    # print(hadamard_matrix(16) @ (hadamard_matrix(16).T))  # Should be close to identity
    main()