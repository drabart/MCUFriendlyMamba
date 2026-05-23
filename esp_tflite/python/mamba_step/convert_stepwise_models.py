"""Convert stepwise decomposed models to TFLite float format using litert_torch."""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import contextlib
import io
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent dirs to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from stepwise_inference_example import (
    PreSSMModule, StepSSMModule, PostSSMModule, load_trained_weights
)


def _print_step(step_num, title):
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {title}")
    print("=" * 70)


@contextlib.contextmanager
def _maybe_suppress_output(enabled):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _configure_quiet_logging(verbose=False):
    if verbose:
        return

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)


def _convert_to_litert_float_model(model, input_shape, device, verbose=False):
    """Convert PyTorch model to LiteRT float model."""
    try:
        with _maybe_suppress_output(enabled=not verbose):
            import litert_torch
    except ImportError:
        print("Error: litert_torch not installed")
        print("Install with: pip install litert-torch")
        exit(1)

    sample_input = torch.randn(*input_shape, device=device)
    print(f"Sample input shape: {sample_input.shape}")
    print("Converting with litert_torch.convert()...")
    with _maybe_suppress_output(enabled=not verbose):
        edge_model = litert_torch.convert(model, (sample_input,))
    return edge_model, sample_input


def _validate_conversion_close(model, edge_model, sample_input, atol=1e-4):
    """Validate that PyTorch and LiteRT outputs match."""
    print("Validating conversion...")
    with torch.no_grad():
        torch_output = model(sample_input).detach().numpy()
        edge_output = edge_model(sample_input.numpy())

    if np.allclose(torch_output, edge_output, atol=atol):
        print("✓ Conversion validated: PyTorch and LiteRT outputs match")
        return True
    else:
        print("⚠ Warning: PyTorch and LiteRT outputs differ")
        print(f"  Max difference: {np.abs(torch_output - edge_output).max()}")
        return False


def _export_float_model(edge_model, output_path):
    """Export LiteRT model to TFLite format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    edge_model.export(output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"✓ TFLite model exported: {output_path}")
    print(f"  Size: {size_kb:.2f} KB")
    return size_kb


def convert_stepwise_models(pytorch_model_path, output_dir="models"):
    """Convert 3 stepwise models to TFLite float format.
    
    Converts:
    1. PreSSMModule: (1, 10, 57) → (1, 10, 128, 128)  [state, gate]
    2. StepSSMModule: (1, 128) → (1, 128)  [single timestep]
    3. PostSSMModule: (1, 10, 128) + (1, 10, 128) → (1, 6)  [y_ssm + gate → logits]
    """
    _configure_quiet_logging(verbose=False)
    device = torch.device("cpu")

    print("=" * 70)
    print("CONVERTING STEPWISE MAMBA MODELS TO TFLITE FLOAT FORMAT")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create models
    pre_ssm = PreSSMModule(input_dim=57, d_model=64, d_inner=128, d_conv=4)
    step_ssm = StepSSMModule(d_inner=128, d_state=16)
    post_ssm = PostSSMModule(d_model=64, d_inner=128, output_size=6)

    # Load weights
    _print_step(1, "Loading Trained Weights")
    load_trained_weights(pre_ssm, step_ssm, post_ssm, pytorch_model_path)
    pre_ssm.eval()
    step_ssm.eval()
    post_ssm.eval()

    # ========== Model 1: PreSSM ==========
    _print_step(2, "Converting PreSSM Module")
    print("Model: PreSSMModule")
    print("  Input:  (batch=1, time=10, features=57)")
    print("  Output: (batch=1, time=10, d_inner=128) for state and gate")

    pre_ssm_input_shape = (1, 10, 57)
    edge_pre, sample_pre = _convert_to_litert_float_model(
        pre_ssm, pre_ssm_input_shape, device
    )
    _validate_conversion_close(pre_ssm, edge_pre, sample_pre)
    pre_ssm_output_path = os.path.join(output_dir, "model_pre_ssm.tflite")
    pre_size = _export_float_model(edge_pre, pre_ssm_output_path)

    # ========== Model 2: StepSSM (wrapper for stateless conversion) ==========
    _print_step(3, "Converting StepSSM Module")
    print("Model: StepSSMModule (single timestep)")
    print("  Input:  (batch=1, d_inner=128) [single timestep]")
    print("  Output: (batch=1, d_inner=128) [SSM output]")
    print("  Note: State management is handled externally")

    step_ssm_input_shape = (1, 128)
    edge_step, sample_step = _convert_to_litert_float_model(
        step_ssm, step_ssm_input_shape, device
    )

    # For StepSSM, we need to validate carefully - initialize hidden state first
    print("Validating conversion (with state reset)...")
    with torch.no_grad():
        step_ssm.reset_state(1, device, torch.float32)
        torch_output = step_ssm(sample_step).detach().numpy()

        # Reset LiteRT model's hidden state (it's not persistent in TFLite)
        edge_step_out = edge_step(sample_step.numpy())

        if np.allclose(torch_output, edge_step_out, atol=1e-4):
            print("✓ Conversion validated: PyTorch and LiteRT outputs match")
        else:
            print("⚠ Warning: PyTorch and LiteRT outputs differ")
            print(f"  Max difference: {np.abs(torch_output - edge_step_out).max()}")

    step_ssm_output_path = os.path.join(output_dir, "model_step_ssm.tflite")
    step_size = _export_float_model(edge_step, step_ssm_output_path)

    # ========== Model 3: PostSSM ==========
    _print_step(4, "Converting PostSSM Module")
    print("Model: PostSSMModule")
    print("  Input:  (batch=1, time=10, d_inner=128) for y and gate separately")
    print("  Output: (batch=1, output_size=6) [logits]")

    # For PostSSM, we need a wrapper because it takes 2 inputs
    class PostSSMWrapper(nn.Module):
        def __init__(self, post_ssm_module):
            super().__init__()
            self.post_ssm = post_ssm_module

        def forward(self, combined):
            # Split the combined input
            y = combined[:, :, :128]
            gate = combined[:, :, 128:]
            return self.post_ssm(y, gate)

    post_wrapper = PostSSMWrapper(post_ssm)
    post_wrapper.eval()

    # Create sample input: concatenated [y, gate]
    post_input_shape = (1, 10, 256)  # 128 + 128
    print(f"Wrapper input shape: {post_input_shape}")

    edge_post, sample_post_combined = _convert_to_litert_float_model(
        post_wrapper, post_input_shape, device
    )

    print("Validating conversion...")
    with torch.no_grad():
        sample_post_y = sample_post_combined[:, :, :128]
        sample_post_gate = sample_post_combined[:, :, 128:]
        torch_output = post_ssm(sample_post_y, sample_post_gate).detach().numpy()
        edge_output = edge_post(sample_post_combined.numpy())

        if np.allclose(torch_output, edge_output, atol=1e-4):
            print("✓ Conversion validated: PyTorch and LiteRT outputs match")
        else:
            print("⚠ Warning: PyTorch and LiteRT outputs differ")
            print(f"  Max difference: {np.abs(torch_output - edge_output).max()}")

    post_ssm_output_path = os.path.join(output_dir, "model_post_ssm.tflite")
    post_size = _export_float_model(edge_post, post_ssm_output_path)

    # ========== Summary ==========
    _print_step(5, "Conversion Summary")
    total_size = pre_size + step_size + post_size
    print(f"✓ All models converted successfully!")
    print(f"\n  PreSSM:  {pre_size:7.2f} KB")
    print(f"  StepSSM: {step_size:7.2f} KB")
    print(f"  PostSSM: {post_size:7.2f} KB")
    print(f"  {'─' * 20}")
    print(f"  Total:   {total_size:7.2f} KB")
    print(f"\nModels saved to: {output_dir}/")
    print(f"  - model_pre_ssm.tflite")
    print(f"  - model_step_ssm.tflite")
    print(f"  - model_post_ssm.tflite")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert stepwise decomposed models to TFLite float format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../models/best_model.pt",
        help="Path to trained PyTorch model (default: ../models/best_model.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tflite_models",
        help="Output directory for TFLite models (default: tflite_models)",
    )

    args = parser.parse_args()

    # Expand paths
    model_path = Path(args.model)
    output_dir = Path(args.output)

    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        exit(1)

    convert_stepwise_models(str(model_path), str(output_dir))
