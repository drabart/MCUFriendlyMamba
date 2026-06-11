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
import json

# Add parent dirs to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from stepwise_inference_example import (
    PreSSMModule, StepSSMModule, PostSSMModule, load_trained_weights, load_model_metadata
)


def _print_step(step_num, title):
    print(f"\n[{step_num}] {title}")


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
    with _maybe_suppress_output(enabled=not verbose):
        edge_model = litert_torch.convert(model, (sample_input,))
    return edge_model, sample_input


def _validate_conversion_close(model, edge_model, sample_input, atol=1e-4):
    """Validate that PyTorch and LiteRT outputs match (supports tuple outputs)."""
    with torch.no_grad():
        # Handle single sample input or tuple of inputs
        if isinstance(sample_input, tuple):
            torch_output = model(*sample_input)
            edge_output = edge_model(*[x.numpy() if isinstance(x, torch.Tensor) else x for x in sample_input])
        else:
            torch_output = model(sample_input)
            edge_output = edge_model(sample_input.numpy())
        
        # Convert torch output to numpy (handles tuples)
        if isinstance(torch_output, tuple):
            torch_output = tuple(t.detach().numpy() if isinstance(t, torch.Tensor) else t for t in torch_output)
            # For validation, compare each output
            if isinstance(edge_output, tuple):
                for i, (torch_out, edge_out) in enumerate(zip(torch_output, edge_output)):
                    if not np.allclose(torch_out, edge_out, atol=atol):
                        return False
            return True
        else:
            torch_output = torch_output.detach().numpy()
            return np.allclose(torch_output, edge_output, atol=atol)


def _export_float_model(edge_model, output_path):
    """Export LiteRT model to TFLite format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    edge_model.export(output_path)
    size_kb = os.path.getsize(output_path) / 1024
    return size_kb


def convert_stepwise_models(pytorch_model_path, output_dir="models"):
    """Convert 3 stepwise models to TFLite float format.
    
    Loads architecture from metadata.json in model directory.
    Supports both HAR and KWS datasets with different dimensions.
    """
    _configure_quiet_logging(verbose=False)
    device = torch.device("cpu")

    print("Converting stepwise models to TFLite float format...\n")

    # Extract dataset name from model filename (e.g., "best_model_kws.pt" -> "kws")
    model_filename = os.path.basename(pytorch_model_path)
    dataset = "har"  # default
    if "_kws" in model_filename:
        dataset = "kws"
    elif "_har" in model_filename:
        dataset = "har"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata to get model architecture
    metadata = load_model_metadata(pytorch_model_path, dataset)
    input_dim = metadata.get('input_dim')
    d_model = metadata.get('d_model')
    output_size = metadata.get('output_size')
    seq_len = metadata.get('input_shape')[1]  # Sequence length
    d_inner = d_model * 2  # Standard expansion
    d_state = 16
    d_conv = 4
    dt_size = 8
    
    print(f"Using model architecture from metadata:")
    print(f"  input_dim={input_dim}, d_model={d_model}, output_size={output_size}")
    print(f"  seq_len={seq_len}, d_inner={d_inner}, d_state={d_state}, d_conv={d_conv}\n")

    # Create models with correct dimensions
    pre_ssm = PreSSMModule(input_dim=input_dim, d_model=d_model, d_inner=d_inner, d_conv=d_conv)
    step_ssm = StepSSMModule(d_inner=d_inner, d_state=d_state, dt_size=dt_size)
    post_ssm = PostSSMModule(d_model=d_model, d_inner=d_inner, output_size=output_size)

    # Load weights
    _print_step(1, "Loading Trained Weights")
    load_trained_weights(pre_ssm, step_ssm, post_ssm, pytorch_model_path)
    pre_ssm.eval()
    step_ssm.eval()
    post_ssm.eval()

    # ========== Model 1: PreSSM ==========
    _print_step(2, "Converting PreSSM Module")
    pre_ssm_input_shape = (1, seq_len, input_dim)
    edge_pre, sample_pre = _convert_to_litert_float_model(
        pre_ssm, pre_ssm_input_shape, device
    )
    pre_ssm_output_path = os.path.join(output_dir, f"model_pre_ssm_{dataset}.tflite")
    pre_size = _export_float_model(edge_pre, pre_ssm_output_path)
    print(f"  ✓ PreSSM (tuple output): {pre_size:.2f} KB")

    # ========== Model 2: StepSSM (multi-input/multi-output) ==========
    _print_step(3, "Converting StepSSM Module")
    
    sample_x_t = torch.randn(1, d_inner, device=device)
    sample_hidden = torch.zeros(1, d_inner, d_state, device=device)
    
    try:
        with _maybe_suppress_output(enabled=True):
            import litert_torch
            edge_step = litert_torch.convert(step_ssm, (sample_x_t, sample_hidden))
    except Exception as e:
        print(f"  ✗ StepSSM failed: {str(e)[:100]}")
        edge_step = None

    if edge_step is not None:
        step_ssm_output_path = os.path.join(output_dir, f"model_step_ssm_{dataset}.tflite")
        step_size = _export_float_model(edge_step, step_ssm_output_path)
        print(f"  ✓ StepSSM (tuple output): {step_size:.2f} KB")
    else:
        step_size = 0

    # ========== Model 3: PostSSM (multi-input) ==========
    _print_step(4, "Converting PostSSM Module")
    
    sample_y = torch.randn(1, seq_len, d_inner, device=device)
    sample_gate = torch.randn(1, seq_len, d_inner, device=device)
    
    try:
        with _maybe_suppress_output(enabled=True):
            import litert_torch
            edge_post = litert_torch.convert(post_ssm, (sample_y, sample_gate))
    except Exception as e:
        print(f"  ✗ PostSSM failed: {str(e)[:100]}")
        edge_post = None

    if edge_post is not None:
        post_ssm_output_path = os.path.join(output_dir, f"model_post_ssm_{dataset}.tflite")
        post_size = _export_float_model(edge_post, post_ssm_output_path)
        print(f"  ✓ PostSSM: {post_size:.2f} KB")
    else:
        post_size = 0

    # ========== Summary ==========
    _print_step(5, "Conversion Complete")
    total_size = pre_size + step_size + post_size
    print(f"  Total: {total_size:.2f} KB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert stepwise decomposed models to TFLite float format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained PyTorch model (default: ../models/best_model_<dataset>.pt)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["har", "kws"],
        default="har",
        help="Dataset type for default model name (default: har)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tflite_models",
        help="Output directory for TFLite models (default: tflite_models)",
    )

    args = parser.parse_args()

    # Determine model path
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = Path(f"../models/best_model_{args.dataset}.pt")
    
    output_dir = Path(args.output)

    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        exit(1)

    convert_stepwise_models(str(model_path), str(output_dir))
