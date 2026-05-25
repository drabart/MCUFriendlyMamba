"""Quantize stepwise decomposed models using ai-edge-quantizer with calibration data.

Supports both HAR and KWS datasets with automatic architecture detection from metadata.

Process:
1. PreSSM: Quantize with input calibration data (from train set)
2. StepSSM: Quantize with calibration data from PreSSM outputs
3. PostSSM: Quantize with calibration data from PreSSM+StepSSM outputs
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import contextlib
import logging
import warnings
import numpy as np
import torch
from pathlib import Path
import sys
import json

# Add parent dirs to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from stepwise_inference_example import (
    PreSSMModule, StepSSMModule, PostSSMModule, load_trained_weights, load_model_metadata
)
from data import load_har_data, load_speechcommands_data


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


def _build_tflite_interpreter(model_path, verbose=False):
    """Build TFLite interpreter from model path."""
    with _maybe_suppress_output(enabled=not verbose):
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    return interpreter


def _quantize_input_if_needed(x_np, input_details):
    input_dtype = input_details["dtype"]
    if input_dtype == np.float32:
        return x_np.astype(np.float32)

    scale, zero_point = input_details["quantization"]
    if scale == 0:
        raise ValueError("Input quantization scale is zero; cannot quantize input.")

    q = np.round(x_np / scale + zero_point)
    if input_dtype == np.int8:
        q = np.clip(q, -128, 127)
    elif input_dtype == np.uint8:
        q = np.clip(q, 0, 255)
    return q.astype(input_dtype)


def _dequantize_output_if_needed(y_np, output_details):
    output_dtype = output_details["dtype"]
    if output_dtype == np.float32:
        return y_np.astype(np.float32)

    scale, zero_point = output_details["quantization"]
    if scale == 0:
        return y_np.astype(np.float32)
    return (y_np.astype(np.float32) - zero_point) * scale


def _evaluate_quantized_split_accuracy(pre_ssm_path, step_ssm_path, post_ssm_path, test_ds, verbose=False):
    """Run the three quantized split models end-to-end and report test accuracy."""
    pre_ssm_interpreter = _build_tflite_interpreter(pre_ssm_path, verbose=verbose)
    step_ssm_interpreter = _build_tflite_interpreter(step_ssm_path, verbose=verbose)
    post_ssm_interpreter = _build_tflite_interpreter(post_ssm_path, verbose=verbose)

    pre_input_details = pre_ssm_interpreter.get_input_details()[0]
    pre_output_details = pre_ssm_interpreter.get_output_details()
    step_input_details = step_ssm_interpreter.get_input_details()
    step_output_details = step_ssm_interpreter.get_output_details()
    post_input_details = post_ssm_interpreter.get_input_details()
    post_output_details = post_ssm_interpreter.get_output_details()[0]

    correct = 0
    total = 0

    for data, target in test_ds:
        x_np = data.unsqueeze(0).numpy().astype(np.float32)

        pre_input = _quantize_input_if_needed(x_np, pre_input_details)
        pre_ssm_interpreter.set_tensor(pre_input_details["index"], pre_input)
        pre_ssm_interpreter.invoke()

        pre_state = _dequantize_output_if_needed(
            pre_ssm_interpreter.get_tensor(pre_output_details[0]["index"]),
            pre_output_details[0],
        )
        pre_gate = _dequantize_output_if_needed(
            pre_ssm_interpreter.get_tensor(pre_output_details[1]["index"]),
            pre_output_details[1],
        )

        hidden_state = np.zeros((1, 128, 16), dtype=np.float32)
        seq_len = pre_state.shape[1]  # Get actual sequence length (10 for HAR, 51 for KWS)
        y_all = np.zeros((1, seq_len, 128), dtype=np.float32)

        for timestep in range(pre_state.shape[1]):
            x_t = pre_state[:, timestep, :].astype(np.float32)

            step_input_0 = _quantize_input_if_needed(x_t, step_input_details[0])
            step_input_1 = _quantize_input_if_needed(hidden_state, step_input_details[1])
            step_ssm_interpreter.set_tensor(step_input_details[0]["index"], step_input_0)
            step_ssm_interpreter.set_tensor(step_input_details[1]["index"], step_input_1)
            step_ssm_interpreter.invoke()

            y_t = _dequantize_output_if_needed(
                step_ssm_interpreter.get_tensor(step_output_details[0]["index"]),
                step_output_details[0],
            )
            hidden_state = _dequantize_output_if_needed(
                step_ssm_interpreter.get_tensor(step_output_details[1]["index"]),
                step_output_details[1],
            )

            if y_t.ndim == 1:
                y_all[:, timestep, :] = y_t[np.newaxis, :]
            else:
                y_all[:, timestep, :] = y_t

        post_input_0 = _quantize_input_if_needed(y_all, post_input_details[0])
        post_input_1 = _quantize_input_if_needed(pre_gate.astype(np.float32), post_input_details[1])
        post_ssm_interpreter.set_tensor(post_input_details[0]["index"], post_input_0)
        post_ssm_interpreter.set_tensor(post_input_details[1]["index"], post_input_1)
        post_ssm_interpreter.invoke()

        logits = _dequantize_output_if_needed(
            post_ssm_interpreter.get_tensor(post_output_details["index"]),
            post_output_details,
        )
        pred = int(np.argmax(logits, axis=-1)[0])
        # Handle both tensor (HAR) and int (KWS) labels
        target_val = int(target.item()) if hasattr(target, 'item') else int(target)
        correct += int(pred == target_val)
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"✓ Quantized split-model test accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    return accuracy


def _get_preSsm_calibration_data(train_ds, num_samples=2000):
    """Get calibration data for PreSSM (raw input samples).
    
    Returns:
        Dict with signature key mapping to list of calibration samples
    """
    from ai_edge_quantizer.utils import tfl_interpreter_utils
    from torch.utils.data import DataLoader

    calibration_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    calibration_samples = []
    count = 0

    for data, _ in calibration_loader:
        if count >= num_samples:
            break
        sample_data = data.numpy().astype(np.float32)
        calibration_samples.append({"args_0": sample_data})
        count += 1

    print(f"✓ Loaded {count} calibration samples for PreSSM")
    return {tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples}


def _get_stepSsm_calibration_data(pre_ssm_interpreter, train_ds, num_samples=2000):
    """Get calibration data for StepSSM by running PreSSM on train data.
    
    StepSSM takes:
    - x_t: (1, 128) - each timestep from PreSSM output
    - hidden: (1, 128, 16) - hidden state (initialized to zero)
    
    Returns:
        Dict with signature key mapping to list of calibration samples
    """
    from ai_edge_quantizer.utils import tfl_interpreter_utils
    from torch.utils.data import DataLoader

    calibration_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    
    input_details = pre_ssm_interpreter.get_input_details()
    output_details = pre_ssm_interpreter.get_output_details()
    
    # Find state and gate outputs
    state_idx = None
    gate_idx = None
    for i, detail in enumerate(output_details):
        if "state" in detail["name"].lower():
            state_idx = i
        elif "gate" in detail["name"].lower():
            gate_idx = i
    
    # Fallback: assume first output is state, second is gate
    if state_idx is None:
        state_idx = 0
    if gate_idx is None:
        gate_idx = 1 if len(output_details) > 1 else 0

    calibration_samples = []
    count = 0

    for data, _ in calibration_loader:
        if count >= num_samples:
            break

        input_data = data.numpy().astype(np.float32)
        
        # Run PreSSM to get intermediate outputs
        pre_ssm_interpreter.set_tensor(input_details[0]["index"], input_data)
        pre_ssm_interpreter.invoke()
        
        state_output = pre_ssm_interpreter.get_tensor(output_details[state_idx]["index"])
        gate_output = pre_ssm_interpreter.get_tensor(output_details[gate_idx]["index"])
        
        # state_output shape: (1, 10, 128)
        # Extract each timestep as calibration sample for StepSSM
        for t in range(state_output.shape[1]):  # 10 timesteps
            x_t = state_output[:, t, :].astype(np.float32)  # (1, 128)
            hidden = np.zeros((1, 128, 16), dtype=np.float32)  # (1, 128, 16)
            
            calibration_samples.append({
                "args_0": x_t,
                "args_1": hidden
            })
            
            if len(calibration_samples) >= num_samples * 10:
                break
        
        count += 1

    # Keep only num_samples worth
    calibration_samples = calibration_samples[:num_samples * 10]
    
    print(f"✓ Generated {len(calibration_samples)} calibration samples for StepSSM")
    return {tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples}


def _get_postSsm_calibration_data(pre_ssm_interpreter, step_ssm_interpreter, 
                                   train_ds, num_samples=2000):
    """Get calibration data for PostSSM by running PreSSM + StepSSM.
    
    PostSSM takes:
    - y: (1, 10, 128) - output from running StepSSM for all timesteps
    - gate: (1, 10, 128) - gate from PreSSM
    
    Returns:
        Dict with signature key mapping to list of calibration samples
    """
    from ai_edge_quantizer.utils import tfl_interpreter_utils
    from torch.utils.data import DataLoader

    calibration_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    
    pre_input_details = pre_ssm_interpreter.get_input_details()
    pre_output_details = pre_ssm_interpreter.get_output_details()
    
    step_input_details = step_ssm_interpreter.get_input_details()
    step_output_details = step_ssm_interpreter.get_output_details()
    
    # Find PreSSM state and gate outputs
    pre_state_idx = None
    pre_gate_idx = None
    for i, detail in enumerate(pre_output_details):
        if "state" in detail["name"].lower():
            pre_state_idx = i
        elif "gate" in detail["name"].lower():
            pre_gate_idx = i
    if pre_state_idx is None:
        pre_state_idx = 0
    if pre_gate_idx is None:
        pre_gate_idx = 1 if len(pre_output_details) > 1 else 0
    
    # Find StepSSM y_t and hidden_new outputs - check shapes
    step_y_idx = None
    step_hidden_idx = None
    for i, detail in enumerate(step_output_details):
        shape = detail["shape"]
        # y_t should have shape ending in (1, 128), hidden should be (1, 128, 16)
        if len(shape) == 3 and shape[1] == 128 and shape[2] == 16:
            step_hidden_idx = i
        elif len(shape) == 2 and shape[1] == 128:
            step_y_idx = i
    
    if step_y_idx is None:
        step_y_idx = 0
    if step_hidden_idx is None:
        step_hidden_idx = 1 if len(step_output_details) > 1 else 0

    calibration_samples = []
    count = 0

    for data, _ in calibration_loader:
        if count >= num_samples:
            break

        input_data = data.numpy().astype(np.float32)
        
        # Step 1: Run PreSSM
        pre_ssm_interpreter.set_tensor(pre_input_details[0]["index"], input_data)
        pre_ssm_interpreter.invoke()
        
        state_output = pre_ssm_interpreter.get_tensor(pre_output_details[pre_state_idx]["index"])
        gate_output = pre_ssm_interpreter.get_tensor(pre_output_details[pre_gate_idx]["index"])
        
        # state_output: (1, 10, 128)
        # gate_output: (1, 10, 128)
        
        # Step 2: Run StepSSM for each timestep to accumulate y
        seq_len = state_output.shape[1]
        y_accumulated = np.zeros_like(state_output)  # (1, 10, 128)
        hidden_state = np.zeros((1, 128, 16), dtype=np.float32)
        
        for t in range(seq_len):
            x_t = state_output[:, t, :].astype(np.float32)  # (1, 128)
            
            # Run StepSSM
            step_ssm_interpreter.set_tensor(step_input_details[0]["index"], x_t)
            step_ssm_interpreter.set_tensor(step_input_details[1]["index"], hidden_state)
            step_ssm_interpreter.invoke()
            
            y_t = step_ssm_interpreter.get_tensor(step_output_details[step_y_idx]["index"])
            hidden_state = step_ssm_interpreter.get_tensor(step_output_details[step_hidden_idx]["index"])
            
            # y_t might be (128,) or (1, 128), handle both
            if len(y_t.shape) == 1:
                y_accumulated[:, t, :] = y_t[np.newaxis, :]
            else:
                y_accumulated[:, t, :] = y_t
        
        # Add as calibration sample for PostSSM
        calibration_samples.append({
            "args_0": y_accumulated.astype(np.float32),
            "args_1": gate_output.astype(np.float32)
        })
        
        count += 1

    print(f"✓ Generated {len(calibration_samples)} calibration samples for PostSSM")
    return {tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples}


def _quantize_float_model(output_float, output_quantized, calibration_data, 
                          model_name="model", verbose=False):
    """Quantize a float TFLite model using ai-edge-quantizer.
    
    Args:
        output_float: Path to float TFLite model
        output_quantized: Output path for quantized model
        calibration_data: Calibration data dict
        model_name: Name for logging
        verbose: Print verbose output
    """
    try:
        with _maybe_suppress_output(enabled=not verbose):
            from ai_edge_quantizer import quantizer
            from ai_edge_quantizer import algorithm_manager
            from ai_edge_quantizer import qtyping
            from ai_edge_quantizer import recipe_manager
    except ImportError:
        print("Error: ai-edge-quantizer not installed")
        print("Install with: pip install ai-edge-quantizer")
        exit(1)

    algorithm = algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT

    rp_manager = recipe_manager.RecipeManager()
    rp_manager.add_static_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=8,
        weight_num_bits=8,
        algorithm_key=algorithm,
    )
    recipe = rp_manager.get_quantization_recipe()
    qt = quantizer.Quantizer(output_float, recipe)

    print(f"  Calibrating quantizer for {model_name}...")
    with _maybe_suppress_output(enabled=not verbose):
        calibration_results = qt.calibrate(calibration_data)
        quant_result = qt.quantize(calibration_results)
        quant_result.export_model(output_quantized, overwrite=True)


def _print_model_size_summary(model_name, float_size_kb, output_quantized):
    quant_size_kb = os.path.getsize(output_quantized) / 1024
    reduction = (1 - quant_size_kb / float_size_kb) * 100

    print(f"  ✓ {model_name} quantized: {quant_size_kb:.2f} KB "
          f"(reduction: {reduction:.1f}%)")


def quantize_stepwise_models(float_model_dir="tflite_models", 
                              output_dir="tflite_models",
                              num_calibration_samples=2000,
                              pytorch_model_path=None,
                              dataset_dir="../../UCI HAR Dataset"):
    """Quantize 3 stepwise models with appropriate calibration data.
    
    Loads architecture from metadata.json in model directory.
    Supports both HAR and KWS datasets.
    
    Args:
        float_model_dir: Directory with float TFLite models
        output_dir: Output directory for quantized models
        num_calibration_samples: Number of calibration samples (2000 recommended)
        pytorch_model_path: Path to PyTorch model (for metadata extraction)
        dataset_dir: Path to dataset root directory for calibration data
    """
    _configure_quiet_logging(verbose=False)
    
    print("Quantizing stepwise models with calibration data...\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata if pytorch model path is provided
    dataset = "har"  # default
    if pytorch_model_path:
        model_filename = os.path.basename(pytorch_model_path)
        if "_kws" in model_filename:
            dataset = "kws"
        elif "_har" in model_filename:
            dataset = "har"
        
        metadata = load_model_metadata(pytorch_model_path)
        d_inner = metadata.get('d_model', 64) * 2
        d_state = 16
    else:
        # Use defaults
        d_inner = 128
        d_state = 16
    
    # Model paths with dataset name
    pre_ssm_float = os.path.join(float_model_dir, f"model_pre_ssm_{dataset}.tflite")
    step_ssm_float = os.path.join(float_model_dir, f"model_step_ssm_{dataset}.tflite")
    post_ssm_float = os.path.join(float_model_dir, f"model_post_ssm_{dataset}.tflite")
    
    pre_ssm_quant = os.path.join(output_dir, f"model_pre_ssm_int8_{dataset}.tflite")
    step_ssm_quant = os.path.join(output_dir, f"model_step_ssm_int8_{dataset}.tflite")
    post_ssm_quant = os.path.join(output_dir, f"model_post_ssm_int8_{dataset}.tflite")
    
    # Verify float models exist
    for model_path in [pre_ssm_float, step_ssm_float, post_ssm_float]:
        if not os.path.exists(model_path):
            print(f"Error: Float model not found at {model_path}")
            print("Run convert_stepwise_models.py first to generate float models")
            exit(1)
    
    # Load calibration dataset
    _print_step(1, "Loading Calibration Data")
    if dataset == "har":
        train_ds, _, _ = load_har_data(dataset_dir)
        print(f"✓ Loaded HAR dataset with {len(train_ds)} training samples")
    else:  # kws
        train_ds, _, _ = load_speechcommands_data(dataset_dir)
        print(f"✓ Loaded SpeechCommands (KWS) dataset with {len(train_ds)} training samples")
    
    # ========== PreSSM Quantization ==========
    _print_step(2, "Quantizing PreSSM")
    
    pre_ssm_calib = _get_preSsm_calibration_data(train_ds, num_calibration_samples)
    
    pre_ssm_float_size = os.path.getsize(pre_ssm_float) / 1024
    _quantize_float_model(pre_ssm_float, pre_ssm_quant, pre_ssm_calib, "PreSSM")
    _print_model_size_summary("PreSSM", pre_ssm_float_size, pre_ssm_quant)
    
    # ========== StepSSM Quantization ==========
    _print_step(3, "Quantizing StepSSM")
    
    # Build interpreters for generating calibration data
    pre_ssm_interp = _build_tflite_interpreter(pre_ssm_float)
    
    step_ssm_calib = _get_stepSsm_calibration_data(
        pre_ssm_interp, train_ds, num_calibration_samples
    )
    
    step_ssm_float_size = os.path.getsize(step_ssm_float) / 1024
    _quantize_float_model(step_ssm_float, step_ssm_quant, step_ssm_calib, "StepSSM")
    _print_model_size_summary("StepSSM", step_ssm_float_size, step_ssm_quant)
    
    # ========== PostSSM Quantization ==========
    _print_step(4, "Quantizing PostSSM")
    
    # Build StepSSM interpreter for PostSSM calibration
    step_ssm_interp = _build_tflite_interpreter(step_ssm_float)
    
    post_ssm_calib = _get_postSsm_calibration_data(
        pre_ssm_interp, step_ssm_interp, train_ds, num_calibration_samples
    )
    
    post_ssm_float_size = os.path.getsize(post_ssm_float) / 1024
    _quantize_float_model(post_ssm_float, post_ssm_quant, post_ssm_calib, "PostSSM")
    _print_model_size_summary("PostSSM", post_ssm_float_size, post_ssm_quant)
    
    # ========== Summary ==========
    _print_step(5, "Quantization Complete")
    
    total_float = pre_ssm_float_size + step_ssm_float_size + post_ssm_float_size
    total_quant = (os.path.getsize(pre_ssm_quant) + 
                   os.path.getsize(step_ssm_quant) + 
                   os.path.getsize(post_ssm_quant)) / 1024
    total_reduction = (1 - total_quant / total_float) * 100
    
    print(f"  Total float: {total_float:.2f} KB")
    print(f"  Total quantized: {total_quant:.2f} KB")
    print(f"  Overall reduction: {total_reduction:.1f}%")
    print(f"\n✓ All models saved to {output_dir}")

    # Run a post-quantization accuracy check on the test split.
    _print_step(6, "Evaluating Quantized Accuracy")
    if dataset == "har":
        _, _, test_ds = load_har_data(dataset_dir)
    else:  # kws
        _, _, test_ds = load_speechcommands_data(dataset_dir)
    _evaluate_quantized_split_accuracy(pre_ssm_quant, step_ssm_quant, post_ssm_quant, test_ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantize stepwise decomposed models with calibration data"
    )
    parser.add_argument(
        "--float-dir",
        type=str,
        default="tflite_models",
        help="Directory with float TFLite models (default: tflite_models)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tflite_models",
        help="Output directory for quantized models (default: tflite_models)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of calibration samples (default: 2000)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["har", "kws"],
        default="har",
        help="Dataset type (default: har)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset root (default: ../../../UCI HAR Dataset or ../../../SpeechCommands)",
    )

    args = parser.parse_args()

    # Expand paths
    float_dir = Path(args.float_dir)
    output_dir = Path(args.output)
    
    # Determine dataset directory
    if args.dataset_dir:
        dataset_path = Path(args.dataset_dir)
    else:
        if args.dataset == "har":
            dataset_path = Path("../../../UCI HAR Dataset")
        else:  # kws
            dataset_path = Path("../../../SpeechCommands")

    if not float_dir.is_absolute():
        float_dir = Path(__file__).parent / float_dir

    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir

    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).parent / dataset_path

    # Construct pytorch model path for dataset detection
    pytorch_model_path = Path(__file__).parent / f"../models/best_model_{args.dataset}.pt"

    quantize_stepwise_models(
        str(float_dir),
        str(output_dir),
        args.samples,
        pytorch_model_path=str(pytorch_model_path),
        dataset_dir=str(dataset_path)
    )
