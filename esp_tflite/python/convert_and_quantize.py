"""Convert PyTorch model to quantized TFLite using ai-edge-quantizer."""
import os

# Set quiet defaults before any TensorFlow/LiteRT-related imports happen.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import argparse
import contextlib
import io
import logging
import warnings
import numpy as np
import torch
from models import HARMamba
from data import load_har_data
from torch.utils.data import DataLoader
import json


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


def _load_pytorch_model(pytorch_model_path, device):
    model = HARMamba(input_dim=57, d_model=64, output_size=6).to(device)
    model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    model.eval()
    return model


def _convert_to_litert_float_model(model, input_shape, device, verbose=False):
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
    print("Validating conversion...")
    with torch.no_grad():
        torch_output = model(sample_input).detach().numpy()
        edge_output = edge_model(sample_input.numpy())

    if np.allclose(torch_output, edge_output, atol=atol):
        print("✓ Conversion validated: PyTorch and LiteRT outputs match")
    else:
        print("⚠ Warning: PyTorch and LiteRT outputs differ")
        print(f"  Max difference: {np.abs(torch_output - edge_output).max()}")


def _export_float_model(edge_model, output_quantized, output_float=None):
    if output_float is None:
        output_float = output_quantized.replace(".tflite", "_float.tflite")

    os.makedirs(os.path.dirname(output_float) or ".", exist_ok=True)
    edge_model.export(output_float)
    float_size_kb = os.path.getsize(output_float) / 1024
    print(f"✓ Float TFLite model exported: {output_float}")
    print(f"  Size: {float_size_kb:.2f} KB")
    return output_float, float_size_kb


def _get_test_loader(dataset_dir):
    _, _, test_ds = load_har_data(dataset_dir)
    return DataLoader(test_ds, batch_size=1, shuffle=False)


def _build_tflite_interpreter(model_path, verbose=False):
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


def _evaluate_tflite_accuracy(model_path, dataset_dir, label, verbose=False):
    test_loader = _get_test_loader(dataset_dir)
    interpreter = _build_tflite_interpreter(model_path, verbose=verbose)

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    correct = 0
    total = 0

    for data, target in test_loader:
        x_np = data.numpy().astype(np.float32)
        model_input = _quantize_input_if_needed(x_np, input_details)

        interpreter.set_tensor(input_details["index"], model_input)
        interpreter.invoke()

        y_np = interpreter.get_tensor(output_details["index"])
        y_np = _dequantize_output_if_needed(y_np, output_details)

        pred = int(np.argmax(y_np, axis=-1)[0])
        correct += int(pred == int(target.item()))
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(
        f"✓ {label} test accuracy: {accuracy * 100:.2f}% "
        f"({correct}/{total})"
    )
    return accuracy


def _get_calibration_data(dataset_dir, num_samples=100):
    """Load and cache calibration data from HAR dataset.

    The calibration data is returned as a dictionary with the signature key
    mapping to a list of sample dictionaries. Each sample dictionary maps
    input tensor names to their values. This format is required by ai_edge_quantizer.

    Args:
        dataset_dir: Path to HAR dataset.
        num_samples: Number of samples to load.

    Returns:
        A dictionary with signature key mapping to list of calibration samples.
    """
    from ai_edge_quantizer.utils import tfl_interpreter_utils

    _, _, test_ds = load_har_data(dataset_dir)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)

    calibration_samples = []
    count = 0

    for data, _ in test_loader:
        if count >= num_samples:
            break

        sample_data = data.numpy().astype(np.float32)
        calibration_samples.append({"args_0": sample_data})
        count += 1

    print(f"✓ Loaded {count} calibration samples for quantization")
    return {tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples}


def _quantize_float_model(output_float, output_quantized, calibration_data, verbose=False):
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

    print("Calibrating quantizer with HAR dataset...")
    with _maybe_suppress_output(enabled=not verbose):
        calibration_results = qt.calibrate(calibration_data)
        quant_result = qt.quantize(calibration_results)
        quant_result.export_model(output_quantized, overwrite=True)


def _print_model_size_summary(float_size_kb, output_quantized):
    quant_size_kb = os.path.getsize(output_quantized) / 1024
    reduction = (1 - quant_size_kb / float_size_kb) * 100

    print(f"✓ Quantized TFLite model exported: {output_quantized}")
    print(f"  Size: {quant_size_kb:.2f} KB")
    print(
        f"  Reduction: {reduction:.1f}% "
        f"({float_size_kb:.2f} KB → {quant_size_kb:.2f} KB)"
    )


def convert_and_quantize(pytorch_model_path, dataset_dir, output_quantized, output_float=None, 
                         num_calibration_samples=100, input_shape=(1, 10, 57),
                         verbose=False):
    """Convert PyTorch model to quantized TFLite using ai-edge-quantizer.
    
    Process:
    1. Load PyTorch model
    2. Convert to LiteRT using litert_torch
    3. Export float TFLite model
    4. Quantize using ai-edge-quantizer with dynamic weight int8
    5. Export quantized TFLite model
    
    Args:
        pytorch_model_path: Path to PyTorch model state_dict
        dataset_dir: Path to HAR dataset for calibration
        output_quantized: Path to save quantized TFLite model
        output_float: Path to save float TFLite model (optional)
        num_calibration_samples: Number of samples for quantization calibration
        input_shape: Input shape (batch, time, features)
    
    Returns:
        Tuple of (float_model_path, quantized_model_path)
    """
    _configure_quiet_logging(verbose=verbose)

    _print_step(1, "Loading PyTorch Model")
    device = torch.device("cpu")
    model = _load_pytorch_model(pytorch_model_path, device)
    print(f"✓ Loaded PyTorch model from: {pytorch_model_path}")

    _print_step(2, "Converting PyTorch to LiteRT Float Model")
    edge_model, sample_input = _convert_to_litert_float_model(
        model, input_shape, device, verbose=verbose
    )
    _validate_conversion_close(model, edge_model, sample_input)

    _print_step(3, "Exporting Float TFLite Model")
    output_float, float_size_kb = _export_float_model(
        edge_model, output_quantized, output_float
    )

    _print_step(4, "Evaluating Float TFLite on Test Set")
    float_acc = _evaluate_tflite_accuracy(
        model_path=output_float,
        dataset_dir=dataset_dir,
        label="Float TFLite",
        verbose=verbose,
    )

    _print_step(5, "Quantizing with ai-edge-quantizer")
    print("Loading calibration data...")
    calibration_data = _get_calibration_data(dataset_dir, num_samples=num_calibration_samples)

    print(f"\nApplying quantization recipe...")
    print(f"  Input: {output_float}")
    print(f"  Output: {output_quantized}")

    _quantize_float_model(
        output_float=output_float,
        output_quantized=output_quantized,
        calibration_data=calibration_data,
        verbose=verbose,
    )

    _print_step(6, "Evaluating Quantized TFLite on Test Set")
    quant_acc = _evaluate_tflite_accuracy(
        model_path=output_quantized,
        dataset_dir=dataset_dir,
        label="Quantized TFLite",
        verbose=verbose,
    )

    _print_step(7, "Verification")
    _print_model_size_summary(float_size_kb, output_quantized)
    print(f"  Accuracy drop: {(float_acc - quant_acc) * 100:.2f} percentage points")

    print("\n" + "=" * 70)
    print("CONVERSION AND QUANTIZATION COMPLETE")
    print("=" * 70)
    

def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to quantized TFLite"
    )
    parser.add_argument(
        "--pytorch-model",
        type=str,
        required=True,
        help="Path to PyTorch model state_dict",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to HAR dataset for calibration",
    )
    parser.add_argument(
        "--output-quantized",
        type=str,
        default="model_quantized.tflite",
        help="Output path for quantized TFLite model",
    )
    parser.add_argument(
        "--output-float",
        type=str,
        default=None,
        help="Output path for float TFLite model (optional)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of samples for calibration",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=3,
        default=[1, 10, 57],
        help="Input shape (batch, time, features)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logs from TensorFlow/LiteRT and dependencies",
    )
    
    args = parser.parse_args()
    
    convert_and_quantize(
        pytorch_model_path=args.pytorch_model,
        dataset_dir=args.dataset_dir,
        output_quantized=args.output_quantized,
        output_float=args.output_float,
        num_calibration_samples=args.calibration_samples,
        input_shape=tuple(args.input_shape),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
