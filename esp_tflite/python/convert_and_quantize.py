"""Convert PyTorch model to quantized TFLite using ai-edge-quantizer."""
import argparse
import os
import numpy as np
import torch
from models_linear import TinyLinear
from data import load_har_data
from torch.utils.data import DataLoader


def convert_and_quantize(pytorch_model_path, dataset_dir, output_quantized, output_float=None, 
                         num_calibration_samples=100, input_shape=(1, 10, 57)):
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
    # Import dependencies
    try:
        import litert_torch
    except ImportError:
        print("Error: litert_torch not installed")
        print("Install with: pip install litert-torch")
        return None, None
    
    try:
        from ai_edge_quantizer import quantizer
        from ai_edge_quantizer import recipe
    except ImportError:
        print("Error: ai-edge-quantizer not installed")
        print("Install with: pip install ai-edge-quantizer")
        return None, None
    
    # Step 1: Load PyTorch model
    print("=" * 70)
    print("STEP 1: Loading PyTorch Model")
    print("=" * 70)
    device = torch.device("cpu")
    model = TinyLinear(input_dim=57, d_model=64, output_size=6, bit_width=8).to(device)
    model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    model.eval()
    print(f"✓ Loaded PyTorch model from: {pytorch_model_path}")
    
    # Step 2: Convert PyTorch to LiteRT
    print("\n" + "=" * 70)
    print("STEP 2: Converting PyTorch to LiteRT Float Model")
    print("=" * 70)
    
    # Create sample input
    sample_input = torch.randn(*input_shape, device=device)
    print(f"Sample input shape: {sample_input.shape}")
    
    # Convert using litert_torch
    print("Converting with litert_torch.convert()...")
    edge_model = litert_torch.convert(model, (sample_input,))
    
    # Validate conversion
    print("Validating conversion...")
    with torch.no_grad():
        torch_output = model(sample_input).detach().numpy()
        edge_output = edge_model(sample_input.numpy())
    
    if np.allclose(torch_output, edge_output, atol=1e-4):
        print("✓ Conversion validated: PyTorch and LiteRT outputs match")
    else:
        print("⚠ Warning: PyTorch and LiteRT outputs differ")
        print(f"  Max difference: {np.abs(torch_output - edge_output).max()}")
    
    # Step 3: Export float TFLite model
    print("\n" + "=" * 70)
    print("STEP 3: Exporting Float TFLite Model")
    print("=" * 70)
    
    if output_float is None:
        output_float = output_quantized.replace(".tflite", "_float.tflite")
    
    os.makedirs(os.path.dirname(output_float) or ".", exist_ok=True)
    edge_model.export(output_float)
    float_size = os.path.getsize(output_float) / 1024
    print(f"✓ Float TFLite model exported: {output_float}")
    print(f"  Size: {float_size:.2f} KB")
    
    # Step 4: Quantize using ai-edge-quantizer
    print("\n" + "=" * 70)
    print("STEP 4: Quantizing with ai-edge-quantizer")
    print("=" * 70)
    
    print("Loading calibration data...")
    _, _, test_ds = load_har_data(dataset_dir)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    
    count = 0
    for data, _ in test_loader:
        if count >= num_calibration_samples:
            break
        count += 1
        if count % 20 == 0:
            print(f"  Loaded {count} calibration samples")
    
    print(f"✓ Loaded {count} calibration samples for quantization")
    
    # Quantize with dynamic weight int8, activation float32
    print(f"\nApplying dynamic_wi8_afp32 quantization recipe...")
    print(f"  Input: {output_float}")
    print(f"  Output: {output_quantized}")
    
    qt = quantizer.Quantizer(output_float, recipe.dynamic_wi8_afp32())
    quant_result = qt.quantize().export_model(output_quantized, overwrite=True)
    
    # Step 5: Verify quantized model
    print("\n" + "=" * 70)
    print("STEP 5: Verification")
    print("=" * 70)
    
    quant_size = os.path.getsize(output_quantized) / 1024
    reduction = (1 - quant_size / float_size) * 100
    
    print(f"✓ Quantized TFLite model exported: {output_quantized}")
    print(f"  Size: {quant_size:.2f} KB")
    print(f"  Reduction: {reduction:.1f}% ({float_size:.2f} KB → {quant_size:.2f} KB)")
    
    print("\n" + "=" * 70)
    print("CONVERSION AND QUANTIZATION COMPLETE")
    print("=" * 70)
    
    return output_float, output_quantized


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
    
    args = parser.parse_args()
    
    convert_and_quantize(
        pytorch_model_path=args.pytorch_model,
        dataset_dir=args.dataset_dir,
        output_quantized=args.output_quantized,
        output_float=args.output_float,
        num_calibration_samples=args.calibration_samples,
        input_shape=tuple(args.input_shape),
    )


if __name__ == "__main__":
    main()
