"""Convert PyTorch model to TensorFlow Lite format using litert_torch."""
import argparse
import os
import torch
from models_linear import TinyLinear


def pytorch_to_tflite(pytorch_model_path, output_path, input_shape=(1, 10, 57)):
    """Convert PyTorch model to TFLite format using litert_torch.
    
    Direct conversion: PyTorch → TFLite (float32)
    
    Args:
        pytorch_model_path: Path to saved PyTorch state_dict
        output_path: Path to save TFLite model
        input_shape: Shape of input tensor (batch, time, features)
    """
    try:
        import litert_torch
    except ImportError:
        print("Error: litert_torch not installed. Install with: pip install litert-torch")
        return None
    
    # Create PyTorch model
    device = torch.device("cpu")
    model = TinyLinear(input_dim=57, d_model=64, output_size=6, bit_width=8).to(device)
    model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    model.eval()
    
    # Create dummy input for conversion
    dummy_input = (torch.randn(*input_shape, device=device), )
    
    print(f"Converting PyTorch model to TFLite float32 using litert_torch...")
    
    # Direct PyTorch to TFLite conversion
    edge_model = litert_torch.convert(
        model,
        dummy_input,
    )

    edge_model.export(output_path)
    
    print(f"✓ TFLite float model saved: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TFLite using litert_torch")
    parser.add_argument(
        "--pytorch-model",
        type=str,
        required=True,
        help="Path to PyTorch model state_dict",
    )
    parser.add_argument(
        "--output-tflite",
        type=str,
        default="model_float.tflite",
        help="Output path for TFLite model",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=3,
        default=[1, 10, 57],
        help="Input shape (batch, time, features)",
    )
    args = parser.parse_args()

    pytorch_to_tflite(
        args.pytorch_model,
        args.output_tflite,
        input_shape=tuple(args.input_shape),
    )


if __name__ == "__main__":
    main()
