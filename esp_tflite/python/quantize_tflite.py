"""Quantize TFLite model to int8 using post-training quantization."""
import argparse
import os
import json
import numpy as np
import tensorflow as tf
from data import load_har_data
from torch.utils.data import DataLoader


def create_representative_dataset(data_loader, num_samples=100):
    """Create representative dataset for calibration.
    
    Args:
        data_loader: PyTorch DataLoader
        num_samples: Number of samples to use for calibration
    
    Yields:
        Numpy arrays of representative data
    """
    count = 0
    for data, _ in data_loader:
        # Convert to numpy (batch, 10, 57)
        if count >= num_samples:
            break
        yield [data.numpy().astype(np.float32)]
        count += 1


def quantize_tflite(tflite_model_dir, dataset_dir, output_path, num_calibration_samples=100):
    """Quantize TFLite model to int8.
    
    Args:
        tflite_model_dir: Path to float TFLite model
        dataset_dir: Path to HAR dataset for calibration
        output_path: Path to save quantized model
        num_calibration_samples: Number of samples for calibration
    """
    # Load calibration data
    print("Loading calibration data...")
    _, _, test_ds = load_har_data(dataset_dir)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    
    def representative_dataset():
        count = 0
        for data, _ in test_loader:
            if count >= num_calibration_samples:
                break
            # Yield as list of numpy arrays
            yield [data.numpy().astype(np.float32)]
            count += 1
            if count % 10 == 0:
                print(f"  Calibrated on {count} samples")
    
    # Load TFLite model
    print(f"Loading TFLite model: {tflite_model_dir}")
    converter = tf.lite.TFLiteConverter.from_saved_model(
        tflite_model_dir
    )
    
    # Set quantization options
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert to quantized model
    print("Quantizing model to int8...")
    quantized_model = converter.convert()
    
    # Save quantized model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(quantized_model)
    
    print(f"Quantized model saved to: {output_path}")
    print(f"Quantized model size: {len(quantized_model) / 1024:.2f} KB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Quantize TFLite model to int8")
    parser.add_argument(
        "--tflite-model",
        type=str,
        required=True,
        help="Path to float TFLite model",
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
        default="model_int8.tflite",
        help="Output path for quantized TFLite model",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of samples for calibration",
    )
    args = parser.parse_args()

    quantize_tflite(
        args.tflite_model,
        args.dataset_dir,
        args.output_quantized,
        num_calibration_samples=args.calibration_samples,
    )


if __name__ == "__main__":
    main()
