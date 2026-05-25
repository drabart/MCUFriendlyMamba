#!/usr/bin/env python3
"""
Export 50 random samples from the KWS (Speech Commands) test set as C const arrays.
Useful for testing on embedded devices like ESP32.
"""

import os
import sys
import random
import numpy as np

# Add current directory to path to import data module
sys.path.insert(0, os.path.dirname(__file__))
from data import load_speechcommands_data, get_data_output_size, CLASSES

def export_kws_samples_to_c(data_dir, num_samples=50, output_file="kws_test_samples.h"):
    """
    Load random samples from KWS test set and export as C const array.
    
    Args:
        data_dir: Directory containing SpeechCommands dataset
        num_samples: Number of random samples to extract (default: 50)
        output_file: Output header file name
    """
    print(f"Loading KWS dataset from {data_dir}...")
    _, _, test_ds = load_speechcommands_data(data_dir)
    
    total_samples = len(test_ds)
    print(f"Total test samples available: {total_samples}")
    
    # Randomly select sample indices
    selected_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    print(f"Extracting {len(selected_indices)} random samples...")
    
    # Extract samples
    data_list = []
    labels_list = []
    
    for idx in selected_indices:
        data, label = test_ds[idx]
        # data shape: (51, 40) - T x F format (time steps x MFCC features)
        data_list.append(data.numpy().flatten())  # Flatten to 1D array (2040 values)
        labels_list.append(int(label))
    
    # Convert to numpy arrays
    data_array = np.array(data_list, dtype=np.float32)  # Shape: (50, 2040)
    labels_array = np.array(labels_list, dtype=np.uint8)
    
    print(f"Data shape: {data_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    print(f"Data range: [{data_array.min():.4f}, {data_array.max():.4f}]")
    
    # Generate C header file
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    print(f"Writing C header file to {output_path}...")
    
    with open(output_path, 'w') as f:
        f.write("#ifndef KWS_TEST_SAMPLES_H\n")
        f.write("#define KWS_TEST_SAMPLES_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        # Write class names
        f.write("// KWS class names\n")
        f.write("const char* kws_classes[] = {\n")
        for i, cls in enumerate(CLASSES):
            f.write(f'    "{cls}"')
            if i < len(CLASSES) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")
        
        # Write data constant array
        f.write(f"// KWS test samples - {len(selected_indices)} random samples\n")
        f.write(f"// Each sample: 51 time frames × 40 MFCC features = 2040 float values\n")
        f.write(f"// Total: {len(selected_indices)} × 2040 = {len(selected_indices) * 2040} float values\n\n")
        
        f.write(f"const float kws_test_data[{len(selected_indices)}][2040] = {{\n")
        
        for i, sample in enumerate(data_array):
            f.write("    {")
            for j, val in enumerate(sample):
                if j > 0 and j % 10 == 0:
                    f.write("\n     ")
                f.write(f"{val:.6f}f")
                if j < len(sample) - 1:
                    f.write(", ")
            f.write("}")
            if i < len(data_array) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        
        f.write("};\n\n")
        
        # Write labels constant array
        f.write(f"// Corresponding labels for each sample (class indices 0-34)\n")
        f.write(f"const uint8_t kws_test_labels[{len(selected_indices)}] = {{\n")
        f.write("    ")
        for i, label in enumerate(labels_array):
            f.write(f"{label}")
            if i < len(labels_array) - 1:
                f.write(", ")
            if (i + 1) % 20 == 0 and i < len(labels_array) - 1:
                f.write("\n    ")
        f.write("\n};\n\n")
        
        # Write metadata
        f.write(f"// Metadata\n")
        f.write(f"const uint16_t kws_num_samples = {len(selected_indices)};\n")
        f.write(f"const uint16_t kws_features_per_sample = 2040;\n")
        f.write(f"const uint16_t kws_time_frames = 51;\n")
        f.write(f"const uint16_t kws_mfcc_features = 40;\n")
        f.write(f"const uint8_t kws_num_classes = {get_data_output_size('kws')};\n\n")
        
        f.write("#endif // KWS_TEST_SAMPLES_H\n")
    
    print(f"✓ Successfully exported {len(selected_indices)} samples to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    return output_path


if __name__ == "__main__":
    # Determine SpeechCommands data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    kws_dir = os.path.join(script_dir, "..", "..", "SpeechCommands")
    
    # Optional: allow command line argument for custom path
    if len(sys.argv) > 1:
        kws_dir = sys.argv[1]
    
    # Optional: allow custom number of samples
    num_samples = 50
    if len(sys.argv) > 2:
        try:
            num_samples = int(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid number of samples: {sys.argv[2]}")
            sys.exit(1)
    
    # Export samples
    export_kws_samples_to_c(kws_dir, num_samples=num_samples)
