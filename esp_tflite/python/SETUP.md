# HAR Model Pipeline - Quick Start Guide

This directory contains the complete pipeline to train, convert, quantize, and deploy a Human Activity Recognition (HAR) model on ESP32.

## Setup

### Option 1: Automatic Setup (Recommended)

Create the conda environment with all dependencies:

```bash
bash create_env.sh
```

This will create a `har_pipeline` environment with:
- PyTorch 2.x
- TensorFlow 2.13+
- litert-torch (direct PyTorch → TFLite conversion)
- Other required packages

### Option 2: Manual Setup

If you prefer to set up manually:

```bash
# Create environment
conda create -n har_pipeline python=3.10 -y

# Activate it
conda activate har_pipeline

# Install dependencies
conda install pytorch pytorch::torchvision pytorch::torchaudio -c pytorch -y
pip install tensorflow litert-torch pandas numpy scipy
```

## Running the Pipeline

### 1. Activate Environment

```bash
conda activate har_pipeline
```

### 2. Run Full Pipeline

The workflow script handles all steps: training → conversion → quantization → C code generation → deployment guide.

```bash
bash workflow.sh
```

This will automatically:
1. **Train** a linear HAR model on the UCI HAR Dataset (50 epochs)
2. **Convert** PyTorch → TFLite (float32) using litert_torch (direct, no ONNX)
3. **Quantize** to int8 using post-training quantization (PTQ)
4. **Generate** C/C++ header files for embedded deployment
5. **Create** deployment guide with model specs

### 3. Output Files

Generated in `./models/`:
- `best_model.pt` - Best PyTorch checkpoint
- `metadata.json` - Training metrics and model info
- `model_float.tflite` - TFLite float32 model (direct from PyTorch)
- `model_int8.tflite` - TFLite int8 quantized model
- `model_int8_model_data.cc` - C array with quantized model
- `model_int8_model_data.h` - Header file for model data
- `DEPLOYMENT_GUIDE.md` - ESP32 deployment instructions

## Model Architecture

**TinyLinear** - Linear replacement for Mamba:
- Input: (batch, 10, 57) - HAR time series
- 3 quantized linear layers (via int8 TFLite PTQ)
- Global average pooling for aggregation
- Output: (batch, 6) - Activity predictions

**Activities**: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING

## Configuration

Edit `workflow.sh` to customize:
- `--epochs` (default: 50) - Training iterations
- `--batch-size` (default: 32) - Batch size
- `--d-model` (default: 64) - Hidden dimension
- `--lr` (default: 0.001) - Learning rate
- `HAR_DATASET` - Path to UCI HAR Dataset
- `TFLITE_MICRO` - Path to TensorFlow Lite Micro

## Troubleshooting

### Missing Dependencies
```bash
# Reinstall in current environment
pip install --force-reinstall tensorflow litert-torch
```

### Conversion Fails
Ensure litert-torch is properly installed:
```bash
pip install --upgrade litert-torch
```

### Model Not Found
Ensure UCI HAR Dataset is at:
```
/home/drabart/Documents/ResearchProject/UCI HAR Dataset/
```

## Next Steps

After successful pipeline execution:

1. **Review** `models/DEPLOYMENT_GUIDE.md` for ESP32 deployment
2. **Use** C array files in ESP-IDF project
3. **Integrate** `esp32_har_inference.cpp` into your ESP32 firmware
4. **Build and flash** to ESP32 using ESP-IDF

## Files Overview

- `create_env.sh` - Conda environment setup script
- `workflow.sh` - Main orchestration script
- `train_linear_har.py` - Training code
- `models_linear.py` - Model architecture
- `pytorch_to_tflite.py` - PyTorch → TFLite conversion
- `quantize_tflite.py` - TFLite int8 quantization
- `data.py` - HAR dataset loading
- `esp32_har_inference.cpp` - ESP32 inference example
- `WORKFLOW_README.md` - Detailed workflow documentation
