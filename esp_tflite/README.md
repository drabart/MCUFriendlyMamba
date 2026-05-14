# ESP-TFLite HAR Inference

Human Activity Recognition (HAR) model trained in PyTorch, converted to TensorFlow Lite with quantization, and deployed on ESP32 using TensorFlow Lite Micro.

## Prerequisites

### Python Environment Setup

This project uses a Conda environment named `litert` with the following dependencies:

```bash
conda create -n litert python=3.12
conda activate litert
pip install pytorch torchvision torchaudio
pip install tensorflow litert-torch ai-edge-quantizer numpy pandas
(TODO: probably also a few more)
```

### ESP-IDF

- ESP-IDF v6.0 or later (for building the ESP32 firmware)
- ESP32 development board

### Data

Place your HAR dataset in a directory accessible to the Python scripts. The dataset should be structured as:
```
dataset/
├── train/
├── val/
└── test/
```

## Directory Structure

```
esp_tflite/
├── python/                          # Python workflow for training and conversion
│   ├── convert_and_quantize.py      # Converts PyTorch → TFLite float → int8 quantized
│   ├── data.py                      # HAR dataset loader
│   ├── generate_cc_arrays.py        # Generates C++ arrays from .tflite models
│   ├── models.py                    # Model definitions (HARMamba)
│   ├── mamba_raw.py                 # Mamba architecture with selective scan
│   ├── train_linear_har.py          # Training script
│   ├── workflow.sh                  # Orchestrates full pipeline
│   └── models/                      # Generated models and C arrays
├── main/                            # ESP32 C++ code
│   ├── main.cc                      # Entry point
│   ├── run_inference.cc/h           # Inference wrapper
│   └── model_int8_model_data.{cc,h} # Copy generated int8 model here
├── components/esp-tflite-micro/     # TFLite Micro framework
└── CMakeLists.txt                   # ESP-IDF build config
```

## Quick Start

### 1. Activate the Conda Environment

```bash
conda activate litert
cd python
```

### 2. Run the Complete Workflow

The `workflow.sh` script handles:
1. **Training** – Trains HARMamba on the HAR dataset
2. **Conversion** – Converts trained model to float TFLite, then quantizes to int8
3. **C Array Generation** – Generates C++ source arrays for embedded deployment

```bash
./workflow.sh
```

Expected outputs in `python/models/`:
- `model_float.tflite` – Float32 model (~200 KB)
- `model_int8.tflite` – Quantized int8 model (~115 KB)
- `model_int8_model_data.cc` – C++ int8 model array
- `model_int8_model_data.h` – C++ int8 model header

**Output will show:**
- Float model accuracy on test set
- Quantized model accuracy on test set
- Model size reduction percentage

### 3. Workflow Options

Skip stages to iterate faster:

```bash
# Skip training, reuse existing checkpoint
./workflow.sh --skip-train

# Skip training and conversion, only generate C arrays
./workflow.sh --skip-train --skip-convert

# Override model paths
./workflow.sh --skip-train --skip-convert --quant-model ./models/model_int8.tflite

# Enable verbose logging (for debugging)
./workflow.sh --verbose

# Copy the generated int8 model arrays into ../main
./workflow.sh --copy-model-arrays
```

## Deployment to ESP32

### 1. Copy Generated Model Files

After running `./workflow.sh`, copy the int8 model arrays to the main code directory, or let the workflow do it for you with `--copy-model-arrays`:

```bash
cp python/models/model_int8_model_data.cc ../main/
cp python/models/model_int8_model_data.h ../main/
```

### 2. Build with ESP-IDF

From the project root:

```bash
idf.py build flash monitor
```

## Model Details

**Architecture:** HARMamba with selective scan mechanism
- Input: (batch=1, timesteps=10, features=57)
- Output: 6 activity classes
- Model size: 115 KB (int8 quantized)

**Quantization:**
- Weights: 8-bit, activations: 8-bit (int8)
- Calibration: MIN_MAX_UNIFORM_QUANT on test set samples
- Typical accuracy preservation: 95%+ of float model

**Test Set Accuracy:**
- Float model: Reported after conversion
- Quantized model: Reported after quantization
