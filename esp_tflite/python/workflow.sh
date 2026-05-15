#!/bin/bash
# Complete workflow: Train → Convert → Quantize → Generate → Deploy

set -e  # Exit on any error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}===== $1 =====${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
    exit 1
}

print_info() {
    echo -e "${YELLOW}$1${NC}"
}

usage() {
    cat <<'EOF'
Usage: ./workflow.sh [options]

Options:
  --skip-train        Reuse an existing PyTorch model instead of training a new one
  --skip-convert      Reuse existing float and quantized TFLite models
  --skip-generate     Skip generating C arrays from TFLite models
  --copy-model-arrays Copy the generated int8 model .cc/.h into ../main
  --pytorch-model P   Path to an existing PyTorch model checkpoint
  --float-model P     Path to an existing float TFLite model
  --quant-model P     Path to an existing quantized TFLite model
  -h, --help          Show this help message
EOF
}

# Stage control
RUN_TRAIN=1
RUN_CONVERT=1
RUN_GENERATE=1
COPY_MODEL_ARRAYS=0

# Artifact paths (defaults can be overridden from the command line)
OUTPUT_DIR="./models"
PYTORCH_MODEL=""
TFLITE_FLOAT=""
TFLITE_INT8=""
MAIN_DIR="../main"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-train)
            RUN_TRAIN=0
            shift
            ;;
        --skip-convert)
            RUN_CONVERT=0
            shift
            ;;
        --skip-generate)
            RUN_GENERATE=0
            shift
            ;;
        --copy-model-arrays)
            COPY_MODEL_ARRAYS=1
            shift
            ;;
        --pytorch-model)
            PYTORCH_MODEL="$2"
            shift 2
            ;;
        --float-model)
            TFLITE_FLOAT="$2"
            shift 2
            ;;
        --quant-model)
            TFLITE_INT8="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Configuration
HAR_DATASET="${HAR_DATASET:=/home/drabart/Documents/ResearchProject/UCI HAR Dataset}"
TFLITE_MICRO="${TFLITE_MICRO:=/home/drabart/Documents/ResearchProject/tflite-micro}"
WORKSPACE="/tmp/har_workflow"

print_info "Using HAR dataset: $HAR_DATASET"
print_info "Using TFLite Micro: $TFLITE_MICRO"
print_info "Workspace: $WORKSPACE"

mkdir -p "$WORKSPACE"
mkdir -p "$OUTPUT_DIR"

# Step 1: Train PyTorch model
if [[ "$RUN_TRAIN" -eq 1 ]]; then
    print_step "Step 1: Training PyTorch Linear Model on HAR Dataset"
    python train_linear_har.py \
        --dataset-dir "$HAR_DATASET" \
        --batch-size 32 \
        --epochs 20 \
        --d-model 64 \
        --lr 0.002 \
        --output-dir "$OUTPUT_DIR"

    PYTORCH_MODEL="$OUTPUT_DIR/best_model.pt"
    [ -f "$PYTORCH_MODEL" ] || print_error "PyTorch model not created"
    print_info "✓ PyTorch model saved: $PYTORCH_MODEL"
else
    PYTORCH_MODEL="${PYTORCH_MODEL:-$OUTPUT_DIR/best_model.pt}"
    [ -f "$PYTORCH_MODEL" ] || print_error "PyTorch model not found: $PYTORCH_MODEL"
    print_info "✓ Reusing existing PyTorch model: $PYTORCH_MODEL"
fi

# Step 2-3: Convert PyTorch to Quantized TFLite (unified script)
if [[ "$RUN_CONVERT" -eq 1 ]]; then
    print_step "Step 2-3: Converting PyTorch → Float TFLite → Quantized int8"
    TFLITE_FLOAT="${TFLITE_FLOAT:-$OUTPUT_DIR/model_float.tflite}"
    TFLITE_INT8="${TFLITE_INT8:-$OUTPUT_DIR/model_int8.tflite}"
    python convert_and_quantize.py \
        --pytorch-model "$PYTORCH_MODEL" \
        --dataset-dir "$HAR_DATASET" \
        --output-float "$TFLITE_FLOAT" \
        --output-quantized "$TFLITE_INT8" \
        --calibration-samples 500 \
        --input-shape 1 10 57

    [ -f "$TFLITE_FLOAT" ] || print_error "TFLite float model not created"
    [ -f "$TFLITE_INT8" ] || print_error "TFLite int8 model not created"
    print_info "✓ Conversion and quantization complete:"
    print_info "  - Float: $TFLITE_FLOAT"
    print_info "  - Quantized: $TFLITE_INT8"

    [ -f "$TFLITE_INT8" ] || print_error "TFLite int8 model not created"
    print_info "✓ Quantized TFLite model saved: $TFLITE_INT8"
else
    TFLITE_FLOAT="${TFLITE_FLOAT:-$OUTPUT_DIR/model_float.tflite}"
    TFLITE_INT8="${TFLITE_INT8:-$OUTPUT_DIR/model_int8.tflite}"
    [ -f "$TFLITE_FLOAT" ] || print_error "TFLite float model not found: $TFLITE_FLOAT"
    [ -f "$TFLITE_INT8" ] || print_error "TFLite int8 model not found: $TFLITE_INT8"
    print_info "✓ Reusing existing TFLite models:"
    print_info "  - Float: $TFLITE_FLOAT"
    print_info "  - Quantized: $TFLITE_INT8"
fi

# Step 3: Generate C arrays
if [[ "$RUN_GENERATE" -eq 1 ]]; then
    print_step "Step 3: Generating C arrays from quantized model"

    # Use local generate_cc_arrays.py
    # Pass output to a different filename so .h is not created, triggering the fallback path
    python generate_cc_arrays.py \
        "" \
        "$TFLITE_INT8"

    python generate_cc_arrays.py \
        "" \
        "$TFLITE_FLOAT"

    print_info "✓ C array files generated:"
else
    print_info "✓ Skipping C array generation"
fi

if [[ "$COPY_MODEL_ARRAYS" -eq 1 ]]; then
    print_step "Step 4: Copying int8 model arrays into ESP32 main directory"
    cp -f "$OUTPUT_DIR/model_int8_model_data.cc" "$MAIN_DIR/model_int8_model_data.cc"
    cp -f "$OUTPUT_DIR/model_int8_model_data.h" "$MAIN_DIR/model_int8_model_data.h"

    [ -f "$MAIN_DIR/model_int8_model_data.cc" ] || print_error "Failed to copy int8 model source to $MAIN_DIR"
    [ -f "$MAIN_DIR/model_int8_model_data.h" ] || print_error "Failed to copy int8 model header to $MAIN_DIR"
    print_info "✓ Copied int8 model arrays to $MAIN_DIR"
fi
