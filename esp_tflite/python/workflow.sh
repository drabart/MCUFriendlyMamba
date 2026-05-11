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

# Configuration
HAR_DATASET="${HAR_DATASET:=/home/drabart/Documents/ResearchProject/UCI HAR Dataset}"
TFLITE_MICRO="${TFLITE_MICRO:=/home/drabart/Documents/ResearchProject/tflite-micro}"
WORKSPACE="/tmp/har_workflow"
OUTPUT_DIR="./models"

print_info "Using HAR dataset: $HAR_DATASET"
print_info "Using TFLite Micro: $TFLITE_MICRO"
print_info "Workspace: $WORKSPACE"

mkdir -p "$WORKSPACE"
mkdir -p "$OUTPUT_DIR"

# Step 1: Train PyTorch model
print_step "Step 1: Training PyTorch Linear Model on HAR Dataset"
python train_linear_har.py \
    --dataset-dir "$HAR_DATASET" \
    --batch-size 32 \
    --epochs 5 \
    --d-model 64 \
    --bit-width 8 \
    --lr 0.001 \
    --output-dir "$OUTPUT_DIR"

PYTORCH_MODEL="$OUTPUT_DIR/best_model.pt"
[ -f "$PYTORCH_MODEL" ] || print_error "PyTorch model not created"
print_info "✓ PyTorch model saved: $PYTORCH_MODEL"

# Step 2-3: Convert PyTorch to Quantized TFLite (unified script)
print_step "Step 2-3: Converting PyTorch → Float TFLite → Quantized int8"
TFLITE_FLOAT="$OUTPUT_DIR/model_float.tflite"
TFLITE_INT8="$OUTPUT_DIR/model_int8.tflite"
python convert_and_quantize.py \
    --pytorch-model "$PYTORCH_MODEL" \
    --dataset-dir "$HAR_DATASET" \
    --output-float "$TFLITE_FLOAT" \
    --output-quantized "$TFLITE_INT8" \
    --calibration-samples 100 \
    --input-shape 1 10 57

[ -f "$TFLITE_FLOAT" ] || print_error "TFLite float model not created"
[ -f "$TFLITE_INT8" ] || print_error "TFLite int8 model not created"
print_info "✓ Conversion and quantization complete:"
print_info "  - Float: $TFLITE_FLOAT"
print_info "  - Quantized: $TFLITE_INT8"

[ -f "$TFLITE_INT8" ] || print_error "TFLite int8 model not created"
print_info "✓ Quantized TFLite model saved: $TFLITE_INT8"

# Step 3: Generate C arrays
print_step "Step 3: Generating C arrays from quantized model"

# Use local generate_cc_arrays.py
# Pass output to a different filename so .h is not created, triggering the fallback path
python generate_cc_arrays.py \
    "" \
    "$OUTPUT_DIR/model_int8.tflite"

python generate_cc_arrays.py \
    "" \
    "$OUTPUT_DIR/model_float.tflite"

print_info "✓ C array files generated:"
