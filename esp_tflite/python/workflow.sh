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
cd "$TFLITE_MICRO" || print_error "Cannot cd to $TFLITE_MICRO"

python tensorflow/lite/micro/tools/generate_cc_arrays.py \
    "$OUTPUT_DIR/model_int8_model_data.cc" \
    "$TFLITE_INT8"

[ -f "$OUTPUT_DIR/model_int8_model_data.cc" ] || print_error "C source file not generated"
[ -f "$OUTPUT_DIR/model_int8_model_data.h" ] || print_error "C header file not generated"

print_info "✓ C array files generated:"
print_info "  - $OUTPUT_DIR/model_int8_model_data.cc"
print_info "  - $OUTPUT_DIR/model_int8_model_data.h"

# Step 4: Create ESP32 integration guide
print_step "Step 4: Creating ESP32 Integration Files"

# Summary
print_step "Workflow Complete!"
echo ""
echo "📊 Training Summary:"
cat "$OUTPUT_DIR/metadata.json" | python -m json.tool
echo ""
echo "📁 Output Files:"
ls -lh "$OUTPUT_DIR"/ | grep -E "\.(cc|h|tflite|json|md)$"
echo ""
print_info "Next steps:"
print_info "1. Review $OUTPUT_DIR/DEPLOYMENT_GUIDE.md"
print_info "2. Clone tflite-micro-esp-examples or create ESP-IDF project"
print_info "3. Copy .cc/.h files to your project"
print_info "4. Build and flash to ESP32"
