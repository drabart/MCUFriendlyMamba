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
    --dataset NAME          Dataset to run (har or kws)
    --dataset-dir P         Override the dataset root directory
    --skip-train            Reuse an existing PyTorch model instead of training a new one
    --skip-convert          Reuse existing float and quantized TFLite models
    --skip-strip            Skip model transformations (SELECT replacement + TFLM optimization)
    --skip-generate         Skip generating C arrays from TFLite models
    --copy-model-arrays     Copy the generated int8 model .cc/.h into ../main
    --pytorch-model P       Path to an existing PyTorch model checkpoint
    --float-model P         Path to an existing float TFLite model
    --quant-model P         Path to an existing quantized TFLite model
    --d-model N             Model dimension used for training/conversion
    -h, --help              Show this help message

Pipeline (Step 3 - Model Transformations):
  1. Replace SELECT (opcode 64) with SELECT_V2 (opcode 123) in original models
  2. Run TFLM model optimizations via bazel (clears buffers, removes quantization data, strips strings, shortens names)
EOF
}

# Stage control
RUN_TRAIN=1
RUN_CONVERT=1
RUN_STRIP=1
RUN_GENERATE=1
COPY_MODEL_ARRAYS=0
DATASET="har"
DATASET_DIR=""
D_MODEL=64

# Artifact paths (defaults can be overridden from the command line)
OUTPUT_DIR="./models"
PYTORCH_MODEL=""
TFLITE_FLOAT=""
TFLITE_INT8=""
ESP32_MODELS_DIR="../components/models/full_har_inference"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --skip-train)
            RUN_TRAIN=0
            shift
            ;;
        --skip-convert)
            RUN_CONVERT=0
            shift
            ;;
        --skip-strip)
            RUN_STRIP=0
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
        --d-model)
            D_MODEL="$2"
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
KWS_DATASET="${KWS_DATASET:=/home/drabart/Documents/ResearchProject/SpeechCommands}"
TFLITE_MICRO="${TFLITE_MICRO:=/home/drabart/Documents/ResearchProject/tflite-micro}"

if [[ -z "$DATASET_DIR" ]]; then
    case "$DATASET" in
        har)
            DATASET_DIR="$HAR_DATASET"
            ;;
        kws)
            DATASET_DIR="$KWS_DATASET"
            ;;
        *)
            print_error "Unsupported dataset: $DATASET"
            ;;
    esac
fi

WORKSPACE="/tmp/esp_tflite_${DATASET}_workflow"

print_info "Using dataset: $DATASET"
print_info "Using dataset root: $DATASET_DIR"
print_info "Using TFLite Micro: $TFLITE_MICRO"
print_info "Workspace: $WORKSPACE"

mkdir -p "$WORKSPACE"
mkdir -p "$OUTPUT_DIR"

# Step 1: Train PyTorch model
if [[ "$RUN_TRAIN" -eq 1 ]]; then
    print_step "Step 1: Training PyTorch Linear Model on ${DATASET^^} Dataset"
    python train_linear.py \
        --dataset "$DATASET" \
        --dataset-dir "$DATASET_DIR" \
        --batch-size 32 \
        --epochs 20 \
        --d-model "$D_MODEL" \
        --lr 0.002 \
        --output-dir "$OUTPUT_DIR" \
        --model-name "best_model_${DATASET}.pt"

    PYTORCH_MODEL="$OUTPUT_DIR/best_model_${DATASET}.pt"
    [ -f "$PYTORCH_MODEL" ] || print_error "PyTorch model not created"
    print_info "✓ PyTorch model saved: $PYTORCH_MODEL"
else
    PYTORCH_MODEL="${PYTORCH_MODEL:-$OUTPUT_DIR/best_model_${DATASET}.pt}"
    [ -f "$PYTORCH_MODEL" ] || print_error "PyTorch model not found: $PYTORCH_MODEL"
    print_info "✓ Reusing existing PyTorch model: $PYTORCH_MODEL"
fi

# Step 2-3: Convert PyTorch to Quantized TFLite (unified script)
if [[ "$RUN_CONVERT" -eq 1 ]]; then
    print_step "Step 2-3: Converting PyTorch → Float TFLite → Quantized int8"
    TFLITE_FLOAT="${TFLITE_FLOAT:-$OUTPUT_DIR/model_full_${DATASET}.tflite}"
    TFLITE_INT8="${TFLITE_INT8:-$OUTPUT_DIR/model_full_int8_${DATASET}.tflite}"
    python convert_and_quantize.py \
        --dataset "$DATASET" \
        --pytorch-model "$PYTORCH_MODEL" \
        --dataset-dir "$DATASET_DIR" \
        --d-model "$D_MODEL" \
        --output-float "$TFLITE_FLOAT" \
        --output-quantized "$TFLITE_INT8" \
        --calibration-samples 500

    [ -f "$TFLITE_FLOAT" ] || print_error "TFLite float model not created"
    [ -f "$TFLITE_INT8" ] || print_error "TFLite int8 model not created"
    print_info "✓ Conversion and quantization complete:"
    print_info "  - Float: $TFLITE_FLOAT"
    print_info "  - Quantized: $TFLITE_INT8"

    [ -f "$TFLITE_INT8" ] || print_error "TFLite int8 model not created"
    print_info "✓ Quantized TFLite model saved: $TFLITE_INT8"
else
    TFLITE_FLOAT="${TFLITE_FLOAT:-$OUTPUT_DIR/model_full_${DATASET}.tflite}"
    TFLITE_INT8="${TFLITE_INT8:-$OUTPUT_DIR/model_full_int8_${DATASET}.tflite}"
    [ -f "$TFLITE_FLOAT" ] || print_error "TFLite float model not found: $TFLITE_FLOAT"
    [ -f "$TFLITE_INT8" ] || print_error "TFLite int8 model not found: $TFLITE_INT8"
    print_info "✓ Reusing existing TFLite models:"
    print_info "  - Float: $TFLITE_FLOAT"
    print_info "  - Quantized: $TFLITE_INT8"
fi

# Step 3: Transform TFLite models (new pipeline)
if [[ "$RUN_STRIP" -eq 1 ]]; then
    print_step "Step 3: Transforming TFLite models (SELECT replacement → TFLM optimization)"
    SCHEMA_PATH="./models/schema.fbs"
    
    if [ ! -f "$SCHEMA_PATH" ]; then
        print_error "Schema file not found: $SCHEMA_PATH"
    fi
    
    # Step 3a: Replace SELECT with SELECT_V2 (required before TFLM transforms)
    print_info "Step 3a: Replacing SELECT with SELECT_V2..."
    python fix_select.py --schema "$SCHEMA_PATH" --tflite "$TFLITE_INT8" || print_error "Failed to replace SELECT in int8 model"
    python fix_select.py --schema "$SCHEMA_PATH" --tflite "$TFLITE_FLOAT" || print_error "Failed to replace SELECT in float model"
    print_info "✓ SELECT replacement complete"
    
    # Step 3b: Run TFLM model transforms (from tflite-micro)
    print_info "Step 3b: Running TFLM model optimizations..."
    print_info "Note: Running from $TFLITE_MICRO directory"
    
    # Use absolute paths since we're changing directories
    TFLITE_INT8_ABS="$(cd "$(dirname "$TFLITE_INT8")" && pwd)/$(basename "$TFLITE_INT8")"
    TFLITE_FLOAT_ABS="$(cd "$(dirname "$TFLITE_FLOAT")" && pwd)/$(basename "$TFLITE_FLOAT")"
    TFLITE_INT8_OPTIMIZED="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")/model_full_int8_${DATASET}_tflm_optimized.tflite"
    TFLITE_FLOAT_OPTIMIZED="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")/model_full_${DATASET}_tflm_optimized.tflite"
    
    print_info "Input INT8:  $TFLITE_INT8_ABS"
    print_info "Output INT8: $TFLITE_INT8_OPTIMIZED"
    
    # Run TFLM transforms for int8 model
    (cd "$TFLITE_MICRO" && bazel run tensorflow/lite/micro/tools:tflm_model_transforms -- --input_model_path="$TFLITE_INT8_ABS" --output_model_path="$TFLITE_INT8_OPTIMIZED") || print_error "Failed to run TFLM transforms on int8 model"
    [ -f "$TFLITE_INT8_OPTIMIZED" ] || print_error "TFLM optimization did not produce: $TFLITE_INT8_OPTIMIZED"
    INT8_ORIG_KB=$(($(stat -f%z "$TFLITE_INT8_ABS" 2>/dev/null || stat -c%s "$TFLITE_INT8_ABS") / 1024))
    INT8_OPT_KB=$(($(stat -f%z "$TFLITE_INT8_OPTIMIZED" 2>/dev/null || stat -c%s "$TFLITE_INT8_OPTIMIZED") / 1024))
    INT8_REDUCTION=$((INT8_ORIG_KB - INT8_OPT_KB))
    INT8_PCT=$((INT8_REDUCTION * 100 / INT8_ORIG_KB))
    print_info "✓ INT8 model optimized: ${INT8_ORIG_KB}KB → ${INT8_OPT_KB}KB (${INT8_PCT}% reduction)"
    
    # Run TFLM transforms for float model
    (cd "$TFLITE_MICRO" && bazel run tensorflow/lite/micro/tools:tflm_model_transforms -- --input_model_path="$TFLITE_FLOAT_ABS" --output_model_path="$TFLITE_FLOAT_OPTIMIZED") || print_error "Failed to run TFLM transforms on float model"
    [ -f "$TFLITE_FLOAT_OPTIMIZED" ] || print_error "TFLM optimization did not produce: $TFLITE_FLOAT_OPTIMIZED"
    FLOAT_ORIG_KB=$(($(stat -f%z "$TFLITE_FLOAT_ABS" 2>/dev/null || stat -c%s "$TFLITE_FLOAT_ABS") / 1024))
    FLOAT_OPT_KB=$(($(stat -f%z "$TFLITE_FLOAT_OPTIMIZED" 2>/dev/null || stat -c%s "$TFLITE_FLOAT_OPTIMIZED") / 1024))
    FLOAT_REDUCTION=$((FLOAT_ORIG_KB - FLOAT_OPT_KB))
    FLOAT_PCT=$((FLOAT_REDUCTION * 100 / FLOAT_ORIG_KB))
    print_info "✓ Float model optimized: ${FLOAT_ORIG_KB}KB → ${FLOAT_OPT_KB}KB (${FLOAT_PCT}% reduction)"
    
    print_info "✓ Model transformation pipeline complete"
    
    # Rename optimized models back to original names (for C array generation)
    mv "$TFLITE_INT8_OPTIMIZED" "$TFLITE_INT8"
    mv "$TFLITE_FLOAT_OPTIMIZED" "$TFLITE_FLOAT"
    print_info "✓ Optimized models renamed to original names"
else
    print_info "✓ Skipping model transformations"
fi

# Step 4: Generate C arrays from optimized model
if [[ "$RUN_GENERATE" -eq 1 ]]; then
    print_step "Step 4: Generating C arrays from optimized model"

    # Use the optimized models (now renamed to original names)
    print_info "Generating C arrays from: $TFLITE_INT8"
    python generate_cc_arrays.py \
        "" \
        "$TFLITE_INT8"

    print_info "Generating C arrays from: $TFLITE_FLOAT"
    python generate_cc_arrays.py \
        "" \
        "$TFLITE_FLOAT"

    print_info "✓ C array files generated:"
else
    print_info "✓ Skipping C array generation"
fi

if [[ "$COPY_MODEL_ARRAYS" -eq 1 ]]; then
    print_step "Step 5: Copying model arrays into ESP32 main directory"
    
    # Copy int8 arrays
    cp -f "$OUTPUT_DIR/model_full_int8_${DATASET}_model_data.cc" "$ESP32_MODELS_DIR/model_full_int8_${DATASET}_model_data.cc"
    cp -f "$OUTPUT_DIR/model_full_int8_${DATASET}_model_data.h" "$ESP32_MODELS_DIR/include/model_full_int8_${DATASET}_model_data.h"
    [ -f "$ESP32_MODELS_DIR/model_full_int8_${DATASET}_model_data.cc" ] || print_error "Failed to copy int8 model source to $ESP32_MODELS_DIR"
    [ -f "$ESP32_MODELS_DIR/include/model_full_int8_${DATASET}_model_data.h" ] || print_error "Failed to copy int8 model header to $ESP32_MODELS_DIR"
    print_info "✓ Copied int8 model arrays to $ESP32_MODELS_DIR"
    
    # Copy full arrays
    cp -f "$OUTPUT_DIR/model_full_${DATASET}_model_data.cc" "$ESP32_MODELS_DIR/model_full_${DATASET}_model_data.cc"
    cp -f "$OUTPUT_DIR/model_full_${DATASET}_model_data.h" "$ESP32_MODELS_DIR/include/model_full_${DATASET}_model_data.h"
    [ -f "$ESP32_MODELS_DIR/model_full_${DATASET}_model_data.cc" ] || print_error "Failed to copy full model source to $ESP32_MODELS_DIR"
    [ -f "$ESP32_MODELS_DIR/include/model_full_${DATASET}_model_data.h" ] || print_error "Failed to copy full model header to $ESP32_MODELS_DIR"
    print_info "✓ Copied full model arrays to $ESP32_MODELS_DIR"
fi
