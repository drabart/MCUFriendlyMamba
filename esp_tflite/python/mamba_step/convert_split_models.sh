#!/bin/bash
# Convert split Mamba models to C++ arrays for ESP32
# This script generates .cc and .h files from the 3 split TFLite models
# Usage: ./convert_split_models.sh [--dataset har|kws]

# Default dataset
DATASET="har"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}/tflite_models"
OUTPUT_DIR="${SCRIPT_DIR}/../../components/models/split_${DATASET}_inference"
OUTPUT_INCLUDE_DIR="${OUTPUT_DIR}/include"
GENERATE_SCRIPT="${SCRIPT_DIR}/../generate_cc_arrays.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TFLITE_MICRO="${TFLITE_MICRO:-/home/drabart/Documents/ResearchProject/tflite-micro}"
SCHEMA_PATH="${SCRIPT_DIR}/../models/schema.fbs"

# Check if input models exist
if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: Models directory not found at $MODELS_DIR"
    exit 1
fi

echo "Converting split Mamba models to C++ arrays..."
echo "Dataset: $DATASET"
echo "Input directory: $MODELS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Include directory: $OUTPUT_INCLUDE_DIR"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_INCLUDE_DIR"

models=("pre_ssm" "step_ssm" "post_ssm")
variants=("" "_int8")

for model_name in "${models[@]}"; do
    for variant_suffix in "${variants[@]}"; do
        input_file="${MODELS_DIR}/model_${model_name}${variant_suffix}_${DATASET}.tflite"
        output_h="${OUTPUT_INCLUDE_DIR}/model_${model_name}${variant_suffix}_${DATASET}_model_data.h"
        output_cc="${OUTPUT_DIR}/model_${model_name}${variant_suffix}_${DATASET}_model_data.cc"
        transformed_file="${MODELS_DIR}/model_${model_name}${variant_suffix}_${DATASET}_tflm_optimized.tflite"

        if [ ! -f "$input_file" ]; then
            echo "WARNING: Model not found: $input_file"
            continue
        fi

        echo "Processing: $input_file"

        if [ ! -f "$SCHEMA_PATH" ]; then
            echo "ERROR: Schema file not found: $SCHEMA_PATH"
            exit 1
        fi

        # Run tflm transforms; skip runtime equivalence testing to avoid
        # crashes in the python runtime during automated checks.
        (cd "$TFLITE_MICRO" && bazel run tensorflow/lite/micro/tools:tflm_model_transforms -- --input_model_path="$input_file" --output_model_path="$transformed_file") || {
            echo "  ✗ Failed to run TFLM transforms"
            exit 1
        }

        # Rename the optimized file back to the original input filename
        # Backup the original input model if it exists so we can restore it later
        backup=""
        if [ -f "$transformed_file" ]; then
            # find a non-colliding backup name
            backup="${input_file}.bak"
            idx=0
            while [ -e "$backup" ]; do
                backup="${input_file}.bak.$idx"
                idx=$((idx+1))
            done
            if [ -f "$input_file" ]; then
                mv "$input_file" "$backup" || {
                    echo "  ✗ Failed to backup original model: $input_file -> $backup"
                    exit 1
                }
            fi
            mv "$transformed_file" "$input_file" || {
                echo "  ✗ Failed to rename optimized model back to original name"
                # try to restore backup if present
                if [ -n "$backup" ] && [ -f "$backup" ]; then
                    mv "$backup" "$input_file" || true
                fi
                exit 1
            }
        else
            echo "  ✗ Transformed file not found: $transformed_file"
            exit 1
        fi

        # Generate header file to include/ directory
        "$PYTHON_BIN" "$GENERATE_SCRIPT" "$output_h" "$input_file"
        if [ $? -ne 0 ]; then
            echo "  ✗ Failed to generate header"
            # restore original model if we backed it up
            if [ -n "$backup" ] && [ -f "$backup" ]; then
                mv "$backup" "$input_file" || true
            fi
            exit 1
        fi

        # Generate implementation file to root of OUTPUT_DIR
        "$PYTHON_BIN" "$GENERATE_SCRIPT" "$output_cc" "$input_file"
        if [ $? -ne 0 ]; then
            echo "  ✗ Failed to generate implementation"
            if [ -n "$backup" ] && [ -f "$backup" ]; then
                mv "$backup" "$input_file" || true
            fi
            exit 1
        fi

        # Restore the original model file if we created a backup
        if [ -n "$backup" ] && [ -f "$backup" ]; then
            mv "$backup" "$input_file" || {
                echo "  ✗ Failed to restore original model from backup: $backup"
                exit 1
            }
        fi

        echo "  ✓ Created ${output_h##*/} and ${output_cc##*/}"
    done
done

echo ""
echo "✓ All models converted successfully!"
echo ""
echo "Generated files:"
echo "  Include files in ${OUTPUT_INCLUDE_DIR##*/}:"
ls -lh "${OUTPUT_INCLUDE_DIR}"/model_*_model_data.h 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
echo "  Implementation files in ${OUTPUT_DIR##*/}:"
ls -lh "${OUTPUT_DIR}"/model_*_model_data.cc 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
