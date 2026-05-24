#!/bin/bash
# Convert split Mamba models to C++ arrays for ESP32
# This script generates .cc and .h files from the 3 split TFLite models

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}/tflite_models"
OUTPUT_DIR="${SCRIPT_DIR}/../../main"
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
echo "Input directory: $MODELS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

models=("pre_ssm" "step_ssm" "post_ssm")
variants=("" "_int8")

for model_name in "${models[@]}"; do
    for variant_suffix in "${variants[@]}"; do
        input_file="${MODELS_DIR}/model_${model_name}${variant_suffix}.tflite"
        output_h="${OUTPUT_DIR}/model_${model_name}${variant_suffix}_model_data.h"
        output_cc="${OUTPUT_DIR}/model_${model_name}${variant_suffix}_model_data.cc"
        transformed_file="${MODELS_DIR}/model_${model_name}${variant_suffix}_tflm_optimized.tflite"

        if [ ! -f "$input_file" ]; then
            echo "WARNING: Model not found: $input_file"
            continue
        fi

        echo "Processing: $input_file"

        if [ ! -f "$SCHEMA_PATH" ]; then
            echo "ERROR: Schema file not found: $SCHEMA_PATH"
            exit 1
        fi

        # Run the same transform pipeline used by workflow.sh.
        "$PYTHON_BIN" "${SCRIPT_DIR}/../strip_names.py" --schema "$SCHEMA_PATH" --tflite "$input_file" || {
            echo "  ✗ Failed to replace SELECT in model"
            exit 1
        }

        # Run tflm transforms; skip runtime equivalence testing to avoid
        # crashes in the python runtime during automated checks.
        (cd "$TFLITE_MICRO" && bazel run tensorflow/lite/micro/tools:tflm_model_transforms -- --input_model_path="$input_file" --output_model_path="$transformed_file" --test_transformed_model=False) || {
            echo "  ✗ Failed to run TFLM transforms"
            exit 1
        }

        # Generate header file
        "$PYTHON_BIN" "$GENERATE_SCRIPT" "$output_h" "$transformed_file"
        if [ $? -ne 0 ]; then
            echo "  ✗ Failed to generate header"
            exit 1
        fi

        # Generate implementation file
        "$PYTHON_BIN" "$GENERATE_SCRIPT" "$output_cc" "$transformed_file"
        if [ $? -ne 0 ]; then
            echo "  ✗ Failed to generate implementation"
            exit 1
        fi

        echo "  ✓ Created ${output_h##*/} and ${output_cc##*/}"
    done
done

echo ""
echo "✓ All models converted successfully!"
echo ""
echo "Generated files:"
ls -lh "${OUTPUT_DIR}"/model_*_model_data.* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
