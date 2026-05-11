/*
 * ESP32 Inference Wrapper for HAR Model
 * 
 * This is a simple example that demonstrates how to run inference on ESP32
 * using TensorFlow Lite Micro with the quantized HAR model.
 * 
 * To use this:
 * 1. Replace the model data array with your generated hello_world_int8_model_data.h
 * 2. Adapt the input/output handling for your specific model
 * 3. Build and flash to ESP32
 */

#include <Arduino.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include generated model data
#include "hello_world_int8_model_data.h"

// Namespace to avoid conflicts
namespace {
constexpr int kTensorArenaSize = 10 * 1024;  // 10 KB arena for inference
uint8_t tensor_arena[kTensorArenaSize];

// Model constants (update these for your model)
constexpr int kInputLength = 10 * 57;  // 10 timesteps * 57 features for HAR
constexpr int kOutputLength = 6;        // 6 activity classes
}

// Model metadata
const char* ACTIVITY_LABELS[] = {
    "WALKING",
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
};

void setup() {
    Serial.begin(115200);
    delay(100);
    Serial.println("\n\n=== ESP32 HAR Inference Test ===");
    
    // Setup TensorFlow Lite
    tflite::InitializeTarget();
    
    // Load model
    const tflite::Model* model = tflite::GetModel(hello_world_int8_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Model version mismatch!");
        return;
    }
    Serial.println("Model loaded successfully");
    
    // Create interpreter
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: Failed to allocate tensors");
        return;
    }
    Serial.println("Tensors allocated successfully");
    
    // Get input/output tensors
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter output(0);
    
    Serial.print("Input shape: ");
    for (int i = 0; i < input->dims->size; i++) {
        Serial.print(input->dims->data[i]);
        Serial.print(" ");
    }
    Serial.println();
    
    Serial.print("Output shape: ");
    for (int i = 0; i < output->dims->size; i++) {
        Serial.print(output->dims->data[i]);
        Serial.print(" ");
    }
    Serial.println();
    
    // Test inference with dummy data
    Serial.println("\n--- Running inference with dummy data ---");
    
    // Fill input with random-ish test data
    if (input->type == kTfLiteInt8) {
        // Quantized input
        int8_t* input_data = tflite::GetTensorData<int8_t>(input);
        for (int i = 0; i < kInputLength; i++) {
            input_data[i] = (int8_t)(i % 128 - 64);  // Range: -64 to 63
        }
    } else if (input->type == kTfLiteFloat32) {
        // Float input
        float* input_data = tflite::GetTensorData<float>(input);
        for (int i = 0; i < kInputLength; i++) {
            input_data[i] = (float)(i % 100) / 100.0f;
        }
    }
    
    // Run inference
    unsigned long start_time = micros();
    TfLiteStatus invoke_status = interpreter.Invoke();
    unsigned long elapsed_time = micros() - start_time;
    
    if (invoke_status != kTfLiteOk) {
        Serial.println("ERROR: Inference failed");
        return;
    }
    
    Serial.print("Inference completed in ");
    Serial.print(elapsed_time);
    Serial.println(" microseconds");
    
    // Process output
    Serial.println("\n--- Inference Results ---");
    
    if (output->type == kTfLiteInt8) {
        int8_t* output_data = tflite::GetTensorData<int8_t>(output);
        for (int i = 0; i < kOutputLength; i++) {
            Serial.print(ACTIVITY_LABELS[i]);
            Serial.print(": ");
            Serial.println((int)output_data[i]);
        }
    } else if (output->type == kTfLiteFloat32) {
        float* output_data = tflite::GetTensorData<float>(output);
        for (int i = 0; i < kOutputLength; i++) {
            Serial.print(ACTIVITY_LABELS[i]);
            Serial.print(": ");
            Serial.println(output_data[i]);
        }
    }
    
    // Find predicted class
    int predicted_class = 0;
    int max_value = -128;
    
    if (output->type == kTfLiteInt8) {
        int8_t* output_data = tflite::GetTensorData<int8_t>(output);
        for (int i = 0; i < kOutputLength; i++) {
            if (output_data[i] > max_value) {
                max_value = output_data[i];
                predicted_class = i;
            }
        }
    }
    
    Serial.print("\nPredicted Activity: ");
    Serial.println(ACTIVITY_LABELS[predicted_class]);
}

void loop() {
    delay(1000);
}
