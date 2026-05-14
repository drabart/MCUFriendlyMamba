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

#include "run_inference.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

// Include generated model data
#include "model_int8_model_data.h"
// #include "model_float_model_data.h"

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"

// Namespace to avoid conflicts
namespace {
constexpr int kTensorArenaSize = 100 * 1024;
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
    printf("\n\n=== ESP32 HAR Inference Test ===\n");
    
    // Setup TensorFlow Lite
    tflite::InitializeTarget();
    
    // Load model
    // const tflite::Model* model = tflite::GetModel(g_model_float_model_data);
    const tflite::Model* model = tflite::GetModel(g_model_int8_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: Model version mismatch!\n");
        return;
    }
    printf("Model loaded successfully\n");
    
    // Create interpreter
    static tflite::MicroMutableOpResolver<20> resolver;
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddMul();
    resolver.AddTranspose();
    resolver.AddSum();

    resolver.AddGatherNd();
    resolver.AddPad();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddDepthwiseConv2D();

    resolver.AddSlice();
    resolver.AddLogistic();
    resolver.AddExp();
    resolver.AddAdd();
    resolver.AddLog();

    resolver.AddSelect();
    resolver.AddSelectV2();
    resolver.AddGreater();
    resolver.AddBroadcastTo();

    tflite::MicroProfiler profiler = tflite::MicroProfiler();
    printf("Profiler size: %d bytes\n", sizeof(profiler));

    static tflite::RecordingMicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler);
    
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: Failed to allocate tensors\n");
        return;
    }
    printf("Tensors allocated successfully\n");

    // Get input/output tensors
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("Input shape: ");
    for (int i = 0; i < input->dims->size; i++) {
        printf("%d ", input->dims->data[i]);
    }
    printf("Input params: %f %ld\n", input->params.scale, input->params.zero_point);
    printf("\n");
    
    printf("Output shape: ");
    for (int i = 0; i < output->dims->size; i++) {
        printf("%d ", output->dims->data[i]);
        printf(" ");
    }
    printf("\n");
    
    // Test inference with dummy data
    printf("\n--- Running inference with dummy data ---\n");

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
    unsigned long start_time = esp_timer_get_time();
    TfLiteStatus invoke_status = interpreter.Invoke();
    unsigned long elapsed_time = esp_timer_get_time() - start_time;
    
    if (invoke_status != kTfLiteOk) {
        printf("ERROR: Inference failed\n");
        return;
    }
    
    printf("Inference completed in \n");
    printf("%lu", elapsed_time);
    printf(" microseconds\n");
    
    // Process output
    printf("\n--- Inference Results ---\n");
    
    if (output->type == kTfLiteInt8) {
        int8_t* output_data = tflite::GetTensorData<int8_t>(output);
        for (int i = 0; i < kOutputLength; i++) {
            printf("%s: %d\n", ACTIVITY_LABELS[i], (int)output_data[i]);
        }
    } else if (output->type == kTfLiteFloat32) {
        float* output_data = tflite::GetTensorData<float>(output);
        for (int i = 0; i < kOutputLength; i++) {
            printf("%s: %f\n", ACTIVITY_LABELS[i], output_data[i]);
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
    
    printf("\nPredicted Activity: ");
    printf("%s\n", ACTIVITY_LABELS[predicted_class]);

    // Print out detailed allocation information:
    interpreter.GetMicroAllocator().PrintAllocations();
    profiler.Log();

    // Check the stack of the current running task
    UBaseType_t uxHighWaterMark = uxTaskGetStackHighWaterMark(NULL);
    printf("Stack High Water Mark: %d bytes remaining\n", (int)uxHighWaterMark * 4);

    printf("\n=== Inference test completed ===\n");
}

void loop() {
    
}
