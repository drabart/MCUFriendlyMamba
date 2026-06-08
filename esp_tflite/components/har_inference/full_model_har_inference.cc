#include "sdkconfig.h"

#include "full_model_har_inference.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

// Include generated model data
#if CONFIG_USE_QUANTIZED_MODEL
#include "model_full_int8_har_model_data.h"
#else
#include "model_full_har_model_data.h"
#endif

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "profiler.h"

// Namespace to avoid conflicts
namespace {
constexpr int kTensorArenaSize = 130 * 1024;
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

void run_inference() {
    printf("\n\n=== ESP32 HAR Inference Test ===\n");
    
    // Setup TensorFlow Lite
    tflite::InitializeTarget();
    
    // Load model
#if CONFIG_USE_QUANTIZED_MODEL
    const tflite::Model* model = tflite::GetModel(g_model_full_int8_har_model_data);
#else
    const tflite::Model* model = tflite::GetModel(g_model_full_har_model_data);
#endif
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: Model version mismatch!\n");
        return;
    }
    printf("Model loaded successfully\n");
    
    // Create interpreter
    static tflite::MicroMutableOpResolver<21> resolver;
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
    resolver.AddRelu();

    resolver.AddConcatenation();

    tflite::CustomProfiler<512, 20> profiler = tflite::CustomProfiler<512, 20>();

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

    // printf("Model Type: %s\n", CONFIG_USE_QUANTIZED_MODEL ? "INT8 Quantized" : "Float32");
    
    printf("Input shape: ");
    for (int i = 0; i < input->dims->size; i++) {
        printf("%d ", input->dims->data[i]);
    }
    printf("\n");
    printf("Input dtype: %s, scale: %f, zero_point: %ld\n", 
           input->type == kTfLiteInt8 ? "INT8" : "FLOAT32", 
           input->params.scale, input->params.zero_point);
    
    printf("Output shape: ");
    for (int i = 0; i < output->dims->size; i++) {
        printf("%d ", output->dims->data[i]);
    }
    printf("\n");
    printf("Output dtype: %s, scale: %f, zero_point: %ld\n", 
           output->type == kTfLiteInt8 ? "INT8" : "FLOAT32", 
           output->params.scale, output->params.zero_point);
    
    // Test inference with dummy data
    printf("\n--- Running inference with dummy data ---\n");

    // Fill input with test data respecting quantization parameters
    if (input->type == kTfLiteInt8) {
        // Quantized input: use quantization parameters
        int8_t* input_data = tflite::GetTensorData<int8_t>(input);
        int8_t zero_point = (int8_t)input->params.zero_point;
        float scale = input->params.scale;
        // Generate values around zero_point with some variation
        for (int i = 0; i < kInputLength; i++) {
            int8_t offset = (int8_t)((i % 50) - 25);  // Range: -25 to 24
            input_data[i] = zero_point + offset;
        }
    } else if (input->type == kTfLiteFloat32) {
        // Float input: use normalized ranges (-1 to 1 is typical for sensor data)
        float* input_data = tflite::GetTensorData<float>(input);
        for (int i = 0; i < kInputLength; i++) {
            // Generate normalized sensor-like data
            input_data[i] = (float)((i % 100) - 50) / 100.0f;  // Range: -0.5 to 0.49
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
    
    // Find predicted class and display results
    int predicted_class = 0;
    
    if (output->type == kTfLiteInt8) {
        int8_t* output_data = tflite::GetTensorData<int8_t>(output);
        int8_t max_value = output_data[0];
        for (int i = 0; i < kOutputLength; i++) {
            printf("%s: %d\n", ACTIVITY_LABELS[i], (int)output_data[i]);
            if (output_data[i] > max_value) {
                max_value = output_data[i];
                predicted_class = i;
            }
        }
    } else if (output->type == kTfLiteFloat32) {
        float* output_data = tflite::GetTensorData<float>(output);
        float max_value = output_data[0];
        for (int i = 0; i < kOutputLength; i++) {
            printf("%s: %f\n", ACTIVITY_LABELS[i], output_data[i]);
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

    // printf("\n--- Precise Profiling Results ---\n");
    // profiler.LogPrecise();
    
    printf("\n--- Grouped Profiling Results ---\n");
    profiler.LogGroupedTotal();

    // Check the stack of the current running task
    UBaseType_t uxHighWaterMark = uxTaskGetStackHighWaterMark(NULL);
    printf("\nStack High Water Mark: %d bytes remaining\n", (int)uxHighWaterMark);

    printf("\n=== Inference test completed ===\n");
}
