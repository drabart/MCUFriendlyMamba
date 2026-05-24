/*
 * ESP32 Stepwise Split Model Inference Implementation
 * 
 * Runs 3 separate TFLite models in sequence to perform HAR inference
 * without memory explosion from unrolled SSM loops.
 */

#include "split_model_inference.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "profiler.h"

#include "har_test_samples.h"
#include "model_pre_ssm_model_data.h"
#include "model_step_ssm_model_data.h"
#include "model_post_ssm_model_data.h"

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include <algorithm>

// Activity labels for output
const char* ACTIVITY_LABELS[] = {
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
};

namespace {
// ========== Dimension Constants ==========
constexpr int kNumFeatures = 57;           // Input feature dimension
constexpr int kNumTimesteps = 10;          // Sequence length
constexpr int kDInner = 128;               // SSM inner dimension
constexpr int kDState = 16;                // SSM state space dimension
constexpr int kDModel = 64;                // Output model dimension
constexpr int kNumClasses = 6;             // Activity classification classes

// ========== Derived Size Constants ==========
constexpr int kInputLength = kNumTimesteps * kNumFeatures;  // 10 * 57 = 570
constexpr int kPreSSMStateSize = kNumTimesteps * kDInner;   // 10 * 128 = 1280
constexpr int kPreSSMGateSize = kNumTimesteps * kDInner;    // 10 * 128 = 1280
constexpr int kPreSSMOutputSize = kPreSSMStateSize + kPreSSMGateSize;  // 10*128*2 = 2560
constexpr int kHiddenStateSize = kDInner * kDState;         // 128 * 16 = 2048
constexpr int kStepSSMInputSize = kDInner + kHiddenStateSize;  // 128 + 2048 = 2176
constexpr int kStepSSMOutputSize = kDInner + kHiddenStateSize; // 128 + 2048 = 2176
constexpr int kYAllSize = kNumTimesteps * kDInner;          // 10 * 128 = 1280
constexpr int kPostSSMInputSize = kYAllSize + kPreSSMGateSize;  // 1280 + 1280 = 2560
constexpr int kOutputLength = kNumClasses;

// ========== Shared Memory ==========
constexpr int kTensorArenaSize = 60 * 1024;  // 60 KB shared arena
uint8_t tensor_arena[kTensorArenaSize];

// Interpreter pointer (only one used at a time)
tflite::RecordingMicroInterpreter* current_interpreter = nullptr;

// Single shared resolver
static tflite::MicroMutableOpResolver<20> resolver;

// Single shared profiler (reused across all models)
static tflite::CustomProfiler<512, 20> profiler;

UBaseType_t uxHighWaterMark = 0;  // For stack checking

// ========== Inter-model Communication Buffers ==========
float pre_ssm_state[kPreSSMStateSize];     // State output from pre_ssm (10 * 128)
float pre_ssm_gate[kPreSSMGateSize];       // Gate output from pre_ssm (10 * 128)
float hidden_state[kHiddenStateSize];      // Hidden state for step_ssm (2048)
float y_all[kYAllSize];                    // Collected y_t outputs (10 * 128)
}

// Helper function to print float tensor data
static void print_tensor(const char* name, const float* data, int size, int max_elements = 10) {
    printf("%s (first %d of %d elements):\n  ", name, std::min(max_elements, size), size);
    for (int i = 0; i < std::min(max_elements, size); i++) {
        printf("%.4f", data[i]);
        if (i < std::min(max_elements, size) - 1) printf(", ");
    }
    if (size > max_elements) printf(", ...");
    printf("\n");
}

// Helper function to create interpreter with shared arena
static bool create_interpreter(
    const uint8_t* model_data,
    const char* model_name) {
    
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: %s model version mismatch!\n", model_name);
        return false;
    }
    
    // Initialize resolver once
    static bool resolver_initialized = false;
    if (!resolver_initialized) {
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
        resolver.AddSelectV2();
        resolver.AddGreater();
        resolver.AddBroadcastTo();
        resolver.AddRelu();
        resolver.AddConcatenation();
        resolver_initialized = true;
    }
    
    // Create new recording interpreter with shared arena and profiler
    current_interpreter = new tflite::RecordingMicroInterpreter(
        model, resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler);
    
    if (current_interpreter->AllocateTensors() != kTfLiteOk) {
        printf("ERROR: Failed to allocate tensors for %s\n", model_name);
        return false;
    }

    return true;
}

// Initialize
void setup_split_model_inference() {
    printf("\n=== Split Mamba Model Inference Setup ===\n");
    tflite::InitializeTarget();
    printf("✓ TensorFlow Lite Micro initialized\n");
    printf("✓ Shared tensor arena: %d KB\n", kTensorArenaSize / 1024);
    printf("✓ Models will be loaded on-demand during inference\n");
    printf("\n");
}

bool run_split_model_inference_raw(const float* input_data, float* output_logits) {
    // ========== STAGE 1: PreSSM ==========
    if (!create_interpreter(g_model_pre_ssm_model_data, "PreSSM")) {
        return false;
    }
    
    TfLiteTensor* pre_input = current_interpreter->input(0);
    float* pre_input_data = tflite::GetTensorData<float>(pre_input);
    memcpy(pre_input_data, input_data, kInputLength * sizeof(float));
    
    if (current_interpreter->Invoke() != kTfLiteOk) {
        printf("ERROR: PreSSM inference failed\n");
        return false;
    }
    // Extract outputs: state and gate are outputs from the model
    TfLiteTensor* pre_output_0 = current_interpreter->output(0);  // state
    TfLiteTensor* pre_output_1 = current_interpreter->output(1);  // gate
    
    float* pre_state_data = tflite::GetTensorData<float>(pre_output_0);
    float* pre_gate_data = tflite::GetTensorData<float>(pre_output_1);
    
    // Copy state and gate to separate buffers
    memcpy(pre_ssm_state, pre_state_data, kPreSSMStateSize * sizeof(float));
    memcpy(pre_ssm_gate, pre_gate_data, kPreSSMGateSize * sizeof(float));
    
    // Initialize hidden state to zeros for the first timestep
    memset(hidden_state, 0, kHiddenStateSize * sizeof(float));
    
    // Print PreSSM memory allocation
    printf("\n--- PreSSM Memory Allocation ---\n");
    current_interpreter->GetMicroAllocator().PrintAllocations();
    
    // ========== STAGE 2: StepSSM Loop ==========
    if (!create_interpreter(g_model_step_ssm_model_data, "StepSSM")) {
        return false;
    }
    
    TfLiteTensor* step_input_0 = current_interpreter->input(0);  // x_t
    TfLiteTensor* step_input_1 = current_interpreter->input(1);  // hidden_state
    TfLiteTensor* step_output_0 = current_interpreter->output(0); // y_t
    TfLiteTensor* step_output_1 = current_interpreter->output(1); // updated_hidden_state
    
    for (int t = 0; t < kNumTimesteps; t++) {
        float* step_x_t = tflite::GetTensorData<float>(step_input_0);
        float* step_h_t = tflite::GetTensorData<float>(step_input_1);
        
        memcpy(step_x_t, &pre_ssm_state[t * kDInner], kDInner * sizeof(float));
        memcpy(step_h_t, hidden_state, kHiddenStateSize * sizeof(float));
        
        if (current_interpreter->Invoke() != kTfLiteOk) {
            printf("ERROR: StepSSM inference failed at timestep %d\n", t);
            return false;
        }
        
        float* step_y_t = tflite::GetTensorData<float>(step_output_0);
        float* step_h_new = tflite::GetTensorData<float>(step_output_1);
        
        memcpy(&y_all[t * kDInner], step_y_t, kDInner * sizeof(float));
        memcpy(hidden_state, step_h_new, kHiddenStateSize * sizeof(float));
    }
    
    printf("\n--- StepSSM Memory Allocation ---\n");
    current_interpreter->GetMicroAllocator().PrintAllocations();
    
    // ========== STAGE 3: PostSSM ==========
    if (!create_interpreter(g_model_post_ssm_model_data, "PostSSM")) {
        return false;
    }
    
    TfLiteTensor* post_input_0 = current_interpreter->input(0);  // y_all
    TfLiteTensor* post_input_1 = current_interpreter->input(1);  // gate
    float* post_y_data = tflite::GetTensorData<float>(post_input_0);
    float* post_gate_data = tflite::GetTensorData<float>(post_input_1);
    
    memcpy(post_y_data, y_all, kYAllSize * sizeof(float));
    memcpy(post_gate_data, pre_ssm_gate, kPreSSMGateSize * sizeof(float));
    
    if (current_interpreter->Invoke() != kTfLiteOk) {
        printf("ERROR: PostSSM inference failed\n");
        return false;
    }
    
    TfLiteTensor* post_output = current_interpreter->output(0);
    float* post_output_data = tflite::GetTensorData<float>(post_output);
    memcpy(output_logits, post_output_data, kOutputLength * sizeof(float));
    
    printf("\n--- PostSSM Memory Allocation ---\n");
    current_interpreter->GetMicroAllocator().PrintAllocations();
    
    printf("\n--- Profiling Results ---\n");
    profiler.LogGrouped();
    
    // Check stack
    uxHighWaterMark = uxTaskGetStackHighWaterMark(NULL);
    printf("\nStack High Water Mark: %d bytes remaining\n", (int)uxHighWaterMark);
    
    return true;
}

bool run_split_model_inference(const float* input_data, int* output_class) {
    float output_logits[kOutputLength];
    if (!run_split_model_inference_raw(input_data, output_logits)) {
        return false;
    }
    
    // Find predicted class
    *output_class = 0;
    float max_value = output_logits[0];
    for (int i = 1; i < kOutputLength; i++) {
        if (output_logits[i] > max_value) {
            max_value = output_logits[i];
            *output_class = i;
        }
    }
    
    return true;
}

void run_inference() {
    setup_split_model_inference();
    
    const int num_samples = 1;
    int correct_predictions = 0;
    
    printf("\n=== Running inference on %d test samples ===\n\n", num_samples);
    
    for (int i = 0; i < num_samples; i++) {
        const float* test_input = har_test_data[i];
        uint8_t true_label = har_test_labels[i];
        int predicted_class = -1;
        
        if (!run_split_model_inference(test_input, &predicted_class)) {
            printf("Sample %d: Inference failed\n", i);
            continue;
        }
        
        bool is_correct = (predicted_class == true_label);
        if (is_correct) {
            correct_predictions++;
        }

        if (is_correct == false) {
            printf("Sample %2d: Predicted %d, True %d %s\n", 
                   i + 1, predicted_class, true_label, is_correct ? "✓" : "✗");
        }
        
        if ((i + 1) % 10 == 0){
            printf("Processed %d samples...\n", i + 1);
        }
    }
    
    // Print accuracy
    float accuracy = (float)correct_predictions / num_samples * 100.0f;
    printf("\n=== Results ===\n");
    printf("Correct predictions: %d / %d\n", correct_predictions, num_samples);
    printf("Accuracy: %.2f%%\n\n", accuracy);
}
