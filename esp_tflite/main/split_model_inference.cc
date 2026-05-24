// Define to choose between quantized (int8) and float models
#define USE_QUANTIZED_MODEL 1

// Set to 1 to print allocator and profiler details after each model step.
#define ENABLE_MODEL_DEBUG_PRINTS 0

#include "split_model_inference.h"

#if ENABLE_MODEL_DEBUG_PRINTS
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#else
#include "tensorflow/lite/micro/micro_interpreter.h"
#endif

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "profiler.h"

#include "har_test_samples.h"

#if USE_QUANTIZED_MODEL
#include "model_pre_ssm_int8_model_data.h"
#include "model_step_ssm_int8_model_data.h"
#include "model_post_ssm_int8_model_data.h"
#else
#include "model_pre_ssm_model_data.h"
#include "model_step_ssm_model_data.h"
#include "model_post_ssm_model_data.h"
#endif

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include <algorithm>
#include <cmath>
#include <cstring>

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
// Match the model tensor element type selected by USE_QUANTIZED_MODEL.
#if USE_QUANTIZED_MODEL
using model_tensor_t = int8_t;
constexpr const char* kModelTypeName = "INT8 Quantized";
#else
using model_tensor_t = float;
constexpr const char* kModelTypeName = "Float32";
#endif

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
#if ENABLE_MODEL_DEBUG_PRINTS
tflite::RecordingMicroInterpreter* current_interpreter = nullptr;
#else
tflite::MicroInterpreter* current_interpreter = nullptr;
#endif

// Single shared resolver
static tflite::MicroMutableOpResolver<20> resolver;

// Single shared profiler (reused across all models)
static tflite::CustomProfiler<512, 20> profiler;

UBaseType_t uxHighWaterMark = 0;  // For stack checking

// ========== Inter-model Communication Buffers ==========
model_tensor_t pre_ssm_state[kPreSSMStateSize];     // State output from pre_ssm (10 * 128)
model_tensor_t pre_ssm_gate[kPreSSMGateSize];       // Gate output from pre_ssm (10 * 128)
model_tensor_t hidden_state[kHiddenStateSize];      // Hidden state for step_ssm (2048)
model_tensor_t y_all[kYAllSize];                    // Collected y_t outputs (10 * 128)
}

static void copy_input_to_tensor(const float* source_data, TfLiteTensor* tensor, int size) {
#if USE_QUANTIZED_MODEL
    int8_t* tensor_data = tflite::GetTensorData<int8_t>(tensor);
    const float scale = tensor->params.scale;
    const int zero_point = tensor->params.zero_point;
    for (int i = 0; i < size; ++i) {
        int32_t quantized_value = zero_point;
        if (scale != 0.0f) {
            quantized_value = static_cast<int32_t>(std::lround(source_data[i] / scale)) + zero_point;
        }
        quantized_value = std::max<int32_t>(-128, std::min<int32_t>(127, quantized_value));
        tensor_data[i] = static_cast<int8_t>(quantized_value);
    }
#else
    float* tensor_data = tflite::GetTensorData<float>(tensor);
    memcpy(tensor_data, source_data, size * sizeof(float));
#endif
}

static void copy_tensor_to_model_buffer(const TfLiteTensor* tensor, model_tensor_t* buffer, int size) {
    const model_tensor_t* tensor_data = tflite::GetTensorData<model_tensor_t>(tensor);
    memcpy(buffer, tensor_data, size * sizeof(model_tensor_t));
}

static void copy_model_buffer_to_tensor(const model_tensor_t* buffer, TfLiteTensor* tensor, int size) {
    model_tensor_t* tensor_data = tflite::GetTensorData<model_tensor_t>(tensor);
    memcpy(tensor_data, buffer, size * sizeof(model_tensor_t));
}

static void copy_output_to_float_buffer(const TfLiteTensor* tensor, float* output, int size) {
#if USE_QUANTIZED_MODEL
    const int8_t* tensor_data = tflite::GetTensorData<int8_t>(tensor);
    const float scale = tensor->params.scale;
    const int zero_point = tensor->params.zero_point;
    for (int i = 0; i < size; ++i) {
        output[i] = static_cast<float>(tensor_data[i] - zero_point) * scale;
    }
#else
    const float* tensor_data = tflite::GetTensorData<float>(tensor);
    memcpy(output, tensor_data, size * sizeof(float));
#endif
}

#if ENABLE_MODEL_DEBUG_PRINTS
static void print_memory_debug(const char* step_name, tflite::RecordingMicroInterpreter* interpreter) {
#else
static void print_memory_debug(const char* step_name, tflite::MicroInterpreter* interpreter) {
#endif
#if ENABLE_MODEL_DEBUG_PRINTS
    printf("\n--- %s Memory Allocation ---\n", step_name);
    interpreter->GetMicroAllocator().PrintAllocations();
#else
    (void)step_name;
    (void)interpreter;
#endif
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

#if ENABLE_MODEL_DEBUG_PRINTS
    // Create new recording interpreter with shared arena and profiler
    current_interpreter = new tflite::RecordingMicroInterpreter(
        model, resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler);
#else
    current_interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
#endif
    
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
    profiler.ClearEvents();
    printf("✓ TensorFlow Lite Micro initialized\n");
    printf("✓ Shared tensor arena: %d KB\n", kTensorArenaSize / 1024);
    printf("✓ Model type: %s\n", kModelTypeName);
    printf("✓ Models will be loaded on-demand during inference\n");
    printf("\n");
}

bool run_split_model_inference_raw(const float* input_data, float* output_logits) {
    // ========== STAGE 1: PreSSM ==========
#if USE_QUANTIZED_MODEL
    if (!create_interpreter(g_model_pre_ssm_int8_model_data, "PreSSM")) {
#else
    if (!create_interpreter(g_model_pre_ssm_model_data, "PreSSM")) {
#endif
        return false;
    }
    
    TfLiteTensor* pre_input = current_interpreter->input(0);
    copy_input_to_tensor(input_data, pre_input, kInputLength);
    
    if (current_interpreter->Invoke() != kTfLiteOk) {
        printf("ERROR: PreSSM inference failed\n");
        return false;
    }
    // Extract outputs: state and gate are outputs from the model
    TfLiteTensor* pre_output_0 = current_interpreter->output(0);  // state
    TfLiteTensor* pre_output_1 = current_interpreter->output(1);  // gate
    
    // Copy state and gate to separate buffers
    copy_tensor_to_model_buffer(pre_output_0, pre_ssm_state, kPreSSMStateSize);
    copy_tensor_to_model_buffer(pre_output_1, pre_ssm_gate, kPreSSMGateSize);
    
    // Initialize hidden state to zeros for the first timestep
    memset(hidden_state, 0, kHiddenStateSize * sizeof(model_tensor_t));
    
    print_memory_debug("PreSSM", current_interpreter);
#if ENABLE_MODEL_DEBUG_PRINTS
    printf("\n--- PreSSM Profiling Results ---\n");
    profiler.LogGroupedSinceLap();
    profiler.AdvanceLap();
#endif
    
    // ========== STAGE 2: StepSSM Loop ==========
#if USE_QUANTIZED_MODEL
    if (!create_interpreter(g_model_step_ssm_int8_model_data, "StepSSM")) {
#else
    if (!create_interpreter(g_model_step_ssm_model_data, "StepSSM")) {
#endif
        return false;
    }
    
    TfLiteTensor* step_input_0 = current_interpreter->input(0);  // x_t
    TfLiteTensor* step_input_1 = current_interpreter->input(1);  // hidden_state
    TfLiteTensor* step_output_0 = current_interpreter->output(0); // y_t
    TfLiteTensor* step_output_1 = current_interpreter->output(1); // updated_hidden_state
    
    for (int t = 0; t < kNumTimesteps; t++) {
        copy_model_buffer_to_tensor(&pre_ssm_state[t * kDInner], step_input_0, kDInner);
        copy_model_buffer_to_tensor(hidden_state, step_input_1, kHiddenStateSize);
        
        if (current_interpreter->Invoke() != kTfLiteOk) {
            printf("ERROR: StepSSM inference failed at timestep %d\n", t);
            return false;
        }

        copy_tensor_to_model_buffer(step_output_0, &y_all[t * kDInner], kDInner);
        copy_tensor_to_model_buffer(step_output_1, hidden_state, kHiddenStateSize);
    }

    print_memory_debug("StepSSM", current_interpreter);
#if ENABLE_MODEL_DEBUG_PRINTS
    printf("\n--- StepSSM Profiling Results ---\n");
    profiler.LogGroupedSinceLap();
    profiler.AdvanceLap();
#endif
    
    // ========== STAGE 3: PostSSM ==========
#if USE_QUANTIZED_MODEL
    if (!create_interpreter(g_model_post_ssm_int8_model_data, "PostSSM")) {
#else
    if (!create_interpreter(g_model_post_ssm_model_data, "PostSSM")) {
#endif
        return false;
    }
    
    TfLiteTensor* post_input_0 = current_interpreter->input(0);  // y_all
    TfLiteTensor* post_input_1 = current_interpreter->input(1);  // gate
    
    copy_model_buffer_to_tensor(y_all, post_input_0, kYAllSize);
    copy_model_buffer_to_tensor(pre_ssm_gate, post_input_1, kPreSSMGateSize);
    
    if (current_interpreter->Invoke() != kTfLiteOk) {
        printf("ERROR: PostSSM inference failed\n");
        return false;
    }
    
    TfLiteTensor* post_output = current_interpreter->output(0);
    copy_output_to_float_buffer(post_output, output_logits, kOutputLength);
    
    print_memory_debug("PostSSM", current_interpreter);
#if ENABLE_MODEL_DEBUG_PRINTS
    printf("\n--- PostSSM Profiling Results ---\n");
    profiler.LogGroupedSinceLap();
    printf("\n--- Total Profiling Results ---\n");
    profiler.LogGroupedTotal();
    
    // Check stack
    uxHighWaterMark = uxTaskGetStackHighWaterMark(NULL);
    printf("\nStack High Water Mark: %d bytes remaining\n", (int)uxHighWaterMark);
#endif
    
    
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
    
    // const int num_samples = 1;
    const int num_samples = 50;
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
