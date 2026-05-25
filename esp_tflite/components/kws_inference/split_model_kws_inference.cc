// Define to choose between quantized (int8) and float models
#define USE_QUANTIZED_MODEL 1

// Set to 1 to print allocator and profiler details after each model step.
#define ENABLE_MODEL_DEBUG_PRINTS 1

#include "split_model_kws_inference.h"

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

#include "kws_test_samples.h"

#if USE_QUANTIZED_MODEL
#include "model_pre_ssm_int8_kws_model_data.h"
#include "model_step_ssm_int8_kws_model_data.h"
#include "model_post_ssm_int8_kws_model_data.h"
#else
#include "model_pre_ssm_kws_model_data.h"
#include "model_step_ssm_kws_model_data.h"
#include "model_post_ssm_kws_model_data.h"
#endif

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Keyword labels for output (35 speech command keywords)
const char* KEYWORD_LABELS[] = {
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero"
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
constexpr int kNumFeatures = 40;           // MFCC feature dimension
constexpr int kNumTimesteps = 51;          // Sequence length (time frames)
constexpr int kDInner = 128;               // SSM inner dimension
constexpr int kDState = 16;                // SSM state space dimension
constexpr int kDModel = 64;                // Output model dimension
constexpr int kNumClasses = 35;            // Keyword classification classes

// ========== Derived Size Constants ==========
constexpr int kInputLength = kNumTimesteps * kNumFeatures;  // 51 * 40 = 2040
constexpr int kPreSSMStateSize = kNumTimesteps * kDInner;   // 51 * 128 = 6528
constexpr int kPreSSMGateSize = kNumTimesteps * kDInner;    // 51 * 128 = 6528
constexpr int kPreSSMOutputSize = kPreSSMStateSize + kPreSSMGateSize;  // 51*128*2 = 13056
constexpr int kHiddenStateSize = kDInner * kDState;         // 128 * 16 = 2048
constexpr int kStepSSMInputSize = kDInner + kHiddenStateSize;  // 128 + 2048 = 2176
constexpr int kStepSSMOutputSize = kDInner + kHiddenStateSize; // 128 + 2048 = 2176
constexpr int kYAllSize = kNumTimesteps * kDInner;          // 51 * 128 = 6528
constexpr int kPostSSMInputSize = kYAllSize + kPreSSMGateSize;  // 6528 + 6528 = 13056
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

#if ENABLE_MODEL_DEBUG_PRINTS
// Single shared profiler (reused across all models)
static tflite::CustomProfiler<1024, 20> profiler;

UBaseType_t uxHighWaterMark = 0;  // For stack checking
#endif

// ========== Inter-model Communication Buffers (float for proper quantization) ==========
float pre_ssm_state_float[kPreSSMStateSize];     // Dequantized state output from pre_ssm
float pre_ssm_gate_float[kPreSSMGateSize];       // Dequantized gate output from pre_ssm
float hidden_state_float[kHiddenStateSize];      // Hidden state (float) for step_ssm
float y_all_float[kYAllSize];                    // Dequantized y_t outputs
}

// Dequantize tensor output to float for inter-model communication
static void dequantize_tensor_to_float(const TfLiteTensor* tensor, float* output, int size) {
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

// Quantize float data to tensor input (matching Python's quantization logic)
static void quantize_float_to_tensor(const float* input, TfLiteTensor* tensor, int size) {
#if USE_QUANTIZED_MODEL
    int8_t* tensor_data = tflite::GetTensorData<int8_t>(tensor);
    const float scale = tensor->params.scale;
    const int zero_point = tensor->params.zero_point;
    for (int i = 0; i < size; ++i) {
        // Python: q = np.round(x_np / scale + zero_point)
        int32_t q = static_cast<int32_t>(std::lround(input[i] / scale)) + zero_point;
        // Clip to int8 range
        q = std::max<int32_t>(-128, std::min<int32_t>(127, q));
        tensor_data[i] = static_cast<int8_t>(q);
    }
#else
    float* tensor_data = tflite::GetTensorData<float>(tensor);
    memcpy(tensor_data, input, size * sizeof(float));
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
    printf("\n=== Split Mamba KWS Model Inference Setup ===\n");
    tflite::InitializeTarget();
#if ENABLE_MODEL_DEBUG_PRINTS
    profiler.ClearEvents();
#endif
    printf("✓ TensorFlow Lite Micro initialized\n");
    printf("✓ Shared tensor arena: %d KB\n", kTensorArenaSize / 1024);
    printf("✓ Model type: %s\n", kModelTypeName);
    printf("✓ Models will be loaded on-demand during inference\n");
    printf("\n");
}

bool run_split_model_inference_raw(const float* input_data, float* output_logits) {
    // ========== STAGE 1: PreSSM ==========
#if USE_QUANTIZED_MODEL
    if (!create_interpreter(g_model_pre_ssm_int8_kws_model_data, "PreSSM")) {
#else
    if (!create_interpreter(g_model_pre_ssm_kws_model_data, "PreSSM")) {
#endif
        return false;
    }
    
    TfLiteTensor* pre_input = current_interpreter->input(0);
    quantize_float_to_tensor(input_data, pre_input, kInputLength);
    
    if (current_interpreter->Invoke() != kTfLiteOk) {
        printf("ERROR: PreSSM inference failed\n");
        return false;
    }
    // Extract outputs: state and gate are outputs from the model
    TfLiteTensor* pre_output_0 = current_interpreter->output(0);  // state
    TfLiteTensor* pre_output_1 = current_interpreter->output(1);  // gate
    
    // Dequantize outputs to float for inter-stage communication
    dequantize_tensor_to_float(pre_output_0, pre_ssm_state_float, kPreSSMStateSize);
    dequantize_tensor_to_float(pre_output_1, pre_ssm_gate_float, kPreSSMGateSize);
    
    // Initialize hidden state to zeros (as float)
    memset(hidden_state_float, 0, kHiddenStateSize * sizeof(float));
    
    print_memory_debug("PreSSM", current_interpreter);
#if ENABLE_MODEL_DEBUG_PRINTS
    printf("\n--- PreSSM Profiling Results ---\n");
    profiler.LogGroupedSinceLap();
    profiler.AdvanceLap();
#endif
    
    // ========== STAGE 2: StepSSM Loop ==========
#if USE_QUANTIZED_MODEL
    if (!create_interpreter(g_model_step_ssm_int8_kws_model_data, "StepSSM")) {
#else
    if (!create_interpreter(g_model_step_ssm_kws_model_data, "StepSSM")) {
#endif
        return false;
    }
    
    TfLiteTensor* step_input_0 = current_interpreter->input(0);  // x_t
    TfLiteTensor* step_input_1 = current_interpreter->input(1);  // hidden_state
    TfLiteTensor* step_output_0 = current_interpreter->output(0); // y_t
    TfLiteTensor* step_output_1 = current_interpreter->output(1); // updated_hidden_state
    
    for (int t = 0; t < kNumTimesteps; t++) {
        // Re-quantize float buffers using StepSSM's input quantization parameters
        quantize_float_to_tensor(&pre_ssm_state_float[t * kDInner], step_input_0, kDInner);
        quantize_float_to_tensor(hidden_state_float, step_input_1, kHiddenStateSize);
        
        if (current_interpreter->Invoke() != kTfLiteOk) {
            printf("ERROR: StepSSM inference failed at timestep %d\n", t);
            return false;
        }

        // Dequantize outputs to float for next iteration
        dequantize_tensor_to_float(step_output_0, &y_all_float[t * kDInner], kDInner);
        dequantize_tensor_to_float(step_output_1, hidden_state_float, kHiddenStateSize);
    }

    print_memory_debug("StepSSM", current_interpreter);
#if ENABLE_MODEL_DEBUG_PRINTS
    printf("\n--- StepSSM Profiling Results ---\n");
    profiler.LogGroupedSinceLap();
    profiler.AdvanceLap();
#endif
    
    // ========== STAGE 3: PostSSM ==========
#if USE_QUANTIZED_MODEL
    if (!create_interpreter(g_model_post_ssm_int8_kws_model_data, "PostSSM")) {
#else
    if (!create_interpreter(g_model_post_ssm_kws_model_data, "PostSSM")) {
#endif
        return false;
    }
    
    TfLiteTensor* post_input_0 = current_interpreter->input(0);  // y_all
    TfLiteTensor* post_input_1 = current_interpreter->input(1);  // gate
    
    // Re-quantize float buffers using PostSSM's input quantization parameters
    quantize_float_to_tensor(y_all_float, post_input_0, kYAllSize);
    quantize_float_to_tensor(pre_ssm_gate_float, post_input_1, kPreSSMGateSize);
    
    if (current_interpreter->Invoke() != kTfLiteOk) {
        printf("ERROR: PostSSM inference failed\n");
        return false;
    }
    
    TfLiteTensor* post_output = current_interpreter->output(0);
    dequantize_tensor_to_float(post_output, output_logits, kOutputLength);
    
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

void run_inference_kws() {
    setup_split_model_inference();
    
#if ENABLE_MODEL_DEBUG_PRINTS
    const int num_samples = 1;
#else
    const int num_samples = 50;
#endif

    int correct_predictions = 0;
    
    printf("\n=== Running KWS inference on %d test samples ===\n\n", num_samples);
    
    for (int i = 0; i < num_samples; i++) {
        const float* test_input = kws_test_data[i];
        uint8_t true_label = kws_test_labels[i];
        int predicted_class = -1;
        
        if (!run_split_model_inference(test_input, &predicted_class)) {
            printf("Sample %d: Inference failed\n", i);
            continue;
        }
        
        bool is_correct = (predicted_class == true_label);
        if (is_correct) {
            correct_predictions++;
        }

        // Only print incorrect predictions
        if (!is_correct) {
            printf("Sample %2d: Predicted %s, True %s %s\n", 
                   i + 1, KEYWORD_LABELS[predicted_class], KEYWORD_LABELS[true_label], 
                   is_correct ? "✓" : "✗");
        }
        
        if ((i + 1) % 10 == 0) {
            printf("Processed %d samples...\n", i + 1);
        }
    }
    
    // Print accuracy
    float accuracy = (float)correct_predictions / num_samples * 100.0f;
    printf("\n=== KWS Results ===\n");
    printf("Correct predictions: %d / %d\n", correct_predictions, num_samples);
    printf("Accuracy: %.2f%%\n\n", accuracy);
}
