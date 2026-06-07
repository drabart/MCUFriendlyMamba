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

#include "freertos/FreeRTOS.h"
#include "esp_timer.h"

template <int kNumTimesteps, int kDInner, int kDState, int kNumFeatures, int kNumClasses>
class SplitInference {
public:

    const uint8_t* pre_model_data;
    const uint8_t* step_model_data;
    const uint8_t* post_model_data;
    const char* name;

    // Initialize
    void setup_split_model_inference(
        const uint8_t* pre_model_data, 
        const uint8_t* step_model_data, 
        const uint8_t* post_model_data,
        const char* name
    ) {
        this->pre_model_data = pre_model_data;
        this->step_model_data = step_model_data;
        this->post_model_data = post_model_data;
        this->name = name;

        printf("\n=== Split Mamba %s Inference Setup ===\n", this->name);
        tflite::InitializeTarget();
#if ENABLE_MODEL_DEBUG_PRINTS
        profiler.ClearEvents();
#endif
        printf("✓ TensorFlow Lite Micro initialized\n");
        printf("✓ Shared tensor arena: %d KB\n", kTensorArenaSize / 1024);
        printf("✓ Model type: %s\n", this->name);
#if USE_QUANTIZED_MODEL
        printf("✓ Inter-model buffers: %d KB (int8)\n", (kPreSSMStateSize + kPreSSMGateSize) / 1024);
#else
        printf("✓ Inter-model buffers: %d KB (float32)\n", ((kPreSSMStateSize + kPreSSMGateSize) * sizeof(float)) / 1024);
#endif
        printf("✓ Models will be loaded on-demand during inference\n");
        printf("\n");
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

private:
    // ========== Derived Size Constants ==========
    static constexpr int kInputLength = kNumTimesteps * kNumFeatures;  // 51 * 40 = 2040
    static constexpr int kPreSSMStateSize = kNumTimesteps * kDInner;   // 51 * 128 = 6528
    static constexpr int kPreSSMGateSize = kNumTimesteps * kDInner;    // 51 * 128 = 6528
    static constexpr int kPreSSMOutputSize = kPreSSMStateSize + kPreSSMGateSize;  // 51*128*2 = 13056
    static constexpr int kHiddenStateSize = kDInner * kDState;         // 128 * 16 = 2048
    static constexpr int kStepSSMInputSize = kDInner + kHiddenStateSize;  // 128 + 2048 = 2176
    static constexpr int kStepSSMOutputSize = kDInner + kHiddenStateSize; // 128 + 2048 = 2176
    static constexpr int kYAllSize = kNumTimesteps * kDInner;          // 51 * 128 = 6528
    static constexpr int kPostSSMInputSize = kYAllSize + kPreSSMGateSize;  // 6528 + 6528 = 13056
    static constexpr int kOutputLength = kNumClasses;

    // ========== Shared Memory ==========
    static constexpr int kTensorArenaSize = 120 * 1024;
    inline static uint8_t tensor_arena[kTensorArenaSize] = {};

    // Interpreter pointer (only one used at a time)
#if ENABLE_MODEL_DEBUG_PRINTS
    tflite::RecordingMicroInterpreter* current_interpreter = nullptr;
#else
    tflite::MicroInterpreter* current_interpreter = nullptr;
#endif

    // Single shared resolver
    tflite::MicroMutableOpResolver<20> resolver;

#if ENABLE_MODEL_DEBUG_PRINTS
    // Single shared profiler (reused across all models)
    tflite::CustomProfiler<1024, 20> profiler;

    UBaseType_t uxHighWaterMark = 0;  // For stack checking
#endif

// ========== Inter-model Communication Buffers ==========
#if USE_QUANTIZED_MODEL

    float pre_ssm_gate_scale = 0.0f;
    int32_t pre_ssm_gate_zero_point = 0;
    float pre_ssm_state_scale = 0.0f;
    int32_t pre_ssm_state_zero_point = 0;
    int8_t pre_ssm_gate[kPreSSMGateSize];
    int8_t pre_ssm_state[kPreSSMStateSize];

    float step_output_y_scale = 0.0f;
    int32_t step_output_y_zero_point = 0;

#else

    float pre_ssm_state[kPreSSMStateSize];
    float pre_ssm_gate[kPreSSMGateSize]; 

#endif

    void process_model_output(const TfLiteTensor* tensor, float* output, size_t size) {
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

    void process_model_input(const float* input, TfLiteTensor* tensor, size_t size) {
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

#if USE_QUANTIZED_MODEL
    void requantize_int8_to_tensor(
        const int8_t* source_data,
        float source_scale,
        int32_t source_zero_point,
        TfLiteTensor* dest_tensor,
        size_t size) {
        int8_t* dest_data = tflite::GetTensorData<int8_t>(dest_tensor);
        const float dest_scale = dest_tensor->params.scale;
        const int32_t dest_zero_point = dest_tensor->params.zero_point;
        
        for (int i = 0; i < size; ++i) {
            // Dequantize from source quantization
            float dequantized_value = static_cast<float>(source_data[i] - source_zero_point) * source_scale;
            // Requantize to destination quantization
            int32_t q = static_cast<int32_t>(std::lround(dequantized_value / dest_scale)) + dest_zero_point;
            // Clip to int8 range
            q = std::max<int32_t>(-128, std::min<int32_t>(127, q));
            dest_data[i] = static_cast<int8_t>(q);
        }
    }
#endif

#if ENABLE_MODEL_DEBUG_PRINTS
    void print_memory_debug(const char* step_name, tflite::RecordingMicroInterpreter* interpreter) {
        printf("\n--- %s Memory Allocation ---\n", step_name);
        interpreter->GetMicroAllocator().PrintAllocations();
    }
#endif

    bool resolver_initialized = false;

    // Helper function to create interpreter with shared arena
    bool create_interpreter(
        const uint8_t* model_data,
        const char* model_name) {
        
        const tflite::Model* model = tflite::GetModel(model_data);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            printf("ERROR: %s model version mismatch!\n", model_name);
            return false;
        }

        if (current_interpreter != nullptr) {
            delete current_interpreter;
            current_interpreter = nullptr;
        }
        
        // Initialize resolver once
        if (!this->resolver_initialized) {
            resolver.AddFullyConnected();
            resolver.AddDepthwiseConv2D();
            
            resolver.AddGatherNd();
            resolver.AddReshape();
            resolver.AddTranspose();
            resolver.AddSlice();
            resolver.AddPad();

            resolver.AddMul();
            resolver.AddAdd();
            resolver.AddSum();

            resolver.AddQuantize();
            resolver.AddDequantize();
            
            resolver.AddRelu();
            resolver.AddExp();
            // resolver.AddLogistic();
            // resolver.AddLog();
            
            this->resolver_initialized = true;
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


    bool run_split_model_inference_raw(const float* input_data, float* output_logits) {
#if ENABLE_MODEL_DEBUG_PRINTS
        int64_t inference_start_time = esp_timer_get_time();
#endif
        // ========== STAGE 1: PreSSM ==========
        if (!create_interpreter(pre_model_data, "PreSSM")) {
            return false;
        }
        
        TfLiteTensor* pre_input = current_interpreter->input(0);
        process_model_input(input_data, pre_input, kInputLength);
        
        if (current_interpreter->Invoke() != kTfLiteOk) {
            printf("ERROR: PreSSM inference failed\n");
            return false;
        }
        // Extract outputs: state and gate are outputs from the model
        TfLiteTensor* pre_output_state = current_interpreter->output(0);
        TfLiteTensor* pre_output_gate = current_interpreter->output(1);
        
#if USE_QUANTIZED_MODEL
        pre_ssm_state_scale = pre_output_state->params.scale;
        pre_ssm_state_zero_point = pre_output_state->params.zero_point;
        pre_ssm_gate_scale = pre_output_gate->params.scale;
        pre_ssm_gate_zero_point = pre_output_gate->params.zero_point;
        memcpy(pre_ssm_state, tflite::GetTensorData<int8_t>(pre_output_state), kPreSSMStateSize * sizeof(int8_t));
        memcpy(pre_ssm_gate, tflite::GetTensorData<int8_t>(pre_output_gate), kPreSSMGateSize * sizeof(int8_t));
#else
        memcpy(pre_ssm_state, tflite::GetTensorData<float>(pre_output_state), kPreSSMStateSize * sizeof(float));
        memcpy(pre_ssm_gate, tflite::GetTensorData<float>(pre_output_gate), kPreSSMGateSize * sizeof(float));
#endif
        
#if ENABLE_MODEL_DEBUG_PRINTS
        print_memory_debug("PreSSM", current_interpreter);
        printf("\n--- PreSSM Profiling Results ---\n");
        profiler.LogGroupedSinceLap();
        profiler.AdvanceLap();
#endif
        
        // ========== STAGE 2: StepSSM Loop ==========
        if (!create_interpreter(step_model_data, "StepSSM")) {
            return false;
        }
        
        TfLiteTensor* step_input_x = current_interpreter->input(0);
        TfLiteTensor* step_input_hidden = current_interpreter->input(1);
        TfLiteTensor* step_output_y = current_interpreter->output(0);
        TfLiteTensor* step_output_updated_hidden = current_interpreter->output(1);

#if USE_QUANTIZED_MODEL
        step_output_y_scale = step_output_y->params.scale;
        step_output_y_zero_point = step_output_y->params.zero_point;
        memset(tflite::GetTensorData<int8_t>(step_input_hidden), step_input_hidden->params.zero_point, kHiddenStateSize * sizeof(int8_t));
#else
        memset(tflite::GetTensorData<float>(step_input_hidden), 0, kHiddenStateSize * sizeof(float));
#endif

        for (int t = 0; t < kNumTimesteps; t++) {
#if USE_QUANTIZED_MODEL
            requantize_int8_to_tensor(&pre_ssm_state[t * kDInner], pre_ssm_state_scale, pre_ssm_state_zero_point, 
                                    step_input_x, kDInner);
#else
            memcpy(tflite::GetTensorData<float>(step_input_x), pre_ssm_state + t * kDInner, kDInner * sizeof(float));
#endif

            if (current_interpreter->Invoke() != kTfLiteOk) {
                printf("ERROR: StepSSM inference failed at timestep %d\n", t);
                return false;
            }

#if USE_QUANTIZED_MODEL
            // Copy y_t output to y_all accumulator
            memcpy(pre_ssm_state + t * kDInner, tflite::GetTensorData<int8_t>(step_output_y), kDInner * sizeof(int8_t));
            // Requantize hidden state for next timestep
            requantize_int8_to_tensor(tflite::GetTensorData<int8_t>(step_output_updated_hidden),
                                    step_output_updated_hidden->params.scale, step_output_updated_hidden->params.zero_point,
                                    step_input_hidden, kHiddenStateSize);
#else
            // Copy outputs to buffers for next iteration
            memcpy(pre_ssm_state + t * kDInner, tflite::GetTensorData<float>(step_output_y), kDInner * sizeof(float));
            memcpy(tflite::GetTensorData<float>(step_input_hidden), tflite::GetTensorData<float>(step_output_updated_hidden), kHiddenStateSize * sizeof(float));
#endif
        }

#if ENABLE_MODEL_DEBUG_PRINTS
        print_memory_debug("StepSSM", current_interpreter);
        printf("\n--- StepSSM Profiling Results ---\n");
        profiler.LogGroupedSinceLap();
        profiler.AdvanceLap();
#endif
        
        // ========== STAGE 3: PostSSM ==========
        if (!create_interpreter(post_model_data, "PostSSM")) {
            return false;
        }
        
        TfLiteTensor* post_input_y = current_interpreter->input(0);
        TfLiteTensor* post_input_gate = current_interpreter->input(1);
        
#if USE_QUANTIZED_MODEL
        // Requantize y_all (accumulated outputs) and gate for PostSSM inputs
        requantize_int8_to_tensor(pre_ssm_state, step_output_y_scale, step_output_y_zero_point,
                                post_input_y, kYAllSize);
        requantize_int8_to_tensor(pre_ssm_gate, pre_ssm_gate_scale, pre_ssm_gate_zero_point,
                                post_input_gate, kPreSSMGateSize);
#else
        // Copy float buffers to PostSSM inputs
        memcpy(tflite::GetTensorData<float>(post_input_y), pre_ssm_state, kYAllSize * sizeof(float));
        memcpy(tflite::GetTensorData<float>(post_input_gate), pre_ssm_gate, kPreSSMGateSize * sizeof(float));
#endif

        if (current_interpreter->Invoke() != kTfLiteOk) {
            printf("ERROR: PostSSM inference failed\n");
            return false;
        }
        
        TfLiteTensor* post_output = current_interpreter->output(0);
        process_model_output(post_output, output_logits, kOutputLength);
        
#if ENABLE_MODEL_DEBUG_PRINTS
        print_memory_debug("PostSSM", current_interpreter);
        printf("\n--- PostSSM Profiling Results ---\n");
        profiler.LogGroupedSinceLap();
        printf("\n--- Total Profiling Results ---\n");
        profiler.LogGroupedTotal();
        
        // Check stack
        uxHighWaterMark = uxTaskGetStackHighWaterMark(NULL);
        printf("\nStack High Water Mark: %d bytes remaining\n", (int)uxHighWaterMark);
        
        // Print total inference time
        int64_t inference_end_time = esp_timer_get_time();
        int64_t total_inference_time_us = inference_end_time - inference_start_time;
        float total_inference_time_ms = total_inference_time_us / 1000.0f;
        printf("\nTotal Inference Time: %.2f ms (%.0f µs)\n", total_inference_time_ms, (float)total_inference_time_us);
#endif
        
        return true;
    }

};
