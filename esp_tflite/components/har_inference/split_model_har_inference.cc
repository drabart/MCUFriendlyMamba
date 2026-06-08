#include "sdkconfig.h"

#include "split_model_har_inference.h"
#include "split_inference.h"

#include "har_test_samples.h"

// Activity labels for output
const char* ACTIVITY_LABELS[] = {
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
};

#if CONFIG_USE_QUANTIZED_MODEL
#include "model_pre_ssm_int8_har_model_data.h"
#include "model_step_ssm_int8_har_model_data.h"
#include "model_post_ssm_int8_har_model_data.h"
#else
#include "model_pre_ssm_har_model_data.h"
#include "model_step_ssm_har_model_data.h"
#include "model_post_ssm_har_model_data.h"
#endif

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Match the model tensor element type selected by CONFIG_USE_QUANTIZED_MODEL.
#if CONFIG_USE_QUANTIZED_MODEL
using model_tensor_t = int8_t;
constexpr const char* kModelTypeName = "HAR INT8";
#else
using model_tensor_t = float;
constexpr const char* kModelTypeName = "HAR Float32";
#endif

// ========== Dimension Constants ==========
constexpr int kNumFeatures = 57;
constexpr int kNumTimesteps = 10;
constexpr int kDInner = 128;
constexpr int kDState = 16;
constexpr int kNumClasses = 6;

SplitInference<kNumTimesteps, kDInner, kDState, kNumFeatures, kNumClasses> model_inference;

void run_inference_har() {
#if CONFIG_USE_QUANTIZED_MODEL
    model_inference.setup_split_model_inference(
        g_model_pre_ssm_int8_har_model_data, 
        g_model_step_ssm_int8_har_model_data, 
        g_model_post_ssm_int8_har_model_data,
        kModelTypeName
    );
#else
    model_inference.setup_split_model_inference(
        g_model_pre_ssm_har_model_data, 
        g_model_step_ssm_har_model_data, 
        g_model_post_ssm_har_model_data,
        kModelTypeName
    );
#endif
    
#if CONFIG_ENABLE_MODEL_DEBUG_PRINTS
    const int num_samples = 1;
#else
    const int num_samples = 50;
#endif

    int correct_predictions = 0;
    
    printf("\n=== Running %s inference on %d test samples ===\n\n", kModelTypeName, num_samples);
    
    for (int i = 0; i < num_samples; i++) {
        const float* test_input = har_test_data[i];
        uint8_t true_label = har_test_labels[i];
        int predicted_class = -1;
        
        if (!model_inference.run_split_model_inference(test_input, &predicted_class)) {
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
                   i + 1, ACTIVITY_LABELS[predicted_class], ACTIVITY_LABELS[true_label], 
                   is_correct ? "✓" : "✗");
        }
        
        if ((i + 1) % 10 == 0) {
            printf("Processed %d samples...\n", i + 1);
        }
    }
    
    // Print accuracy
    float accuracy = (float)correct_predictions / num_samples * 100.0f;
    printf("\n=== %s Results ===\n", kModelTypeName);
    printf("Correct predictions: %d / %d\n", correct_predictions, num_samples);
    printf("Accuracy: %.2f%%\n\n", accuracy);
}
