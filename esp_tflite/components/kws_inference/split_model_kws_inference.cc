#include "sdkconfig.h"

#include "split_model_kws_inference.h"
#include "split_inference.h"

#include "kws_test_samples.h"

#if CONFIG_USE_QUANTIZED_MODEL
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

// Match the model tensor element type selected by CONFIG_USE_QUANTIZED_MODEL.
#if CONFIG_USE_QUANTIZED_MODEL
using model_tensor_t = int8_t;
constexpr const char* kModelTypeName = "KWS INT8";
#else
using model_tensor_t = float;
constexpr const char* kModelTypeName = "KWS Float32";
#endif

// ========== Dimension Constants ==========
constexpr int kNumFeatures = 40;           // MFCC feature dimension
constexpr int kNumTimesteps = 51;          // Sequence length (time frames)
constexpr int kDInner = 128;               // SSM inner dimension
constexpr int kDState = 16;                // SSM state space dimension
constexpr int kNumClasses = 35;            // Keyword classification classes

SplitInference<kNumTimesteps, kDInner, kDState, kNumFeatures, kNumClasses> model_inference;

void run_inference_kws() {
#if CONFIG_USE_QUANTIZED_MODEL
    model_inference.setup_split_model_inference(
        g_model_pre_ssm_int8_kws_model_data, 
        g_model_step_ssm_int8_kws_model_data, 
        g_model_post_ssm_int8_kws_model_data,
        kModelTypeName
    );
#else
    model_inference.setup_split_model_inference(
        g_model_pre_ssm_kws_model_data, 
        g_model_step_ssm_kws_model_data, 
        g_model_post_ssm_kws_model_data,
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
        const float* test_input = kws_test_data[i];
        uint8_t true_label = kws_test_labels[i];
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
                   i + 1, KEYWORD_LABELS[predicted_class], KEYWORD_LABELS[true_label], 
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
