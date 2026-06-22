/* Copyright 2020-2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "main_functions.h"
#include "esp_log.h"
#include "driver/gpio.h"

#include "audio_provider.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

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

namespace
{
    FeatureProvider *feature_provider = nullptr;
    int32_t previous_time = 0;

    float feature_buffer[kFeatureElementCount];

    // Keyword labels for output (35 speech command keywords)
    const char *KEYWORD_LABELS[] = {
        "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
        "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
        "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
        "tree", "two", "up", "visual", "wow", "yes", "zero"};

    const int id_to_number[] = {
        -1, -1, -1, -1, -1, -1, 8, 5, -1,
        -1, 4, -1, -1, -1, -1, -1, -1, 9,
        -1, -1, -1, 1, -1, 7, -1, 6, -1, 3,
        -1, 2, -1, -1, -1, -1, 0};

    std::vector<int> accumulated_digits;

    // Helper function to print the current state of the running number
    void print_accumulated_number()
    {
        // \r moves the cursor to the beginning of the CURRENT line.
        // \033[K clears the rest of that line so old numbers don't ghost.
        printf("\r\033[K>>> CURRENT NUMBER: ");
        if (accumulated_digits.empty())
        {
            printf("[Empty]\n");
        }
        else
        {
            for (int digit : accumulated_digits)
            {
                printf("%d", digit);
            }
            printf("\n");
        }
        // Force the output to show immediately without waiting for a newline (\n)
        fflush(stdout);
    }
}

#if CONFIG_USE_QUANTIZED_MODEL
using model_tensor_t = int8_t;
constexpr const char *kModelTypeName = "KWS INT8";
#else
using model_tensor_t = float;
constexpr const char *kModelTypeName = "KWS Float32";
#endif

// ========== Dimension Constants ==========
constexpr int kNumFeatures = 40;  // MFCC feature dimension
constexpr int kNumTimesteps = 49; // Sequence length (time frames)
constexpr int kDInner = 128;      // SSM inner dimension
constexpr int kDState = 16;       // SSM state space dimension
constexpr int kNumClasses = 35;   // Keyword classification classes

#define BUTTON_PIN GPIO_NUM_15

SplitInference<kNumTimesteps, kDInner, kDState, kNumFeatures, kNumClasses> model_inference;

// The name of this function is important for Arduino compatibility.
void setup()
{
#if CONFIG_USE_QUANTIZED_MODEL
    model_inference.setup_split_model_inference(
        g_model_pre_ssm_int8_kws_model_data,
        g_model_step_ssm_int8_kws_model_data,
        g_model_post_ssm_int8_kws_model_data,
        kModelTypeName);
#else
    model_inference.setup_split_model_inference(
        g_model_pre_ssm_kws_model_data,
        g_model_step_ssm_kws_model_data,
        g_model_post_ssm_kws_model_data,
        kModelTypeName);
#endif

    static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                   feature_buffer);
    feature_provider = &static_feature_provider;

    previous_time = 0;

    // Configure button on GPIO 15
    gpio_set_direction(BUTTON_PIN, GPIO_MODE_INPUT);
    gpio_set_pull_mode(BUTTON_PIN, GPIO_PULLUP_ONLY);
}

// The name of this function is important for Arduino compatibility.
void audio_process()
{
    int button_level = gpio_get_level(BUTTON_PIN);
    const int32_t current_time = LatestAudioTimestamp();

    // Fetch the spectrogram for the current time.
    int how_many_new_slices = 0;
    TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
        previous_time, current_time, &how_many_new_slices);
    if (feature_status != kTfLiteOk)
    {
        MicroPrintf("Feature generation failed");
        return;
    }
    previous_time = current_time;

    if (button_level == 1 || how_many_new_slices == 0)
    {
        // button not pressed or no new audio
        return;
    }

    int predicted_class = -1;
    float confidence = 0.0f;
    if (!model_inference.run_split_model_inference(feature_buffer, &predicted_class, &confidence))
    {
        printf("Inference failed\n");
    }

    if (predicted_class < 0 || predicted_class >= 35)
    {
        printf("Error: predicted_class (%d) is out of bounds!\n", predicted_class);
        return;
    }

    static int32_t last_added_time = 0;
    const char *keyword = KEYWORD_LABELS[predicted_class];
    int mapped_number = id_to_number[predicted_class];
    if (confidence >= 0.20f)
    {
        printf("Predicted: %s (Confidence: %.2f)\n", keyword, confidence);
    }
}

void kws_test()
{
    int sample_count = 50;
    int correct_predictions = 0;

    for (int i = 0; i < sample_count; i++)
    {
        int predicted_class = -1;

        if (!model_inference.run_split_model_inference(kws_test_data[i], &predicted_class))
        {
            printf("Inference failed\n");
        }

        if (predicted_class != kws_test_labels[i])
        {
            printf("Sample %2d: Predicted %s, True %s ✗\n",
                   i + 1, KEYWORD_LABELS[predicted_class], KEYWORD_LABELS[kws_test_labels[i]]);
        }
        else
        {
            correct_predictions++;
        }
    }

    printf("Correct predictions: %d/%d\n", correct_predictions, sample_count);

    for (;;)
    {
    }
}

void loop()
{
    audio_process();
    // kws_test();
}
