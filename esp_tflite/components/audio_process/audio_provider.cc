/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "audio_provider.h"

#include <cstdlib>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "ringbuf.h"
#include "micro_model_settings.h"

using namespace std;

static const char* TAG = "TF_LITE_AUDIO_PROVIDER";

/* Ringbuffer to hold incoming audio data */
ringbuf_t* g_audio_capture_buffer;
volatile int32_t g_latest_audio_timestamp = 0;

/* Model requires 20ms new data and 10ms old data each time */
constexpr int32_t history_samples_to_keep =
    ((kFeatureDurationMs - kFeatureStrideMs) * (kAudioSampleFrequency / 1000));
constexpr int32_t new_samples_to_get =
    (kFeatureStrideMs * (kAudioSampleFrequency / 1000));

const int32_t kAudioCaptureBufferSize = 40000;
// 1024 samples * 2 bytes per sample (16-bit PCM) = 2048 bytes
const int32_t i2s_bytes_to_read = 2048; 

namespace {
int16_t g_audio_output_buffer[kMaxAudioSampleSize * 32];
bool g_is_audio_initialized = false;
int16_t g_history_buffer[history_samples_to_keep];

// New I2S channel driver handle
i2s_chan_handle_t rx_handle = NULL;
uint8_t g_i2s_read_buffer[i2s_bytes_to_read] = {};
}  // namespace

static void modern_i2s_init(void) {
    ESP_LOGI(TAG, "Initializing modern I2S standard driver for ICS43434...");

    /* Configure channel as Master RX */
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    /* Setup standard Philips 16-bit audio configuration with your working 32-bit slot width trick */
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(kAudioSampleFrequency), // 16000Hz
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = GPIO_NUM_5,
            .ws = GPIO_NUM_7,
            .dout = I2S_GPIO_UNUSED,
            .din = GPIO_NUM_6,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    // Apply specific parameters tailored to your clean hardware performance
    std_cfg.slot_cfg.slot_bit_width = I2S_SLOT_BIT_WIDTH_32BIT;
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

static void CaptureSamples(void* arg) {
    size_t bytes_read = 0;
    modern_i2s_init();

    while (1) {
        /* Read audio frames directly into the intermediate buffer */
        esp_err_t err = i2s_channel_read(rx_handle, 
                                         (void*)g_i2s_read_buffer, 
                                         i2s_bytes_to_read, 
                                         &bytes_read, 
                                         100);

        if (err != ESP_OK || bytes_read <= 0) {
            ESP_LOGE(TAG, "Error in I2S read or timeout occurred");
            continue;
        }

        if (bytes_read < i2s_bytes_to_read) {
            ESP_LOGW(TAG, "Partial I2S read: fetched %d bytes", bytes_read);
        }

        /* Write audio bytes directly into ring buffer. 
           No raw bitwise shifting needed since driver abstracts it to 16-bit */
        int bytes_written = rb_write(g_audio_capture_buffer, 
                                     (uint8_t*)g_i2s_read_buffer, 
                                     bytes_read, 
                                     pdMS_TO_TICKS(100));
        
        if (bytes_written <= 0) {
            ESP_LOGE(TAG, "Could Not Write in Ring Buffer: %d", bytes_written);
            continue;
        } else if (bytes_written < bytes_read) {
            ESP_LOGW(TAG, "Partial Ring Buffer Write: %d out of %d", bytes_written, bytes_read);
        }

        /* Update the timestamp (in ms) to alert the ML pipeline that new audio is processed */
        g_latest_audio_timestamp += ((1000 * (bytes_written / 2)) / kAudioSampleFrequency);
    }

    // Clean up if thread leaves main execution loop (failsafe)
    i2s_channel_disable(rx_handle);
    i2s_del_channel(rx_handle);
    vTaskDelete(NULL);
}

TfLiteStatus InitAudioRecording() {
    g_audio_capture_buffer = rb_init("tf_ringbuffer", kAudioCaptureBufferSize);
    if (!g_audio_capture_buffer) {
        ESP_LOGE(TAG, "Error creating ring buffer");
        return kTfLiteError;
    }

    /* Spawn the background Task to pump data from I2S to Ring Buffer */
    xTaskCreate(CaptureSamples, "CaptureSamples", 1024 * 4, NULL, 10, NULL);
    
    while (!g_latest_audio_timestamp) {
        vTaskDelay(1); // Yield execution briefly to satisfy watchdog timers
    }
    
    ESP_LOGI(TAG, "Audio Recording started successfully");
    return kTfLiteOk;
}

TfLiteStatus GetAudioSamples1(int* audio_samples_size, int16_t** audio_samples) {
    if (!g_is_audio_initialized) {
        TfLiteStatus init_status = InitAudioRecording();
        if (init_status != kTfLiteOk) {
            return init_status;
        }
        g_is_audio_initialized = true;
    }

    int bytes_read = rb_read(g_audio_capture_buffer, (uint8_t*)(g_audio_output_buffer), 16000, 1000);
    if (bytes_read < 0) {
        ESP_LOGI(TAG, "Couldn't read data in time");
        bytes_read = 0;
    }
    *audio_samples_size = bytes_read;
    *audio_samples = g_audio_output_buffer;
    return kTfLiteOk;
}

TfLiteStatus GetAudioSamples(int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
    if (!g_is_audio_initialized) {
        TfLiteStatus init_status = InitAudioRecording();
        if (init_status != kTfLiteOk) {
            return init_status;
        }
        g_is_audio_initialized = true;
    }

    /* Copy historical stride data into output buffer front */
    memcpy((void*)(g_audio_output_buffer), (void*)(g_history_buffer),
           history_samples_to_keep * sizeof(int16_t));

    /* Populate standard inference window size directly behind history offset */
    int bytes_read = rb_read(g_audio_capture_buffer,
                             ((uint8_t*)(g_audio_output_buffer + history_samples_to_keep)),
                             new_samples_to_get * sizeof(int16_t), pdMS_TO_TICKS(200));
    if (bytes_read < 0) {
        ESP_LOGE(TAG, "Model Could not read data from Ring Buffer");
    } else if (bytes_read < new_samples_to_get * sizeof(int16_t)) {
        ESP_LOGD(TAG, "RB FILLED RIGHT NOW IS %d", rb_filled(g_audio_capture_buffer));
        ESP_LOGD(TAG, "Partial Read of Data by Model");
    }

    /* Cascade newer buffer chunks back down into the history frame */
    memcpy((void*)(g_history_buffer),
           (void*)(g_audio_output_buffer + new_samples_to_get),
           history_samples_to_keep * sizeof(int16_t));

    *audio_samples_size = kMaxAudioSampleSize;
    *audio_samples = g_audio_output_buffer;
    return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { 
    return g_latest_audio_timestamp; 
}
