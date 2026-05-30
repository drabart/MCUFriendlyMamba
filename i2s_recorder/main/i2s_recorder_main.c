/*
 * SPDX-FileCopyrightText: 2021-2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Unlicense OR CC0-1.0
 */

/* I2S Digital Microphone Recording Example */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/unistd.h>
#include <sys/stat.h>
#include "sdkconfig.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_system.h"
#include "esp_vfs_fat.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "driver/spi_common.h"
#include "format_wav.h"
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "pdm_rec_example";

#define NUM_CHANNELS (1)
#define SAMPLE_SIZE (CONFIG_EXAMPLE_BIT_SAMPLE * 64)

i2s_chan_handle_t rx_handle = NULL;

static int16_t i2s_readraw_buff[SAMPLE_SIZE];
size_t bytes_read;
const int WAVE_HEADER_SIZE = 44;

static void print_loudness_bar(const int16_t *samples, size_t sample_count)
{
    const int bar_width = 40;
    int32_t sum = 0;

    for (size_t i = 0; i < sample_count; i++)
    {
        sum += abs(samples[i]);
    }

    float level = (float)sum / (float)(((1 << 15) - 1) * sample_count);
    if (level > 1.0f)
    {
        level = 1.0f;
    }
    int filled = (int)(level * bar_width);

    char buffer[128u];
    int written = snprintf(buffer, sizeof(buffer), "[");
    for (int i = 0; i < bar_width; i++)
    {
        sprintf(buffer + written + i, "%c", (i < filled) ? '#' : ' ');
    }
    sprintf(buffer + written + bar_width, "] %3d%%\n", (int)(level * 100.0f));
    printf("%s", buffer);
}

void record_wav()
{
    int64_t time_sum = 0;
    int64_t time_count = 0;

    for (;;)
    {
        int64_t time_start = esp_timer_get_time();

        ESP_ERROR_CHECK(i2s_channel_read(rx_handle, (char *)i2s_readraw_buff, SAMPLE_SIZE, &bytes_read, 1000));

        size_t sample_count = bytes_read / sizeof(int16_t);
        print_loudness_bar(i2s_readraw_buff, sample_count);

        int64_t time_end = esp_timer_get_time();
        int64_t time_diff = time_end - time_start;
        time_sum += time_diff;
        time_count++;

        // if (time_count % 10 == 0)
        // {
        //     int64_t avg_time = time_sum / time_count;
        //     printf("\nAverage read time: %.2f ms\n", avg_time / 1000.0);
        //     time_sum = 0;
        //     time_count = 0;
        // }
    }

    printf("\n");
}

void init_microphone(void)
{
    /* Create I2S channel for ICS43434 standard I2S microphone */
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    /* Configure standard I2S mode for ICS43434 */
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(CONFIG_EXAMPLE_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PCM_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .bclk = 5,
            .din = 6,
            .ws = 7,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

void app_main(void)
{
    printf("PDM microphone recording example start\n--------------------------------------\n");
    // Acquire a I2S PDM channel for the PDM digital microphone
    init_microphone();
    // Start Recording
    record_wav();
    // Stop I2S driver and destroy
    ESP_ERROR_CHECK(i2s_channel_disable(rx_handle));
    ESP_ERROR_CHECK(i2s_del_channel(rx_handle));
}
