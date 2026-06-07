/*
 * SPDX-FileCopyrightText: 2021-2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Unlicense OR CC0-1.0
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/unistd.h>
#include <sys/stat.h>
#include "sdkconfig.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

static const char *TAG = "ics43434_rec";

#define SAMPLE_SIZE (1024)
i2s_chan_handle_t rx_handle = NULL;

// Define recording parameters
#define SAMPLE_RATE 16000
#define BITS_PER_SAMPLE 16
#define CHANNELS 1        // Mono
#define RECORD_TIME_SEC 3 // How many seconds to record

// Calculate total buffer size needed
#define TOTAL_DATA_SIZE (SAMPLE_RATE * (BITS_PER_SAMPLE / 8) * CHANNELS * RECORD_TIME_SEC)
#define WAV_HEADER_SIZE 44
#define TOTAL_RECORDBUF_SIZE (TOTAL_DATA_SIZE + WAV_HEADER_SIZE)

// Function to generate the WAV header
void generate_wav_header(uint8_t *header, uint32_t data_size, uint32_t sample_rate, uint8_t channels, uint8_t bits_per_sample)
{
    uint32_t total_data_len = data_size + WAV_HEADER_SIZE - 8;
    uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;

    header[0] = 'R';
    header[1] = 'I';
    header[2] = 'F';
    header[3] = 'F';
    header[4] = (total_data_len & 0xff);
    header[5] = ((total_data_len >> 8) & 0xff);
    header[6] = ((total_data_len >> 16) & 0xff);
    header[7] = ((total_data_len >> 24) & 0xff);
    header[8] = 'W';
    header[9] = 'A';
    header[10] = 'V';
    header[11] = 'E';
    header[12] = 'f';
    header[13] = 'm';
    header[14] = 't';
    header[15] = ' ';
    header[16] = 16;
    header[17] = 0;
    header[18] = 0;
    header[19] = 0; // Subchunk1Size
    header[20] = 1;
    header[21] = 0; // AudioFormat (PCM)
    header[22] = channels;
    header[23] = 0;
    header[24] = (sample_rate & 0xff);
    header[25] = ((sample_rate >> 8) & 0xff);
    header[26] = ((sample_rate >> 16) & 0xff);
    header[27] = ((sample_rate >> 24) & 0xff);
    header[28] = (byte_rate & 0xff);
    header[29] = ((byte_rate >> 8) & 0xff);
    header[30] = ((byte_rate >> 16) & 0xff);
    header[31] = ((byte_rate >> 24) & 0xff);
    header[32] = (channels * bits_per_sample / 8);
    header[33] = 0; // BlockAlign
    header[34] = bits_per_sample;
    header[35] = 0; // BitsPerSample
    header[36] = 'd';
    header[37] = 'a';
    header[38] = 't';
    header[39] = 'a';
    header[40] = (data_size & 0xff);
    header[41] = ((data_size >> 8) & 0xff);
    header[42] = ((data_size >> 16) & 0xff);
    header[43] = ((data_size >> 24) & 0xff);
}

void record_wav()
{
    // Allocate buffer. Using MALLOC_CAP_INTERNAL because 3 seconds at 16kHz Mono is ~96KB,
    // which easily fits inside standard ESP32-S3 internal RAM.
    uint8_t *recording_buffer = (uint8_t *)heap_caps_malloc(TOTAL_RECORDBUF_SIZE, MALLOC_CAP_INTERNAL);

    if (recording_buffer == NULL)
    {
        printf("Failed to allocate memory for recording!\n");
        return;
    }

    printf("Recording started for %d seconds...\n", RECORD_TIME_SEC);

    size_t total_bytes_written = WAV_HEADER_SIZE;
    size_t bytes_read = 0;

    while (total_bytes_written < TOTAL_RECORDBUF_SIZE)
    {
        size_t bytes_to_read = SAMPLE_SIZE;
        if (total_bytes_written + bytes_to_read > TOTAL_RECORDBUF_SIZE)
        {
            bytes_to_read = TOTAL_RECORDBUF_SIZE - total_bytes_written;
        }

        esp_err_t err = i2s_channel_read(rx_handle,
                                         (char *)(recording_buffer + total_bytes_written),
                                         bytes_to_read,
                                         &bytes_read,
                                         1000);

        if (err == ESP_OK && bytes_read > 0)
        {
            total_bytes_written += bytes_read;
        }
        else
        {
            printf("I2S Read Error or timeout\n");
            break;
        }
    }

    printf("\nRecording stopped. Generating WAV Header...\n");

    uint32_t actual_data_size = total_bytes_written - WAV_HEADER_SIZE;
    generate_wav_header(recording_buffer, actual_data_size, SAMPLE_RATE, CHANNELS, BITS_PER_SAMPLE);

    printf("=== WAV FILE START ===\n");

#define CHUNK_SIZE 2048
    char hex_chunk[CHUNK_SIZE * 2 + 1];

    size_t chunk_index = 0;
    for (size_t i = 0; i < total_bytes_written; i++)
    {
        sprintf(&hex_chunk[chunk_index * 2], "%02X", recording_buffer[i]);
        chunk_index++;

        if (chunk_index == CHUNK_SIZE)
        {
            hex_chunk[CHUNK_SIZE * 2] = '\0';
            printf("%s", hex_chunk);
            chunk_index = 0;
        }
    }

    if (chunk_index > 0)
    {
        hex_chunk[chunk_index * 2] = '\0';
        printf("%s", hex_chunk);
    }

    printf("\n=== WAV FILE END ===\n");

    heap_caps_free(recording_buffer);
}

void init_microphone(void)
{
    /* Create I2S channel for standard hardware routing */
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    /* Fix: Setup Philips standard timing, but explicitly set 32-bit slot width while receiving 16-bit data precision */
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .bclk = GPIO_NUM_5,
            .din = GPIO_NUM_6,
            .ws = GPIO_NUM_7,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    // This tells the underlying driver to use 32 bits per channel frame (required by ICS43434)
    std_cfg.slot_cfg.slot_bit_width = I2S_SLOT_BIT_WIDTH_32BIT;
    // Ensure it targets the Left channel (assuming physical L/R pin on your mic is tied to GND)
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

void app_main(void)
{
    init_microphone();
    record_wav();

    ESP_ERROR_CHECK(i2s_channel_disable(rx_handle));
    ESP_ERROR_CHECK(i2s_del_channel(rx_handle));
}