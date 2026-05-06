#include <stdio.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

#include "data.h"
#include "esp_timer.h"

static const char *TAG = "MAIN_APP";

void app_main(void)
{
    ESP_LOGI(TAG, "Starting ESP32-WROOM-32D...");

    for (;;)
    {
        uint32_t free_heap_size = esp_get_free_heap_size();
        ESP_LOGI(TAG, "Free Heap Size: %d", free_heap_size);

        // Measure data_const speed
        int64_t t0 = esp_timer_get_time();
        uint32_t sum_const = 0;
        for (size_t i = 0; i < DATA_SIZE; i++)
        {
            sum_const += data_const[i];
        }
        int64_t t1 = esp_timer_get_time();
        int64_t elapsed_const_us = t1 - t0;
        ESP_LOGI(TAG, "data_const: sum=%u, elapsed: %lld us (%.3f ms)", sum_const, (long long)elapsed_const_us, ((double)elapsed_const_us) / 1000.0);

    #ifdef DATA_MUT_AVAILABLE
        // Measure data_mut speed
        t0 = esp_timer_get_time();
        uint32_t sum_mut = 0;
        for (size_t i = 0; i < DATA_SIZE; i++)
        {
            sum_mut += data_mut[i];
        }
        t1 = esp_timer_get_time();
        int64_t elapsed_mut_us = t1 - t0;
        ESP_LOGI(TAG, "data_mut:   sum=%u, elapsed: %lld us (%.3f ms)", sum_mut, (long long)elapsed_mut_us, ((double)elapsed_mut_us) / 1000.0);
    #endif

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
