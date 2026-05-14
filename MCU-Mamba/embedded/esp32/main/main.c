#include <stdio.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

#include "weights.h"

static const char *TAG = "MAIN_APP";

void app_main(void)
{
    for (;;)
    {
        ESP_LOGI(TAG, "Linear Weights Matrix: %d x %d", linear_weights.rows, linear_weights.cols);

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
