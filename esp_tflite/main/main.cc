#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "esp_random.h"

// #include "full_model_inference.h"
#include "split_model_inference.h"

extern "C" {

void app_main(void) {
    run_inference();
}

}
