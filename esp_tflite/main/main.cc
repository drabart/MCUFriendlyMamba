#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "run_inference.h"

extern "C" {

void app_main(void) {
  setup();
  while (true) {
    loop();
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}

}
