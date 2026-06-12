// #include "full_model_har_inference.h"
// #include "split_model_har_inference.h"
// #include "split_model_kws_inference.h"
#include "main_functions.h"

extern "C" {

void app_main(void) {
    // run_inference();
    setup();
    while (true) {
        loop();
    }
}

}
