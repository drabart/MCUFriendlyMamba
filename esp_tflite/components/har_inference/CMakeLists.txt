idf_component_register(
    SRCS
        "full_model_inference.cc"
        "split_model_inference.cc"
    INCLUDE_DIRS "include"
    REQUIRES esp-tflite-micro models sample_data esp_timer
)
