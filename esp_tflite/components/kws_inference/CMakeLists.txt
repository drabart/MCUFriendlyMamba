idf_component_register(
    SRCS
        "split_model_kws_inference.cc"
    INCLUDE_DIRS "include"
    REQUIRES esp-tflite-micro models sample_data esp_timer split_inference
)
