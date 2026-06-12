idf_component_register(
    SRCS 
        "audio_provider.cc"    
        "audio_preprocessor_float_model_data.cc"
        "feature_provider.cc" 
        "micro_features_generator.cc" 
        "main_functions.cc" 
        "ringbuf.c"
    INCLUDE_DIRS "include"
    PRIV_REQUIRES esp-tflite-micro spi_flash driver esp_driver_i2s esp_timer kws_inference models split_inference
)