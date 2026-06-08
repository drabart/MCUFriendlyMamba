idf_register_component(
    SRCS audio_provider.cc feature_provider.cc micro_features_generator.cc main_functions.cc ringbuf.cc
    INCLUDE_DIRS "include"
    PRIV_REQUIRES spi_flash driver esp_driver_i2s esp_timer 
)