idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_driver_i2s fatfs i2s_examples_common esp_timer
)
