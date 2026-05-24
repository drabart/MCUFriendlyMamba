idf_component_register(
    SRCS 
        main.cc 
        debug_log.cc 
        full_model_inference.cc 
        split_model_inference.cc 
        model_int8_model_data.cc 
        model_float_model_data.cc
        model_pre_ssm_model_data.cc
        model_step_ssm_model_data.cc
        model_post_ssm_model_data.cc
        model_pre_ssm_int8_model_data.cc
        model_step_ssm_int8_model_data.cc
        model_post_ssm_int8_model_data.cc
    INCLUDE_DIRS "."
)
