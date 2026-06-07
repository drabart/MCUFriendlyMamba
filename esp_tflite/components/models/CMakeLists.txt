idf_component_register(
    SRCS
        "audio_process/audio_preprocessor_float_model_data.cc"
        "audio_process/audio_preprocessor_int8_model_data.cc"

        "full_har_inference/model_full_har_model_data.cc"
        "full_har_inference/model_full_int8_har_model_data.cc"

        "split_har_inference/model_pre_ssm_har_model_data.cc"
        "split_har_inference/model_step_ssm_har_model_data.cc"
        "split_har_inference/model_post_ssm_har_model_data.cc"
        "split_har_inference/model_pre_ssm_int8_har_model_data.cc"
        "split_har_inference/model_step_ssm_int8_har_model_data.cc"
        "split_har_inference/model_post_ssm_int8_har_model_data.cc"

        "split_kws_inference/model_pre_ssm_kws_model_data.cc"
        "split_kws_inference/model_step_ssm_kws_model_data.cc"
        "split_kws_inference/model_post_ssm_kws_model_data.cc"
        "split_kws_inference/model_pre_ssm_int8_kws_model_data.cc"
        "split_kws_inference/model_step_ssm_int8_kws_model_data.cc"
        "split_kws_inference/model_post_ssm_int8_kws_model_data.cc"
    INCLUDE_DIRS 
        "audio_process/include"
        "full_har_inference/include"
        "split_har_inference/include"
        "split_kws_inference/include"
    REQUIRES esp-tflite-micro
)
