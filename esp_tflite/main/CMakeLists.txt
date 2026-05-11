
#
# Main component of TF Micro project 'micro_speech'.
#

idf_component_register(
    SRCS main.cc run_inference.cc model_int8_model_data.cc
    INCLUDE_DIRS ".")
