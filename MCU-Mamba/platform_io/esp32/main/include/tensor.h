#pragma once

#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef float TensorType;
    typedef struct
    {
        uint16_t d1;
        uint16_t d2;
        uint16_t d3;
        TensorType *data;
    } Tensor;

    Tensor *tensor_zeros(uint16_t d1, uint16_t d2, uint16_t d3);
    void tensor_free(Tensor *tensor);

#ifdef __cplusplus
}
#endif
