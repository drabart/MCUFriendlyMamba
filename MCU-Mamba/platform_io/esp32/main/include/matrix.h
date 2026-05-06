/**
 * @file matrix.h
 *
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef float MatType;
    typedef enum
    {
        MAT_OK,
        MAT_ERR_DIM_MISMATCH,
        MAT_ERR_MEMORY,
        MAT_ERR_SINGULAR
    } MatError;

    typedef struct
    {
        uint16_t rows;
        uint16_t cols;
        MatType *data;
    } Matrix;

    MatError matrix_init(Matrix *mat, uint16_t rows, uint16_t cols, MatType *buffer);
    MatError matrix_add(const Matrix *a, const Matrix *b, Matrix *result);
    MatError matrix_sub(const Matrix *a, const Matrix *b, Matrix *result);
    MatError matrix_mul(const Matrix *a, const Matrix *b, Matrix *result);

    MatError matrix_sigmoid(Matrix *mat, Matrix *result);
    MatError matrix_hadamard_prod(Matrix *a, Matrix *b, Matrix *result);

    Matrix *matrix_zeros(uint16_t rows, uint16_t cols);
    Matrix *matrix_identity(uint16_t rows);

    void matrix_free(Matrix *mat);

#ifdef __cplusplus
}
#endif
