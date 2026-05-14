/**
 * @file matrix.h
 *
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#pragma once

#include <stdint.h>

typedef enum {
    MAT_OK,
    MAT_ERR_DIM_MISMATCH,
    MAT_ERR_MEMORY,
    MAT_ERR_SINGULAR
} MatError;

typedef struct {
    uint16_t rows;
    uint16_t cols;
    float* data;
} f32_Matrix;

MatError matrix_init(f32_Matrix* mat, uint16_t rows, uint16_t cols, float* buffer);

// TODO in place operations?
MatError matrix_add(const f32_Matrix* a, const f32_Matrix* b, f32_Matrix* result);
MatError matrix_sub(const f32_Matrix* a, const f32_Matrix* b, f32_Matrix* result);
MatError matrix_mul(const f32_Matrix* a, const f32_Matrix* b, f32_Matrix* result);

MatError matrix_zeros(f32_Matrix* mat);
MatError matrix_identity(f32_Matrix* mat);

void matrix_free(f32_Matrix* mat);
