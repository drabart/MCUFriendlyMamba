#include "matrix.h"

#include <string.h>
#include <math.h>

MatError matrix_init(f32_Matrix *mat, uint16_t rows, uint16_t cols, float *buffer)
{
    if (!buffer)
    {
        return MAT_ERR_MEMORY;
    }
    mat->rows = rows;
    mat->cols = cols;
    mat->data = buffer;
    return MAT_OK;
}

// Generate Identity matrix
MatError matrix_identity(f32_Matrix *mat)
{
    if (mat->rows != mat->cols)
    {
        return MAT_ERR_DIM_MISMATCH;
    }

    for (uint16_t i = 0; i < mat->rows; i++)
    {
        mat->data[i * mat->rows + i] = 1;
    }
    return MAT_OK;
}

// Generate zero matrix, allocate new space
MatError matrix_zeros(f32_Matrix *mat)
{
    size_t total = mat->rows * mat->cols;
    for (uint16_t i = 0; i < total; i++)
    {
        mat->data[i] = 0;
    }
    return MAT_OK;
}

void matrix_free(f32_Matrix *mat)
{
    free(mat);
}

MatError matrix_add(const f32_Matrix *a, const f32_Matrix *b, f32_Matrix *result)
{
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != result->rows || a->cols != result->cols)
        return MAT_ERR_DIM_MISMATCH;

    const uint16_t total = a->rows * a->cols;
    for (uint16_t i = 0; i < total; i++)
    {
        result->data[i] = a->data[i] + b->data[i];
    }
    return MAT_OK;
}

MatError matrix_sub(const f32_Matrix *a, const f32_Matrix *b, f32_Matrix *result)
{
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != result->rows || a->cols != result->cols)
        return MAT_ERR_DIM_MISMATCH;

    const uint16_t total = a->rows * a->cols;
    for (uint16_t i = 0; i < total; i++)
    {
        result->data[i] = a->data[i] - b->data[i];
    }
    return MAT_OK;
}

MatError matrix_mul(const f32_Matrix *a, const f32_Matrix *b, f32_Matrix *result)
{
    if (a->cols != b->rows || result->rows != a->rows || result->cols != b->cols)
        return MAT_ERR_DIM_MISMATCH;

    for (uint16_t i = 0; i < a->rows; i++)
    {
        for (uint16_t j = 0; j < b->cols; j++)
        {
            float sum = 0;
            for (uint16_t k = 0; k < a->cols; k++)
            {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }
    return MAT_OK;
}

MatError conv1d(const f32_Matrix *input, const f32_Matrix *weight, const float *bias, f32_Matrix *output)
{
    int kernel_size = self->d_conv;
    int padding = kernel_size - 1;
    int padded_len = seqlen + 2 * padding;
    int groups = self->d_inner;
    float *padded = (float *)calloc(padded_len, sizeof(float));
    if (!padded)
        return Mamba_ERR_MEMORY;
    for (int b = 0; b < batch; b++)
    {
        for (int c = 0; c < groups; c++)
        {
            memset(padded, 0, sizeof(float) * padded_len);
            for (int i = 0; i < seqlen; i++)
            {
                padded[i + padding] = x->data[b * groups * seqlen + c * seqlen + i];
            }
            for (int i = 0; i < seqlen; i++)
            {
                float sum = 0.0f;
                for (int k = 0; k < kernel_size; k++)
                {
                    sum += padded[i + k] * self->conv1d_weight->data[c * kernel_size + k];
                }
                sum += self->conv1d_bias[c];
                float activated = silu(sum);
                x->data[b * groups * seqlen + c * seqlen + i] = activated;
            }
        }
    }
    free(padded);
}
