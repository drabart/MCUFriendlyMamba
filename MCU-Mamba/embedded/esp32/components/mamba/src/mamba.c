/**
 * @file mamba.c
 * @brief mamba inference kernel
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#include "mamba.h"

#include "matrix.h"
#include "mamba_weights.h"
#include <stdlib.h>
#include <math.h>

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}
float silu(float x)
{
    return x * sigmoid(x);
}
float softplus(float x)
{
    return logf(1.0f + expf(x));
}

typedef struct
{
    uint16_t d_model;
    uint16_t d_state;
    uint16_t d_conv;
    uint16_t expand;
    uint16_t d_inner;
    uint16_t dt_rank;

    // expand projection
    f32_Matrix *state_weight;
    f32_Matrix *gate_weight;

    // convolution kernel
    f32_Matrix *conv1d_weight;
    float *conv1d_bias;

    // SSM
    f32_Matrix *A_log;
    f32_Matrix *B_proj_weight;
    f32_Matrix *C_proj_weight;
    float *D;

    // dt
    f32_Matrix *xdt_proj_weight;
    float *dt_proj_bias;
    f32_Matrix *dt_proj_weight;

    // out
    f32_Matrix *out_proj_weight;
} Mamba_Variables;

Mamba_Variables *Mamba_Init()
{
    Mamba_Variables *m = (Mamba_Variables *)malloc(sizeof(Mamba_Variables));
    // TODO
    // m->d_model = mamba_d_model;
    // m->d_state = mamba_d_state;
    // m->d_conv = mamba_d_conv;
    // m->d_inner = mamba_expand * mamba_d_model;
    // m->dt_rank = mamba_dt_rank;
    // m->in_proj_weight = &mamba_in_proj_weight_matrix;
    // m->A_log = &mamba_A_log_matrix;
    // m->conv1d_weight = &mamba_conv1d_weight_matrix;
    // m->conv1d_bias = (float *)mamba_conv1d_bias;
    // m->x_proj_weight = &mamba_x_proj_weight_matrix;
    // m->dt_proj_weight = &mamba_dt_proj_weight_matrix;
    // m->dt_proj_bias = (float *)mamba_dt_proj_bias;
    // m->D = (float *)mamba_D;
    // m->out_proj_weight = &mamba_out_proj_weight_matrix;
    return m;
}

MambaError Mamba_Forward(Mamba_Variables *self, f32_Matrix *input, f32_Matrix *out)
{
    int seqlen = input->rows;
    int dim = input->cols;

    // state path
    f32_Matrix state;
    matrix_mul(input, self->state_weight, state);

    f32_Matrix conv_out;
    matrix_conv1d(state, self->conv1d_weight, self->conv1d_bias, &conv_out);

    matrix_silu(&conv_out, &state);

    f32_Matrix A;
    f32_Matrix B;
    f32_Matrix C;

    // calculate dt
    f32_Matrix xdt;
    f32_Matrix dt;
    matrix_mul(state, self->xdt_proj_weight, xdt);
    matrix_add(&xdt, self->dt_proj_bias, &xdt);
    matrix_mul(state, self->dt_proj_weight, dt);
    matrix_softplus(&dt);

    // -expf(self->A_log->data[idx]);
    matrix_mul(&A)

        matrix_mul

            f32_Matrix ssm_out = ; // TODO

    // gate path
    f32_Matrix gate;
    matrix_mul(input, self->gate_weight, gate);

    // combine
    matrix_pointwise_mul(&state, &gate, out);

    return Mamba_OK;
}