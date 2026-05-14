#pragma once

#include "matrix.h"

float linear_in[] = {
    0.1f, 0.2f, 0.3f,
    0.4f, 0.5f, 0.6f,
    0.7f, 0.8f, 0.9f};
const f32_Matrix linear_weights = {
    3, 3, linear_in};