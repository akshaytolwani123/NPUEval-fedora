# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "A kernel that performs vector-matrix multiplication of a vector with shape (16,) and matrix of (16,16)."

def behavioral(vector, matrix):
    vector_size = 16
    matrix_cols = 16
    
    m = matrix.reshape((vector_size, matrix_cols))
    result = np.zeros(matrix_cols, dtype=np.int32)
    for i in range(matrix_cols):
        for j in range(vector_size):
            result[i] += vector[j] * m[j, i]
    return result

vector_size = 16
matrix_cols = 16
vector = np.random.randint(-10, 10, size=(vector_size,), dtype=np.int32)
matrix = np.random.randint(-10, 10, size=(vector_size, matrix_cols), dtype=np.int32)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[vector, matrix],
    rtp_values=None
)

pc.write_json("kernel.json")