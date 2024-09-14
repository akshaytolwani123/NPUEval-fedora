# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = (
    "A kernel that computes the matrix-vector multiplication of an int8 matrix (16x16, row-major) and "
    "an int8 vector (length 16). Result is an int32 vector of length 16."
)

def behavioral(matrix, vector):
    mat = matrix.reshape((16, 16)).astype(np.int32)
    vec = vector.astype(np.int32)
    result = np.zeros(16, dtype=np.int32)
    for i in range(16):
        acc = 0
        for j in range(16):
            acc += mat[i, j] * vec[j]
        result[i] = acc
    return result

matrix = np.random.randint(-20, 20, (16, 16), dtype=np.int8)
vector = np.random.randint(-20, 20, 16, dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[matrix.flatten(), vector],
    rtp_values=None
)

pc.write_json("kernel.json")