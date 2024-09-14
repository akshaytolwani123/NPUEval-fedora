# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "A kernel that performs a 2D convolution on a bfloat16 input matrix (16x16) with a bfloat16 2x2 kernel and a runtime stride parameter. The output is also in bfloat16. No padding."

def behavioral(input_matrix, kernel, stride):
    input_rows = input_matrix.shape[0]
    input_cols = input_matrix.shape[1]
    kernel_size = kernel.shape[0]
    output_rows = (input_rows - kernel_size) // stride + 1
    output_cols = (input_cols - kernel_size) // stride + 1

    output_matrix = np.zeros((output_rows, output_cols), dtype=np.float32)

    for i in range(output_rows):
        for j in range(output_cols):
            region = input_matrix[i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size].astype(np.float32)
            kv = kernel.astype(np.float32)
            output_matrix[i, j] = np.sum(region * kv)
    return output_matrix.astype(bfloat16)

input_matrix = (np.random.randn(16, 16) * 2).astype(bfloat16)
kernel = (np.random.randn(2, 2) * 2).astype(bfloat16)
stride = 1

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_matrix, kernel],
    rtp_values=[stride]
)

pc.write_json("kernel.json")