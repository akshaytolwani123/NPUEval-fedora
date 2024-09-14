# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "A kernel that performs a 2D convolution operation on an input matrix with a given kernel and stride as runtime parameter. Both matrix and kernel are assumed to have a square shape."

def behavioral(input_matrix, kernel, stride):
    input_rows = input_matrix.shape[0]
    input_cols = input_matrix.shape[1]
    kernel_size = kernel.shape[0]
    output_rows = (input_rows - kernel_size) // stride + 1
    output_cols = (input_cols - kernel_size) // stride + 1
    output_matrix = np.zeros((output_rows, output_cols), dtype=np.int32)
    for i in range(output_rows):
        for j in range(output_cols):
            conv_sum = 0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    conv_sum += input_matrix[i * stride + ki, j * stride + kj] * kernel[ki, kj]
            output_matrix[i, j] = conv_sum
    return output_matrix

input_rows = 16
input_cols = 16
kernel_size = 3
stride = 1

input_matrix = np.random.randint(-10, 10, size=(input_rows, input_cols), dtype=np.int32)
kernel = np.random.randint(-10, 10, size=(kernel_size, kernel_size), dtype=np.int32)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_matrix, kernel],
    rtp_values=[stride]
)

pc.write_json("kernel.json")