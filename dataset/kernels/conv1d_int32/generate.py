# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "A kernel that performs a 1D convolution operation on an input vector with a given kernel and stride as runtime parameters."

def behavioral(in_buffer, kernel, stride):
    vector_size = in_buffer.shape[0]
    kernel_size = kernel.shape[0]
    output_size = (vector_size - kernel_size) // stride + 1
    out_buffer = np.zeros(output_size, dtype=np.int32)
    for i in range(output_size):
        conv_sum = 0
        for j in range(kernel_size):
            conv_sum += in_buffer[i * stride + j] * kernel[j]
        out_buffer[i] = conv_sum
    return out_buffer

in_buffer = np.random.randint(-10, 10, size=(256,), dtype=np.int32)
kernel = np.random.randint(-10, 10, size=(3,), dtype=np.int32)
stride = 1

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer, kernel],
    rtp_values=[stride]
)

pc.write_json("kernel.json")