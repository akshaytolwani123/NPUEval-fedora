# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = (
    "A kernel that performs 1D convolution with kernel size 2 and stride 1 on a bfloat16 input vector, "
    "adds a bfloat16 bias, applies ReLU, and writes bfloat16 output. The input is 256-wide and the kernel has size 2."
)

def behavioral(in_buffer, kernel, bias):
    in_buffer = in_buffer.astype(np.float32)
    kernel = kernel.astype(np.float32)
    bias = float(bias)
    vector_size = in_buffer.shape[0]
    kernel_size = kernel.shape[0]
    stride = 1
    output_size = (vector_size - kernel_size) // stride + 1
    out = np.zeros(output_size, dtype=bfloat16)
    for i in range(output_size):
        acc = 0.0
        for j in range(kernel_size):
            acc += in_buffer[i * stride + j] * kernel[j]
        acc += bias
        acc = acc if acc > 0.0 else 0.0
        out[i] = bfloat16(acc)
    return out

in_buffer = (np.random.randn(256) * 2).astype(bfloat16)
kernel = (np.random.randn(2)).astype(bfloat16)
bias = bfloat16(np.random.randn())

# Use atol/rtol 0.03/0.03 since this is an activation function (has ReLU)
tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer, kernel],
    rtp_values=[float(bias)],
    tolerances=tolerances
)

pc.write_json("kernel.json")