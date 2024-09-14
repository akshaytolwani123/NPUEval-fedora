# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "A kernel that performs a 1D convolution with a bfloat16 vector of length 256. Uses a convolution kernel of size 4, stride 1, adds a bfloat16 bias after convolution, applies ReLU, and writes bfloat16 output. Output vector is length 253."

def behavioral(in_buffer, kernel, bias):
    VECTOR_SIZE = in_buffer.shape[0]
    KERNEL_SIZE = kernel.shape[0]
    STRIDE = 1
    num_windows = (VECTOR_SIZE - KERNEL_SIZE) // STRIDE + 1
    output = np.zeros(num_windows, dtype=bfloat16)
    for i in range(num_windows):
        acc = 0.0
        for j in range(KERNEL_SIZE):
            acc += float(in_buffer[i * STRIDE + j]) * float(kernel[j])
        acc += float(bias)
        out_val = max(acc, 0.0)
        output[i] = bfloat16(out_val)
    return output

in_buffer = np.random.randn(256).astype(bfloat16)
kernel = np.random.randn(4).astype(bfloat16)
bias = bfloat16(np.random.randn()+1)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer, kernel],
    rtp_values=[bias],
    tolerances=tolerances
)

pc.write_json("kernel.json")