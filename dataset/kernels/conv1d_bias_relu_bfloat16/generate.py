# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = (
    "A 1D convolution kernel operating on a 256-length bfloat16 vector input "
    "using a 4-element bfloat16 kernel, bfloat16 bias, and reported stride (runtime parameter). "
    "The sum is computed, biased, ReLU applied, and written to the output buffer. "
    "Output size = (256-4)//stride+1."
)

def behavioral(in_buffer, kernel, bias, stride):
    vector_size = in_buffer.shape[0]
    kernel_size = kernel.shape[0]
    output_size = (vector_size - kernel_size) // stride + 1
    out_buffer = np.zeros(output_size, dtype=bfloat16)
    for i in range(output_size):
        acc = bfloat16(0.0)
        for j in range(kernel_size):
            prod = bfloat16(float(in_buffer[i*stride + j]) * float(kernel[j]))
            acc = bfloat16(float(acc) + float(prod))
        acc = bfloat16(float(acc) + float(bias))
        if float(acc) < 0.0:
            acc = bfloat16(0.0)
        out_buffer[i] = acc
    return out_buffer

in_buffer = np.random.randn(256).astype(bfloat16)
kernel = np.random.randn(4).astype(bfloat16)
bias = bfloat16(np.random.randn(1).item() + 1)
stride = 2

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer, kernel],
    rtp_values=[bias, stride]
)

pc.write_json("kernel.json")