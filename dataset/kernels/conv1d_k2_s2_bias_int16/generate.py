# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = (
    "A kernel that performs a 1D convolution with kernel size 2 and stride 2 on an int16 vector "
    "with an int16 kernel and adds a scalar int16 bias to the accumulator before writing the result. "
    "Input vector is 128 elements (int16), kernel is length 2 (int16), and bias is scalar int16. "
    "The output is ((128 - 2) // 2) + 1 = 64 elements (int16), with result saturated to int16 range."
)

def behavioral(in_buffer, kernel, bias):
    vector_size = in_buffer.shape[0]
    kernel_size = kernel.shape[0]
    stride = 2
    num_windows = (vector_size - kernel_size) // stride + 1
    out = np.zeros(num_windows, dtype=np.int16)
    for i in range(num_windows):
        acc = int(bias)
        for j in range(kernel_size):
            acc += int(in_buffer[i * stride + j]) * int(kernel[j])
        acc = np.clip(acc, -32768, 32767)
        out[i] = acc
    return out

in_buffer = np.random.randint(-100, 100, size=128, dtype=np.int16)
kernel    = np.random.randint(-10, 10,   size=2,   dtype=np.int16)
bias      = np.int16(np.random.randint(-10, 10))

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer, kernel],
    rtp_values=[int(bias)]
)

pc.write_json("kernel.json")