# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = (
    "A kernel that performs 2D convolution with a 2x2 bfloat16 kernel on a 16x16 bfloat16 input, stride 1, "
    "adds a bfloat16 bias, and applies ReLU activation to each output. Output is a 15x15 bfloat16 matrix."
)

def behavioral(input, kernel, bias):
    IN_ROWS, IN_COLS = input.shape
    KERNEL_SIZE = kernel.shape[0]
    OUT_ROWS = IN_ROWS - KERNEL_SIZE + 1
    OUT_COLS = IN_COLS - KERNEL_SIZE + 1
    output = np.zeros((OUT_ROWS, OUT_COLS), dtype=bfloat16)
    for i in range(OUT_ROWS):
        for j in range(OUT_COLS):
            acc = float(bias)
            for ki in range(KERNEL_SIZE):
                for kj in range(KERNEL_SIZE):
                    acc += float(input[i + ki, j + kj]) * float(kernel[ki, kj])
            relu = max(acc, 0.0)
            output[i, j] = bfloat16(relu)
    return output

IN_ROWS, IN_COLS = 16, 16
KERNEL_SIZE = 2

input = (np.random.randn(IN_ROWS, IN_COLS) * 2).astype(bfloat16)
kernel = (np.random.randn(KERNEL_SIZE, KERNEL_SIZE)).astype(bfloat16)
bias = bfloat16(np.random.randn())

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input, kernel],
    rtp_values=[bias]
)

pc.write_json("kernel.json")