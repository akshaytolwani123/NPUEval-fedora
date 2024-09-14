# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = (
    "A kernel that performs 2D convolution on a bfloat16 input (16x16) with a 4x4 bfloat16 kernel, "
    "stride 2 (runtime parameter), adds scalar bias, and applies ReLU. Produces a 7x7 bfloat16 matrix."
)

def behavioral(in_buffer, kernel, bias, stride):
    in_rows, in_cols = in_buffer.shape
    kernel_size = kernel.shape[0]
    out_rows = (in_rows - kernel_size) // stride + 1
    out_cols = (in_cols - kernel_size) // stride + 1
    out = np.zeros((out_rows, out_cols), dtype=bfloat16)
    for i in range(out_rows):
        for j in range(out_cols):
            window = in_buffer[i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size].astype(np.float32)
            k = kernel.astype(np.float32)
            acc = np.sum(window * k) + float(bias)
            relu = max(acc, 0.0)
            out[i, j] = bfloat16(relu)
    return out

IN_ROWS = 16
IN_COLS = 16
KERNEL_SIZE = 4
stride = 2

in_buffer = np.random.randn(IN_ROWS, IN_COLS).astype(bfloat16)
kernel = np.random.randn(KERNEL_SIZE, KERNEL_SIZE).astype(bfloat16)
bias = bfloat16(np.random.randn(1).item() + 1)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer, kernel],
    rtp_values=[bias, stride],
    tolerances=tolerances
)

pc.write_json("kernel.json")