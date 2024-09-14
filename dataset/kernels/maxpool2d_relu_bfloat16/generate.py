# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = (
    "A kernel that performs 2D max pooling and applies ReLU on an input bfloat16 matrix (32x32), "
    "using a 2x2 window and stride of 2. The output is a 16x16 matrix in the output buffer where "
    "each element is the maximum of each 2x2 patch with ReLU applied."
)

def behavioral(input):
    rows, cols = input.shape
    out_rows = rows // 2
    out_cols = cols // 2
    output = np.zeros((out_rows, out_cols), dtype=input.dtype)
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            window = input[i:i+2, j:j+2]
            maxval = np.max(window).astype(np.float32)
            relu = max(maxval, 0.0)
            output[i//2, j//2] = bfloat16(relu)
    return output

input = (np.random.randn(32, 32) * 3).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input],
    rtp_values=None
)

pc.write_json("kernel.json")