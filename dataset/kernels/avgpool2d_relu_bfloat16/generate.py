# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)
description = "A kernel that performs 2D average pooling with ReLU activation on an input matrix (32x32), using a 2x2 window and stride 2. The output is a 16x16 matrix written to the output buffer. All operations use bfloat16."

def behavioral(input):
    rows, cols = input.shape
    out_rows = rows // 2
    out_cols = cols // 2
    output = np.zeros((out_rows, out_cols), dtype=input.dtype)
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            window = input[i:i+2, j:j+2].astype(np.float32)
            avg = window.mean()
            relu = max(avg, 0.0)
            output[i//2, j//2] = bfloat16(relu)
    return output

in_buffer = (np.random.randn(32,32) * 4).astype(bfloat16)
tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")