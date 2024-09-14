# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = (
    "This AIE kernel performs 2D max pooling with a 2x2 window and stride=2 on a 16x16 input int8 tensor. "
    "After max-pooling, ReLU is applied (all negative values are set to zero). "
    "The result is written to a 8x8 output tensor."
)

def behavioral(input):
    rows, cols = input.shape
    pooled = np.zeros((rows // 2, cols // 2), dtype=input.dtype)
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            window = input[i:i+2, j:j+2]
            max_val = np.max(window)
            pooled[i//2, j//2] = max(max_val, 0)
    return pooled

input = np.random.randint(-128, 128, size=(16, 16), dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input],
    rtp_values=None
)

pc.write_json("kernel.json")