# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel computes the mean squared error (MSE) between two bfloat16 input vectors (length 256), returning the result as a single bfloat16."

def behavioral(input_a, input_b):
    diff = input_a.astype(np.float32) - input_b.astype(np.float32)
    mse = np.mean(diff ** 2)
    return np.array([bfloat16(mse)], dtype=bfloat16)

input_a = (np.random.randn(256) * 2).astype(bfloat16)
input_b = (np.random.randn(256) * 2).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_a, input_b],
    rtp_values=None
)

pc.write_json("kernel.json")