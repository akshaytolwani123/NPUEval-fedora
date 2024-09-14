# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel computes the base-2 logarithm (log2) of each element in an input bfloat16 vector of size 256 and writes the result to the output vector. For zero or negative inputs, it should return -inf."

def behavioral(input):
    with np.errstate(divide='ignore'):
        out = np.log2(input.astype(np.float32))
        out = np.where(input <= 0, -np.inf, out)
    return out.astype(bfloat16)

input = (np.abs(np.random.randn(256)) + 0.001).astype(bfloat16)

tolerances = {"atol": 0.02, "rtol": 0.02}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")