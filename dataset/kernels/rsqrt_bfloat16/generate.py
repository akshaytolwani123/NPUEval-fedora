# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel computes the reciprocal square root (1/sqrt(x)) for each element of the bfloat16 input vector of length 256, producing a bfloat16 output vector. Nonpositive inputs yield 0 by convention."

def behavioral(input):
    result = np.zeros_like(input)
    mask = input > 0
    result[mask] = 1.0 / np.sqrt(input[mask])
    return result.astype(bfloat16)

input = (np.abs(np.random.randn(256)) + 0.1).astype(bfloat16)
input[5] = 0.0
input[10] = -1.0

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")