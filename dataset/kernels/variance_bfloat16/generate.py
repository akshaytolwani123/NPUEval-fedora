# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = (
    "This kernel computes the variance of a bfloat16 input vector (length 256) using the definition "
    "Var(X) = E[X^2] - (E[X])^2, and writes the scalar (bfloat16) result to the output buffer (length 1)."
)

def behavioral(input):
    vals = input.astype(np.float32)
    mean = np.mean(vals)
    mean_sq = np.mean(vals ** 2)
    var = mean_sq - mean ** 2
    if var < 0:
        var = 0.0
    return np.array([bfloat16(var)])

input_array = (np.random.randn(256) * 3).astype(bfloat16)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_array],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")