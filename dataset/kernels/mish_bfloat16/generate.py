# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(7)

description = "This AIE kernel computes the elementwise Mish activation, Mish(x) = x * tanh(ln(1 + exp(x))), for a bfloat16 input vector of length 256, storing results in the bfloat16 output buffer."

def behavioral(input):
    x = input
    softplus = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    mish = x * np.tanh(softplus)
    return mish.astype(bfloat16)

input = (np.random.randn(256) * 4).astype(bfloat16)

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