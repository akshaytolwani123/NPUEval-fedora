# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = (
    "This AIE kernel computes the SiLU (Sigmoid Linear Unit, also known as Swish) activation, elementwise on a bfloat16 input vector (length 256). "
    "Each output is computed as x*sigmoid(x) with sigmoid(x)=1/(1+exp(-x))."
)

def behavioral(input):
    x = input.astype(np.float32)
    return (x / (1 + np.exp(-x))).astype(bfloat16)

input = (np.random.randn(256) * 3).astype(bfloat16)
tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input],
    rtp_values=None,
    tolerances=tolerances,
)

pc.write_json("kernel.json")