# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = "This AIE kernel computes the elementwise vector addition of two bfloat16 vectors of length 256, followed by the ReLU activation. Each output is given by out[i] = max(0, in0[i] + in1[i])."

def behavioral(in0, in1):
    sum = in0 + in1
    return np.maximum(sum, bfloat16(0)).astype(bfloat16)

in0 = (np.random.randn(256)).astype(bfloat16)
in1 = (np.random.randn(256)).astype(bfloat16)

tolerances = {"atol": 0.02, "rtol": 0.02}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in0, in1],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")