# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = (
    "This AIE kernel computes the base-10 logarithm (log10) elementwise for a bfloat16 input vector; "
    "negative and zero inputs should return -infinity."
)

def behavioral(input):
    out = np.where(input > 0, np.log10(input.astype(np.float32)), -np.inf)
    return out.astype(bfloat16)

input = np.random.uniform(0.001, 100.0, 256).astype(bfloat16)

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