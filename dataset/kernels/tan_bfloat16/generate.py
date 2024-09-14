# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel computes the elementwise tangent (tan) function for a bfloat16 input vector of size 256."

def behavioral(input):
    x = np.clip(input.astype(np.float32), -15, 15)
    return np.tan(x).astype(bfloat16)

input_array = np.random.uniform(-3, 3, 256).astype(bfloat16)

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