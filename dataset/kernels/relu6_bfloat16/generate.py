# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel applies the ReLU6 activation function elementwise to a bfloat16 input vector of size 256. ReLU6 clamps each value to [0, 6]."

def behavioral(input):
    return np.minimum(np.maximum(input.astype(np.float32), 0.0), 6.0).astype(bfloat16)

input = (np.random.randn(256) * 8 - 2).astype(bfloat16)

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