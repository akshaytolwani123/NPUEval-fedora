# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = (
    "This AIE kernel computes the elementwise cumulative sum (cumsum) of a bfloat16 input vector of length 256."
    " Each output element is the sum of all input elements up to and including that position."
)

def behavioral(in_buffer):
    sum_ = np.cumsum(in_buffer.astype(np.float32)).astype(bfloat16)
    return sum_

in_buffer = (np.random.randn(256) * 3).astype(bfloat16)

tolerances = {"atol": 0.02, "rtol": 0.02}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=None,
    tolerances=tolerances,
)

pc.write_json("kernel.json")