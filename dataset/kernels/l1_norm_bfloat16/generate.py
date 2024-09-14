# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = "This AIE kernel computes the L1 norm (sum of absolute values) of a bfloat16 input vector of length 256. The result is placed in the output buffer (bfloat16)."

def behavioral(in_buffer):
    abs_sum = np.sum(np.abs(in_buffer))
    return np.array([abs_sum], dtype=bfloat16)

in_buffer = (np.random.randn(256) * 3).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=None
)

pc.write_json("kernel.json")