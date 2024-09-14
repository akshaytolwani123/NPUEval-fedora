# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = "This AIE kernel finds the maximum absolute value in a bfloat16 input vector of size 256 and writes the result as a scalar bfloat16 in the output buffer."

def behavioral(input):
    absmax = np.max(np.abs(input.astype(np.float32)))
    return np.array([bfloat16(absmax)])

input = (np.random.randn(256) * 7).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input],
    rtp_values=None
)

pc.write_json("kernel.json")