# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = "This AIE kernel takes two bfloat16 input vectors of size 256 and writes their elementwise maximum to an output bfloat16 vector."

def behavioral(input0, input1):
    return np.maximum(input0, input1).astype(bfloat16)

input0 = (np.random.randn(256) * 3).astype(bfloat16)
input1 = (np.random.randn(256) * 2).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input0, input1],
    rtp_values=None
)

pc.write_json("kernel.json")