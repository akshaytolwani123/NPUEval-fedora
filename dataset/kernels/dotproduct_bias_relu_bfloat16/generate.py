# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This kernel computes the dot product of two bfloat16 vectors of size 64, adds a bfloat16 bias (passed as an RTP), applies a ReLU, and writes the single bfloat16 output."

def behavioral(in0, in1, bias, vector_size):
    s = bfloat16(np.dot(in0[:vector_size], in1[:vector_size])) + bias
    s_relu = max(s, bfloat16(0.0))
    return np.array([s_relu], dtype=bfloat16)

vector_size = 64
in0 = (np.random.randn(vector_size)).astype(bfloat16)
in1 = (np.random.randn(vector_size)).astype(bfloat16)
bias = bfloat16(np.random.randn())

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in0, in1],
    rtp_values=[bias, vector_size]
)

pc.write_json("kernel.json")