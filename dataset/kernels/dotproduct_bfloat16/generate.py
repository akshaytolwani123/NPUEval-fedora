# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = "An AIE kernel that performs a dot product operation on two bfloat16 input vectors of the same size."

def behavioral(vector1, vector2, vector_size):
    return np.array([np.dot(vector1[:vector_size], vector2[:vector_size])], dtype=bfloat16)

vector_size = 64
vector1 = np.random.randn(vector_size).astype(bfloat16)
vector2 = np.random.randn(vector_size).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[vector1, vector2],
    rtp_values=[vector_size]
)

pc.write_json("kernel.json")