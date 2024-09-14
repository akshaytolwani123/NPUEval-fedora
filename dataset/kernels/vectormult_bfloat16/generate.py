# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This is the implementation of an AIE kernel that performs an elementwise vector multiplication of 2 input vectors and stores the result in 1 output vector."

def behavioral(in_buffer1, in_buffer2, vector_size):
    return (in_buffer1[:vector_size] * in_buffer2[:vector_size]).astype(bfloat16)

vector_size = 64
in_buffer1 = (np.random.randn(vector_size) * 2).astype(bfloat16)
in_buffer2 = (np.random.randn(vector_size) * 2).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer1, in_buffer2],
    rtp_values=[vector_size]
)

pc.write_json("kernel.json")