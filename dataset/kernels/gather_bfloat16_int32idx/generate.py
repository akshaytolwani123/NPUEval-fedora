# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = (
    "This AIE kernel implements a gather operation: for each idx in the int32 indices buffer, "
    "output_data[i] = input_data[indices[i]], all with bfloat16 inputs/outputs and int32 indices. "
    "Vectors are length 256."
)

def behavioral(input_data, indices):
    return input_data[indices]

src_buffer_size = 256
input_data = np.random.randn(src_buffer_size).astype(bfloat16)
indices = np.random.randint(0, src_buffer_size, size=src_buffer_size).astype(np.int32)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_data, indices],
    rtp_values=None
)

pc.write_json("kernel.json")