# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = "This AIE kernel calculates the argmax of a given bfloat16 vector, returning a single uint32 value for the maximum argument."

def behavioral(input_vector):
    return np.argmax(input_vector).astype(np.uint32)

input_vector = np.random.randn(64).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None
)

pc.write_json("kernel.json")