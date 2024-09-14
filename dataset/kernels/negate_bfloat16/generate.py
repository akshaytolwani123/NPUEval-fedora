# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "A kernel that negates each element of a bfloat16 input vector and writes the result to an output vector."

def behavioral(input_vector):
    return (-input_vector).astype(bfloat16)

input_vector = (np.random.randn(256) * 4).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None
)

pc.write_json("kernel.json")