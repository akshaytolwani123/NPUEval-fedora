# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel rounds each element of a bfloat16 input vector (to nearest integer, ties to +inf), writing the result as bfloat16."

def behavioral(input):
    rounded = np.floor(input + 0.5)
    return rounded.astype(bfloat16)

input = (np.random.randn(256) * 4).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input],
    rtp_values=None
)

pc.write_json("kernel.json")