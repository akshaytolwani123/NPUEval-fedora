# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel performs a ReLU activation on a bfloat16 input vector of fixed size."

def behavioral(in_buffer):
    return np.where(in_buffer > bfloat16(0), in_buffer, bfloat16(0))

in_buffer = np.random.randn(256).astype(bfloat16)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")