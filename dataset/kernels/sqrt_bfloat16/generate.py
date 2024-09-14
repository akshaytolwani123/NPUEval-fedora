# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "A kernel that computes the elementwise square root of a bfloat16 vector. Negative values are clamped to 0.0 in the output."

def behavioral(input):
    out = np.sqrt(np.clip(input.astype(np.float32), 0.0, None)).astype(bfloat16)
    return out

input = (np.random.standard_normal(256) * 3).astype(bfloat16)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")