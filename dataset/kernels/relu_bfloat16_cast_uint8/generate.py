# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel applies ReLU to a bfloat16 input vector (negative values become zero), then rounds to the nearest positive integer, clamps between 0 and 255, and casts to uint8_t."

def behavioral(input_vector):
    relu = np.maximum(input_vector, 0)
    rounded = np.round(relu)
    clipped = np.clip(rounded, 0, 255)
    return clipped.astype(np.uint8)

input_vector = (np.random.randn(256) * 128).astype(bfloat16)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")