# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = "This AIE kernel computes the elementwise reciprocal (1/x) of a bfloat16 input vector of size 256, writing the result to an output buffer."

def behavioral(input_vector):
    # Prevent division by zero for behavioral model, though canonical has no check.
    result = np.empty_like(input_vector)
    result[:] = 1.0 / input_vector.astype(np.float32)
    result[input_vector == 0] = np.inf
    return result.astype(bfloat16)

input_vector = np.random.uniform(-10, 10, size=256).astype(np.float32)
input_vector = input_vector.astype(bfloat16)

tolerances = {"atol": 0.02, "rtol": 0.02}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")