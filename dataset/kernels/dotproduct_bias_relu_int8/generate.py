# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = (
    "This kernel computes the dot product of two int8 vectors of length 256, adds a scalar int32 bias, "
    "applies ReLU, clamps the result to int8_t range, and writes the result to a single-element int8_t output buffer."
)

def behavioral(in0, in1, bias):
    acc = np.dot(in0.astype(np.int32), in1.astype(np.int32)) + int(bias)
    acc = max(acc, 0)
    acc = np.clip(acc, -128, 127)
    return np.int8(acc)

in0 = np.random.randint(-20, 20, 256, dtype=np.int8)
in1 = np.random.randint(-20, 20, 256, dtype=np.int8)
bias = np.random.randint(-10, 10, dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in0, in1],
    rtp_values=[int(bias)]
)

pc.write_json("kernel.json")