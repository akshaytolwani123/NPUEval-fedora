# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = (
    "This AIE kernel applies the hardsigmoid activation function on a vector of int8 values. "
    "The hardsigmoid is computed as: y = max(0, min(1, x/6 + 0.5)), "
    "and the result is rescaled and quantized into int8 output with the range [0,127]."
)

def behavioral(input_vector):
    x = input_vector.astype(np.float32)
    y = x / 6.0 + 0.5
    y = np.clip(y, 0.0, 1.0)
    out = np.round(y * 127.0).clip(0, 127).astype(np.int8)
    return out

input_vector = np.random.randint(-128, 128, size=(256,), dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None
)

pc.write_json("kernel.json")