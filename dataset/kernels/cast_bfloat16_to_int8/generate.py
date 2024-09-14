# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel casts each element of a bfloat16 vector input to int8, performing rounding to nearest integer and saturating to the int8 value range [-128, 127]. The output is a vector of int8."

def behavioral(input):
    x = input.astype(np.float32)
    xi = np.trunc(x).astype(np.int32)
    floor_x = np.where(
        x >= 0,
        xi,
        np.where(x == xi, xi, xi - 1)
    )
    frac = x - floor_x.astype(np.float32)
    even_floor = (floor_x & 1) == 0
    r = np.where(
        frac < 0.5,
        floor_x,
        np.where(
            frac > 0.5,
            floor_x + 1,
            np.where(even_floor, floor_x, floor_x + 1)
        )
    )
    r = np.clip(r, -128, 127)
    return r.astype(np.int8)

input = (np.random.randn(256) * 70).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input],
    rtp_values=None
)

pc.write_json("kernel.json")