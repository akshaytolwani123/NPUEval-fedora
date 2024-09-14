# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel compares two bfloat16 vectors elementwise for equality, and writes a uint8_t mask output (1 if equal, 0 if not)."

def behavioral(input_a, input_b):
    return (input_a == input_b).astype(np.uint8)

input_a = np.random.randn(256).astype(bfloat16)
input_b = input_a.copy()
flip_idx = np.random.choice(256, size=20, replace=False)
input_b[flip_idx] = (input_b[flip_idx] + np.float32(1.234)).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_a, input_b],
    rtp_values=None
)

pc.write_json("kernel.json")