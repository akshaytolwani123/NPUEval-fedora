# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel takes two int32 input vectors and compares them element-wise for equality, producing a uint8 output vector: 1 if equal, 0 if not."

def behavioral(a, b):
    return (a == b).astype(np.uint8)

a = np.random.randint(-1000, 1000, size=256, dtype=np.int32)
b = a.copy()
mask = np.random.rand(256) > 0.1
b[mask] = np.random.randint(-1000, 1000, size=np.sum(mask), dtype=np.int32)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[a, b],
    rtp_values=None
)

pc.write_json("kernel.json")