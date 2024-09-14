# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "An AIE kernel that performs a dot product operation on two int32 input vectors of the same size."

def behavioral(vector1, vector2):
    return np.dot(vector1, vector2).astype(np.int32)

vector1 = np.random.randint(-10, 10, size=(256,), dtype=np.int32)
vector2 = np.random.randint(-10, 10, size=(256,), dtype=np.int32)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[vector1, vector2],
    rtp_values=None
)

pc.write_json("kernel.json")