# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = "This AIE kernel casts each element of a signed int8 input vector to signed int32, storing the result in the output buffer. The casting should preserve the sign."

def behavioral(input_vector):
    return input_vector.astype(np.int32)

input_vector = np.random.randint(-128, 128, 256, dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None
)

pc.write_json("kernel.json")