# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel negates each element in a vector of int8_t (length 256). The input is a buffer of 256 int8, and the output buffer receives the elementwise negation."

def behavioral(input_vector):
    return (-input_vector).astype(np.int8)

input_vector = np.random.randint(-128, 128, size=256, dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None
)

pc.write_json("kernel.json")