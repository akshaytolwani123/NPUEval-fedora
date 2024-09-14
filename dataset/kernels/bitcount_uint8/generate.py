# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel counts the number of set bits (population count) for every entry in a uint8 vector of length 256, and returns a uint8 output vector of the same length containing the bit counts per element."

def behavioral(input_vector):
    in_arr = input_vector.astype(np.uint8)
    return np.array([bin(x).count('1') for x in in_arr]).astype(np.uint8)

input_vector = np.random.randint(0, 256, size=256, dtype=np.uint8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None
)

pc.write_json("kernel.json")