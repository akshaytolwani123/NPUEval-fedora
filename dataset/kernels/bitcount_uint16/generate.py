# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel counts the number of 1-bits (Hamming weight) in each entry of a uint16_t vector, writing out a uint16_t vector of the same length where each entry holds the bit count for the corresponding input value."

def behavioral(input_vector):
    input_vector = np.asarray(input_vector, dtype=np.uint16)
    arr8 = input_vector.view(np.uint8).reshape(-1,2)
    bits = np.unpackbits(arr8, axis=1)
    return np.sum(bits, axis=1).astype(np.uint16)

input_vector = np.random.randint(0, 65536, size=256, dtype=np.uint16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None
)

pc.write_json("kernel.json")