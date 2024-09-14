# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "A kernel that performs a 1D padding operation on an input vector with given padding size and padding value as runtime parameters."

def behavioral(input_vector, pad_size, pad_value):
    padded_length = len(input_vector) + 2 * pad_size
    padded_vector = np.full(padded_length, pad_value, dtype=input_vector.dtype)
    padded_vector[pad_size:pad_size + len(input_vector)] = input_vector
    return padded_vector

in_buffer = np.random.randint(-10, 10, size=(256,), dtype=np.int32)
pad_size = 2
pad_value = 0

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=[pad_size, pad_value]
)

pc.write_json("kernel.json")