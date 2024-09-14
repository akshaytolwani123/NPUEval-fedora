# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "A kernel that performs a 2D padding operation on an input matrix with given padding size and padding value as runtime parameters."

def behavioral(input_matrix, pad_size, pad_value):
    input_rows, input_cols = input_matrix.shape
    padded_rows = input_rows + 2 * pad_size
    padded_cols = input_cols + 2 * pad_size
    padded_matrix = np.full((padded_rows, padded_cols), pad_value, dtype=input_matrix.dtype)
    padded_matrix[pad_size:pad_size + input_rows, pad_size:pad_size + input_cols] = input_matrix
    return padded_matrix

input_rows = 16
input_cols = 16
pad_size = 2
pad_value = 0

input_matrix = np.random.randint(-10, 10, size=(input_rows, input_cols), dtype=np.int32)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_matrix],
    rtp_values=[pad_size, pad_value]
)

pc.write_json("kernel.json")