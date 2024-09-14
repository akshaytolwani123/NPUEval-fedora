# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = "A kernel that performs a 1D average pooling operation on an input vector with window_size and stride as runtime parameters."

def behavioral(in_buffer, window_size=2, stride=2):
    output_length = (len(in_buffer) - window_size) // stride + 1
    out_buffer = np.zeros(output_length, dtype=bfloat16)
    output_idx = 0
    for i in range(0, len(in_buffer) - window_size + 1, stride):
        window = in_buffer[i:i+window_size]
        out_buffer[output_idx] = np.mean(window)
        output_idx += 1
    return out_buffer

in_buffer = (np.random.randn(1024) * 2).astype(bfloat16)
window_size = 2
stride = 2

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=[window_size, stride]
)

pc.write_json("kernel.json")