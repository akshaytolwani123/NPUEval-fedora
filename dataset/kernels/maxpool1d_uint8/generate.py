# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "A kernel that performs a 1D maxpool operation on an input vector with window_size and stride as runtime parameters."

def behavioral(in_buffer, window_size=2, stride=2):
    num_windows = (len(in_buffer) - window_size) // stride + 1
    out_buffer = np.zeros(num_windows, dtype=np.uint8)
    for i in range(num_windows):
        window = in_buffer[i*stride : i*stride + window_size]
        out_buffer[i] = np.max(window)
    return out_buffer

in_buffer = np.random.randint(0, 255, size=(1024,), dtype=np.uint8)
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