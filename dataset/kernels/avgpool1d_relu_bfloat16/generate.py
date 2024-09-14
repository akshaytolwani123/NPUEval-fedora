# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)

description = "A kernel that performs 1D average pooling with a 4-element window and stride of 4 on a 256-element bfloat16 vector, followed by elementwise ReLU activation, writing the results to a 64-element bfloat16 output buffer."

def behavioral(in_buffer):
    window_size = 4
    stride = 4
    length = in_buffer.shape[0]
    output_size = length // stride
    out = np.zeros(output_size, dtype=bfloat16)
    for i in range(output_size):
        window = in_buffer[i * stride : i * stride + window_size].astype(np.float32)
        avg = window.mean()
        relu_avg = np.maximum(avg, 0.0)
        out[i] = relu_avg.astype(bfloat16)
    return out

in_buffer = (np.random.randn(256) * 3).astype(bfloat16)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")