# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel sums all 128 elements of an int8_t input buffer, applies a ReLU (sets negative sums to zero), and outputs the saturated result as a single int8_t to out_buffer."

def behavioral(in_buffer):
    s = np.sum(in_buffer.astype(np.int32))
    s = max(s, 0)
    s = np.clip(s, -128, 127)
    return np.int8(s)

in_buffer = np.random.randint(-40, 40, 128, dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=None
)

pc.write_json("kernel.json")