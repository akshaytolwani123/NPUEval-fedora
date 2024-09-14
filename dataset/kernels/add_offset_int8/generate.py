# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel adds a scalar int8 offset to every element of the input int8_t vector (length 256), and writes the result to the output buffer."

def behavioral(in_buffer, offset):
    res = in_buffer.astype(np.int16) + int(offset)
    res = np.clip(res, -128, 127)
    return res.astype(np.int8)

in_buffer = np.random.randint(-100, 100, 256).astype(np.int8)
offset = np.random.randint(-40, 40, dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=[int(offset)]
)

pc.write_json("kernel.json")