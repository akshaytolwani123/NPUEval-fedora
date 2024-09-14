# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This is the implementation of an AIE kernel that performs a vector subtraction of 2 input vectors and stores the result in 1 output vector."

def behavioral(in_buffer1, in_buffer2):
    return (in_buffer1 - in_buffer2).astype(np.int8)

in_buffer1 = np.random.randint(-128, 127, size=(512,), dtype=np.int8)
in_buffer2 = np.random.randint(-128, 127, size=(512,), dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer1, in_buffer2],
    rtp_values=None
)

pc.write_json("kernel.json")