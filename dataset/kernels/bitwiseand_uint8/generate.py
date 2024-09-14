# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This is the implementation of an AIE kernel that performs a bitwise AND operation on 2 input vectors and stores the result in 1 output vector."

def behavioral(in_buffer1, in_buffer2):
    return np.bitwise_and(in_buffer1, in_buffer2)

in_buffer1 = np.random.randint(0, 255, size=(1024,), dtype=np.uint8)
in_buffer2 = np.random.randint(0, 255, size=(1024,), dtype=np.uint8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer1, in_buffer2],
    rtp_values=None
)

pc.write_json("kernel.json")