# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel compares two int8 input vectors of length 256 and produces an int8 output vector, where each output element is 1 if the corresponding element of the first input is greater than that of the second, and 0 otherwise."

def behavioral(in_buffer0, in_buffer1):
    return (in_buffer0 > in_buffer1).astype(np.int8)

in_buffer0 = np.random.randint(-128, 127, 256, dtype=np.int8)
in_buffer1 = np.random.randint(-128, 127, 256, dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer0, in_buffer1],
    rtp_values=None
)

pc.write_json("kernel.json")