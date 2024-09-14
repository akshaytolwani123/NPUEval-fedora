# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel computes the elementwise minimum of two int8 input vectors of size 256 and writes the results to the output buffer."

def behavioral(input1, input2):
    return np.minimum(input1, input2).astype(np.int8)

input1 = np.random.randint(-128, 127, size=256, dtype=np.int8)
input2 = np.random.randint(-128, 127, size=256, dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input1, input2],
    rtp_values=None
)

pc.write_json("kernel.json")