# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel computes the elementwise maximum between two int8 vectors of length 512 and places the result in the output buffer."

def behavioral(in_buffer1, in_buffer2):
    return np.maximum(in_buffer1, in_buffer2).astype(np.int8)

in_buffer1 = np.random.randint(-128, 128, 512, dtype=np.int8)
in_buffer2 = np.random.randint(-128, 128, 512, dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer1, in_buffer2],
    rtp_values=None
)

pc.write_json("kernel.json")