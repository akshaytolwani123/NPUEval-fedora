# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel performs an in-place Fisher–Yates shuffle of a length-256 int32 vector, using a runtime-provided seed parameter. The input is copied to output and then shuffled."

def behavioral(input, seed):
    arr = input.copy()
    N = arr.shape[0]
    state = np.uint32(seed)
    for i in range(N - 1, 0, -1):
        state ^= (state << np.uint32(13))
        state ^= (state >> np.uint32(17))
        state ^= (state << np.uint32(5))
        j = int(state % np.uint32(i + 1))
        arr[i], arr[j] = arr[j], arr[i]
    return arr

input = np.random.randint(-1000, 1000, size=256, dtype=np.int32)
seed = 12345

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input],
    rtp_values=[seed]
)

pc.write_json("kernel.json")