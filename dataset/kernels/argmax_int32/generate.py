# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel calculates the argmax (the index of the largest element) of a given int32 vector of length 256 and stores the result in a single uint32 element output buffer."

def behavioral(input_vector):
    # Return as shape (1,) array for correct output serialization
    return np.array([np.argmax(input_vector)], dtype=np.uint32)

input_vector = np.random.randint(-32768, 32768, size=256, dtype=np.int32)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None
)

pc.write_json("kernel.json")