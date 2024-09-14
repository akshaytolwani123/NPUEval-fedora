# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel computes the elementwise sign of a vector of int8 values. For each input value: if positive return 1, if negative return -1, otherwise 0."

def behavioral(input_vector):
    result = np.zeros_like(input_vector)
    result[input_vector > 0] = 1
    result[input_vector < 0] = -1
    return result.astype(np.int8)

input_vector = (np.random.randint(-130, 130, size=256) // 2).astype(np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None
)

pc.write_json("kernel.json")