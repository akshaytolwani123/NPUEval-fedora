# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel compares two int8_t input vectors elementwise (length 256), outputting 1 at index i if input_a[i] < input_b[i], else 0. Results are int8_t."

def behavioral(input_a, input_b):
    return (input_a < input_b).astype(np.int8)

input_a = np.random.randint(-100, 100, size=256, dtype=np.int8)
input_b = np.random.randint(-100, 100, size=256, dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_a, input_b],
    rtp_values=None
)

pc.write_json("kernel.json")