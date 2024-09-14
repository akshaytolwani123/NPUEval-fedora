# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel performs an elementwise minimum operation between two input bfloat16 vectors, writing the minimum of each pair to the output buffer."

def behavioral(in_buffer_a, in_buffer_b):
    return np.minimum(in_buffer_a, in_buffer_b).astype(bfloat16)

in_buffer_a = (np.random.randn(256) * 3.2).astype(bfloat16)
in_buffer_b = (np.random.randn(256) * 3.2).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer_a, in_buffer_b],
    rtp_values=None
)

pc.write_json("kernel.json")