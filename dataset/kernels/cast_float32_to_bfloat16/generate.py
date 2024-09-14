# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel casts a float32 input vector of length 256 to bfloat16, writing results to the output buffer."

def behavioral(in_buffer):
    return in_buffer.astype(bfloat16)

in_buffer = (np.random.randn(256) * 5.0).astype(np.float32)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=None
)

pc.write_json("kernel.json")