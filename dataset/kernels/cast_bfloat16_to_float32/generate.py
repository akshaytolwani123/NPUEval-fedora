# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel casts a bfloat16 input vector of length 512 into an output vector of float32, performing an elementwise conversion."

def behavioral(in_buffer):
    return in_buffer.astype(np.float32)

in_buffer = np.random.randn(512).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=None
)

pc.write_json("kernel.json")