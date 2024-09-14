# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel implements an abs function on an input vector."

def behavioral(in_buffer):
    return np.abs(in_buffer).astype(np.int8)

in_buffer = np.random.randint(-128, 128, size=(1024,), dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=None
)

pc.write_json("kernel.json")