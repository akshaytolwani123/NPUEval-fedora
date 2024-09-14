# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel inverts all input values of a buffer containing a uint8 array."

def behavioral(in_buffer):
    return (255 - in_buffer).astype(np.uint8)

in_buffer = np.random.randint(0, 255, size=(1024,), dtype=np.uint8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=None
)

pc.write_json("kernel.json")