# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel adds an offset to each element of an input vector containing uint8 data."

def behavioral(in_buffer, offset=1):
    return (in_buffer + offset).astype(np.uint8)

offset = 2
in_buffer = np.random.randint(0, 255, size=(1024,), dtype=np.uint8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=[offset]
)

pc.write_json("kernel.json")