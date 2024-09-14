# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = (
    "This AIE kernel computes the cumulative sum (cumsum) of a 256-element int16_t input vector. "
    "The result is written to the output buffer, with saturation to int16_t range on overflow."
)

def behavioral(in_buffer):
    acc = np.cumsum(in_buffer.astype(np.int32))
    acc = np.clip(acc, -32768, 32767)
    return acc.astype(np.int16)

in_buffer = np.random.randint(-500, 500, 256, dtype=np.int16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=None
)

pc.write_json("kernel.json")