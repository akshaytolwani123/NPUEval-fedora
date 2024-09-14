# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This kernel divides two bfloat16 vectors elementwise, stores the quotient in out. Division by zero returns 0.0 as output for that element."

def behavioral(a, b):
    out = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return out.astype(bfloat16)

a = (np.random.randn(256) * 5).astype(bfloat16)
b = (np.random.randn(256) * 5).astype(bfloat16)
b[::51] = 0.0

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[a, b],
    rtp_values=None
)

pc.write_json("kernel.json")