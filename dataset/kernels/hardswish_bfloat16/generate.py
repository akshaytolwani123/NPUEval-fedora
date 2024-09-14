# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel implements the hard swish activation function on a bfloat16 vector"

def behavioral(input_vector):
    x = input_vector
    relu6 = np.minimum(np.maximum(x + 3, 0).astype(bfloat16), 6).astype(bfloat16)
    return (x * relu6 / 6).astype(bfloat16)

input_vector = (np.random.randn(256) * 2).astype(bfloat16)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")