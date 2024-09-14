# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel computes the elementwise cosine (cos) function of a bfloat16 input vector. Input and output shapes are (256,) and the result is written as bfloat16."

def behavioral(input_vector):
    return np.cos(input_vector).astype(bfloat16)

input_vector = np.random.uniform(-6, 6, (256,)).astype(bfloat16)

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