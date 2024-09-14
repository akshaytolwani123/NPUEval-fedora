# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel implements the sine function on a bfloat16 vector"

def behavioral(input_vector):
    return np.sin(input_vector).astype(bfloat16)

input_vector = (np.random.rand(256) * 4 * np.pi - 2 * np.pi).astype(bfloat16)

tolerances = {"atol": 0.02, "rtol": 0.02}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input_vector],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")