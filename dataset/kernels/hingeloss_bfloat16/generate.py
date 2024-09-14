# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = (
    "An AIE kernel that computes the element-wise hinge loss for two bfloat16 arrays of length 256. "
    "For each index i, output[i] = max(0, 1 - label[i] * pred[i]), where label is typically +1 or -1."
)

def behavioral(pred, label):
    t = 1.0 - label * pred
    h = np.maximum(0.0, t)
    return h.astype(bfloat16)

# Generate predictions in [-2, 2], labels are strictly -1 or 1 (as expected for hinge)
pred = (np.random.uniform(-2, 2, 256)).astype(bfloat16)
label = np.random.choice([-1.0, 1.0], 256).astype(bfloat16)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[pred, label],
    rtp_values=None,
    tolerances=tolerances
)

pc.write_json("kernel.json")