# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel computes the absolute value of complex numbers. real_vector contains the real parts, and imag_vector contains the imaginary parts. The result is stored in output_vector."

def behavioral(real_vector, imag_vector):
    return np.sqrt(real_vector**2 + imag_vector**2).astype(bfloat16)

real_vector = (np.random.randn(64) * 2).astype(bfloat16)
imag_vector = (np.random.randn(64) * 2).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[real_vector, imag_vector],
    rtp_values=None
)

pc.write_json("kernel.json")