# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor, extract_canonical

def behavioral(in_buffer1, in_buffer2, nbytes=None):
    return np.dot(in_buffer1, in_buffer2)

tile_size = 1024
total_transfer = 1024
trace_size = 8192

input_buffer1 = np.random.rand(total_transfer).astype(bfloat16)
input_buffer2 = np.random.rand(total_transfer).astype(bfloat16)
output_buffer = behavioral(input_buffer1, input_buffer2)

with open("canonical_scalar.cc", 'r') as f:
    src = f.read()

canonical_solution = extract_canonical(src)

pc = PromptConstructor(src)
signature = pc.construct_signature()
program_code = pc.construct_wrapper()

kernel_name = pc.name
description = "An AIE kernel that performs a dot product operation on two bfloat16 input vectors of the same size."

prompt = f"""/*
{description}
>>> {kernel_name}({input_buffer1[:4]}, {input_buffer2[:4]})
{behavioral(input_buffer1[:4], input_buffer2[:4])}
>>>> {kernel_name}({input_buffer1[-4:]}, {input_buffer2[-4:]})
{behavioral(input_buffer1[-4:], input_buffer2[-4:])}
*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

void {signature} {{
"""

data_movement = {"total_transfer": total_transfer,
                 "tile_size": tile_size,
                 "trace_size": trace_size}


test_vectors = {"inputs": [{"in_buffer1": input_buffer1.tolist(), "dtype": input_buffer1.dtype.name},
                           {"in_buffer2": input_buffer2.tolist(), "dtype": input_buffer2.dtype.name}],
                "outputs": [{"out_buffer1": output_buffer.tolist(), "dtype": output_buffer.dtype.name}],
                "rtps": [{"vector_size": tile_size, "dtype": "int32"}]}

sample = {"kernel_name": kernel_name,
          "prompt": prompt,
          "canonical_solution": canonical_solution,
          "program_code": program_code,
          "data_movement": data_movement,
          "test_vectors": test_vectors}

with open("kernel.json", 'w') as f:
    json.dump(sample, f)