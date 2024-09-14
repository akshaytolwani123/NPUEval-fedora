# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import json

def save_kernel(sample):
    with open("kernel.json", 'w') as f:
        json.dump(sample, f)


typemap = {
        "bfloat16" : "bfloat16",
        "uint8"    : "uint8_t",
        "uint16"   : "uint16_t",
        "uint32"   : "uint32_t",
        "uint64"   : "uint64_t",
        "int8"     : "int8_t",
        "int16"    : "int16_t",
        "int32"    : "int32_t",
        "int64"    : "int64_t",
}