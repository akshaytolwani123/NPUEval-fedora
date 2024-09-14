# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import os

kernels_dir = 'kernels/'
output_file = 'npueval.jsonl'

def extract_number(folder_name):
    return int(folder_name.split('_')[0])

# Get a list of folders and sort them based on preceding digit
#folders = sorted(os.listdir(kernels_dir), key=extract_number)
folders = sorted(os.listdir(kernels_dir))

with open(output_file, 'w') as outfile:
    for kernel_folder in folders:
        kernel_json = os.path.join(kernels_dir, kernel_folder, 'kernel.json')
        if os.path.exists(kernel_json):
            with open(kernel_json, 'r') as infile:
                data = json.load(infile)
                json.dump(data, outfile)
                outfile.write('\n')
