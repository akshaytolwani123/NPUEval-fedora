#!/bin/bash
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

KERNELS_DIR="kernels"

for kernel_folder in "$KERNELS_DIR"/*; do
    if [ -d "$kernel_folder" ]; then  # Check if it is a directory
        generate_script="$kernel_folder/generate.py"
        if [ -f "$generate_script" ]; then  # Check if the generate.py file exists
            echo "Processing $kernel_folder..."
            (cd "$kernel_folder" && python3 generate.py)
        else
            echo "No generate.py found in $kernel_folder"
        fi
    fi
done

echo "All kernels have been processed."