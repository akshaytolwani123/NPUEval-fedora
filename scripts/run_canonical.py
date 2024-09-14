# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import json
from npueval import run_functional_tests

with open("dataset/npueval.jsonl", 'r') as f:
    tests = [json.loads(line) for line in f]

results_path_functional = "results/evaluations/canonical"
run_functional_tests(tests, results_path=results_path_functional, overwrite=True, verbose=False, generate_assembly=False)
