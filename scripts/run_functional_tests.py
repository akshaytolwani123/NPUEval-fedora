# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import json
from npueval import run_functional_tests

with open("dataset/npueval.jsonl", 'r') as f:
    tests = [json.loads(line) for line in f]

# OpenAI
N = [1, 2]
models = ["gpt-4o-mini", "gpt-4o"]

for attempts in N:
    for MODEL in models:
        print(f"{MODEL} N={attempts}")
        solutions = f"results/solutions/{MODEL}_attempts_{attempts}/"
        results_path = f"results/evaluations/{MODEL}_attempts_{attempts}"
        run_functional_tests(tests, solutions, results_path=results_path)

# RAG
num_retrieved = [1]
for attempts in N:
    for MODEL in models:
        for k in num_retrieved:
            print(f"{MODEL} N={attempts} k={k}")
            solutions = f"results/solutions/{MODEL}_attempts_{attempts}_rag_{k}/"
            results_path = f"results/evaluations/{MODEL}_attempts_{attempts}_rag_{k}"
            run_functional_tests(tests, solutions, results_path=results_path)
