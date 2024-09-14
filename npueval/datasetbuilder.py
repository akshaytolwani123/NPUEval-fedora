# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from CppHeaderParser import CppHeader
from ml_dtypes import bfloat16
import re
import numpy as np
import json

C_TO_NUMPY_DTYPE = {
    "int8_t": "int8",
    "uint8_t": "uint8",
    "int16_t": "int16",
    "uint16_t": "uint16",
    "int32_t": "int32",
    "uint32_t": "uint32",
    "int64_t": "int64",
    "uint64_t": "uint64",
    "bfloat16": "bfloat16"
}

def extract_canonical(src):
    match = re.search(r'void\s+\w+\s*\(.*?\)\s*{(.*)}', src, re.DOTALL)
    if match:
        return match.group(1).strip() + "\n}"
    else:
        raise Exception("Couldn't parse canonical solution.")

class PromptConstructor:
    def __init__(self, source_path, description, behavioral, input_arrays, rtp_values=None, tolerances=None):
        # --- Parse source + function ---
        with open(source_path, "r") as f:
            source_code = f.read()
        parsedcpp = CppHeader(source_code, argType='string')

        if len(parsedcpp.functions) > 1:
            raise RuntimeError("There should be only 1 function in the source code.")

        func = parsedcpp.functions[0]
        self.name = func['name']
        self.buffers = [p for p in func['parameters'] if p['pointer']]
        self.rtps = [p for p in func['parameters'] if not p['pointer']]
        self.signature = self._construct_signature()
        self.call = self._construct_call()

        # --- Extract canonical solution ---
        with open(source_path, "r") as f:
            src = f.read()
        canonical_solution = extract_canonical(src)

        # --- Generate test_vectors ---
        outputs = behavioral(*input_arrays, *rtp_values) if rtp_values else behavioral(*input_arrays)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        test_vectors = {"inputs": [], "outputs": []}
        for i, arr in enumerate(input_arrays):
            buf_name = self.buffers[i]['name']
            test_vectors["inputs"].append({
                buf_name: arr.tolist(),
                "dtype": str(arr.dtype)
            })
        for i, out in enumerate(outputs):
            # assume last one because we only support 1 output buf
            buf_name = self.buffers[-1]['name']
            test_vectors["outputs"].append({
                buf_name: out.tolist(),
                "dtype": str(out.dtype)
            })
        if rtp_values:
            test_vectors["rtps"] = []
            for i, val in enumerate(rtp_values):
                c_type = self.rtps[i]['type']
                try:
                    np_type = C_TO_NUMPY_DTYPE[c_type]
                except KeyError:
                    raise ValueError(f"Unsupported rtp type '{c_type}' in kernel '{self.name}'")
                if isinstance(val, bfloat16):
                    val = float(val)
                test_vectors["rtps"].append({self.rtps[i]['name']: val, "dtype": np_type})

        program_code = self._construct_wrapper()
        dataflow = self._construct_dataflow(test_vectors)
        
        # Some edge case kernels may fail to generate examples, we'll just skip for those
        try:
            examples = self._construct_examples(behavioral, input_arrays, rtp_values)
        except:
            print(f"Warning: {self.name} prompt won't have examples")
            examples = None
        
        if examples:
            prompt = f"""/*
{description}
{examples}
{dataflow}
*/
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"

void {self.signature} {{
    // Implementation goes here
}}
"""
        else:
            prompt = f"""/*
{description}
{dataflow}
*/
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"

void {self.signature} {{
    // Implementation goes here
}}
"""

        self.sample = {
            "kernel_name": self.name,
            "prompt": prompt,
            "canonical_solution": canonical_solution,
            "program_code": program_code,
            "test_vectors": test_vectors,
        }

        if tolerances:
            self.sample['tolerances'] = tolerances

    def write_json(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.sample, f)

    # Internal methods
    def _construct_signature(self):
        buf = ", ".join([f"{b['type']}{b['name']}" for b in self.buffers])
        rtp = ", ".join([f"{r['type']} {r['name']}" for r in self.rtps])
        return f"{self.name}({buf}{', ' + rtp if rtp else ''})"

    def _construct_call(self):
        buf = ", ".join([b['name'] for b in self.buffers])
        rtp = ", ".join([r['name'] for r in self.rtps])
        return f"{self.name}({buf}{', ' + rtp if rtp else ''})"

    def _construct_wrapper(self):
        buf = ", ".join([f"{b['type']}{b['name']}" for b in self.buffers])
        rtp = ", ".join([f"{r['type']} {r['name']}" for r in self.rtps])
        params = buf if not self.rtps else f"{buf}, {rtp}"
        return f"""extern "C" {{
    void {self.name}_wrapper({params}) {{
        ::aie::set_rounding(aie::rounding_mode::positive_inf);
        event0();
        {self.call};
        event1();
    }}
}}"""

    def _construct_dataflow(self, test_vectors):
        df = ""
        for d in test_vectors["inputs"]:
            for k, v in d.items():
                if k != "dtype":
                    df += f"{k} size: {np.size(v)}\n"
                    break
        for d in test_vectors["outputs"]:
            for k, v in d.items():
                if k != "dtype":
                    df += f"{k} size: {np.size(v)}\n"
                    break
        if "rtps" in test_vectors:
            for i, d in enumerate(test_vectors["rtps"]):
                for k, v in d.items():
                    if k != "dtype":
                        df += f"{k}: {v}\n"
        return f"This kernel should be optimized for the following input/output buffer shapes and parameters:\n{df.strip()}"

    def _construct_examples(self, behavioral, input_arrays, rtp_values=None, num_examples=1, slice_size=8):
        examples = []
        arr0 = input_arrays[0]
        indices = [0, max(0, arr0.shape[0] - slice_size)] if arr0.ndim == 1 else [0]

        for idx in indices[:num_examples]:
            sliced = [a[idx:idx+slice_size] if a.ndim == 1 else a[:4, :4] for a in input_arrays]
            out = behavioral(*sliced, *rtp_values) if rtp_values else behavioral(*sliced)
            inputs_str = ', '.join(str(a.tolist()) for a in sliced)
            rtp_str = ', ' + ', '.join(map(str, rtp_values)) if rtp_values else ''
            examples.append(f">>> {self.name}({inputs_str}{rtp_str})\n{out.tolist()}")
        return "\n".join(examples)
