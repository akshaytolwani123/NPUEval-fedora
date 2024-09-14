# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import traceback
from typing import List, Dict, Optional, Any

import numpy as np

from aie.utils.xrt import AIE_Application, write_out_trace

from .utils import (trace_to_json, 
                    get_cycles, 
                    get_vector_time)

class NPUExecutor:
    """Handles execution and validation of kernels on the NPU."""
    
    def __init__(self, 
                 xclbin: str,
                 instr: str,
                 xrt_kernel_name: str = "MLIR_AIE",
                 atol: float = 1e-2,
                 rtol: float = 1e-2,
                 verbose: bool = False):
        self.xclbin = xclbin
        self.instr = instr
        self.xrt_kernel_name = xrt_kernel_name
        self.atol = atol
        self.rtol = rtol
        self.verbose = verbose
        self.app = None
        
    def run(self, 
            in_buffers: List[np.ndarray],
            out_buffers: List[np.ndarray],
            trace_size: int = 0,
            trace_name: str = "",
            padding: int=0):
        """Execute kernel on NPU and validate results.
        
        Parameters
        ----------
        in_buffers : List[np.ndarray]
            Input buffers to process
        out_buffers : List[np.ndarray]
            Reference output buffers for validation
        trace_size : int
            Size of trace buffer if tracing enabled
        trace_name : str
            Path to save trace data if tracing enabled
        padding : int
            For non 4-byte aligned outputs we may need to pad the output buffer. We can get
            this information from the build_1in1out_app and build_2in1out_app functions.
            
        Returns
        -------
        bool
            Whether execution was successful
        """
        
        try:
            self.app = AIE_Application(self.xclbin, self.instr, self.xrt_kernel_name)
            
            # Register input buffers
            self.app.register_buffer(3, shape=in_buffers[0].shape, dtype=in_buffers[0].dtype)
            if len(in_buffers) == 2:
                self.app.register_buffer(4, shape=in_buffers[1].shape, dtype=in_buffers[1].dtype)
            
            # For reduce ops with bfloat16, add padding for 32-bit alignment
            # padding = 3 if out_buffers[0].size == 1 and out_buffers[0].dtype == bfloat16 else 0
            total_elements = out_buffers[0].size + padding + trace_size
            
            # Register output buffer
            self.app.register_buffer(5, shape=(total_elements,), dtype=out_buffers[0].dtype)
            
            # Write input data
            self.app.buffers[3].write(in_buffers[0])
            if len(in_buffers) == 2:
                self.app.buffers[4].write(in_buffers[1])
            
            # Execute
            self.app.run()
            entire_buffer = self.app.buffers[5].read()
            
            # Process output and trace data
            if trace_size > 0:
                data_size = out_buffers[0].size
                if data_size == 1:
                    # For reduce ops, only compare first element
                    result = entire_buffer[0:1]
                else:
                    result = entire_buffer[:data_size].reshape(out_buffers[0].shape)
                    
                trace_buffer = entire_buffer[data_size + padding:]
                write_out_trace(trace_buffer.view(np.uint32), trace_name)
                
                # Process trace data
                mlir_path = f"{self.xclbin.strip('.xclbin')}.mlir"
                trace_json_path = f"{self.xclbin.strip('.xclbin')}_trace.json"
                trace_to_json(
                    trace_name,
                    mlir_path,
                    trace_json_path,
                    dev = os.environ['NPU']
                )
                total_cycles = get_cycles(trace_json_path)
                vector_cycles = get_vector_time(trace_json_path, return_score=False)
            else:
                if data_size == 1:
                    result = entire_buffer[0:1]
                else:
                    result = entire_buffer.reshape(out_buffers[0].shape)
            
            if self.verbose:
                print(f"Expected: {out_buffers[0]}")
                print(f"Result: {result}")
            
            eval_output = self.evaluate_result(result, out_buffers[0])
            
        except Exception as e:
            print("NPU execution failed:", str(e))
            traceback.print_exc()
            
        finally:
            self.cleanup()
            
        if trace_size > 0:
            return eval_output, total_cycles, vector_cycles
        else:
            return eval_output
    
    def evaluate_result(self, expected, result):
        expected = expected.astype(np.float32)
        result = result.astype(np.float32)
        
        # Calculate absolute and relative errors
        abs_errors = np.abs(expected - result)
        rel_errors = np.abs((expected - result) / (expected + np.finfo(float).eps))
        
        # Find maximum error locations
        max_abs_idx = int(np.argmax(abs_errors))
        max_rel_idx = int(np.argmax(rel_errors))
        
        # Check if results are within tolerances
        within_tolerance = bool(np.all(abs_errors <= (self.atol + self.rtol * np.abs(expected))))
        
        return {
            'success': within_tolerance,
            'stats': {
                # Maximum errors
                'max_absolute_error': float(np.max(abs_errors)),
                'max_relative_error': float(np.max(rel_errors)),
                'max_abs_error_idx': max_abs_idx,
                'max_rel_error_idx': max_rel_idx,
                # Stats
                'abs_error_mean': float(np.mean(abs_errors)),
                'abs_error_std': float(np.std(abs_errors)),
                'rel_error_mean': float(np.mean(rel_errors)),
                'rel_error_std': float(np.std(rel_errors))
            }
        }
    
    def cleanup(self):
        """Clean up NPU resources."""
        if self.app:
            self.app.buffers = None
            self.app.insts_buffer = None
            del self.app
            self.app = None