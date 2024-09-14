# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import json
import pathlib
import traceback
from typing import List, Dict, Optional, Any

from .executor import NPUExecutor
from .iron import build_app
from .tools import aie_compiler, build_single_kernel_app
from .utils import (extract_buffers, 
                    get_kernel_code, 
                    parse_stack_sizes,
                    report_xdna_version)

def save_results(result: dict, results_path: str, results_filename: str):
    """Helper function to save current result status to a json file in results_path."""
    pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)
    with open(f"{results_path}/{results_filename}", 'w') as file:
        json.dump(result, file, indent=4)

def run_functional_tests(tests: List[Dict[str, Any]],
                        solutions: Optional[str] = None,
                        results_path: str = "results/evaluations",
                        allow_continue: bool = True,
                        overwrite: bool = False,
                        verbose: bool = False,
                        generate_assembly: bool = False,
                        compiler: str = "peano"):
    """Run functional tests for AIE kernels.
    
    Parameters
    ----------
    tests : List[Dict[str, Any]]
        List of test configurations
    solutions : Optional[str]
        Path to solutions directory
    results_path : str
        Where to store results
    allow_continue : bool
        Skip existing results unless overwrite is True
    overwrite : bool
        Force overwrite of existing results
    verbose : bool
        Enable verbose output
    generate_assembly : bool
        Requires 2nd compiler run, but enables microcode and
        stack size extraction
    compiler : str
        Options are peano or chess
    """
    trace_size = 8192 # large default, won't change between kernels
    
    passed = 0
    for test in tests:
        kernel_name = f"{test['kernel_name']}_wrapper"
        
        if allow_continue:
            if os.path.isfile(f"{results_path}/{kernel_name}.json") and not overwrite:
                print(f"{results_path}/{kernel_name}.json already exists, skipping...")
                continue
        
        results = {'result': 'Fail'}
        print(f"\nKernel: {kernel_name}")
        
        try:
            # Get and validate kernel code
            if solutions is None:
                print("Using canonical solution...")
                kernel_code = test['prompt'] + test['canonical_solution'] + test['program_code']
            else:
                kernel_code = get_kernel_code(test, solutions) + test['program_code']
            
            if not kernel_code:
                save_results(results, results_path, f"{kernel_name}.json")
                continue
            
            # Compile kernel
            compile_result = aie_compiler(kernel_code, 
                                        kernel_name=kernel_name,
                                        output_dir=results_path,
                                        compiler=compiler,
                                        dev=os.environ['NPU'],
                                        generate_assembly=generate_assembly,
                                        verbose_output=verbose)
            if compile_result.split('\n')[0] != 'Compilation successful.':
                print("Failed to compile kernel")
                results['Error'] = compile_result
                save_results(results, results_path, f"{kernel_name}.json")
                continue

            if generate_assembly:
                if verbose:
                    print("stack size (bytes): ", parse_stack_sizes(f"{results_path}/{kernel_name}.s"))
                stack_sizes = parse_stack_sizes(f"{results_path}/{kernel_name}.s")
                results['stack_size'] = stack_sizes
            
            # Generate MLIR
            in_buffers, out_buffers, rtps = extract_buffers(test)
            
            # Calculate tile size based on largest input buffer
            tile_size = max(in_buffer.size for in_buffer in in_buffers)

            mlir, padding = build_app(
                kernel_name, in_buffers, out_buffers[0], rtps,
                tile_size=tile_size,
                trace_size=trace_size,
                dev=os.environ['NPU']
            )
            
            if mlir:
                with open(f"{results_path}/{kernel_name}.mlir", 'w') as f:
                    f.write(mlir)
                print(f"{results_path}/{kernel_name}.mlir generated successfully")
            else:
                print("Failed to generate MLIR")
                raise Exception("MLIR generation failed")
            
            # Build application
            build_result = build_single_kernel_app(
                f"{results_path}/{kernel_name}.mlir",
                f"{results_path}/{kernel_name}.o",
                output_dir=results_path,
                xclbin_name=kernel_name,
                compiler_backend=compiler
            )
            if build_result.returncode != 0:
                raise Exception(f"Build failed with return code {build_result.returncode}")
            
            # Run on NPU and validate
            # Use specific tolerance in test set if exists
            if "tolerances" in test:
                atol = test['tolerances']['atol']
                rtol = test['tolerances']['rtol']
            
                executor = NPUExecutor(
                    xclbin=f"{results_path}/{kernel_name}.xclbin",
                    instr=f"{results_path}/{kernel_name}.bin",
                    verbose=verbose,
                    atol=atol,
                    rtol=rtol
                )
            else:
                executor = NPUExecutor(
                    xclbin=f"{results_path}/{kernel_name}.xclbin",
                    instr=f"{results_path}/{kernel_name}.bin",
                    verbose=verbose
                )
            
            outputs = executor.run(
                in_buffers=in_buffers,
                out_buffers=out_buffers,
                trace_size=trace_size,
                trace_name=f"{results_path}/{kernel_name}_trace.txt",
                padding=padding
            )

            if isinstance(outputs, tuple):
                eval_output, total_cycles, vector_cycles = outputs
            else:
                eval_output = outputs
                total_cycles = None
                vector_cycles = None
            
            results['stats'] = eval_output['stats']
            results['total_cycles'] = total_cycles
            results['vector_cycles'] = vector_cycles
            results['vector_score'] = vector_cycles/total_cycles
            if eval_output['success']:
                results['result'] = 'Pass'
                passed += 1
            
            if verbose:
                print(results['stats'])
                
        except Exception as e:
            error_msg = str(e)
            if "qds_device::wait() unexpected command state" in error_msg or "Failed to open KMQ device" in error_msg:
                print("Driver in unstable state")
                print("Stopping execution")
                return
            print(f"Test failed: {error_msg}")
            results['Error'] = error_msg
            results['Trace'] = traceback.format_exc()
            
        results['xdna_info'] = report_xdna_version()

        print(f"Result: {results['result']}")
        save_results(results, results_path, f"{kernel_name}.json")
    print(f"Passed: {passed}/{len(tests)}")

