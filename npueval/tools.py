# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import re
import subprocess
import pathlib
import shutil
import tempfile

def aie_compiler(src: str,
                 kernel_name: str="kernel",
                 output_dir: str="output", 
                 compiler: str="peano", 
                 dev="npu1", 
                 generate_assembly: bool=False,
                 verbose_output: bool=False) -> str:
    """Function that calls a single kernel AIE compiler. The resulting .o file 
    gets stored in output_dir - by default ./output/kernel.o
    
    Parameters
    ----------
    src : str
        Source code written as C++
    kernel_name : str
        Name given to outputs files
    output_dir : str
        Directory to store compilation outputs, relative to current working dir,
        by default the kernel .o file will be stored in ./output/kernel.o
    compiler : str
        Which compiler to use in backend, valid options are: peano, chess.
    dev : str
        NPU device, options are "npu1" and "npu2" corresponding to Phoenix and
        Strix respectively.
    generate_assembly : bool
        If True, also generates an assembly file kernel.s alongside kernel.o. This
        requires a second clang++ compilation which doubles compile time, hence
        False by default.
    verbose_output : bool
        If true will generate extra outputs. You might want this disabled to save
        LLM tokens. If set to False it will concisely only produce error messages
        and not output anything on successful compiles.

    Returns
    -------
    result : str
        Result message or log of errors in the case of a failure. 
    """

    if compiler=="peano":
        peano_dir = os.environ.get("PEANO_INSTALL_DIR", "")
        if not peano_dir:
            raise EnvironmentError("PEANO_INSTALL_DIR environment variable is not set.")
    
    # Create output dir if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    tmp_src_file = os.path.join(os.getcwd(), output_dir, kernel_name + ".cc")
    output_object = os.path.join(os.getcwd(), output_dir, kernel_name + ".o")
    
    with open(tmp_src_file, "w") as f:
        f.write(src)

    if compiler=="peano":
        if dev=="npu1":
            compile_flags = os.environ['PEANOWRAP2_FLAGS'].split(' ')
        elif dev=="npu2":
            compile_flags = os.environ['PEANOWRAP2P_FLAGS'].split(' ')
        else:
            raise Exception(f"Unsupported device: {dev}")
        base_command = [f"{peano_dir}/bin/clang++", *compile_flags]
    elif compiler=="chess":
        if dev=="npu1":
            compile_flags = os.environ['CHESSCCWRAP2_FLAGS'].split(' ')
        elif dev=="npu2":
            compile_flags = os.environ['CHESSCCWRAP2P_FLAGS'].split(' ')
        else:
            raise Exception(f"Unsupported device: {dev}")
        base_command = ["xchesscc_wrapper", *compile_flags]
    else:
        raise Exception(f"Unsupported single core compiler: {compiler}, choose 'peano' or 'chess'.")

    try:
        # Generate assembly
        if generate_assembly:
            output_assembly = os.path.join(os.getcwd(), output_dir, kernel_name + ".s")
            asm_command = [*base_command, "-S", "-fverbose-asm", "-fstack-size-section", "-c", tmp_src_file, "-o", output_assembly]
            subprocess.check_output(asm_command, stderr=subprocess.STDOUT, text=True)
        
        # Compile kernel
        full_command = [*base_command, "-c", tmp_src_file, "-o", output_object]
        
        if not verbose_output:
            # suppress warnings only show errors
            full_command.append('-w')
        
        if verbose_output:
            full_command.append('-v')

        result = subprocess.check_output(full_command, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        return e.output
    
    return f"Compilation successful.\nObject file generated at {output_dir}/{kernel_name}.o"

def build_single_kernel_app(mlir_file: str,
                            kernel_file: str,
                            xclbin_name: str="app",
                            output_dir: str="output",
                            workdir: str=None,
                            compiler_backend: str= "peano"):
    """Calls aiecc.py as a subprocesses. Specifically for building a single kernel app,
    which is why it takes exactly 1 kernel object as a parameter.

    Parameters
    ----------
    mlir_file : str
        The MLIR definition of the AIE2 application design graph. This can be hand-written
        or produced by IRON Python bindings.
    kernel_file : str
        Path to compiled kernel object (.o) file.
    xclbin_name : str, optional
        Name of the final xclbin name. It will also generate an instruction sequence .txt
        of the same name as xclbin_name.
    workdir : str or None, optional
        By default the subprocess will run in a temp dir, but if you want to see intermediary
        outputs set this to a different path.
    compiler_backend : str, optional
        Choose compiler backend, defaults to peano.

    Returns
    -------
    returncode : int
        0 if process finished successfully
    """
    if not workdir:
        tmp_workdir = tempfile.TemporaryDirectory(suffix=None, prefix="build_", dir=".")
        workdir = tmp_workdir.name
    else:
        tmp_workdir = None

    try:
        # Copy kernel.o and aie.mlir to the work directory
        shutil.copy2(kernel_file, os.path.join(workdir, pathlib.Path(kernel_file).parts[-1]))
        shutil.copy2(mlir_file, os.path.join(workdir, os.path.basename(mlir_file)))

        # Check if peano dir is set
        peano_dir = os.environ.get("PEANO_INSTALL_DIR", "")
        if not peano_dir:
            raise EnvironmentError("PEANO_INSTALL_DIR environment variable is not set.")
    
        aiecc_flags = ["--aie-generate-cdo",
                       "--aie-generate-npu-insts",
                       "--aie-generate-xclbin",
                       "--no-compile-host",
                       "--no-xchesscc",
                       "--no-xbridge",
                       "--peano", f"{os.environ['PEANO_INSTALL_DIR']}"]

        if compiler_backend == "chess":
            aiecc_flags = ["--aie-generate-cdo",
                       "--aie-generate-npu",
                       "--no-compile-host",
                       "--peano", f"{os.environ['PEANO_INSTALL_DIR']}"]
        
        command = ["aiecc.py",
                   *aiecc_flags,
                   f"--xclbin-name={xclbin_name}.xclbin",
                   f"--npu-insts-name={xclbin_name}.bin",
                   f"{os.path.basename(mlir_file)}"]

        # Outputs of build should be xclbin_name.xclbin and xclbin_name.bin
        result = subprocess.run(command, cwd=workdir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        print(f"{xclbin_name}.xclbin, {xclbin_name}.bin built")
        
        # Return the paths to the generated files
        xclbin_path = os.path.join(workdir, f"{xclbin_name}.xclbin")
        instr_path = os.path.join(workdir, f"{xclbin_name}.bin")

        xclbin_output = os.path.join(output_dir, f"{xclbin_name}.xclbin")
        instr_output = os.path.join(output_dir, f"{xclbin_name}.bin")

        # Create output dir if it doesn't exist
        pathlib.Path(xclbin_output).parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(xclbin_path, xclbin_output)
        shutil.copy2(instr_path, instr_output)
        
        return result
        
    except subprocess.CalledProcessError as e:
        raise Exception("Build failed:", e.stderr)
