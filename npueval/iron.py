# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np  
import sys  
  
from aie.dialects.aie import *  
from aie.dialects.aiex import *  
from aie.extras.context import mlir_mod_ctx  
from aie.helpers.dialects.ext.scf import _for as range_  

import aie.utils.trace as trace_utils

def _get_device(dev: str):
    """Helper function to get AIE device based on device string."""
    if dev == "npu1":
        return AIEDevice.npu1_1col
    elif dev == "npu2":
        return AIEDevice.npu2
    else:
        raise Exception("Unsupported device")

def _prepare_output_buffer(out_buffer: np.ndarray):
    """Helper function to prepare output buffer with padding if needed (e.g. for reduce ops)."""
    pad_elems = 0
    pad_bytes = (-out_buffer.nbytes) % 4

    if pad_bytes:
        pad_elems = pad_bytes // out_buffer.dtype.itemsize
        out_buffer = np.concatenate((out_buffer.flatten(), np.zeros(pad_elems, dtype=out_buffer.dtype)), axis=0)
    
    return out_buffer, pad_elems

def _process_rtps(rtps: list, verbose: bool = False):
    """Helper function to process runtime parameters."""
    rtp_types = [getattr(np, f'int{int(rtp.dtype.name[4:])}') if rtp.dtype.name.startswith('uint') else rtp.dtype.type for rtp in rtps]
    rtp_values = [rtp.item() for rtp in rtps]
    
    if verbose:
        print(f"{rtp_types=}, {rtp_values=}")
    
    return rtp_types, rtp_values

def build_app(kernel_name: str,
              in_buffers: list,
              out_buffer: np.ndarray,
              rtps: list,
              tile_size: int = 1024,
              trace_size: int = 0,
              dev: str = "npu1",
              verbose: bool = False):
    """Generic abstraction for single tile kernel graph in IRON.
    Supports both 1-in-1-out and 2-in-1-out configurations.
    
    Parameters
    ----------
    kernel_name : str
        The kernel name should match the compiled object name (the C++ kernel function
        name) that will be linked in the MLIR.
    in_buffers : list
        List of numpy input buffers representing the data transferred to the NPU.
    out_buffer : np.ndarray
        The numpy output buffer representing the data that will be moved from the NPU
        back to host memory.
    rtps : list
        List of runtime parameters required by the kernel.
    tile_size : int, optional
        How many elements to pass into the compute tile. Default: 1024
    trace_size : int, optional
        Setting this to >0 will enable tracing. Default: 0
    dev : str, optional
        Defaults to npu1 (Phoenix/Hawk), set to npu2 (Strix/Strix Halo/Krackan)
    verbose : bool, optional
        Enable verbose output. Default: False

    Returns
    -------
    str or bool
        Returns the MLIR string or False if the verification failed.
    """
    aie_dev = _get_device(dev)
    buffer_depth = 2
    
    # Calculate number of tiles based on largest input buffer
    num_tiles = max(in_buffer.size // tile_size for in_buffer in in_buffers)
    
    # Prepare output buffer
    out_buffer, pad_elems = _prepare_output_buffer(out_buffer)
    
    # Print verbose information
    if verbose:
        for i, in_buffer in enumerate(in_buffers):
            print(f"{in_buffer.shape=}, {in_buffer.dtype.type=}, {in_buffer.size=}")
        print(f"{out_buffer.shape=}, {out_buffer.dtype.type=}, {out_buffer.size=}")
    
    with mlir_mod_ctx() as ctx:
        @device(aie_dev)
        def device_body():
            # Create type definitions for all buffers
            in_types = [np.ndarray[(in_buffer.size,), np.dtype[in_buffer.dtype.type]] for in_buffer in in_buffers]
            out_ty = np.ndarray[(out_buffer.size,), np.dtype[out_buffer.dtype.type]]
            
            if verbose:
                for in_ty in in_types:
                    print(in_ty)
                print(out_ty)
            
            # Process runtime parameters
            rtp_types, rtp_values = _process_rtps(rtps, verbose)
            
            # AIE Core Function declarations
            kernel_func = external_func(
                kernel_name, inputs=[*in_types, out_ty, *rtp_types]
            )
            
            # Tile declarations
            ShimTile = tile(0, 0)
            ComputeTile2 = tile(0, 2)
            
            # Create object fifos
            ofs = []
            
            # Create input object fifos
            for i, in_buffer in enumerate(in_buffers):
                in_ty = np.ndarray[(in_buffer.size,), np.dtype[in_buffer.dtype.type]]
                of_name = f"in{i+1}" if len(in_buffers) > 1 else "in"
                of_in = object_fifo(of_name, ShimTile, ComputeTile2, buffer_depth, in_ty)
                ofs.append(of_in)
            
            # Create output object fifo
            out_ty = np.ndarray[(out_buffer.size,), np.dtype[out_buffer.dtype.type]]
            of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, out_ty)
            ofs.append(of_out)
            
            # Set up compute tiles
            @core(ComputeTile2, kernel_name+".o")
            def core_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(num_tiles):
                        # Acquire output element
                        elem_out = ofs[-1].acquire(ObjectFifoPort.Produce, 1)
                        
                        # Acquire input elements
                        elem_ins = []
                        for of_in in ofs[:-1]:
                            elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                            elem_ins.append(elem_in)
                        
                        # Call kernel function
                        kernel_func(*elem_ins, elem_out, *rtp_values)
                        
                        # Release all elements
                        for of_in in ofs[:-1]:
                            of_in.release(ObjectFifoPort.Consume, 1)
                        ofs[-1].release(ObjectFifoPort.Produce, 1)
            
            # Set up a packet-switched flow from core to shim for tracing information
            tiles_to_trace = [ComputeTile2, ShimTile]
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)
            
            # To/from AIE-array data movement
            sequence_types = [*in_types, out_ty]
            @runtime_sequence(*sequence_types)
            def sequence(*sequence_params):
                offset = out_buffer.nbytes

                if out_buffer.nbytes < 4:
                    offset = 4*out_buffer.dtype.itemsize
                
                if trace_size > 0:
                    trace_utils.configure_packet_tracing_aie2(
                        tiles_to_trace=tiles_to_trace,
                        shim=ShimTile,
                        trace_size=trace_size,
                        ddr_id=len(in_buffers),
                        trace_offset=offset
                    )
                
                # Create DMA tasks
                tasks = []
                
                # Create input DMA tasks
                for i, (of_in, in_buffer) in enumerate(zip(ofs[:-1], in_buffers)):
                    in_task = shim_dma_single_bd_task(
                        of_in, sequence_params[i], sizes=[1, 1, 1, in_buffer.size], issue_token=True
                    )
                    tasks.append(in_task)
                
                # Create output DMA task
                if out_buffer.nbytes < 4:
                    out_task = shim_dma_single_bd_task(
                        ofs[-1], sequence_params[-1], sizes=[1, 1, 1, 4], issue_token=True
                    )
                else:
                    out_task = shim_dma_single_bd_task(
                        ofs[-1], sequence_params[-1], sizes=[1, 1, 1, out_buffer.size], issue_token=True
                    )
                tasks.append(out_task)
                
                # Start and await all tasks
                dma_start_task(*tasks)
                dma_await_task(*tasks)

                trace_utils.gen_trace_done_aie2(ShimTile)
    
    res = ctx.module.operation.verify()

    if res:
        return ctx.module.__str__(), pad_elems
    else:
        return False