# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from .npueval import run_functional_tests
from .tools import aie_compiler, build_single_kernel_app
from .utils import extract_buffers, trace_to_json, report_peano_version
from .dataset import dataset

import warnings

# This is the expected compiler version from the docker install.
# Compiler version can influence results so we want this on lock.
__compiler_commit__= "b2a279c1939604e2ee82a651683dd995decc25ee"
__compiler_version__ = "19.0.0"

try:
    current_version = report_peano_version()
    if current_version['commit_hash'] != __compiler_commit__:
        warnings.warn(
            f"Compiler version mismatch: expected {__compiler_version__}, commit hash: {__compiler_commit__}, "
            f"but found {current_version['version']}, {current_version['commit_hash']}."
            f"Your results will not match baseline setup.",
            UserWarning,
            stacklevel=2
        )
except Exception as e:
    warnings.warn(
        f"Could not verify compiler version: {e}",
        UserWarning,
        stacklevel=2
    )