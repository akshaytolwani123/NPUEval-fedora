#!/bin/bash
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Copy running host version info from host to docker env
cp /opt/xilinx/xrt/share/amdxdna/version.json $(pwd)

docker run -it \
    --device=/dev/accel/accel0:/dev/accel/accel0 \
    --cap-add=NET_ADMIN \
    --ulimit memlock=-1 \
    -v $(pwd):/host:Z \
    -p 8888:8888 \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    npueval \
    bash -c "cd /host && python3 scripts/check_xrt_versions.py && python3 -m jupyterlab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
