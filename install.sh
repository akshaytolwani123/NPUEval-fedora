#!/bin/bash
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

# Check kernel version
kernel_version=$(uname -r | cut -d'-' -f1)

if [[ "$kernel_version" < "6.10" ]]; then
    echo "Current kernel version ${kernel_version} not supported. Required kernel >6.10."
    exit 1
fi

# Build xrt docker and .rpm files to be installed on the host
docker build -t xrt -f docker/Dockerfile.xrt .

# Host machine setup
if rpm -q xrt_plugin-amdxdna; then
    echo "xrt_plugin-amdxdna is already installed, skipping host installs..."
else
    echo "Installing dependencies"
    # OpenCL-ICD-Loader which is installed by default on Fedora conflicts with ocl-icd
    # dnf --allowerasing to remove OpenCL-ICD-Loader works but dunno if that should be the default
    # Use kernel-devel-matched to bring in packages that ensure /lib/modules/$(uname -r)/build exists
    sudo dnf install -y git openssl jq wget python3.12 gcc-c++ cmake rpm-build \
    curl-devel boost-devel rapidjson-devel libdrm-devel ocl-icd-devel  \
    ncurses-devel protobuf-devel python3.12-devel gtest-devel \
    libuuid-devel systemtap-sdt-devel libstdc++-static \
    glibc-static python3-pybind11 dkms kernel-headers kernel-devel-matched
    
    echo "Copying xrt and xrt_plugin rpms from xrt:latest"
    docker run --rm -v $(pwd)/docker:/host_dir:Z xrt:latest bash -c "cp -v /XDNA/rpms/*.rpm /host_dir/"

    echo "Installing rpm packages..."
    sudo dnf install -y docker/xrt*-base.rpm docker/xrt_plugin*amdxdna.rpm
fi

if ! lsmod | grep -q amdxdna; then
    echo "amdxdna kernel module is not loaded."
    echo "Make sure xrt_plugin-amdxdna is installed, if so try rebooting your machine and run this script again."
    exit 1
fi

# NPUEval docker setup
docker build -t npueval -f docker/Dockerfile.npueval .