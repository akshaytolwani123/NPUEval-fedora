#!/bin/bash
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

# Check Ubuntu version
ubuntu_ver=$(lsb_release -rs | awk '{print $1}')

if [ "$ubuntu_ver" == "24.04" ]; then
    BASE_IMAGE="ubuntu:24.04"
elif [ "$ubuntu_ver" == "24.10" ]; then
    BASE_IMAGE="ubuntu:24.10"
else
    echo "Not supported Ubuntu release: $ubuntu_ver"
    echo "Only Ubuntu 24.04 and Ubuntu 24.10 are supported."
    exit 1
fi

# Check kernel version
kernel_version=$(uname -r | cut -d'-' -f1)

if [[ "$kernel_version" < "6.10" ]]; then
    echo "Current kernel version ${kernel_version} not supported. Required kernel >6.10."
    exit 1
fi

# Build xrt docker and .deb files to be installed on the host
docker build -t xrt -f docker/Dockerfile.xrt --build-arg BASE_IMAGE=$BASE_IMAGE .

# Host machine setup
if dpkg-query -W xrt_plugin-amdxdna; then
    echo "xrt_plugin-amdxdna is already installed, skipping host installs..."
else
    echo "Installing dependencies"
    sudo apt install -y build-essential gcc-x86-64-linux-gnu libgl-dev libxdmcp-dev \
	    bzip2 libalgorithm-diff-perl libglx-dev lto-disabled-list dkms libalgorithm-diff-xs-perl \
	    libhwasan0 make dpkg-dev libalgorithm-merge-perl libitm1 ocl-icd-opencl-dev fakeroot libasan8 \
	    liblsan0 opencl-c-headers g++ libboost-filesystem1.83.0 libquadmath0 opencl-clhpp-headers g++-14 \
	    libboost-program-options1.83.0 libstdc++-14-dev uuid-dev g++-14-x86-64-linux-gnu libcc1-0 libtsan2 \
	    x11proto-dev g++-x86-64-linux-gnu libdpkg-perl libubsan1 xorg-sgml-doctools gcc libfakeroot \
	    libx11-dev xtrans-dev gcc-14 libfile-fcntllock-perl libxau-dev gcc-14-x86-64-linux-gnu \
	    libgcc-14-dev libxcb1-dev
    
    echo "Copying xrt and xrt_plugin debians from xrt:latest"
    docker run --rm -v $(pwd)/docker:/host_dir xrt:latest bash -c "cp -v /XDNA/debs/*.deb /host_dir/"

    echo "Installing debian packages..."
    if ! sudo dpkg -i docker/xrt*-base.deb; then
        sudo apt -y install --fix-broken
    fi

    if ! sudo dpkg -i docker/xrt_plugin*amdxdna.deb; then
        sudo apt -y install --fix-broken
    fi
fi


if ! lsmod | grep -q amdxdna; then
    echo "amdxdna kernel module is not loaded."
    echo "Make sure xrt_plugin-amdxdna is installed, if so try rebooting your machine and run this script again."
    exit 1
fi

# NPUEval docker setup
docker build -t npueval -f docker/Dockerfile.npueval .
