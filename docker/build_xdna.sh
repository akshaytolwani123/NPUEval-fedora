#!/bin/bash

set -e

if [ ! -d "xdna-driver" ]; then
    echo "Cloning repository..."
    git clone https://github.com/amd/xdna-driver
else
    echo "Repository already exists, skipping clone..."
fi

cd xdna-driver
git checkout 0ad5aa3
git submodule update --init --recursive

./tools/amdxdna_deps.sh

cd xrt/build/
./build.sh -npu -opt

cd ../../build
./build.sh -release
./build.sh -package
