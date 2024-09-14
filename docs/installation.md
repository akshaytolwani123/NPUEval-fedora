# Getting started

The NPUEval evaluation harness is based entirely on open-source compiler toolchains. All you need is an AI PC machine like a Phoenix or Strix laptop or miniPC with a Ubuntu image installed on it. The installation scripts will setup a docker environment for you with all necessary components to build and execute NPU applications on your hardware.

## System requirements

* Ubuntu 24.04.2 or Ubuntu 24.10 (must have supported Linux kernel version >6.10)
* Docker - follow instructions in [docs.docker.com](https://docs.docker.com/engine/install/ubuntu/) for setup.

## Clone the repository

To get started, clone the repository and run the installation script:

```bash
git clone https://github.com/AMDResearch/NPUEval
cd NPUEval
./install.sh
```

## Run tests

To make sure your docker environment is setup correctly run the canonical tests script.

```sh
docker_run_script.sh scripts/run_canonical.py
```

