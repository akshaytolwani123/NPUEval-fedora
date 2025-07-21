[ [arxiv]() ] [ [blog](https://amdresearch.github.io/NPUEval/blog.html) ] [ [demo](notebooks/03_use_your_own_model.ipynb) ] [ [bibtex]() ]

![](docs/header_small.png)

# NPUEval

NPUEval is an LLM evaluation dataset written specifically to target AIE kernel code generation on RyzenAI hardware.

## Getting started

Requirements:
* Ubuntu 24.04.2 or Ubuntu 24.10 (must have supported Linux kernel version >6.10)
* Disable secure boot on your machine - this is needed because we'll be working with an experimental (unsigned) kernel module.
* Docker - follow instructions in [docs.docker.com](https://docs.docker.com/engine/install/ubuntu/) for setup.

Once you have prerequisites use the install script:
```
./install.sh
```

This will bring up an XRT docker image that will build the XRT and XDNA debian packages which will be installed on your host machine. Then it will setup the NPUEval docker with all the tools required for NPU application compilation.

## Starter notebooks

Launch the JupyterLab environment to open the notebooks and get familiar with using the dataset

```
./scripts/launch_jupyter.sh
```

You'll be able to connect from your browser on port 8888, e.g. `http://localhost:8888/lab` or give it an IP address if you're using the machine remotely.

## Reproducing results

Currently there are 3 simple scripts to reproduce AIECoder results for gpt-4o and gpt-4o-mini. You can run these as regular scripts from your Jupyterlab or interactive docker session, or use `docker_run_script.sh` to run as individual docker sessions.

```
docker_run_script.sh scripts/run_completions.py
docker_run_script.sh scripts/run_functional_tests.py
```

`run_completions` script will feed all the prompts to the AIECoder agent and generate solutions for each test. Make sure to set your `OPENAI_API_KEY` since it will be making requests to `gpt-4o` and `gpt-4o-mini`. 
`run_functional_tests` will evaluate the LLM generated solutions. Since this is just the evaluator it only requires the NPU and no access to an LLM.

## Known issues limitations

* `Failed to open KMQ device (err=22): Invalid argument` -- if you see this just reboot the machine, the driver can get into an unstable state. Hopefully this won't happen with newer versions of the NPU driver.
* Only targeting **AIE2** and **AIE2P** kernels. Phoenix/Hawk for AIE2 and Strix/Krackan for AIE2P.
* Currently only single output kernels are supported, i.e. 1-in-1-out and 2-in-1-out.

## References

* [AI Engine API User Guide](https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/AI-Engine-API-Overview)
* [MLIR-AIE](https://github.com/Xilinx/mlir-aie)
* [LLVM-AIE](https://github.com/Xilinx/llvm-aie)
