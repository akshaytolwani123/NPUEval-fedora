# Install additional compilers

By default kernels are compiled using the open-source [llvm-aie](https://github.com/Xilinx/llvm-aie) compiler toolchain, however to experiment with different solutions we provide options to switch between backends using the `aie_compiler` abstraction.

## Setting up Chess

To setup your NPUEval environment with the Chess compiler you will have to download the vitis_aie_essentials package. Follow these steps:

* Download vitis_aie_essentials (`ryzen_ai-1.3.0ea1.tgz`) from the [Ryzen AI SW Early Access](https://account.amd.com/en/member/ryzenai-sw-ea.html) site.
* Get an AIE build license from https://www.xilinx.com/getlicense.

Make sure `ryzen_ai-1.3.0ea1.tgz` and `Xilinx.lic` (rename the license file if you have to), are moved under `docker/`, finally run

```
./install
```

from the top level directory of NPUEval. The script will automatically detect the package in the docker/ directory and setup Chess for you.
