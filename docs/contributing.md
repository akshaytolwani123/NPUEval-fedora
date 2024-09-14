# Contributing to NPUEval

We welcome contributions to NPUEval! Whether you're fixing bugs, adding new features or adding new kernels, your contributions are greatly appreciated.

## Adding kernels to the dataset

Add a new directory in `kernels/` and create a generate.py script to produce a `kernel.json` file following this schema:

* kernel_name -- the unique kernel identifier, this will typically just be the kernel name.
* prompt -- this is the C++ function definition that will be fed into the LLM.
* canonical_solution -- **a** solution, can be a scalar kernel for sanity runs without LLM generated code.
* program_code -- wrapper C++ code around the kernel call with event generators (i.e. event0, event1).
* test_vectors
    * inputs -- list of input vectors.
    * outputs -- list of output vectors.
    * rtps -- list of runtime parameters.

You can also optionally pass arbitrary tolerances, e.g.

```
tolerances = {"atol": 0.02, "rtol": 0.02}
```

Kernel prompt generation is greatly simplified using the PromptConstructor class, which ingests the python behavioral model and canonical scalar solution and automatically produces the json file with the required schema. Check out one of the existing kernels for examples on how to use it.
