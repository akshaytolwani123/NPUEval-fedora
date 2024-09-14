# NPUEval dataset

| # | Category | What goes in it | Kernel count | Kernels |
| -- | -- | -- | -- | -- |
| 1 | Element-wise & Activation | Single-input or pairwise math that doesn’t depend on spatial structure. Includes classic arithmetic (add, sub, div, sign, √/rsqrt, exp/log/trig, max/min, etc.) plus all nonlinear activation functions. | 46 | abs_int8, add_offset_int8, add_offset_uint8, ceil_bfloat16, complexabs_bfloat16, cos_bfloat16, divide_bfloat16, elementwise_max_bfloat16, elementwise_max_int8, elementwise_min_bfloat16, elementwise_min_int8, exp_bfloat16, floor_bfloat16, gelu_bfloat16, hardsigmoid_bfloat16, hardsigmoid_int8, hardswish_bfloat16, inverse_uint8, leaky_relu_bfloat16, log10_bfloat16, log2_bfloat16, log_bfloat16, mish_bfloat16, negate_bfloat16, negate_int8, reciprocal_bfloat16, relu6_bfloat16, relu_bfloat16, relu_bfloat16_cast_uint8, relu_int8, round_bfloat16, rsqrt_bfloat16, sigmoid_bfloat16, sign_int8, sin_bfloat16, softmax_bfloat16, softplus_bfloat16, sqrt_bfloat16, tan_bfloat16, tanh_bfloat16, vectoradd_bfloat16, vectoradd_int16, vectoradd_relu_bfloat16, vectormult_bfloat16, vectorsubtract_bfloat16, vectorsubtract_int8
| 2 | Spatial / Linear-Algebra | Anything that exploits neighborhood or tensor layout: convolutions, pooling, GEMM / matrix-matrix, dot products, vector-matrix multiplies. | 24 | avgpool1d_bfloat16, avgpool1d_relu_bfloat16, avgpool2d_bfloat16, avgpool2d_relu_bfloat16, conv1d_bfloat16, conv1d_bias_relu_bfloat16, conv1d_int32, conv1d_k2_s1_bias_relu_bfloat16, conv1d_k2_s2_bias_int16, conv1d_k4_s1_bias_relu_bfloat16, conv2d_bfloat16, conv2d_int32, conv2d_k2_s1_bias_relu_bfloat16, conv2d_k4_s2_bias_relu_bfloat16, dotproduct_bfloat16, dotproduct_bias_relu_bfloat16, dotproduct_bias_relu_int8, dotproduct_int32, gather_bfloat16_int32idx, maxpool1d_uint8, maxpool2d_bfloat16, maxpool2d_relu_bfloat16, maxpool2d_relu_int8, vectormatrix_mult_int32 |
| 3 | Reductions & Statistics / Loss | Operations that collapse dimensions or compute aggregate stats or distances, plus loss functions. | 13 | argmax_bfloat16, argmax_int32, argmin_bfloat16, euclidean_dist_bfloat16, l1_norm_bfloat16, max_abs_bfloat16, mse_loss_bfloat16, reduce_add_relu_int8, reducemax_int32, reducemin_bfloat16, reducemin_int32, reducesum_int32, variance_bfloat16 |
| 4 | Bitwise, Comparison, Casting & Data movement / Padding | Low-level logical ops, popcount, equality/greater/less predicates, type-conversions, padding and shuffles. | 17 | bitcount_uint16, bitcount_uint8, bitwiseand_uint8, bitwisenot_uint8, bitwiseor_uint8, bitwisexor_uint8, cast_bfloat16_to_float32, cast_bfloat16_to_int8, cast_float32_to_bfloat16, cast_int8_to_int32, compare_equal_bfloat16, compare_equal_int32, compare_gt_int8, compare_lt_int8, pad1d_int32, pad2d_int32, shuffle_int32 |

Use these scripts to reproduce the NPUEval dataset npueval.jsonl file.

## Reproduction steps

The makefile will reproduce the whole dataset and store each kernel sample in a npueval.jsonl file. Just run make
```
make
```

## Adding new kernels

Add a new directory in `kernels/` and create a generate.py script to produce a `kernel.json` file following this schema:

* kernel_name -- the unique kernel identifier, this will typically just be the kernel name.
* prompt -- this is the C++ function definition that will be fed into the LLM.
* canonical_solution -- **a** solution, can be a scalar kernel for sanity runs without LLM generated code.
* program_code -- wrapper C++ code around the kernel call with event generators (i.e. event0, event1).
* test_vectors
    * inputs -- list of input vectors.
    * outputs -- list of output vectors.
    * rtps -- list of runtime parameters.
* data_movement
    * tile_size -- amount of data sent to the compute tile per kernel call.
    * total_transfer -- total transfer of data from host memory to NPU.
    * trace_size -- if this is set >0 it will enable trace (cycle count) generation for that kernel.
