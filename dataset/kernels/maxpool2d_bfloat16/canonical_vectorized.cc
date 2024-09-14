// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

void max_pool2d_aligned_bfloat16(bfloat16* input, bfloat16* output) {
    #define ROWS 32
    #define COLS 32
    #define LANES 32
    
    for (unsigned i = 0; i < (ROWS * COLS); i += (2*COLS)) {

        auto row_n = ::aie::load_v<LANES>(input);
        input += COLS;
        auto row_n_even = ::aie::filter_even(row_n); // filter by even indices
        auto row_n_odd = ::aie::filter_odd(row_n);  // filter by  odd indices
        auto row_n_max = aie::max(row_n_even, row_n_odd); // Find max value in the top half

        auto row_n_1 = ::aie::load_v<LANES>(input);
        input += COLS;
        auto row_n_1_even = ::aie::filter_even(row_n_1); // filter by even indices
        auto row_n_1_odd = ::aie::filter_odd(row_n_1);  // filter by  odd indices
        auto row_n_1_max = ::aie::max(row_n_1_even, row_n_1_odd); // Find max value in the lower half

        auto boxmax = ::aie::max(row_n_max, row_n_1_max); // Find max value in the 2x2 kernel

        ::aie::store_v(output, boxmax);
        output += COLS/2;
    }
}