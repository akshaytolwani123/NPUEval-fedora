// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>  // provides bfloat16

// Softplus(x) = ln(1 + e^x) for N=256 bfloat16 inputs.
// Single-function, no externals, with higher-order exp & ln1p series for extra precision.
void softplus_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int N = 256;
    // constants for exp2-approximation
    const float LOG2E = 1.4426950408889634f;  // log2(e)
    const float C1    = 0.6931471805599453f;  // ln2
    const float C2    = 0.2402265069591010f;  // (ln2)^2/2
    const float C3    = 0.05550410866482158f; // (ln2)^3/6
    const float C4    = 0.009618129107628477f;// (ln2)^4/24

    for (int i = 0; i < N; ++i) {
        float x = float(input[i]);
        float sp;

        if (x > 16.0f) {
            // for large x, ln(1+e^x) ≈ x
            sp = x;
        }
        else if (x < -16.0f) {
            // for very negative x, ln(1+e^x) ≈ e^x
            // inline exp_approx(x) with 4-term polynomial
            float t = x * LOG2E;
            int   ip = int(t);
            float fp = t - float(ip);
            unsigned bits = unsigned(ip + 127) << 23;
            union { unsigned u; float f; } pun; pun.u = bits;
            float p2i = pun.f;
            float p2f = 1.0f
                      + C1 * fp
                      + C2 * fp * fp
                      + C3 * fp * fp * fp
                      + C4 * fp * fp * fp * fp;
            sp = p2i * p2f;
        }
        else {
            // mid-range: compute z = e^x or e^{-x}
            float z;
            bool  plus = (x >= 0.0f);

            // compute exponent argument and exp_approx
            float t = (plus ? -x : x) * LOG2E;
            int   ip = int(t);
            float fp = t - float(ip);
            unsigned bits = unsigned(ip + 127) << 23;
            union { unsigned u; float f; } pun; pun.u = bits;
            float p2i = pun.f;
            float p2f = 1.0f
                      + C1 * fp
                      + C2 * fp * fp
                      + C3 * fp * fp * fp
                      + C4 * fp * fp * fp * fp;
            z = p2i * p2f;

            // series ln1p(z) ≈ z - ½z² + ⅓z³ - ¼z⁴ + ⅕z⁵ - ⅙z⁶ + ⅐z⁷
            float z2 = z*z, z3 = z2*z, z4 = z3*z, z5 = z4*z, z6 = z5*z, z7 = z6*z;
            float ln1p =  z
                        - 0.5f       * z2
                        + 0.3333333f * z3
                        - 0.25f      * z4
                        + 0.2f       * z5
                        - 0.1666667f * z6
                        + 0.1428571f * z7;

            // combine
            sp = plus ? (x + ln1p) : ln1p;
        }

        output[i] = bfloat16(sp);
    }
}
