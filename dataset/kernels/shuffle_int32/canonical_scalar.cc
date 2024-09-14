// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void shuffle_int32(int32_t* input, int32_t* output, int32_t seed) {
    constexpr int32_t N = 256;

    for (int i = 0; i < N; ++i) {
        output[i] = input[i];
    }

    uint32_t state = static_cast<uint32_t>(seed);

    for (int i = N - 1; i > 0; --i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        int j = static_cast<int>(state % uint32_t(i + 1));
        int32_t tmp = output[i];
        output[i] = output[j];
        output[j] = tmp;
    }
}