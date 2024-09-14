// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void cumsum_int16(int16_t* in_buffer, int16_t* out_buffer) {
    constexpr int N = 256;
    int32_t acc = 0;
    for (int i = 0; i < N; ++i) {
        acc += static_cast<int32_t>(in_buffer[i]);
        // Clamp to int16_t range
        if (acc > 32767) acc = 32767;
        if (acc < -32768) acc = -32768;
        out_buffer[i] = static_cast<int16_t>(acc);
    }
}