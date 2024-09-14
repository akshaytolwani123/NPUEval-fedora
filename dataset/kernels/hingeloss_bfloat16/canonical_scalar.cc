// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void hingeloss_bfloat16(bfloat16* pred, bfloat16* label, bfloat16* out) {
    constexpr int N = 256;
    for (int i = 0; i < N; ++i) {
        float y = float(label[i]);
        float p = float(pred[i]);
        float t = 1.0f - y * p;
        float h = t > 0.0f ? t : 0.0f;
        out[i] = (bfloat16)h;
    }
}