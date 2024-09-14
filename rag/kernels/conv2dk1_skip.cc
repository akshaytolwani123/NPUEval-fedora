//===- conv2dk1_skip.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 skip - vector
// act: uint8, wts: int8, skip: int8, out: uint8
//
// Assume IC >= 16 as that gives ideal inner loop schedule
//
// TODO - Restricting input_width is mutiple of 32
// Because each VMAC works on 4 inputs at a time and we store intermediate
// results in 8 accumulators, having input_width be a multiple of 4*8=32 is
// ideal. However, we should be able to support input_width that is only a
// multiple of 4 but there is some strange scheduling happening now so for
// now, we do not.
//*****************************************************************************
void conv2dk1_skip_i8_vector(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                             uint8_t *output, int8_t *skip,
                             const int32_t input_width,
                             const int32_t input_channels,
                             const int32_t output_channels, const int scale,
                             const int skip_scale) {

  using MMUL4x8x8 = aie::mmul<4, 8, 8, uint8, int8>;
  ::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
  ::aie::set_rounding(
      aie::rounding_mode::positive_inf); // Needed to saturate properly to uint8

  uint8_t *restrict out_ptr = output;
  int8_t *i_out_ptr = (int8_t *)output;
  int8_t *restrict skip_ptr = skip;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;

  constexpr int NUM_ACC = 8;

  const int iw_32 = (input_width / 4) / 8;
  const int iw = input_width;
  // const int iw_32_rem = (input_width / 4) % 8;
  assert((input_width / 4) % 8 == 0);
  const int iw_32_rem = 0; // TODO - See restriction

  assert((input_channels / 8) > 2); // Assume IC >= 16

  int input_offset1 = 0;
  int input_offset2 = 0;

  if (iw_32 > 0) {

    for (int oc = 0; oc < (output_channels / 8); oc++) {
      for (int x = 0; x < iw_32; x++) {
        MMUL4x8x8 acc_tmp[NUM_ACC];
        for (int i = 0; i < NUM_ACC; i++) {
          acc_tmp[i] = aie::zeros<acc32, 32>();
        }
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(2)
        for (int ic = 0; ic < (input_channels / 16); ic++) {
          aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
          kernels += 64; // wts ic0..7(oc0..7)

          for (int x8 = 0; x8 < NUM_ACC; x8++) {
            aie::vector<uint8, 32> in_a =
                aie::load_v<32>(input0 + input_offset1);
            input_offset1 += 32; // act oc0..3(ic0..7)
            acc_tmp[x8].mac(in_a, in_b);
          }
          input_offset1 +=
              (iw * 8) -
              256; // Move to next ic/8 position. 256 = 32 input * 8 ic
        }
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(2)
        for (int ic = 0; ic < (input_channels / 16); ic++) {
          aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
          kernels += 64; // wts ic0..7(oc0..7)

          for (int x8 = 0; x8 < NUM_ACC; x8++) {
            aie::vector<uint8, 32> in_a =
                aie::load_v<32>(input1 + input_offset2);
            input_offset2 += 32; // act oc0..3(ic0..7)
            acc_tmp[x8].mac(in_a, in_b);
          }
          input_offset2 +=
              (iw * 8) -
              256; // Move to next ic/8 position. 256 = 32 input * 8 ic
        }
        // input ptr just moves to next section
        for (int x8 = 0; x8 < NUM_ACC; x8++) {
          aie::vector<int8, 32> skip1 = aie::load_v<32>(skip_ptr);
          skip_ptr += 32;

          aie::accum<acc32, 32> accj;
          accj.from_vector(skip1, 0);
          accj = aie::add(accj, acc_tmp[x8].to_vector<int8>(scaleT));
          // accj = aie::mac(accj, acc_tmp[x8].to_vector<int8>(scaleT),
          // (uint8_t)1);
          aie::vector<uint8, 32> o1 = accj.to_vector<uint8>(skip_scaleT);
          aie::store_v(out_ptr, o1);
          out_ptr += 32;
          // acc_tmp[x8] = aie::zeros<acc32,32>();
        }
        input_offset1 -=
            ((input_channels / 16) * iw * 8) -
            256; // reset to next input_width/32 block. 256 = 32 input * 8 ic
        input_offset2 -=
            ((input_channels / 16) * iw * 8) -
            256; // reset to next input_width/32 block. 256 = 32 input * 8 ic
        kernels -=
            (input_channels / 8) * 64; // reset kernel back to beginning of ic/8
      }                                // for(int x=0; x<iw_32; x++) {
      // input_offset -= (iw_32) * 256; // 8*32, reset beginning of input ptr
      input_offset1 = 0;                    // reset beginning of input ptr
      input_offset2 = 0;                    // reset beginning of input ptr
      kernels += (input_channels / 8) * 64; // move to next oc/8 weights
      out_ptr += (iw_32_rem *
                  32); // move to next oc/8 (skip remainder section if present)
    }                  // for(int oc=0; oc<(output_channels/8); oc++) {

    out_ptr -= output_channels *
               iw; // output_channels/8*iw_32*8*32 = 256/8*(iw/4/8)*8*32

    out_ptr -= (output_channels - 1) * iw + (iw_32_rem * 32);
    skip_ptr -= (output_channels - 1) * iw + (iw_32_rem * 32);

  }
}

#else // UINT8_ACT

//*****************************************************************************
// conv2d 1x1 skip - vector
// act: uint8, wts: int8, skip: uint8, out: uint8
//
// Assume IC >= 16 as that gives ideal inner loop schedule
//
// TODO - Restricting input_width is mutiple of 32
// Because each VMAC works on 4 inputs at a time and we store intermediate
// results in 8 accumulators, having input_width be a multiple of 4*8=32 is
// ideal. However, we should be able to support input_width that is only a
// multiple of 4 but there is some strange scheduling happening now so for
// now, we do not.
//*****************************************************************************
void conv2dk1_skip_ui8_vector(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                              uint8_t *output, uint8_t *skip,
                              const int32_t input_width,
                              const int32_t input_channels,
                              const int32_t output_channels, const int scale,
                              const int skip_scale) {

  using MMUL4x8x8 = aie::mmul<4, 8, 8, uint8, int8>;
  ::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
  ::aie::set_rounding(
      aie::rounding_mode::positive_inf); // Needed to saturate properly to uint8

  uint8_t *restrict out_ptr = output;
  int8_t *i_out_ptr = (int8_t *)output;
  uint8_t *restrict skip_ptr = skip;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;

  constexpr int NUM_ACC = 8;

  const int iw_32 = (input_width / 4) / 8;
  const int iw = input_width;
  // const int iw_32_rem = (input_width / 4) % 8;
  assert((input_width / 4) % 8 == 0);
  const int iw_32_rem = 0; // TODO - See restriction

  assert((input_channels / 8) > 2); // Assume IC >= 16

  int input_offset1 = 0;
  int input_offset2 = 0;

  if (iw_32 > 0) {

    for (int oc = 0; oc < (output_channels / 8); oc++) {
      for (int x = 0; x < iw_32; x++) {
        MMUL4x8x8 acc_tmp[NUM_ACC];
        for (int i = 0; i < NUM_ACC; i++) {
          acc_tmp[i] = aie::zeros<acc32, 32>();
        }
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(2)
        for (int ic = 0; ic < (input_channels / 16); ic++) {
          aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
          kernels += 64; // wts ic0..7(oc0..7)

          for (int x8 = 0; x8 < NUM_ACC; x8++) {
            aie::vector<uint8, 32> in_a =
                aie::load_v<32>(input0 + input_offset1);
            input_offset1 += 32; // act oc0..3(ic0..7)
            acc_tmp[x8].mac(in_a, in_b);
          }
          input_offset1 +=
              (iw * 8) -
              256; // Move to next ic/8 position. 256 = 32 input * 8 ic
        }
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(2)
        for (int ic = 0; ic < (input_channels / 16); ic++) {
          aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
          kernels += 64; // wts ic0..7(oc0..7)

          for (int x8 = 0; x8 < NUM_ACC; x8++) {
            aie::vector<uint8, 32> in_a =
                aie::load_v<32>(input1 + input_offset2);
            input_offset2 += 32; // act oc0..3(ic0..7)
            acc_tmp[x8].mac(in_a, in_b);
          }
          input_offset2 +=
              (iw * 8) -
              256; // Move to next ic/8 position. 256 = 32 input * 8 ic
        }
        // input ptr just moves to next section
        for (int x8 = 0; x8 < NUM_ACC; x8++) {
          aie::vector<uint8, 32> skip1 = aie::load_v<32>(skip_ptr);
          skip_ptr += 32;

          aie::accum<acc32, 32> accj;
          accj.from_vector(skip1, 0);
          accj = aie::add(accj, acc_tmp[x8].to_vector<int8>(scaleT));
          // accj = aie::mac(accj, acc_tmp[x8].to_vector<int8>(scaleT),
          // (uint8_t)1);
          aie::vector<uint8, 32> o1 = accj.to_vector<uint8>(skip_scaleT);
          aie::store_v(out_ptr, o1);
          out_ptr += 32;
          // acc_tmp[x8] = aie::zeros<acc32,32>();
        }
        input_offset1 -=
            ((input_channels / 16) * iw * 8) -
            256; // reset to next input_width/32 block. 256 = 32 input * 8 ic
        input_offset2 -=
            ((input_channels / 16) * iw * 8) -
            256; // reset to next input_width/32 block. 256 = 32 input * 8 ic
        kernels -=
            (input_channels / 8) * 64; // reset kernel back to beginning of ic/8
      }                                // for(int x=0; x<iw_32; x++) {
      // input_offset -= (iw_32) * 256; // 8*32, reset beginning of input ptr
      input_offset1 = 0;                    // reset beginning of input ptr
      input_offset2 = 0;                    // reset beginning of input ptr
      kernels += (input_channels / 8) * 64; // move to next oc/8 weights
      out_ptr += (iw_32_rem *
                  32); // move to next oc/8 (skip remainder section if present)
    }                  // for(int oc=0; oc<(output_channels/8); oc++) {

    out_ptr -= output_channels *
               iw; // output_channels/8*iw_32*8*32 = 256/8*(iw/4/8)*8*32

    out_ptr -= (output_channels - 1) * iw + (iw_32_rem * 32);
    skip_ptr -= (output_channels - 1) * iw + (iw_32_rem * 32);

  }
}

#endif // UINT8_ACT

//*****************************************************************************
// conv2d 1x1 skip wrappers
//*****************************************************************************
extern "C" {

#ifdef INT8_ACT

void conv2dk1_skip_i8(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                      uint8_t *output, int8_t *skip, const int32_t input_width,
                      const int32_t input_channels,
                      const int32_t output_channels, const int scale,
                      const int skip_scale) {
  conv2dk1_skip_i8_vector(input0, input1, kernels, output, skip, input_width,
                          input_channels, output_channels, scale, skip_scale);
}

#else // UINT8_ACT

void conv2dk1_skip_ui8(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                       uint8_t *output, uint8_t *skip,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels, const int scale,
                       const int skip_scale) {
  conv2dk1_skip_ui8_vector(input0, input1, kernels, output, skip, input_width,
                           input_channels, output_channels, scale, skip_scale);
}

#endif // UINT8_ACT

} // extern "C"