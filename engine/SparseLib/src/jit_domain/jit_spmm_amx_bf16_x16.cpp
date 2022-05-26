//  Copyright (c) 2021 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "jit_domain/jit_spmm_amx_bf16_x16.hpp"

#define TILE_M 16  // Number of rows in an A or C tile
#define TILE_K 32  // Number of columns in an A tile or rows in a B tile
#define TILE_N 16  // Number of columns in a B or C tile
#define KPACK 2    // Vertical K packing into dword
#define MZ 64      // (M / MT)
#define NUM_M 4    // (MZ / TILE_N)

#define GET_OFF(field) offsetof(ssd::amx_bf16f32_inputs_t, field)

namespace jd {
void jit_spmm_amx_bf16_x16_t::read_inputs() {
  mov(reg_weight, ptr[reg_param + GET_OFF(weight)]);
  mov(reg_src, ptr[reg_param + GET_OFF(src)]);
  mov(reg_dst, ptr[reg_param + GET_OFF(dst)]);
  mov(reg_bs, ptr[reg_param + GET_OFF(bs)]);
}

void jit_spmm_amx_bf16_x16_t::main_compute(dim_t mstart) {
  for (int b_row = 0; b_row < nrowptr - 1; ++b_row) {
    tilezero(tmm0);
    tilezero(tmm1);
    tilezero(tmm2);
    tilezero(tmm3);

    for (int group = group_rowptr[b_row]; group < group_rowptr[b_row + 1]; ++group) {
      dim_t* my_rows = colidxs + group * 32;

      mov(r12, sizeof(src_t));
      mov(r13, TILE_K);
      imul(r13, r12);
      tileloadd(tmm6, ptr[reg_weight + r13 + group * 512 * sizeof(src_t)]);

      for (int m = mstart; m < mstart + tileM; m += TILE_M) {
        for (int k = 0; k < 32; k += 2) {
          vmovdqu(ymm0, ptr[reg_src + (m + my_rows[k] * M) * sizeof(src_t)]);
          vmovdqu(ymm1, ptr[reg_src + (m + my_rows[k + 1] * M) * sizeof(src_t)]);
          vinserti32x8(zmm0, zmm0, ymm1, 1);
          vpermw(zmm0, reg_mask, zmm0);
          vmovdqu32(qword[rsp + 0x40 + ((m - mstart) / TILE_M * 512 + k / 2 * 32) * 2], zmm0);
        }
      }
      mov(rax, 64);
      tileloadd(tmm4, ptr[rsp + rax + (0x40)]);
      tdpbf16ps(tmm0, tmm6, tmm4);
      tileloadd(tmm4, ptr[rsp + rax + (1024 + 0x40)]);
      tdpbf16ps(tmm1, tmm6, tmm4);
      tileloadd(tmm4, ptr[rsp + rax + (2048 + 0x40)]);
      tdpbf16ps(tmm2, tmm6, tmm4);
      tileloadd(tmm4, ptr[rsp + rax + (3072 + 0x40)]);
      tdpbf16ps(tmm3, tmm6, tmm4);
    }
    mov(r12, sizeof(dst_t));
    mov(rax, reg_bs);
    imul(rax, r12);
    mov(r12, b_row * 16 * M * sizeof(dst_t));
    add(reg_dst, r12);
    tilestored(ptr[reg_dst + rax], tmm0);
    tilestored(ptr[reg_dst + rax + TILE_N * sizeof(dst_t)], tmm1);
    tilestored(ptr[reg_dst + rax + TILE_N * sizeof(dst_t) * 2], tmm2);
    tilestored(ptr[reg_dst + rax + TILE_N * sizeof(dst_t) * 3], tmm3);
    sub(reg_dst, r12);
  }
}

void jit_spmm_amx_bf16_x16_t::loop_N() {
  dim_t mstart = 0;
  main_compute(mstart);
}

void jit_spmm_amx_bf16_x16_t::init_param() {
  mov(reg_K, K);
  mov(reg_N, N);
  mov(reg_mstart, 0);
  mov(reg_nstart, 0);
  mov(reg_temp, loopMask);
  vmovups(reg_mask, zword[reg_temp]);
}

void jit_spmm_amx_bf16_x16_t::generate() {
  {
    sub(rsp, stack_space_needed_);

    mov(ptr[rsp + 0x00], rbx);
    mov(ptr[rsp + 0x08], rbp);
    mov(ptr[rsp + 0x10], r12);
    mov(ptr[rsp + 0x18], r13);
    mov(ptr[rsp + 0x20], r14);
    mov(ptr[rsp + 0x28], r15);

    read_inputs();
    init_param();
    loop_N();

    mov(rbx, ptr[rsp + 0x00]);
    mov(rbp, ptr[rsp + 0x08]);
    mov(r12, ptr[rsp + 0x10]);
    mov(r13, ptr[rsp + 0x18]);
    mov(r14, ptr[rsp + 0x20]);
    mov(r15, ptr[rsp + 0x28]);

    add(rsp, stack_space_needed_);

    ret();
  }
  align(64);
  L(loopMask);
  int num = 32;
  int wordlen = 2;
  const src_t mask[32] = {0, 16, 1, 17, 2,  18, 3,  19, 4,  20, 5,  21, 6,  22, 7,  23,
                          8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
  for (int i = 0; i < num; ++i) {
    db(mask[i], wordlen);
  }
}
}  // namespace jd
