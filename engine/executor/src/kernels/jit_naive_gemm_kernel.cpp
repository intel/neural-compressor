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

#include "jit_naive_gemm_kernel.hpp"

namespace executor {

jit_naive_gemm_kernel::~jit_naive_gemm_kernel() = default;

void jit_naive_gemm_kernel::generate() {
  this->preamble();

  mov(reg_src, ptr[reg_param + GET_OFF(src)]);
  mov(reg_wgt, ptr[reg_param + GET_OFF(weight)]);
  mov(reg_dst, ptr[reg_param + GET_OFF(dst)]);

  // Init accumulator registers
  for (int n = 0; n < nb_; n++) {
    for (int m = 0; m < m_block_; m++) {
      vpxord(accum(m, n), accum(m, n), accum(m, n));
    }
  }
  // Computations
  for (int k = 0; k < k_block_; k++) {
    for (int n = 0; n < nb_; n++) {
      vmovups(load(n), EVEX_compress_addr(reg_wgt, sizeof(float) * weight_offset(k, n)));
    }
    for (int m = 0; m < m_block_; m++) {
      vbroadcastss(bcst(), ptr[reg_src + sizeof(float) * src_offset(m, k)]);
      for (int n = 0; n < nb_; n++) {
        vfmadd231ps(accum(m, n), load(n), bcst());
      }
    }
  }
  // Store result
  for (int n = 0; n < nb_; n++) {
    for (int m = 0; m < m_block_; m++) {
      vmovups(EVEX_compress_addr(reg_dst, sizeof(float) * dst_offset(m, n)), accum(m, n));
    }
  }

  this->postamble();
}

}  // namespace executor
