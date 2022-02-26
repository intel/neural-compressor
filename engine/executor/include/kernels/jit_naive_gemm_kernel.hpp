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

#ifndef ENGINE_EXECUTOR_INCLUDE_KERNELS_JIT_NAIVE_GEMM_KERNEL_HPP_
#define ENGINE_EXECUTOR_INCLUDE_KERNELS_JIT_NAIVE_GEMM_KERNEL_HPP_

#include "kernels/lib_generator.hpp"

namespace executor {

typedef struct naive_gemm_params {
  const void* src;
  const void* weight;
  void* dst;
} naive_gemm_params_t;
#define GET_OFF(field) offsetof(naive_gemm_params_t, field)

struct jit_naive_gemm_kernel : public LibGenerator {
  jit_naive_gemm_kernel(const int m_block, const int n_block, const int k_block)
      : LibGenerator(MAX_CODE_SIZE, nullptr, true), m_block_(m_block), n_block_(n_block), k_block_(k_block) {
    simd_w_ = 16;
    unroll_factor_ = m_block;
    if (n_block_ % simd_w_ || m_block_ > unroll_factor_) return;
    nb_ = n_block_ / simd_w_;
    if (nb_ > 4) return;
  }

  ~jit_naive_gemm_kernel();

  void generate();

 private:
  int nb_;
  int m_block_;
  int n_block_;
  int k_block_;

  int simd_w_;
  int unroll_factor_;

  using reg64_t = const Xbyak::Reg64;
  reg64_t reg_param = rdi;  // Always mimic the Unix ABI
  reg64_t reg_src = r15;
  reg64_t reg_wgt = r14;
  reg64_t reg_dst = r13;

  Xbyak::Zmm accum(int m, int n) { return Xbyak::Zmm(m * 4 + n); }
  Xbyak::Zmm load(int n) { return Xbyak::Zmm(unroll_factor_ * 4 + n); }
  // Xbyak::Zmm load(int n) { return Xbyak::Zmm(n);  }
  Xbyak::Zmm bcst() { return Xbyak::Zmm(31); }

  int weight_offset(int k, int n) { return k * n_block_ + n * simd_w_; }
  int src_offset(int m, int k) { return m * k_block_ + k; }
  int dst_offset(int m, int n) { return m * n_block_ + n * simd_w_; }
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_KERNELS_JIT_NAIVE_GEMM_KERNEL_HPP_
