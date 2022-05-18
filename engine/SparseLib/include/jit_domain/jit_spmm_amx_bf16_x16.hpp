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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_SPMM_AMX_BF16_X16_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_SPMM_AMX_BF16_X16_HPP_

#include <glog/logging.h>
#include <omp.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../kernels/sparse_data.hpp"
#include "../kernels/spmm_types.hpp"
#include "jit_generator.hpp"
#include "utils.hpp"

namespace jd {
typedef bfloat16_t src_t;
typedef float dst_t;
/**
 * @brief jit_spmm_amx_bf16_x16_t calculates this kind matmul: sparse x dense =
 * dst. weight(N, K) * activation(K, M) + bias(N, 1) = dst(N, M) detailed
 * description please refer to
 * https://gist.github.com/airMeng/020cc034ece43f0ba3d3cf8a2f9ecd0a
 */
class jit_spmm_amx_bf16_x16_t : public jit_generator {
 public:
  explicit jit_spmm_amx_bf16_x16_t(const ssd::amx_bf16_params_t& param) : jit_generator(), param_(param) {
    N = param_.shape[0];
    K = param_.shape[1];
    M = param_.bs;
    nnz_group = param_.nnz_group;
    nrowptr = param_.nrowptr;
    colidxs = param_.colidxs;
    group_rowptr = param_.group_rowptr;
    weight_ = param_.weight;
  }
  virtual ~jit_spmm_amx_bf16_x16_t() {}

 public:
  const bfloat16_t* weight() const { return weight_; }

 private:
  ssd::amx_bf16_params_t param_;

 private:
  void generate() override;

 private:
  const Xbyak::uint8* jit_ker_ = nullptr;

  const Xbyak::Reg64& reg_param = rdi;
  const Xbyak::Reg64& reg_weight = r15;
  const Xbyak::Reg64& reg_src = rdx;
  const Xbyak::Reg64& reg_dst = rcx;
  const Xbyak::Reg64& reg_bs = r8;
  const Xbyak::Reg64& reg_K = rbx;
  const Xbyak::Reg64& reg_N = rbp;
  const Xbyak::Reg64& reg_mstart = r9;
  const Xbyak::Reg64& reg_nstart = r10;
  const Xbyak::Reg64& reg_temp = rsi;
  const Xbyak::Zmm& reg_mask = zmm31;
  Xbyak::Label loopMask;

  dim_t N;
  dim_t K;
  dim_t M;
  const dim_t blocks_per_group = 32;
  dim_t nnz_group;
  dim_t nrowptr;
  dim_t* colidxs;
  dim_t* group_rowptr;
  bfloat16_t* weight_;
  const dim_t tileM = 64;  // 4x16

  static constexpr int stack_space_needed_ = 5120;

  void read_inputs();
  void main_compute(dim_t mstart);
  void loop_N();
  void init_param();
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_SPMM_AMX_BF16_X16_HPP_
