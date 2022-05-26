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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_AMX_BF16_X16_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_AMX_BF16_X16_HPP_

#include <memory>
#include <vector>

#include "jit_domain/jit_spmm_amx_bf16_x16.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "kernels/sparse_data.hpp"
#include "kernels/spmm_types.hpp"
#include "operator_desc.hpp"
#include "amx_utils.hpp"

namespace jd {
// By convention,
//   1. xxxx_kd_t is the descriptor of a specific derived primitive/kernel.
//   2. xxxx_k_t is a specific derived primitive/kernel.
//   3. jit_xxxx_t is JIT assembly implementation of a specific derived
//   primitive/kernel. where, "xxxx" represents an algorithm, such as brgemm,
//   GEMM and so on.
class spmm_amx_bf16_x16_k_t;
/**
 * @brief a derived kernel descriptor. amx_bf16_params_t is its class member.
 */
class spmm_amx_bf16_x16_kd_t : public kernel_desc_t {
 public:
  explicit spmm_amx_bf16_x16_kd_t(const jd::operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::sparse_matmul), op_desc_(op_desc) {}
  virtual ~spmm_amx_bf16_x16_kd_t() {}

 public:
  bool init() override;
  // kernel_desc_t::create_primitive() override.
  DECLARE_COMMON_PD_T(spmm_amx_bf16_x16_k_t, spmm_amx_bf16_x16_kd_t);

 public:
  const jd::operator_desc& operator_desc() const override { return op_desc_; }
  const ssd::amx_bf16_params_t& params() const { return params_; }

 private:
  bool spmm_params_init(ssd::amx_bf16_params_t& param_ref, // NOLINT
                        const jd::operator_desc& op_cfg);

 private:
  jd::operator_desc op_desc_;
  ssd::amx_bf16_params_t params_;
};

/**
 * @brief a derived kernel. kd_t and jit_domain are its class members.
 */
class spmm_amx_bf16_x16_k_t : public kernel_t {
 public:
  using kd_t = spmm_amx_bf16_x16_kd_t;
  explicit spmm_amx_bf16_x16_k_t(const std::shared_ptr<const kd_t>& kd) : kernel_t(kd) {}
  virtual ~spmm_amx_bf16_x16_k_t() {}

 public:
  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;
  // bf16 output TBD
  // bool execute(const amx_bf16bf16_inputs_t& rt_data) const override;

 public:
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  bool spmm_kernel_create(jit_spmm_amx_bf16_x16_t** ker_pp, const ssd::amx_bf16_params_t& param);
  dim_t tileM = 64;

 private:
  jit_spmm_amx_bf16_x16_t* jit_kers_;
  const tile_param_t tile_param_ = {16, 16, 32, true, 2};
  amx_tile_config_t* amx_config_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_AMX_BF16_X16_HPP_
