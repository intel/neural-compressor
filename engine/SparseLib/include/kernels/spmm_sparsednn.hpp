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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_SPARSEDNN_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_SPARSEDNN_HPP_
#include <vector>
#include <memory>
#include "operator_config.hpp"
#include "kernel_desc.hpp"
#include "kernel_framework.hpp"
#include "utils.hpp"
#include "kernels/spmm_types.hpp"
#include "kernels/sparse_data.hpp"
#include "jit_domain/jit_spmm_sparsednn.hpp"

namespace jd {
// By convention,
//   1. xxxx_kd_t is the descriptor of a specific derived primitive/kernel.
//   2. xxxx_k_t is a specific derived primitive/kernel.
//   3. jit_xxxx_t is JIT assembly implementation of a specific derived primitive/kernel.
//   where, "xxxx" represents an algorithm, such as brgemm, GEMM and so on.
class spmm_sparsednn_k_t;
/**
 * @brief a derived kernel descriptor. flat_param_t is its class member.
 */
class spmm_sparsednn_kd_t : public kernel_desc_t {
 public:
  explicit spmm_sparsednn_kd_t(const operator_config& op_cfg)
    : kernel_desc_t(kernel_kind::sparse_matmul), op_cfg_(op_cfg) {}
  virtual ~spmm_sparsednn_kd_t() {}

 public:
  bool init() override;
  // kernel_desc_t::create_primitive() override.
  DECLARE_COMMON_PD_T(spmm_sparsednn_k_t, spmm_sparsednn_kd_t);

 public:
  const operator_config& operator_cfg() const override { return op_cfg_; }
  const std::vector<ssd::flat_param_t>& params() const { return params_; }

 private:
  bool spmm_params_init(ssd::flat_param_t& param_ref, const operator_config& op_cfg);  // NOLINT

 private:
  operator_config op_cfg_;
  std::vector<ssd::flat_param_t> params_;
};

/**
 * @brief a derived kernel. kd_t and jit_domain are its class members.
 */
class spmm_sparsednn_k_t : public kernel_framework_t {
 public:
  using kd_t = spmm_sparsednn_kd_t;
  explicit spmm_sparsednn_k_t(const std::shared_ptr<const kd_t>& kd) : kernel_framework_t(kd) {}
  virtual ~spmm_sparsednn_k_t() {}

 public:
  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;

 public:
  const std::shared_ptr<const kd_t> derived_kd() const {
    return std::static_pointer_cast<const kd_t>(kd_);
  }

 private:
  bool spmm_kernel_create(jit_spmm_sparsednn_t** ker_pp, const ssd::flat_param_t& param);

 private:
  std::vector<jit_spmm_sparsednn_t*> jit_kers_;
  int64_t nthr_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_SPARSEDNN_HPP_
