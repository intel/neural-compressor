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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_DEFAULT_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_DEFAULT_HPP_
#include <vector>
#include <memory>
#include "operator_desc.hpp"
#include "kernel_desc.hpp"
#include "kernel.hpp"
#include "utils.hpp"
#include "kernels/spmm_types.hpp"
#include "kernels/sparse_data.hpp"
#include "jit_domain/jit_spmm_default.hpp"

namespace jd {
// By convention,
//   1. xxxx_kd_t is the descriptor of a specific derived primitive/kernel.
//   2. xxxx_k_t is a specific derived primitive/kernel.
//   3. jit_xxxx_t is JIT assembly implementation of a specific derived primitive/kernel.
//   where, "xxxx" represents an algorithm, such as brgemm, GEMM and so on.
class spmm_default_k_t;
/**
 * @brief a derived kernel descriptor. flat_param_t is its class member.
 */
class spmm_default_kd_t : public kernel_desc_t {
 public:
  explicit spmm_default_kd_t(const jd::operator_desc& op_desc)
    : kernel_desc_t(kernel_kind::sparse_matmul), op_desc_(op_desc) {}
  virtual ~spmm_default_kd_t() {}

 public:
  bool init() override;
  // kernel_desc_t::create_primitive() override.
  DECLARE_COMMON_PD_T(spmm_default_k_t, spmm_default_kd_t);

 public:
  const jd::operator_desc& operator_desc() const override { return op_desc_; }
  const std::vector<ssd::flat_param_t>& params() const { return params_; }

 private:
  bool spmm_params_init(ssd::flat_param_t& param_ref, const jd::operator_desc& op_desc, int nthr, int ithr);  // NOLINT
  std::vector<int64_t> get_avg_group(const csrp_data_t<int8_t>* sparse_ptr, int nthr, int ithr);

 private:
  jd::operator_desc op_desc_;
  std::vector<ssd::flat_param_t> params_;
};

/**
 * @brief a derived kernel. kd_t and jit_domain are its class members.
 */
class spmm_default_k_t : public kernel_t {
 public:
  using kd_t = spmm_default_kd_t;
  explicit spmm_default_k_t(const std::shared_ptr<const kd_t>& kd) : kernel_t(kd) {}
  virtual ~spmm_default_k_t() {}

 public:
  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;

 public:
  const std::shared_ptr<const kd_t> derived_kd() const {
    return std::static_pointer_cast<const kd_t>(kd_);
  }

 private:
  bool spmm_kernel_create(jit_spmm_default_t** ker_pp, const ssd::flat_param_t& param);

 private:
  std::vector<jit_spmm_default_t*> jit_kers_;
  int64_t nthr_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_SPMM_DEFAULT_HPP_
