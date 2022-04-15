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

#include "kernels/spmm_sparsednn.hpp"

namespace jd {
//// Part1: class spmm_sparsednn_kd_t
bool spmm_sparsednn_kd_t::init() {
  using dt = jd::data_type;
  const auto& wei_cfg = op_cfg_.tensor_cfgs()[ssd::WEI];
  const auto& src_cfg = op_cfg_.tensor_cfgs()[ssd::SRC];
  const auto& bias_cfg = op_cfg_.tensor_cfgs()[ssd::BIAS];
  const auto& dst_cfg = op_cfg_.tensor_cfgs()[ssd::DST];
  bool has_bias = !bias_cfg.shape().empty();
  bool is_supported = (op_cfg_.kernel_hypotype() == kernel_hypotype::spmm_sd) &&
    is_any_of({dt::s8, dt::fp32}, [&](const dt& a){ return wei_cfg.dtype() == a; }) &&
    is_any_of({dt::u8, dt::fp32}, [&](const dt& a){ return src_cfg.dtype() == a; }) &&
    (!has_bias || is_any_of({dt::s32, dt::fp32}, [&](const dt& a){ return bias_cfg.dtype() == a; })) &&
    is_any_of({dt::s8, dt::fp32}, [&](const dt& a){ return dst_cfg.dtype() == a; });
  if (!is_supported) {
    return false;
  }
  if (wei_cfg.shape().back() != src_cfg.shape().front()) {
    return false;
  }

  int nthr = op_cfg_.impl_nthr();
  params_.resize(nthr);
  for (int idx = 0; idx < nthr; ++idx) {
    ssd::flat_param_t& param = params_[idx];
    spmm_params_init(param, op_cfg_);
  }
  return true;
}

bool spmm_sparsednn_kd_t::spmm_params_init(ssd::flat_param_t& param_ref,
  const operator_config& op_cfg) {
  const auto& wei_cfg = op_cfg.tensor_cfgs()[ssd::WEI];
  const auto& src_cfg = op_cfg.tensor_cfgs()[ssd::SRC];
  const auto& bias_cfg = op_cfg.tensor_cfgs()[ssd::BIAS];
  param_ref.M = wei_cfg.shape()[0];
  param_ref.K = wei_cfg.shape()[1];
  param_ref.N = src_cfg.shape()[1];
  param_ref.has_bias = !bias_cfg.shape().empty();
  auto op_attrs = op_cfg.attrs();
  param_ref.append_sum = (op_attrs["append_sum"] == "true");
  const auto& temp1 = split_str<int64_t>(op_attrs["mkn_blocks"]);
  param_ref.mkn_blocks = temp1.empty() ? std::vector<int64_t>{1, 1, 1} : temp1;
  const auto& temp2 = split_str<int64_t>(op_attrs["tile_shape"]);
  param_ref.tile_shape = temp2.empty() ? std::vector<int64_t>{4, 4} : temp2;
  param_ref.start = 0;
  param_ref.end = param_ref.mkn_blocks[2];
  const auto& temp_addr = str_to_num<uint64_t>(op_attrs["sparse_ptr"]);
  param_ref.sparse_ptr = reinterpret_cast<csrp_data_t<int8_t>*>(temp_addr);
  param_ref.sub_iperm = param_ref.sparse_ptr->iperm();
  return true;
}

//// Part2: class spmm_sparsednn_k_t
bool spmm_sparsednn_k_t::init() {
  int nthr = kd()->operator_cfg().impl_nthr();
  jit_kers_.resize(nthr);
  for (int idx = 0; idx < nthr; ++idx) {
    jit_spmm_sparsednn_t* ker = nullptr;
    auto status = spmm_kernel_create(&ker, derived_kd()->params()[idx]);
    if (!status) {
      return false;
    }
    jit_kers_[idx] = ker;
  }
  return true;
}

bool spmm_sparsednn_k_t::spmm_kernel_create(jit_spmm_sparsednn_t** ker_pp, const ssd::flat_param_t& param) {
  *ker_pp = new jit_spmm_sparsednn_t(param);
  if (*ker_pp == nullptr) {
    return false;
  }
  return (*ker_pp)->create_kernel();
}

bool spmm_sparsednn_k_t::execute(const std::vector<const void*>& rt_data) const {
  int nthr = kd()->operator_cfg().impl_nthr();
  std::vector<ssd::flat_data_t> td(nthr);
  #pragma omp parallel for num_threads(nthr)
  for (int idx = nthr - 1; idx >= 0; --idx) {
    const auto& spmm_sd_kernel = jit_kers_[idx];
    const auto& param = derived_kd()->params()[idx];
    td[idx].ptr_seq_vals = spmm_sd_kernel->sequence_vals();
    td[idx].ptr_dense = rt_data[ssd::SRC];
    td[idx].ptr_bias = rt_data[ssd::BIAS];
    td[idx].ptr_dst = const_cast<void*>(rt_data[ssd::DST]);
    td[idx].ptr_scales = rt_data[ssd::SCALES];
    td[idx].start = param.start;
    td[idx].end = param.end;
    (*spmm_sd_kernel)(td.data());
  }
  return true;
}
}  // namespace jd
