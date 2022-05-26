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

#include "kernels/spmm_amx_bf16_x16.hpp"

namespace jd {
//// Part1: class spmm_amx_bf16_x16_kd_t
bool spmm_amx_bf16_x16_kd_t::init() {
  using dt = jd::data_type;
  const auto& wei_desc = op_desc_.tensor_descs()[ssd::WEI];
  const auto& src_desc = op_desc_.tensor_descs()[ssd::SRC];
  const auto& bias_desc = op_desc_.tensor_descs()[ssd::BIAS];
  const auto& dst_desc = op_desc_.tensor_descs()[ssd::DST];
  bool has_bias = !bias_desc.shape().empty();
  // TBD(hengyu): int8 support
  bool is_supported =
      (op_desc_.kernel_prop() == kernel_prop::forward_inference) &&
      is_any_of({dt::bf16}, [&](const dt& a) { return wei_desc.dtype() == a; }) &&
      is_any_of({dt::bf16}, [&](const dt& a) { return src_desc.dtype() == a; }) &&
      (!has_bias || is_any_of({dt::bf16, dt::fp32}, [&](const dt& a) { return bias_desc.dtype() == a; })) &&
      is_any_of({dt::bf16, dt::fp32}, [&](const dt& a) { return dst_desc.dtype() == a; });
  if (!is_supported) {
    return false;
  }
  if (wei_desc.shape().back() != src_desc.shape().front()) {
    return false;
  }

  return spmm_params_init(params_, op_desc_);
}

bool spmm_amx_bf16_x16_kd_t::spmm_params_init(ssd::amx_bf16_params_t& param_ref, const jd::operator_desc& op_desc) {
  const auto& wei_desc = op_desc.tensor_descs()[0];
  const auto& src_desc = op_desc.tensor_descs()[1];
  const auto& dst_desc = op_desc.tensor_descs()[3];
  param_ref.bs = src_desc.shape()[1];
  param_ref.shape[0] = wei_desc.shape()[0];
  param_ref.shape[1] = wei_desc.shape()[1];
  param_ref.nrowptr = wei_desc.shape()[1] + 1;
  // param_ref.has_bias = !bias_desc.shape().empty();
  auto op_attrs = op_desc.attrs();
  const auto& temp_addr = str_to_num<uint64_t>(op_attrs["sparse_ptr"]);
  const auto& bsr_data = reinterpret_cast<bsr_data_t<bfloat16_t>*>(temp_addr);
  param_ref.nnz_group = bsr_data->nnz_group();
  param_ref.nrowptr = bsr_data->indptr().size();
  param_ref.colidxs = const_cast<dim_t*>(bsr_data->indices().data());
  param_ref.group_rowptr = const_cast<dim_t*>(bsr_data->indptr().data());
  param_ref.weight = const_cast<bfloat16_t*>(bsr_data->data().data());
  return true;
}

//// Part2: class spmm_amx_bf16_x16_k_t
bool spmm_amx_bf16_x16_k_t::init() {
  if (!init_amx()) return false;
  jit_spmm_amx_bf16_x16_t* ker = nullptr;
  bool status = spmm_kernel_create(&ker, derived_kd()->params());
  if (!status) return false;
  jit_kers_ = ker;
  amx_config_ = amx_tile_config_t::GetInstance();
  return true;
}

bool spmm_amx_bf16_x16_k_t::spmm_kernel_create(jit_spmm_amx_bf16_x16_t** ker_pp, const ssd::amx_bf16_params_t& param) {
  *ker_pp = new jit_spmm_amx_bf16_x16_t(param);
  if (*ker_pp == nullptr) {
    return false;
  }
  return (*ker_pp)->create_kernel();
}

bool spmm_amx_bf16_x16_k_t::execute(const std::vector<const void*>& rt_data) const {
  dim_t bs = derived_kd()->params().bs;
  bfloat16_t* weight = derived_kd()->params().weight;
  #pragma omp parallel for
  for (dim_t micro_bs = 0; micro_bs < bs; micro_bs += tileM) {
    amx_config_->amx_tile_configure(tile_param_);
    jd::ssd::amx_bf16f32_inputs_t inputs;
    inputs.weight = weight;
    inputs.src = static_cast<bfloat16_t*>(const_cast<void*>(rt_data[1]));
    inputs.dst = static_cast<float*>(const_cast<void*>(rt_data[3]));
    inputs.bs = tileM;
    (*jit_kers_)(inputs);
  }
  return true;
}
}  // namespace jd
