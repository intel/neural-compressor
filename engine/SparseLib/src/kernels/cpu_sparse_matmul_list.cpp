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

#include <map>
#include <tuple>
#include "cpu_engine.hpp"
#include "param_types.hpp"
#include "impl_list_item.hpp"
#include "kernels/spmm_sparsednn.hpp"

namespace jd {
using hypo = jd::kernel_hypotype;
using dt = jd::data_type;
using map_key_t = std::tuple<hypo, dt, dt, dt>;
/**
 * @param kernel_kind, point to this file "cpu_sparse_matmul_list.cpp".
 * @param kernel_hypotype, point to [KEY] of impl_list_map. A specific function or scenario.
 * @param kernel_algorithm, point to [VAL] of impl_list_map. Different algorithms of a
 *   specific function, e.g.: gemm, brgemm.
 * @note Use (kernel_kind->kernel_hypotype->kernel_algorithm) to denote a specific/derived kernel.
 *   Ref onednn's cpu_inner_product_list.cpp, a map's [VAL] is a derived "struct primitive_t",
 *   e.g.: gemm_inner_product_fwd_t<f32>.
 */
static const std::map<map_key_t, std::vector<impl_list_item_t>> impl_list_map = {
  {{hypo::spmm_sd, dt::s8, dt::u8, dt::s8}, {
    CPU_INSTANCE(spmm_sparsednn_k_t),
    NULL_INSTANCE()
  }},
};

const std::vector<impl_list_item_t>* get_sparse_matmul_impl_list(const operator_config& op_cfg) {
  const auto& tensor_cfgs = op_cfg.tensor_cfgs();
  const auto& src0_dtype = tensor_cfgs[0].dtype();
  const auto& src1_dtype = tensor_cfgs[1].dtype();
  const auto& dst_dtype = tensor_cfgs[3].dtype();
  map_key_t key{op_cfg.kernel_hypotype(), src0_dtype, src1_dtype, dst_dtype};
  const auto impl_list_it = impl_list_map.find(key);
  return (impl_list_it != impl_list_map.end()) ? &(impl_list_it->second) : &cpu_engine::empty_list;
}
}  // namespace jd
