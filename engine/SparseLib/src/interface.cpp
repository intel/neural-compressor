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

#include "interface.hpp"

namespace jd {
kernel_desc_proxy::kernel_desc_proxy(const operator_desc& op_desc) {
  std::shared_ptr<const kernel_desc_t> result = nullptr;
  auto status = create_proxy_object(result, op_desc);
  reset_sp(result);
}

bool kernel_desc_proxy::create_proxy_object(std::shared_ptr<const kernel_desc_t>& result_ref,
                                            const operator_desc& op_desc) {
  // Step 1: Get the pd (or kernel_desc_t) if it's in the cache.
  auto& global_primitive_cache = kernel_cache::instance();
  std::shared_ptr<const kernel_desc_t> candidate_kd = global_primitive_cache.get_kd(op_desc);
  if (candidate_kd != nullptr) {
    result_ref = candidate_kd;
    return true;
  }

  // Step 2.1: get impl_list_
  const auto& eng_kind = op_desc.engine_kind();
  const engine* eng = engine_factory::instance().create(eng_kind);
  impl_list_ = eng->get_implementation_list(op_desc);
  if (impl_list_ == nullptr) {
    return false;
  }
  // Step 2.2: Get the first && success object in impl_list_.
  auto& impl_list = (*impl_list_);
  for (int i = 0; i < impl_list.size(); ++i) {
    candidate_kd = nullptr;
    auto status = impl_list[i](candidate_kd, op_desc);  // kd->create() + kd->init()
    if (status) {
      break;
    }
  }
  result_ref = candidate_kd;
  return true;
}

kernel_proxy::kernel_proxy(const kernel_desc_proxy& kdp) {
  std::shared_ptr<const kernel_t> result = nullptr;
  auto status = create_proxy_object(result, kdp.get_sp());
  reset_sp(result);
}

bool kernel_proxy::create_proxy_object(std::shared_ptr<const kernel_t>& result_ref,
                                       const std::shared_ptr<const kernel_desc_t>& kd) {
  auto& global_primitive_cache = kernel_cache::instance();
  const auto& callback = std::bind(&kernel_desc_t::create_primitive, kd, std::placeholders::_1,
                                   kd);  // k_t->create() + k_t->init()
  std::shared_ptr<const kernel_t> value = global_primitive_cache.find_or_construct(kd->operator_desc(), callback);
  if (value == nullptr) {
    return false;
  }
  result_ref = value;
  return true;
}

void kernel_proxy::execute(const std::vector<const void*>& rt_data) {
  auto status = get_sp()->execute(rt_data);
  return;
}
}  // namespace jd
