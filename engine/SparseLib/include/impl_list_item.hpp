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

#ifndef ENGINE_SPARSELIB_INCLUDE_IMPL_LIST_ITEM_HPP_
#define ENGINE_SPARSELIB_INCLUDE_IMPL_LIST_ITEM_HPP_
#include <memory>
#include "param_types.hpp"
#include "operator_desc.hpp"
#include "kernel_desc.hpp"

namespace jd {
template <typename T>
struct type_deduction_helper_t {
  type_deduction_helper_t() {}
  ~type_deduction_helper_t() {}
  using type = T;
};

class impl_list_item_t {
 public:
  impl_list_item_t() {}
  explicit impl_list_item_t(std::nullptr_t nu) {}
  virtual ~impl_list_item_t() {}

 public:
  template <typename derived_kd_t>
  explicit impl_list_item_t(const type_deduction_helper_t<derived_kd_t> a) {
    using deduced_t = typename type_deduction_helper_t<derived_kd_t>::type;
    create_kd_func_ = &kernel_desc_t::create<deduced_t>;
  }

 public:
  bool operator()(std::shared_ptr<const kernel_desc_t>& kd_ref, const operator_desc& op_desc) const {  // NOLINT
    if (create_kd_func_ == nullptr) {
      return false;
    }
    auto status = create_kd_func_(kd_ref, op_desc);
    return status;
  }

 private:
  using create_kd_func_t = bool (*)(std::shared_ptr<const kernel_desc_t>&, const operator_desc&);
  create_kd_func_t create_kd_func_ = nullptr;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_IMPL_LIST_ITEM_HPP_
