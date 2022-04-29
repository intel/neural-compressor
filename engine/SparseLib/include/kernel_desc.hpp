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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNEL_DESC_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNEL_DESC_HPP_
#include <omp.h>
#include <vector>
#include <cstdint>
#include <memory>
#include "param_types.hpp"
#include "operator_desc.hpp"

namespace jd {
class engine;
class kernel_t;
/**
 * @brief kernel descriptor implementation real class.
 */
class kernel_desc_t {
 public:
  kernel_desc_t() {}
  explicit kernel_desc_t(const jd::kernel_kind& ker_kind);
  virtual ~kernel_desc_t() {}

 public:
  // Self-created API, provided for external users to call.
  template <typename derived_kd_t>
  static bool create(std::shared_ptr<const kernel_desc_t>& kd_ref, const operator_desc& op_desc) {  // NOLINT
    std::shared_ptr<derived_kd_t> derived_kd = std::make_shared<derived_kd_t>(op_desc);
    if (derived_kd == nullptr) {
      return false;
    }
    auto status = derived_kd->init();
    if (!status) {
      derived_kd.reset();  // desc failed and destroy.
      return false;
    }
    kd_ref = derived_kd;
    return true;
  }
  // init kernel_desc_t
  virtual bool init() = 0;
  virtual bool create_primitive(std::shared_ptr<const kernel_t>& k_ref,  // NOLINT
    const std::shared_ptr<const kernel_desc_t>& kd) const = 0;

 public:
  virtual const jd::operator_desc& operator_desc() const = 0;
  inline const jd::kernel_kind& kernel_kind() const { return ker_kind_; }

 protected:
  jd::kernel_kind ker_kind_;
};

// kernel_desc_t::create_primitive() override.
#define DECLARE_COMMON_PD_T(derived_k_t, derived_kd_t, ...) \
  bool create_primitive(std::shared_ptr<const kernel_t>& k_ref, \
    const std::shared_ptr<const kernel_desc_t>& kd) const override { \
    return kernel_t::create<derived_k_t, derived_kd_t>(k_ref, kd); \
  }

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNEL_DESC_HPP_
