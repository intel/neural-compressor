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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNEL_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNEL_HPP_
#include <memory>
#include <vector>

#include "kernel_desc.hpp"

namespace jd {
/**
 * @brief kernel/primitive implementation real class.
 */
class kernel_t {
 public:
  explicit kernel_t(const std::shared_ptr<const kernel_desc_t>& kd);
  virtual ~kernel_t() {}

 public:
  // Self-created API, provided for external users to call.
  template <typename derived_k_t, typename derived_kd_t>
  static bool create(std::shared_ptr<const kernel_t>& k_ref,  // NOLINT
                     const std::shared_ptr<const kernel_desc_t>& kd) {
    const auto& derived_kd_temp = std::dynamic_pointer_cast<const derived_kd_t>(kd);
    std::shared_ptr<derived_k_t> prim = std::make_shared<derived_k_t>(derived_kd_temp);
    if (prim == nullptr) {
      return false;
    }
    auto status = prim->init();
    if (!status) {
      prim.reset();  // prim failed and destroy.
      return false;
    }
    k_ref = prim;
    return true;
  }
  // init kernel_t
  virtual bool init() = 0;
  virtual bool execute(const std::vector<const void*>& rt_data) const = 0;

 public:
  const std::shared_ptr<const kernel_desc_t>& kd() const { return kd_; }

 protected:
  // kernel_desc_t has no cache management. So use shared_ptr to cache and
  // destruct automatically.
  std::shared_ptr<const kernel_desc_t> kd_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNEL_HPP_
