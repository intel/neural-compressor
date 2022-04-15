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

#ifndef ENGINE_SPARSELIB_INCLUDE_OPERATOR_CONFIG_HPP_
#define ENGINE_SPARSELIB_INCLUDE_OPERATOR_CONFIG_HPP_
#include <omp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "param_types.hpp"
#include "tensor_config.hpp"

namespace jd {
/**
 * @brief The kernel/operator config class, describing a specific kind of kernel/operator.
 */
class operator_config {
 public:
  operator_config() : ker_kind_(jd::kernel_kind::undef), ker_hypotype_(jd::kernel_hypotype::undef),
    eng_kind_(jd::engine_kind::undef), impl_nthr_(0), tensor_cfgs_({}), attrs_({}) {}
  operator_config(const jd::kernel_kind& ker_kind, const jd::kernel_hypotype& ker_hypotype,
    const jd::engine_kind& eng_kind, const std::vector<tensor_config>& tensor_cfgs,
    const std::unordered_map<std::string, std::string>& attrs)
  : ker_kind_(ker_kind), ker_hypotype_(ker_hypotype), eng_kind_(eng_kind),
    impl_nthr_((omp_get_max_threads() == omp_get_num_procs()) ? 1 : omp_get_max_threads()),
    tensor_cfgs_(tensor_cfgs), attrs_(attrs) {}
  virtual ~operator_config() {}

 public:
  bool operator==(const operator_config& rhs) const {
    return (ker_kind_ == rhs.ker_kind_) && (ker_hypotype_ == rhs.ker_hypotype_) &&
      (eng_kind_ == rhs.eng_kind_) && (impl_nthr_ == rhs.impl_nthr_) &&
      (tensor_cfgs_ == rhs.tensor_cfgs_) && (attrs_ == rhs.attrs_);
  }

 public:
  inline const jd::kernel_kind& kernel_kind() const { return ker_kind_; }
  inline const jd::kernel_hypotype& kernel_hypotype() const { return ker_hypotype_; }
  inline const jd::engine_kind& engine_kind() const { return eng_kind_; }
  inline const uint64_t& impl_nthr() const { return impl_nthr_; }
  inline const std::vector<tensor_config>& tensor_cfgs() const { return tensor_cfgs_; }
  inline const std::unordered_map<std::string, std::string>& attrs() const { return attrs_; }

 private:
  jd::kernel_kind ker_kind_;
  jd::kernel_hypotype ker_hypotype_;
  jd::engine_kind eng_kind_;
  uint64_t impl_nthr_;
  std::vector<tensor_config> tensor_cfgs_;
  std::unordered_map<std::string, std::string> attrs_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_OPERATOR_CONFIG_HPP_
