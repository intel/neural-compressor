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

#ifndef ENGINE_SPARSELIB_INCLUDE_OPERATOR_DESC_HPP_
#define ENGINE_SPARSELIB_INCLUDE_OPERATOR_DESC_HPP_
#include <omp.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "param_types.hpp"
#include "tensor_desc.hpp"

namespace jd {
/**
 * @brief The operator descriptor class, describing a specific kind of operator.
 */
class operator_desc {
 public:
  operator_desc()
      : ker_kind_(jd::kernel_kind::undef),
        ker_prop_(jd::kernel_prop::undef),
        eng_kind_(jd::engine_kind::undef),
        impl_nthr_(0),
        ts_descs_({}),
        attrs_({}) {}
  operator_desc(const jd::kernel_kind& ker_kind, const jd::kernel_prop& ker_prop, const jd::engine_kind& eng_kind,
                const std::vector<tensor_desc>& ts_descs, const std::unordered_map<std::string, std::string>& attrs)
      : ker_kind_(ker_kind),
        ker_prop_(ker_prop),
        eng_kind_(eng_kind),
        impl_nthr_((omp_get_max_threads() == omp_get_num_procs()) ? 1 : omp_get_max_threads()),
        ts_descs_(ts_descs),
        attrs_(attrs) {}
  virtual ~operator_desc() {}

 public:
  bool operator==(const operator_desc& rhs) const {
    return (ker_kind_ == rhs.ker_kind_) && (ker_prop_ == rhs.ker_prop_) && (eng_kind_ == rhs.eng_kind_) &&
           (impl_nthr_ == rhs.impl_nthr_) && (ts_descs_ == rhs.ts_descs_) && (attrs_ == rhs.attrs_);
  }

 public:
  inline const jd::kernel_kind& kernel_kind() const { return ker_kind_; }
  inline const jd::kernel_prop& kernel_prop() const { return ker_prop_; }
  inline const jd::engine_kind& engine_kind() const { return eng_kind_; }
  inline const uint64_t& impl_nthr() const { return impl_nthr_; }
  inline const std::vector<tensor_desc>& tensor_descs() const { return ts_descs_; }
  inline const std::unordered_map<std::string, std::string>& attrs() const { return attrs_; }

 private:
  jd::kernel_kind ker_kind_;
  jd::kernel_prop ker_prop_;
  jd::engine_kind eng_kind_;
  uint64_t impl_nthr_;
  std::vector<tensor_desc> ts_descs_;
  std::unordered_map<std::string, std::string> attrs_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_OPERATOR_DESC_HPP_
