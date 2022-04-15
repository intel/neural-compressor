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

#ifndef ENGINE_SPARSELIB_INCLUDE_ENGINE_HPP_
#define ENGINE_SPARSELIB_INCLUDE_ENGINE_HPP_
#include <vector>
#include "param_types.hpp"
#include "impl_list_item.hpp"

namespace jd {
class engine {
 public:
  explicit engine(const engine_kind& eng_kind) : eng_kind_(eng_kind) {}
  virtual ~engine() {}

 public:
  inline const engine_kind& kind() const { return eng_kind_; }
  virtual const std::vector<impl_list_item_t>* get_implementation_list(const operator_config& op_cfg) const = 0;

 protected:
  engine_kind eng_kind_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_ENGINE_HPP_
