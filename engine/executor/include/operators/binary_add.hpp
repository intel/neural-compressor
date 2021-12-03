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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_BINARY_ADD_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_BINARY_ADD_HPP_
#include <unordered_map>
#include <vector>

#include "../operator.hpp"
#include "oneapi/dnnl/dnnl.hpp"

namespace executor {
using dnnl::algorithm;
using dnnl::engine;
using dnnl::memory;
using dnnl::prop_kind;

/**
 * @brief A BinaryAdd operator.
 *
 */

class BinaryAddOperator : public Operator {
 public:
  explicit BinaryAddOperator(const OperatorConfig& conf);
  virtual ~BinaryAddOperator() {}

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  bool append_sum_;
  dnnl::engine eng_ = engine(engine::kind::cpu, 0);
  dnnl::binary::primitive_desc binary_pd_;
  dnnl::binary binary_p_;
  memory user_src0_m_;
  memory user_src1_m_;
  memory user_dst_m_;
  unordered_map<int, memory> memory_args_;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_BINARY_ADD_HPP_
