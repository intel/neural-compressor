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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_RESHAPE_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_RESHAPE_HPP_
#include <vector>

#include "../common.hpp"
#include "../operator.hpp"
#include "oneapi/dnnl/dnnl.hpp"

namespace executor {

/**
 * @brief A Reshape operator.
 *
 */

class ReshapeOperator : public Operator {
 public:
  explicit ReshapeOperator(const OperatorConfig& conf);
  virtual ~ReshapeOperator();

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

  vector<int64_t> shape_;
  vector<int64_t> dims_;
  vector<int64_t> mul_;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_RESHAPE_HPP_
