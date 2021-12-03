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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_INPUT_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_INPUT_HPP_

#include <vector>

#include "operator.hpp"

namespace executor {

/**
 * @brief Provides data to the Model by assigning output directly.
 *
 * This Input operator is a container that merely holds the data assigned to it;
 */

class InputOperator : public Operator {
 public:
  explicit InputOperator(const OperatorConfig& config) : Operator(config) {}

  // Input operator only have no output, do nothing with Reshape .
  virtual void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {}

  // Input operator's output is assigned outside, Forward do nothing.
  virtual void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {}
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_INPUT_HPP_
